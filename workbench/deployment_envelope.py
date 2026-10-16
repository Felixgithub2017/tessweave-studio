"""Conditional per-rank admission envelope; never a throughput or OOM guarantee."""
import math

GIB = 2**30


def deployment_envelope(body, model, report, single):
    from .resources import number
    cfg = (model or {}).get('config', {})
    d = report['dims']
    selected = report['selected']
    # A target that overflows can still be evaluated on the same fixed hardware
    # using the one-request layout. Never borrow a layout from more GPUs.
    base = report
    if not selected['topology'] and selected['allocated_gpus'] == single['selected']['allocated_gpus']:
        base = single
    topo = base['selected']['topology']
    seq = number(body, 'seq', 4096, 128, 10000000, True)
    concurrency = number(body, 'batch', 1, 1, 65536, True)
    block = number(body, 'kv_block_tokens', 16, 1, 1024, True)
    limit = number(body, 'context_limit', cfg.get('max_position_embeddings') or cfg.get('model_max_length') or 0, 0, 10000000, True)
    from .hybrid_cache import hybrid_shape, coefficients, activation_gib, moe_scratch_bytes
    hybrid=hybrid_shape(cfg)
    from .mla_planning import supported as mla_supported, token_bytes
    mla=mla_supported(cfg)
    from .layered_cache import layers as cache_layers, token_bytes as layered_token_bytes
    layered=cache_layers(cfg)
    if mla:block=max(64,block)
    gdn=bool(hybrid and hybrid.get('kind')=='gdn')
    chunk=number(body,'prefill_chunk_tokens',2048,16,65536,True)
    out = dict(schema='tessweave.deployment-envelope.v1', evidence='analytic_unverified',
               minimum_gpus_one_request=single['selected']['conditional_min_gpus'],
               minimum_gpus_target=selected['conditional_min_gpus'],
               minimum_basis=f'上下文 {seq} tokens；搜索范围内、给定预算与 TP/PP 约束下的条件最小卡数',
               allocated_gpus=selected['allocated_gpus'], context_limit=limit or None,
               memory_concurrency=None, slo_concurrency=None, context_at_target_concurrency=None,
               curve=[], blockers=[], assumptions=[
                   '同构 GPU、均匀权重/层分片、标准全注意力 MHA/GQA/MQA、BF16 KV；不是后端支持认证。',
                   '上下文包含 prompt、历史、生成及处理器展开后的 tokens；并发指驻留序列，不是用户数或每日请求数。',
                   '不计前缀共享收益；每请求按 KV 页向上取整；最忙 rank 决定容量。',
                   '权重使用已有账本，量化元数据为估计；模型加载、图捕获、prefill 峰值仍须实测。',
                   'MLA、混合/滑窗/状态空间、多模态架构不套用标准 KV 公式。',
                   '上下文配置上限不是长上下文质量认证；内存上限不等于满足延迟目标的并发。'],
               operations=[
                   '准入：校验输入+最大输出≤上下文限制；限制驻留序列和总预留 tokens，有限队列，过载返回 429/503。',
                   'OOM：分开测加载/图捕获/prefill/decode；降低 prefill token 批量、Graph 桶或并发；量化须质量评估。',
                   '可用性：权重加载和探测生成成功后才 readiness；故障摘除、跨节点副本、预热和滚动升级。',
                   '重试：有截止时间与次数上限；已输出部分 SSE 不透明重试；取消与超时及时回收 KV。',
                   '观测：队列等待、TTFT/ITL、p95/p99、错误率、KV 使用/驱逐、峰值显存；演练节点故障。'])
    from .deployment_recipes import assessment
    unsupported = not assessment(model)['analytic_supported']
    if hybrid:
        out['assumptions']=[
            'Kimi Linear专用：BF16压缩MLA按token计且TP复制；FP32 KDA矩阵+BF16卷积历史按请求计且按头TP分片。',
            f'全部MoE专家权重驻留；PP=EP=1；必须分块prefill≤{chunk} tokens，路由/专家临时张量使用代理预算。',
            '不计MTP、前缀状态快照和缓存共享；需关闭这些功能或另加预算。量化仅容量假设，不保证存在兼容产物。',
            '内存并发不是吞吐/SLO；加载、prefill、decode、通信与图捕获需实测；不是TessWeave原生核支持。']
        if gdn:
            out['assumptions'][0]='Qwen Gated DeltaNet专用：仅full_attention层计BF16 KV，按KV头分片；FP32 GDN矩阵+BF16卷积历史按请求计、按头TP分片。'
            out['assumptions'].append('仅文本请求；总检查点权重计入，但图片/视频编码峰值、视觉token展开须另测，不能外推多模态准入。')
    if cfg.get('model_type')=='gemma3_text' and not unsupported:
        out['assumptions'].append('Gemma 3 27B专用保守口径：所有层保留全历史KV，不假定滑窗回收；仅文本请求，图片编码激活未建模。')
    if mla:
        out['assumptions']=[f'MLA压缩latent+RoPE在TP卡间复制；DSA额外索引键/scale按BF16保守预算；页至少64 tokens，逐请求另留一尾页。',f'全部专家权重驻留；分块prefill≤{chunk}；无MTP/投机、无前缀共享；跨节点TP/等层PP只是容量候选，需兼容外部引擎预检。','只提供显存准入估算，不保证吞吐、SLO或量化质量。']
        out['cache_adapter']='mla'
    if unsupported or not topo or not all(d[k] for k in ('layers', 'hidden', 'heads', 'kv_heads')):
        out['blockers'].append('给定GPU数量与预留预算下未找到可容纳布局；请调整上下文、并发或预算，或增加GPU。' if not unsupported and not topo and all(d[k] for k in ('layers','hidden','heads','kv_heads')) else '结构缺失或不是标准全注意力缓存；需架构专用/实测逐 rank 缓存与峰值适配器。')
        out['minimum_gpus_one_request'] = out['minimum_gpus_target'] = None
        return out
    head = cfg.get('head_dim') or (d['hidden'] // d['heads'] if d['hidden'] % d['heads'] == 0 else 0)
    tp, pp = topo['tp'], topo['pp']
    # Compressed MLA has its own dimensions; hidden/query_heads need not be
    # integral (e.g. GLM-4.7-Flash). Do not reject it with a standard-GQA guard.
    if d['layers'] % pp or (not mla and (not head or (d['kv_heads'] % tp and tp % d['kv_heads']))):
        out['blockers'].append('层数/头维度/TP 分片不满足当前解析模型，不能推断容量。')
        return out
    rows = base['detailed_budget']['rows']
    fixed = sum(r['per_gpu_gib'] or 0 for r in rows if r['item'] not in ('kv', 'transient', 'activation_extra'))
    usable = report['detailed_budget']['usable_gib']
    act_floor = number(body, 'activation_gib', 2, 0, 100000)
    kv_bytes = 2 * 2 * (d['layers'] // pp) * max(1, d['kv_heads'] // tp) * head
    state_bytes=0
    if hybrid:kv_bytes,state_bytes=coefficients(cfg,tp)
    if mla:kv_bytes,state_bytes=token_bytes(cfg,pp),block*token_bytes(cfg,pp)
    if layered:
        kv_bytes=layered_token_bytes(cfg,tp,pp)
        out['cache_adapter']='layered_full_history'
        out['layered_cache']=dict(layers=layered,pp=pp,tp=tp,scope='text_only_full_history_upper_bound')
        out['assumptions']=['逐层使用各自KV头数及K/V维度；滑窗按全历史预算，不计驱逐、跨层KV共享或K=V复用收益。',
            '等层PP取缓存最大的阶段，不以平均层大小代替；权重仍为均匀分片代理，需逐rank实测。',
            f'分块prefill≤{chunk} tokens；仅文本，无MTP/前缀共享，视觉与音频编码峰值未纳入。',
            '内存容量候选，不代表外部后端TP/PP支持、吞吐或质量已验证。']
    def activation_for(c,s):
        return max(act_floor,activation_gib(cfg,c,s,chunk) if hybrid or mla or layered else c*s*(12*d['hidden']+moe_scratch_bytes(cfg))/GIB)
    def peak(c, s):
        padded = math.ceil(s / block) * block
        kv = c * (padded * kv_bytes+state_bytes) / GIB
        activation = activation_for(c,s)
        return fixed + kv + activation, kv, activation
    def maximum(predicate, ceiling):
        lo, hi = 0, ceiling
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if predicate(mid): lo = mid
            else: hi = mid - 1
        return lo
    for length in sorted(set([512, 2048, 4096, 8192, 32768, 131072, seq])):
        c = maximum(lambda n: peak(n, length)[0] <= usable, 65536)
        allowed = not limit or length <= limit
        out['curve'].append(dict(context_tokens=length, memory_concurrency=c,
                                 admitted_concurrency=c if allowed and limit else None,
                                 within_context_limit=allowed if limit else None,
                                 search_ceiling_reached=c == 65536))
    c = maximum(lambda n: peak(n, seq)[0] <= usable, 65536)
    memory_len = maximum(lambda s: peak(concurrency, s)[0] <= usable, 10000000)
    total, kv, act = peak(concurrency, seq)
    # Counterfactuals share this exact TP/PP layout. More concurrent residents
    # share weights, not KV. These are memory estimates, never utilization/SLO.
    def memory_row(count, length):
        amount, cache, activation = peak(count, length)
        return dict(concurrency=count, context_tokens=length, fixed_gib=fixed,
                    kv_gib=cache, activation_gib=activation, peak_gib=amount,
                    extra_vs_one_gib=amount-peak(1, length)[0],
                    usable_gib=usable, fits_memory=amount <= usable,
                    within_context_limit=length <= limit if limit else None)
    out['concurrency_memory'] = [memory_row(n, seq) for n in sorted({1, 2, 4, 8, 16, 32, concurrency})]
    mix=body.get('request_mix','')
    if mix:
        if not isinstance(mix,str) or len(mix)>2048:raise ValueError('混合请求格式：总token数:请求数，以逗号分隔')
        groups=[]
        for entry in mix.replace('，',',').split(','):
            parts=entry.strip().split(':')
            if len(parts)!=2:raise ValueError('混合请求格式：2048:8,8192:2')
            length=number({'n':parts[0]},'n',0,1,10000000,True)
            count=number({'n':parts[1]},'n',0,1,65536,True)
            groups.append(dict(context_tokens=length,concurrency=count,padded_tokens=math.ceil(length/block)*block))
        residents=sum(g['concurrency'] for g in groups)
        if residents>65536:raise ValueError('混合请求总数不能超过65536')
        reserved=sum(g['padded_tokens']*g['concurrency'] for g in groups)
        tokens=sum(g['context_tokens']*g['concurrency'] for g in groups)
        # Sum independently rounded pages, NOT padded mean length. Reserve the
        # full prompt+maximum-output per request, with no speculative sharing.
        cache=(reserved*kv_bytes+residents*state_bytes)/GIB
        activation=activation_for(1,tokens)
        repeats=maximum(lambda n: fixed+n*cache+activation_for(1,n*tokens)<=usable,65536//residents)
        allowed=all(g['context_tokens']<=limit for g in groups) if limit else None
        out['mixed_requests']=dict(groups=groups,concurrency=residents,reserved_tokens=reserved,
            fixed_gib=fixed,kv_gib=cache,activation_gib=activation,peak_gib=fixed+cache+activation,
            usable_gib=usable,fits_memory=fixed+cache+activation<=usable,within_context_limit=allowed,
            maximum_whole_groups=repeats,maximum_concurrency_same_mix=repeats*residents,
            admitted_concurrency_same_mix=repeats*residents if allowed else None,
            search_ceiling_reached=repeats==65536//residents,
            uniform_max_length_kv_gib=peak(residents,max(g['context_tokens'] for g in groups))[1],
            formula='KV = Σ(请求数 × ceil((输入+最大输出)/页大小) × 页大小) × 每rank每token字节；权重只计一次')
    declared = cfg.get('max_position_embeddings') or cfg.get('model_max_length')
    out['declared_context'] = memory_row(concurrency, declared) if isinstance(declared, int) and not isinstance(declared, bool) and 0 < declared <= 10000000 else None
    out['comparison_layout'] = dict(tp=tp, pp=pp, scope='per_rank_fixed_layout_analytic')
    # Counterfactual on this exact TP/PP layout: never compare independently
    # optimized plans, whose GPU count/weight sharding could hide context cost.
    baseline_tokens = 2048
    base_total, base_kv, base_act = peak(concurrency, baseline_tokens)
    out['context_memory_delta'] = dict(
        baseline_tokens=baseline_tokens, target_tokens=seq, concurrency=concurrency,
        tp=tp, pp=pp, kv_precision='BF16 (2 bytes)', block_tokens=block,
        baseline_kv_gib=base_kv, target_kv_gib=kv, kv_delta_gib=kv-base_kv,
        activation_delta_gib=act-base_act, peak_delta_gib=total-base_total,
        scope='per_rank_fixed_layout_analytic',
        formula=f'{concurrency} × (ceil({seq}/{block}) − ceil({baseline_tokens}/{block})) × {block} × {kv_bytes} / 2^30')
    out.update(memory_concurrency=c, context_memory_bound=memory_len,
               context_at_target_concurrency=min(memory_len, limit) if limit else None,
               target=dict(concurrency=concurrency, context_tokens=seq, peak_gib=total,
                           usable_gib=usable, headroom_gib=usable-total,
                           fits_memory=total <= usable, within_context_limit=seq <= limit if limit else None),
               ledger=[dict(item='权重/运行时固定预算', gib=fixed, formula='原逐卡账本求和，排除 KV 与激活'),
                       dict(item='BF16 KV', gib=kv, formula=f'{concurrency} × ceil({seq}/{block}) × {block} × {kv_bytes} bytes / 2^30'),
                       dict(item='prefill 激活代理', gib=act, formula=f'max({act_floor}, 12 × {concurrency} × {seq} × {d["hidden"]} / 2^30)')],
               kv_bytes_per_token_per_rank=kv_bytes)
    if not limit: out['blockers'].append('模型/后端上下文上限未知；仅给显存边界，不批准上下文配置。')
    if limit and seq > limit: out['blockers'].append('请求上下文超过声明上限；不能靠增加 GPU 自动扩展模型上下文。')
    if total > usable: out['blockers'].append('目标并发×长度超过预算；降低长度/并发或增加模型分片 GPU。')
    if hybrid:
        out['hybrid_cache']=dict(mla_bytes_per_token_per_rank=kv_bytes,kda_bytes_per_request_per_rank=state_bytes,
            mla_gib=concurrency*math.ceil(seq/block)*block*kv_bytes/GIB,kda_gib=concurrency*state_bytes/GIB,
            prefill_chunk_tokens=chunk,pp_supported=False,ep_enabled=False)
        out['hybrid_cache'].update(family='qwen_gdn' if gdn else 'kimi_linear',
            attention_gib=out['hybrid_cache']['mla_gib'],state_gib=out['hybrid_cache']['kda_gib'])
        if gdn:
            for key in ('mla_bytes_per_token_per_rank','kda_bytes_per_request_per_rank','mla_gib','kda_gib'):out['hybrid_cache'].pop(key)
            out['hybrid_cache'].update(kv_bytes_per_token_per_rank=kv_bytes,gdn_bytes_per_request_per_rank=state_bytes)
        out['ledger'][1].update(item='MLA + KDA',formula=f'请求数 × (向上取整tokens × {kv_bytes} + {state_bytes}) bytes / 2^30')
        if gdn:out['ledger'][1]['item']='全注意力KV + GDN'
        out['ledger'][2]['formula']=f'max({act_floor}, 分块prefill/MoE临时张量代理)，chunk={chunk}'
        if out.get('mixed_requests'):out['mixed_requests']['formula']=('全注意力KV' if gdn else 'MLA')+' = Σ(请求数×分页tokens)×每token字节；'+('GDN' if gdn else 'KDA')+' = 驻留请求数×状态字节；权重只计一次'
    if mla:
        out['ledger'][1].update(item='MLA / DSA（含尾页）',formula=f'请求数 × (分页tokens + {block}) × {kv_bytes} bytes / 2^30；TP不分片')
        out['ledger'][2]['formula']=f'max({act_floor}, 分块prefill/MoE临时张量代理)，chunk={chunk}'
        if out.get('mixed_requests'):out['mixed_requests']['formula']='Σ(请求数 × (分页tokens + 尾页)) × 每rank MLA/DSA字节；权重只计一次'
    if layered:
        out['ledger'][1].update(item='逐层KV（全历史上界）',formula=f'请求数 × 分页tokens × 最忙PP阶段每token字节 {kv_bytes} / 2^30')
        out['ledger'][2]['formula']=f'max({act_floor}, 分块prefill/MoE临时张量代理)，chunk={chunk}'
    return out
