"""Inspectable capacity planning with explicit per-rank assumptions, not an OOM oracle."""
import math
from .model_reference import model_identity

GIB = 2**30
GPUS = {
    "4090": {"name":"RTX 4090", "gib":24, "fabric":"PCIe；无 NVLink，先测 P2P / NCCL"},
    "5090": {"name":"RTX 5090", "gib":32, "fabric":"PCIe；无 NVLink，核对驱动与内核支持"},
    "a100": {"name":"A100 80GB SXM", "gib":80, "fabric":"NVLink / NVSwitch（以服务器实际拓扑为准）"},
    "h100": {"name":"H100 80GB SXM", "gib":80, "fabric":"NVLink / NVSwitch（以服务器实际拓扑为准）"},
    "h200": {"name":"H200 141GB SXM", "gib":141, "fabric":"NVLink / NVSwitch（以服务器实际拓扑为准）"},
    "b200": {"name":"B200 180GB", "gib":180, "fabric":"NVLink / NVSwitch（以服务器实际拓扑为准）"},
}


def number(body, key, default, low, high, integer=False):
    raw=body.get(key,default)
    if isinstance(raw,bool):
        raise ValueError(key+" 不能是布尔值")
    try:
        v=float(raw)
    except (ValueError,TypeError):
        raise ValueError(key+" 必须为数字")
    if not math.isfinite(v) or not low<=v<=high or (integer and v!=int(v)):
        raise ValueError(key+" 超出范围")
    return int(v) if integer else v


def plan_resources(body, model=None):
    """Recommend a stage, rather than requiring users to know ZeRO first.

    Objective: smallest feasible allocation, then least state sharding.
    This is a capacity/complexity heuristic, not a throughput prediction.
    A fixed GPU count is evaluated without silently adding more GPUs.
    """
    if body.get('task','train')!='train':
        return _plan_resources(body,model)
    runs=[_plan_resources(dict(body,zero_stage=stage),model) for stage in range(4)]
    def score(candidate,stage):
        if candidate['feasible_under_assumptions']:
            return (0,candidate['allocated_gpus'],stage)
        peak=(candidate.get('topology') or {}).get('peak_gib',float('inf'))
        return (1,peak,stage)
    winners=[]
    for index in range(len(GPUS)):
        stage=min(range(4),key=lambda z:score(runs[z]['candidates'][index],z))
        candidate=dict(runs[stage]['candidates'][index])
        candidate['recommended_zero_stage']=stage if candidate['feasible_under_assumptions'] else None
        winners.append(candidate)
    selected=body.get('gpu','h100')
    winner=next(c for c in winners if c['gpu']==selected)
    chosen=min(range(4),key=lambda z:score(runs[z]['selected'],z))
    result=runs[chosen]
    result['candidates']=winners
    result['selected']=winner
    result['assumptions']=dict(body,zero_stage='auto')
    result['zero_recommendation']={
        'mode':'automatic','stage':chosen if winner['feasible_under_assumptions'] else None,
        'objective':'fewest_cards_then_least_sharding',
        'fixed_hardware':number(body,'gpu_count',0,0,4096,True)>0,
        'alternatives':[{
            'stage':z,'minimum_gpus':run['selected']['conditional_min_gpus'],
            'allocated_gpus':run['selected']['allocated_gpus'],
            'peak_gib':(run['selected']['topology'] or {}).get('peak_gib'),
            'headroom_gib':run['detailed_budget']['headroom_gib'],
            'feasible':run['selected']['feasible_under_assumptions'],
            'reason':run['selected']['reason'],
            'selected':z==chosen and winner['feasible_under_assumptions'],
        } for z,run in enumerate(runs)],
        'throughput_verified':False,
    }
    if not winner['feasible_under_assumptions']:
        result['warnings'].append('所有ZeRO阶段均未满足当前约束；账本仅展示诊断候选，不是推荐可运行方案。')
    from .training_design import design_training
    from .planning_trace import explain
    result['training_design']=explain(design_training(body,result,model))
    from .length_curriculum import length_curriculum
    curriculum=length_curriculum(body,model)
    # Independent per-stage search; no assumption that one topology fits all lengths.
    for stage in curriculum['stages']:
        stage['resource_design']=explain(design_training(dict(body,**stage['parameter_patch']),result,model))
    result['length_curriculum']=curriculum
    return result


def _plan_resources(body, model=None):
    cfg=(model or {}).get("config",{})
    observed=(model or {}).get("parameter_estimate")
    if observed is not None:
        p=observed
    else:
        if 'parameters_b' in body and body['parameters_b'] in ('',None,0):
            raise ValueError('未获得可靠参数量，请填写总参数（单位十亿：7B填7，321B填321）；不能填参数个数。')
        p=number(body,"parameters_b",7,.000001,100000)*1e9
    task=body.get("task","train")
    mode=body.get("mode","full")
    zero=number(body,'zero_stage',3,0,3,True)
    if task not in ("train","inference") or mode not in ("full","lora","qlora"):
        raise ValueError("不支持的任务或训练方式")
    bits=number(body,"bits",16,4,16,True)
    if bits not in (4,8,16):
        raise ValueError("精度必须为4/8/16位")
    seq=number(body,"seq",4096,128,1000000,True)
    batch=number(body,"batch",1,1,65536,True)
    reserve=number(body,"reserve_percent",20,5,50)/100
    per_node=number(body,"gpus_per_node",8,1,16,True)
    extra=number(body,"extra_gpus",0,0,4096,True)
    fixed=number(body,"gpu_count",0,0,4096,True)
    workspace=number(body,"workspace_gib",2 if task=="inference" else 4,.1,1024)
    extras={key:number(body,key,0,0,100000) for key in
            ('runtime_gib','communication_gib','cuda_graph_gib','other_models_gib','rollout_gib')}
    extra_total=sum(extras.values())
    activation_budget=number(body,"activation_gib",8 if task=="train" else 2,0,100000)
    kv_budget=number(body,"kv_gib",4,0,100000)
    adapter_m=number(body,"adapter_million",20,.000001,1000000)
    dims={}
    for key,field in (("layers","num_hidden_layers"),("hidden","hidden_size"),("ffn","intermediate_size"),
                      ("heads","num_attention_heads"),("kv_heads","num_key_value_heads"),("vocab","vocab_size")):
        dims[key]=number(body,key,cfg.get(field) or 0,0,10000000,True)
    L,d,hq,hkv=dims["layers"],dims["hidden"],dims["heads"],dims["kv_heads"]
    notes=["这些是带明确假设的容量方案，不是已实测最低配置；GPU铭牌容量暂以GiB代入，租赁前用实测memory.total复算。",
           "参数规模采用总驻留参数，而非MoE每token激活参数。数据/图像/音频的最终token预算须由processor实测。"]
    if model and observed is None:
        notes.append("使用标称或手填参数量估算，非权重实测；不是从压缩文件元素数推导。")
    if task=="train" and mode!="full":
        notes.append("可训练adapter参数采用手填数值，不自动猜rank/target；LoRA梯度与Adam状态按该数计算。")
    if L and d:
        # Explicit conservative checkpoint-boundary + transient workload proxy, kept unsharded per rank.
        activation=(2*batch*seq*d*L + 12*batch*seq*max(d,dims['ffn']) + 4*batch*seq*dims['vocab'])/GIB if task=="train" else 12*batch*seq*d/GIB
        activation=max(activation_budget,activation)
        activation_basis="max(手填预算, checkpoint边界2BSdL + 临时张量12BS·max(d,ffn) + 未分块logits4BSV)" if task=="train" else "max(手填预算, prefill临时张量12BSd)"
    else:
        activation=activation_budget
        activation_basis="结构未知，采用手填的每卡激活预算；不是测得值"
    kv=kv_budget
    kv_basis="结构未知，采用手填的整实例KV预算"
    if L and d and hq and hkv and d%hq==0 and not cfg.get("layer_types"):
        if cfg.get("kv_lora_rank"):
            kv=2*batch*seq*L*(cfg["kv_lora_rank"]+cfg.get("qk_rope_head_dim",64))/GIB
            kv_basis="MLA压缩缓存假设：2BSL(rank+RoPE维度)；后端若展开缓存须重算"
        else:
            kv=4*batch*seq*L*hkv*cfg.get("head_dim",d//hq)/GIB
            kv_basis="BF16 K/V：4BSL·Hkv·dh bytes；批量B表示同时驻留序列数"
    if task=="train":
        if mode=="full":
            ledger={"权重":2*p/GIB,"梯度":2*p/GIB,"FP32主权重":4*p/GIB,"Adam m":4*p/GIB,"Adam v":4*p/GIB}
        else:
            ledger={"冻结基座":p*(.5625 if mode=="qlora" else 2)/GIB,"adapter权重/梯度/Adam":adapter_m*1e6*16/GIB}
        kv=0
    else:
        ledger={"权重及量化scale估算":p*(bits/8+(.0625 if bits==4 else 0))/GIB,"KV缓存":kv}
    static=sum(ledger.values())
    def state_divisor(key,n):
        if key in ('master','adam_m','adam_v'):return n if zero>=1 else 1
        if key=='gradients':return n if zero>=2 else 1
        return n if zero==3 else 1
    def training_resident(n):
        count=p if mode=='full' else adapter_m*1e6
        base=0 if mode=='full' else p*(.5625 if mode=='qlora' else 2)/GIB/state_divisor('frozen',n)
        return base+sum(count*b/GIB/state_divisor(k,n) for k,b in [('weights',2),('gradients',2),('master',4),('adam_m',4),('adam_v',4)])
    # Whole-block gather proxy; permit expert models to specify a measured/known wrap-unit budget.
    gather=number(body,"gather_gib",2*p/max(L,1)/GIB if L else 2,0,100000)
    if task=="train" and mode!="qlora" and not L:
        notes.append("最大参数聚合单元未知；当前聚合峰值是假设值，不能证明ZeRO-3一定装得下。")
    def topology(n, gpu):
        if task=="train":
            if mode=="qlora" and n>1:
                return None
            peak=training_resident(n)+activation+workspace+(gather if n>1 and zero==3 else 0)+extra_total
            return {"tp":1,"pp":1,"dp":n,"zero_stage":zero,"peak_gib":peak,"strategy":"单卡" if n==1 else ("DDP / ZeRO-0" if zero==0 else f"ZeRO-{zero}"+(' / FSDP FULL_SHARD' if zero==3 else ''))}
        best=None
        for tp in range(1,min(n,per_node)+1):
            if n%tp or (per_node%tp and n>per_node):
                continue
            pp=n//tp
            if tp>1 and (not hq or hq%tp or (hkv and hkv%tp and tp%hkv)):
                continue
            if tp>1 and any(v and v%tp for v in (d,dims['ffn'])):
                continue
            if pp>1 and (not L or L%pp):
                continue
            # Include GQA KV replication when TP exceeds KV heads.
            kv_local=kv/pp*(max(1,hkv/tp)/hkv if hkv else 1/tp)
            weight=ledger["权重及量化scale估算"]/n
            peak=weight+kv_local+activation+workspace+extra_total
            cand={"tp":tp,"pp":pp,"dp":1,"peak_gib":peak,"strategy":"单卡" if n==1 else "节点内TP + 按层PP"}
            if peak<=gpu["gib"]*(1-reserve) and (best is None or pp<best["pp"]):
                best=cand
        return best
    def candidate(key, gpu):
        lower=max(1,math.ceil(static/(gpu["gib"]*(1-reserve))))
        minimum=None
        choices=list(range(1,min(per_node,4096)+1))+list(range(per_node*2,4097,per_node))
        for n in choices:
            t=topology(n,gpu)
            if t and t["peak_gib"]<=gpu["gib"]*(1-reserve):
                minimum=n;break
        wanted=fixed or ((minimum+extra) if minimum else 0)
        allocated=fixed if fixed else (math.ceil(wanted/per_node)*per_node if wanted>per_node else wanted)
        t=topology(allocated,gpu) if allocated else None
        feasible=bool(t and t["peak_gib"]<=gpu["gib"]*(1-reserve))
        nodes=math.ceil(allocated/per_node) if allocated else 0
        reason=None if feasible else "当前预算/结构约束下没有可行候选；补充结构、降低上下文/并发，或审计TP/PP/Offload方案"
        if mode=="qlora" and task=="train" and not minimum:
            reason="单卡放不下；未把QLoRA假定为可任意ZeRO-3切分，需专用分布式量化后端审计"
        return {"gpu":key,"name":gpu["name"],"vram_gib":gpu["gib"],"capacity_lower_bound":lower,
                "conditional_min_gpus":minimum,"allocated_gpus":allocated,"nodes":nodes,"gpus_per_node":min(per_node,allocated),
                "feasible_under_assumptions":feasible,"reason":reason,"topology":t,
                "intra_node":gpu["fabric"],"inter_node":"不涉及跨节点" if nodes<=1 else "建议100/200/400Gbps RDMA；速率不是容量可行的证明，需NCCL all_reduce/all_gather及P2P实测",
                "host_ram_gib_guidance":math.ceil(max(32,p*2/GIB/max(nodes,1)*1.5)),
                "storage_gib_guidance":math.ceil(p*(32 if task=="train" else 4)/GIB),
                "status":"条件候选，未GPU验证","command_note":"规划不自动启动；TP/PP和分布式量化支持还需对应模型/引擎预检"}
    choices=[candidate(k,g) for k,g in GPUS.items()]
    selected=body.get("gpu","h100")
    if selected not in GPUS:
        raise ValueError("未知GPU型号")
    result=next(c for c in choices if c["gpu"]==selected)
    if cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts"):
        notes.append("MoE按总参数计驻留；此页不自动生成EP路由/专家映射。聚合峰值请按实际wrap粒度填写。")
    notes.append("本页只规划单模型CPT/SFT或推理，不把DPO/PPO/GRPO多角色资源混入单模型账本；它们需额外参考/奖励/价值模型与生成池。")
    # Every row below participates exactly once in the selected per-rank peak.
    n=result['allocated_gpus']; topo=result['topology']; rows=[]
    def row(key,total,local,basis):
        rows.append({'item':key,'whole_instance_gib':total,'per_gpu_gib':local,'basis':basis})
    divisor=n if n and topo else None
    if task=='train':
        count=p if mode=='full' else adapter_m*1e6
        if mode!='full':row('frozen',p*(.5625 if mode=='qlora' else 2)/GIB,p*(.5625 if mode=='qlora' else 2)/GIB/state_divisor('frozen',n) if divisor else None,f'Frozen base: BF16 2P bytes; QLoRA proxy 0.5625P; ZeRO-{zero}, divisor={state_divisor("frozen",n) if n else "unknown"}')
        for key,b in [('weights',2),('gradients',2),('master',4),('adam_m',4),('adam_v',4)]:
            total=count*b/GIB;div=state_divisor(key,n) if n else None
            row(key,total,total/div if div and divisor else None,f'{count:,.0f} × {b} bytes / 2^30 / {div or "unknown"}; ZeRO-{zero}: '+('sharded / 分片' if div and div>1 else 'replicated / 每卡复制'))
        row('kv',0,0,'Teacher-forced CPT/SFT: persistent generation KV cache disabled; saved K/V tensors are counted as activations')
    else:
        total=p*bits/8/GIB;over=p*.0625/GIB if bits==4 else 0
        row('weights',total,total/divisor if divisor else None,'P × weight bits / 8; packed format and retained layers can differ')
        row('quant_meta',over,over/divisor if divisor else None,'INT4 scale/zero proxy 0.0625 byte per weight; inspect actual format')
        local=kv/topo['pp']*(max(1,hkv/topo['tp'])/hkv if hkv else 1/topo['tp']) if topo else None
        row('kv',kv,local,kv_basis)
        for key in ('gradients','master','adam_m','adam_v'):row(key,0,0,'Inference: no backward pass or optimizer')
    boundary=2*batch*seq*d*L/GIB if L and d and task=='train' else 0
    transient=(12*batch*seq*max(d,dims['ffn'])/GIB if task=='train' else 12*batch*seq*d/GIB) if L and d else 0
    logits=4*batch*seq*dims['vocab']/GIB if L and d and task=='train' else 0
    for key,value,basis in [('saved_activations',boundary,'2BSdL: BF16 checkpoint boundaries; assumes activation checkpointing'),('transient',transient,'Temporary tensor proxy; not a layer-by-layer measured liveness trace'),('logits',logits,'4BSV: materialized FP32 training logits; chunked/fused loss can reduce this'),('activation_extra',max(0,activation-boundary-transient-logits),'Remaining per-GPU user activation budget above the structural proxy')]:
        row(key,None,value,basis)
    row('gather',None,gather if task=='train' and n>1 and zero==3 else 0,'ZeRO-3 extra parameter all-gather allocation; not divided by DP. Other collective buffers belong in communication budget.')
    row('workspace',None,workspace,'User budget for kernel scratch/workspace; exclude allocations assigned to other rows')
    for key,value in extras.items():row(key,None,value,'Additional resident per-GPU budget; 0 means NOT budgeted, not proven free')
    # Explanations travel with the numeric ledger: the UI never re-estimates memory.
    count=p if mode=='full' else adapter_m*1e6
    weight_bits=16 if task=='train' else bits
    for entry in rows:
        key=entry['item'];value=entry['per_gpu_gib']
        precision='N/A';formula=f'{value or 0:.6f} GiB'
        source='manual_budget'
        if key in ('weights','gradients','master','adam_m','adam_v'):
            b=weight_bits/8 if key=='weights' else 2 if key=='gradients' else 4
            precision=f'{weight_bits}-bit' if key=='weights' and task=='inference' else ('BF16 · 2 bytes' if b==2 else 'FP32 · 4 bytes')
            source='parameter_formula'
            if task=='inference' and key!='weights':
                precision='N/A';formula='0 (inference: no gradients / optimizer)'
            else:
                div=state_divisor(key,n) if task=='train' and n else n
                formula=f'{count if task=="train" else p:,.0f} × {b:g} bytes ÷ {div or "?"} ÷ 2^30'
        elif key=='frozen':
            b=.5625 if mode=='qlora' else 2
            precision='4-bit + metadata proxy' if mode=='qlora' else 'BF16 · 2 bytes'
            source='format_proxy' if mode=='qlora' else 'parameter_formula'
            formula=f'{p:,.0f} × {b:g} bytes ÷ {state_divisor(key,n) if n else "?"} ÷ 2^30'
        elif key=='quant_meta':
            precision='format-dependent';source='format_proxy'
            formula=f'{p:,.0f} × {0.0625 if bits==4 else 0} bytes ÷ {n or "?"} ÷ 2^30'
        elif key=='kv':
            if task=='train':formula='0 (teacher-forced training; K/V activations counted separately)'
            else:
                precision='BF16 · 2 bytes';source='shape_proxy' if L and d and hq and hkv and d%hq==0 and not cfg.get('layer_types') else 'manual_budget'
                total_formula=f'{kv:.6f} GiB (input budget)'
                if source=='shape_proxy':
                    total_formula=(f'2 × {batch} × {seq} × {L} × ({cfg["kv_lora_rank"]} + {cfg.get("qk_rope_head_dim",64)}) ÷ 2^30' if cfg.get('kv_lora_rank') else f'2(K,V) × 2 bytes × {batch} × {seq} × {L} × {hkv} × {cfg.get("head_dim",d//hq)} ÷ 2^30')
                factor=max(1,hkv/topo['tp'])/hkv if topo and hkv else 1/topo['tp'] if topo else None
                formula=f'({total_formula}) ÷ PP {topo["pp"] if topo else "?"} × head fraction {factor if factor is not None else "?"}'
        elif key in ('saved_activations','transient','logits'):
            source='shape_proxy';precision={'saved_activations':'BF16 · 2 bytes','transient':'mixed · proxy coefficient 12','logits':'FP32 · 4 bytes'}[key]
            formula={'saved_activations':f'2 × {batch} × {seq} × {d} × {L} ÷ 2^30',
                     'transient':f'12 × {batch} × {seq} × {max(d,dims["ffn"]) if task=="train" else d} ÷ 2^30',
                     'logits':f'4 × {batch} × {seq} × {dims["vocab"]} ÷ 2^30'}[key]
            if not (L and d) or (task=='inference' and key!='transient'):
                formula='0 (not modeled / not applicable; see activation budget)';precision='N/A'
        elif key=='activation_extra':
            formula=f'max(0, {activation_budget:.6f} − {boundary+transient+logits:.6f}) GiB'
        elif key=='gather':
            source='manual_budget' if 'gather_gib' in body or not L else 'shape_proxy'
            precision='BF16 proxy' if source=='shape_proxy' else 'N/A'
            formula=(f'2 × {p:,.0f} ÷ {L} ÷ 2^30' if source=='shape_proxy' else f'{gather:.6f} GiB (input/default budget)') if task=='train' and n>1 and zero==3 else '0 (parameter gather extra: only multi-GPU ZeRO-3)'
        entry.update(precision=precision,calculation=formula+f' = {value:.6f} GiB' if value is not None else formula+' = unknown',source=source)
    total_rows=sum(x['per_gpu_gib'] for x in rows if x['per_gpu_gib'] is not None) if topo else None
    budget={'rows':rows,'peak_gib':total_rows,'usable_gib':result['vram_gib']*(1-reserve),
            'reserve_gib':result['vram_gib']*reserve,'headroom_gib':result['vram_gib']*(1-reserve)-total_rows if total_rows is not None else None,
            'notes':['B=micro-batch during training, resident sequences during inference; S includes prompt and generated tokens.',
                     'Adam convention: BF16 weights/gradients + FP32 master/m/v = 16 bytes/trainable parameter. Actual optimizers differ.',
                     'Activation proxy assumes checkpointing and does not include a materialized B×H×S×S attention matrix. Unfused attention, media encoders and MoE dispatch require extra measured budgets.',
                     'DPO may double paired sequences per micro-batch and need reference weights; PPO may need actor/reference/reward/critic; GRPO needs grouped rollout KV and rewards. Enter their measured colocated per-GPU costs in other-model/rollout budgets; separate pools require separate plans.',
                     'CPU RAM, dataset prefetch, checkpoint disk, offload bandwidth and distributed synchronization are separate resources, not GPU bytes. Host RAM/disk figures are guidance, not guarantees.',
                     'Do not add allocation peaks from different execution phases unless budgeting conservatively. Never count the same cache in both rollout and KV rows.']}
    derivation=[
        {'step':'参数与单位','calculation':f'P={p:,.0f}; 1 GiB=2^30 bytes; B={batch}, S={seq}, L={L}, d={d}'},
        {'step':'整实例常驻状态','calculation':' + '.join(f'{k} {v:.3f} GiB' for k,v in ledger.items())+f' = {static:.3f} GiB'},
        {'step':'每卡可用预算','calculation':f'{result["vram_gib"]} × (1 − {reserve:.2f}) = {budget["usable_gib"]:.3f} GiB'},
        {'step':'仅按常驻状态的容量下界','calculation':f'ceil({static:.3f} / {budget["usable_gib"]:.3f}) = {result["capacity_lower_bound"]} cards; 尚未加入激活、工作区和聚合峰值'},
        {'step':'实际候选卡数','calculation':f'{n} cards, {result["nodes"]} nodes; {topo["strategy"] if topo else result["reason"]}'},
        {'step':'分片规则','calculation':f'TP={topo["tp"]}, PP={topo["pp"]}, DP={topo["dp"]}; '+(f'ZeRO-{zero}：按阶段决定状态分片，具体除数见逐项账本；激活/工作区不随DP均分' if task=='train' else '权重除以TP×PP；KV按PP和KV头分片，TP超过KV头时计入复制') if topo else '没有满足预算和整除约束的候选'},
        {'step':'逐卡验证','calculation':f'账本逐行求和={total_rows:.3f} GiB；可用{budget["usable_gib"]:.3f} GiB；余量={budget["headroom_gib"]:.3f} GiB' if total_rows is not None else '不可行或结构不足，无法证明装得下'},
    ]
    return {"derivation":derivation,"parameters":p,"parameter_basis":("远端报告存储元素（未下载验证）" if (model or {}).get('kind')=='remote' else "扫描存储元素") if observed is not None else "用户手填总参数",
            "task":task,"mode":mode,"zero_stage":zero if task=='train' else None,"ledger_gib":ledger,"static_total_gib":static,
            "per_rank_activation_gib":activation,"activation_basis":activation_basis,"workspace_gib":workspace,
            "gather_gib":gather,"kv_basis":kv_basis,"reserve_percent":reserve*100,"dims":dims,
            "candidates":choices,"selected":result,"warnings":notes,
            "model_identity":model_identity(model),
            "planning_scope":{"limited":bool(cfg.get('n_routed_experts') or cfg.get('num_experts') or cfg.get('num_local_experts') or cfg.get('layer_types') or result['nodes']>1),
                              "search":"ZeRO-only training; TP/PP-only inference","official_equivalent":False},
            "detailed_budget":budget,"placement":placement(result,budget,dims,task,batch,seq),
            "assumptions":dict(body),"code":"workbench.resources.plan_resources"}


def placement(selected, budget, dims, task, batch, seq):
    """Logical rank mapping, NOT a hardware inventory or measured layer partition.

    TP ranks are contiguous. PP stages follow TP groups; training ranks form
    one FULL_SHARD DP group. All memory figures reuse the audited ledger.
    """
    topo = selected['topology']
    if not topo:
        return {'status':'unavailable', 'ranks':[], 'groups':[]}
    tp, pp, dp = (topo[k] for k in ('tp','pp','dp'))
    n = selected['allocated_gpus']
    per_node = selected['gpus_per_node']
    layers = dims['layers']
    groups = []
    if tp > 1:
        for stage in range(pp):
            groups.append({'axis':'tp','ranks':list(range(stage*tp,(stage+1)*tp))})
    if pp > 1:
        for lane in range(tp):
            groups.append({'axis':'pp','ranks':list(range(lane,n,tp))})
    if dp > 1:
        groups.append({'axis':'dp','ranks':list(range(n))})
    ranks = []
    for rank in range(n):
        stage = rank//tp if task == 'inference' else 0
        ranks.append({
            'rank':rank, 'node':rank//per_node, 'local_gpu':rank%per_node,
            'tp_rank':rank%tp, 'pp_rank':stage,
            'dp_rank':rank if task == 'train' else 0, 'cp_rank':0,
            'layer_start':stage*(layers//pp) if layers else None,
            'layer_end_exclusive':(stage+1)*(layers//pp) if layers else None,
            'weight_fraction':1/n if task=='inference' or topo.get('zero_stage')==3 else 1,
            'query_heads':dims['heads']//tp if dims['heads'] else None,
            'kv_heads':max(1,dims['kv_heads']//tp) if dims['kv_heads'] else None,
            'tokens_per_sequence':seq, 'sequences':batch,
            'peak_gib':budget['peak_gib'], 'capacity_gib':selected['vram_gib'],
            'reserve_gib':budget['reserve_gib'], 'headroom_gib':budget['headroom_gib'],
        })
    return {'status':'estimated', 'axes':{'tp':tp,'pp':pp,'dp':dp,'cp':1},
            'ranks':ranks,'groups':groups,'memory_layout':'uniform_proxy',
            'layer_partition':'equal_layer_count_proxy',
            'global_microbatch':batch*dp if task=='train' else batch,
            'cp_supported':False, 'measured':False}
