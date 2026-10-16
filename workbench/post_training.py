"""Model-grounded SFT -> GRPO design. Pure, conditional, never launches jobs."""
import math
from .resources import number
from .data import dataset_guide


def sharding_brief(p, trainable, mode, body):
    dp=number(body,'dp',8,1,4096,True)
    tp=number(body,'tp',1,1,256,True)
    pp=number(body,'pp',1,1,256,True)
    rows=[]
    for z in range(4):
        parts=[]
        for label,elements,bpe,sharded in [('权重（含适配器）',None if p is None or trainable is None else p+(trainable if mode=='lora' else 0),2,z==3),('梯度',trainable,4,z>=2),('主副本＋Adam m/v',trainable,12,z>=1)]:
            divisor=tp*pp*(dp if sharded else 1)
            parts.append(dict(item=label,elements=elements,bytes_per_element=bpe,divisor=divisor,
                              gib=None if elements is None else elements*bpe/divisor/2**30,
                              formula=f'{elements if elements is not None else "未知"} × {bpe} / ({tp} TP × {pp} PP × {dp if sharded else 1} DP分片) / 2^30'))
        rows.append(dict(zero=z,parts=parts,state_gib=None if any(x['gib'] is None for x in parts) else sum(x['gib'] for x in parts)))
    return dict(dp=dp,tp=tp,pp=pp,gpus=dp*tp*pp,rows=rows,
                assumptions='教学用均匀分片状态下界：BF16 权重、FP32 梯度、FP32 主副本和 Adam m/v。实际 BF16 梯度或无主副本时需重算，不把每参数固定为 16 bytes。',
                caveat='不是拓扑推荐或每卡峰值：未包括激活、临时 all-gather、通信桶、工作区、碎片及不均匀层／专家。TP/PP 是否支持和可整除需模型适配器核验。GRPO 参考、rollout 另算。')


def design_course(model, body):
    c = model.get('config', {})
    mode = body.get('mode', 'lora')
    if mode not in ('lora', 'full'):
        raise ValueError('此连续方案目前只审计 Full / LoRA；QLoRA 需独立量化加载契约')
    seq = number(body, 'seq', 2048, 128, 131072, True)
    prompt = number(body, 'prompt_tokens', 512, 1, 131072, True)
    output = number(body, 'output_tokens', 512, 1, 131072, True)
    group = number(body, 'group', 4, 2, 32, True)
    rank = number(body, 'rank', 8, 1, 256, True)
    samples = number(body, 'samples', 1000, 1, 1e9, True)
    mean_tokens = number(body, 'mean_tokens', 512, 1, seq, True)
    epochs = number(body, 'epochs', 1, 1, 10, True)
    batch = number(body, 'batch', 1, 1, 128, True)
    accum = number(body, 'grad_acc', 8, 1, 4096, True)
    capacity = number(body, 'capacity_gib', 80, 1, 2048)
    reserve = number(body, 'reserve_percent', 15, 0, 50)
    activation = number(body, 'activation_gib', 0, 0, 2048)
    workspace = number(body, 'workspace_gib', 0, 0, 2048)
    reward = body.get('reward', 'unknown')
    if reward not in ('math', 'subjective', 'unknown'):
        raise ValueError('未知奖励类型')
    p = model.get('parameter_estimate')
    if not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
        p = None
    targets = []
    # Explicit target subset: no fabricated all-linear coverage / embeddings / router.
    for tensor in model.get('tensors', []):
        name, shape = tensor['name'], tensor['shape']
        if name.endswith(tuple('.'+x+'.weight' for x in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'))) and len(shape) == 2:
            targets.append((name, rank * sum(shape)))
    trainable = p if mode == 'full' else sum(n for _, n in targets) or None
    blockers = list(model.get('errors', []))
    if p is None: blockers.append('原始浮点参数量未知或权重已打包量化，不能从文件大小猜原参数量。')
    if trainable is None: blockers.append('没有识别到本方案 LoRA 目标线性层；需架构专用适配。')
    experts = model.get('experts', 0)
    mt = model.get('model_type', '')
    standard = mt in ('llama','qwen2','qwen3','mistral') and not experts and model.get('modality') == 'text'
    dims = {k:c.get(k) for k in ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','head_dim','max_position_embeddings')}
    if not dims['head_dim'] and dims['hidden_size'] and dims['num_attention_heads']:
        dims['head_dim'] = dims['hidden_size'] // dims['num_attention_heads']
    dims['num_key_value_heads'] = dims['num_key_value_heads'] or dims['num_attention_heads']
    if c.get('max_position_embeddings') and max(seq,prompt+output)>c['max_position_embeddings']:
        blockers.append('输入长度超过 config 声明上下文；不自动进行 RoPE 扩展。')
    if not standard: blockers.append('MoE／多模态／混合或未审计架构：只展示已知状态账本；激活、KV 和执行需专用适配。')
    if reward != 'math': blockers.append('GRPO 尚无已审核奖励：主观偏好通常先考虑 DPO；不能只凭“想增强推理”启动 RL。')
    rows = []
    def add(stage, label, precision, elements, bpe, equation):
        rows.append(dict(stage=stage,item=label,precision=precision,elements=elements,bytes_per_element=bpe,
                         gib=None if elements is None else elements*bpe/2**30,formula=equation))
    for stage in ('sft','grpo'):
        add(stage,'策略权重','BF16',p,2,'P × 2 / 2^30')
        if mode == 'lora':add(stage,'LoRA 权重','BF16',trainable,2,'T = Σ r(d_in+d_out); T × 2 / 2^30')
        add(stage,'可训练梯度','FP32（保守预算）',trainable,4,'T × 4 / 2^30')
        add(stage,'优化器主副本','FP32（保守预算）',trainable,4,'T × 4 / 2^30')
        add(stage,'Adam m、v','FP32 × 2',trainable,8,'T × (4+4) / 2^30')
    add('grpo','冻结参考权重','BF16',p,2,'保守假设独立参考副本 P × 2 / 2^30；β=0 或 adapter 复用时须重算')
    kv = None
    if standard and all(dims[k] for k in ('num_hidden_layers','num_key_value_heads','head_dim')):
        kv = 2 * dims['num_hidden_layers'] * dims['num_key_value_heads'] * dims['head_dim'] * group * (prompt+output)
    add('grpo','rollout KV 上界预算','BF16',kv,2,'2(K,V) × L × Hkv × Dhead × G × (prompt+output) × 2 / 2^30；同时驻留一组，无共享前缀折扣')
    budgets = []
    for stage in ('sft','grpo'):
        items=[r for r in rows if r['stage']==stage]
        known=sum(r['gib'] for r in items if r['gib'] is not None)
        complete=all(r['gib'] is not None for r in items) and standard and activation>0 and workspace>0
        peak=known+activation+workspace
        fit=complete and peak <= capacity*(1-reserve/100)
        budgets.append(dict(stage=stage,known_gib=known,activation_budget_gib=activation,workspace_budget_gib=workspace,
                            peak_budget_gib=peak if complete else None,capacity_gib=capacity,reserve_gib=capacity*reserve/100,
                            status='conditional_fit' if fit else 'over_budget' if complete else 'incomplete',
                            next='先在该单卡做短程实测，再依据峰值与吞吐放大。' if fit else '先补齐激活／工作区并实测；不据此给出卡数。' if not complete else '先减微批／长度／rollout 并发或重计算，再交联合规划器搜索分片；不能简单用总内存除卡容量。'))
    return dict(status='design_only',sharding=sharding_brief(p,trainable,mode,body),model=dict(name=model.get('model_card_title') or model.get('name'),path=model['path'],type=mt,
                   modality=model.get('modality'),experts=experts,parameters=p,parameter_basis=model.get('count_basis'),dims=dims),
                corrections=[
                    'SFT → GRPO 是可选路线，不是必须流程。已有合格指令模型且奖励可靠，可直接评估 RL；能力或格式不足时先 SFT。',
                    'SFT 与 GRPO 通常不改变网络拓扑：改变的是数据生成方式、损失和参数。LoRA 改变可训练参数集合，不是另一种训练目标。',
                    'GRPO 不要求单独 critic，但仍需 rollout、梯度和优化器；参考副本是否需要取决于 KL 与实现。旧策略通常保存 logprob，不必常驻第三份模型。',
                    'MoE 的激活参数量不能用于权重驻留账本；多模态还要说明视觉编码器／投影层冻结策略。',
                    '提高奖励不等于能力提高；模型可能迎合错误奖励，必须保留独立评估和作弊测试。'],
                architecture_note='token → embedding → 每层 attention（混合上下文）→ FFN（特征变换；MoE 为路由专家）→ LM head → 下一 token 概率。残差、归一化与位置编码仍在；实际顺序以模型实现为准。多模态还需输入处理器与编码器。',
                update=dict(mode=mode,rank=rank,trainable=trainable,targets=len(targets),target_examples=[x[0] for x in targets[:8]],
                            scope='只预算 q/k/v/o/gate/up/down 的二维投影；当前执行器 all-linear 可能更广，提交前必须对齐，不能直接当执行显存。'),
                rows=rows,budgets=budgets,blockers=blockers,
                schedule=dict(samples=samples,mean_tokens=mean_tokens,epochs=epochs,seq=seq,batch=batch,grad_acc=accum,
                              effective_batch=batch*accum,sft_updates=math.ceil(samples/(batch*accum))*epochs,
                              sft_tokens=samples*mean_tokens*epochs,group=group,prompt_tokens=prompt,output_tokens=output,
                              grpo_generated_tokens_per_group=group*output,
                              assumptions='单 GPU、无 packing、每 epoch 最后不足 batch 仍更新；实际 packing / drop_last / 分布式数据加载会改变步数。GRPO 无法用 SFT token/s 推算时间。'),
                data={s:dataset_guide(s,model,'zh') for s in ('sft','grpo')},
                sources=['https://huggingface.co/docs/trl/sft_trainer','https://huggingface.co/docs/trl/grpo_trainer','https://arxiv.org/abs/2402.03300'],
                execution_note='仅设计，不创建任务。SFT 通过验证后，需要显式选择 checkpoint；LoRA 必须绑定同一底座 revision 并经验证加载／合并，不能把适配器目录冒充完整模型。')
