"""Dated upstream recipe evidence, not executable or locally verified presets."""
from copy import deepcopy

CHECKED = '2026-10-13'


def coverage(model):
    """Catalog visibility is not architecture support or execution evidence."""
    cfg=(model or {}).get('config') or {}
    cfg=cfg.get('text_config') or cfg
    from .deployment_routes import recipes
    has_recipe=any(r['repo']==(model or {}).get('repo') for r in recipes())
    if has_recipe:
        return dict(status='upstream_recipe',label='已有上游部署配方 · 待本机验证',missing=[],execution_verified=False)
    missing=[k for k in ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads') if not cfg.get(k)]
    if missing:
        return dict(status='metadata_required',label='需读取结构元数据',missing=missing,execution_verified=False)
    a=assessment(dict(model or {},config=cfg))
    return dict(status='analytic_unverified' if a['analytic_supported'] else 'adapter_required',
        label='容量可规划 · 未实机验证' if a['analytic_supported'] else '架构适配待完成',
        missing=[] if a['analytic_supported'] else ['逐层缓存/循环状态与后端布局'],
        reason=a['reason'],execution_verified=False)

def entry(repo, architecture, deployment, caveat, evidence='upstream_recipe'):
    return dict(repo=repo, architecture=architecture, deployment=deployment,
                caveat=caveat, evidence=evidence, checked=CHECKED,
                source='https://recipes.vllm.ai/'+repo,
                locally_verified=False, executable=False, revision=None)

RECIPES = [
    entry('moonshotai/Kimi-K3', 'KDA + gated MLA / MoE / vision',
          'Blackwell 专用配方；MXFP4 权重 / MXFP8 激活',
          '配方仍有 pre-release 文案；必须核对检查点与镜像版本。', 'conflicting_status'),
    entry('Qwen/Qwen3.8-2.4T-A95B', 'Gated DeltaNet + attention / MoE',
          '配方容量表：BF16 24×B300；NVFP4 8×B300',
          '容量表不是实测最低卡数或等吞吐对照；需分别计算循环状态和 KV。'),
    entry('deepseek-ai/DeepSeek-V4-Pro', 'CSA/HCA / MoE / mHC',
          'NVIDIA / AMD 专用配方，混合 FP4/FP8 与 DSpark',
          'preview 与 0813 不得共享性能结论；草稿模块另计。'),
    entry('deepseek-ai/DeepSeek-V4.1-Flash', 'encoder-decoder / Engram / MoE',
          'Blackwell；Engram CPU/UVA；另有 NIXL P/D 配方',
          '552B 主干外还有 Engram；主存与传输成本不可遗漏。'),
    entry('XiaomiMiMo/MiMo-V2.6-Pro-RL', 'hybrid attention / MoE / multimodal',
          '8×H200 TP8 或 4×MI355X TP4；MXFP4 存储转 FP8 计算',
          '566GB 检查点不等于显存；680GB 是配方容量估计。RL/MOPD 不混用。'),
    entry('XiaomiMiMo/MiMo-V2.6-Flash-RL', 'hybrid attention / MoE',
          'H200 专用配方；按检查点及硬件核对引擎版本',
          '页面默认检查点及 TP 示例存在版本差异，未批准自动启动。'),
    entry('tencent/Hy4-preview', 'sparse attention / MoE / MTP',
          'vLLM 专用配方', 'preview；MTP 内存另计；不推断最低卡数。'),
    entry('zai-org/GLM-5.3', 'MoE / MTP',
          'FP8 8×H200 TP8 示例', 'BF16 不继承 FP8 卡数；MTP 和 KV 精度需匹配版本。'),
    entry('zai-org/GLM-5.3-Flash', 'KDA + sparse MLA / MoE / vision',
          'H200 TP8；Blackwell TP4；另有 P/D 配方',
          'Hopper 与 Blackwell 的 FP8 KV 支持不同；打印的 KV 并发不是 SLO 并发。'),
    entry('MiniMaxAI/MiniMax-M3', 'MSA / MoE / vision',
          'SGLang：B200 TP8，B300 TP4；Hopper BF16 路径',
          '参数口径有冲突；专用 dev 镜像、128-token 页；GB200 部分为推断支持。'),
    entry('nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B', 'Mamba + attention / MoE',
          'Dynamo：4×B200 / worker，TP4 + EP，MTP',
          '验证限配方负载与缓存复用条件；非任意业务 SLA。', 'upstream_validated_workload'),
    entry('stepfun-ai/Step-3.7-Flash', 'full + sliding attention / MoE / vision',
          '官方模型卡提供引擎部署入口', '未登记可复现服务压测，不推断并发。'),
]
for r in RECIPES:
    if r['repo'].startswith('nvidia/'):
        r['source']='https://docs.nvidia.com/dynamo/v1.3.0/recipes/nemotron-3-ultra'
    elif r['repo'].startswith('MiniMax'):
        r['source']='https://docs.sglang.io/cookbook/autoregressive/MiniMax/MiniMax-M3'
    elif r['repo'].startswith('stepfun'):
        r['source']='https://huggingface.co/'+r['repo']


def assessment(model):
    """Never infer checkpoint identity from a local directory or family substring."""
    model=model or {}; cfg=model.get('config') or {}
    matches=[deepcopy(r) for r in RECIPES if r['repo']==model.get('repo')]
    kind=cfg.get('model_type') or model.get('model_type')
    special=any(cfg.get(k) for k in (
        'kv_lora_rank','layer_types','vision_config','text_config',
        'ssm_cfg','num_experts','n_routed_experts','num_local_experts',
        'moe_intermediate_size','linear_num_key_heads','mamba_d_state'))
    # Conservative analytic adapters, not a claim of serving-engine compatibility.
    known=kind in ('llama','qwen2','qwen3','mistral','phi3','phi','gemma','gemma2','gemma3_text')
    # Sliding attention is budgeted as full history: a conservative capacity
    # upper bound, not a claim that the backend allocates/evicts this way.
    standard_moe=kind in ('qwen2_moe','qwen3_moe','mixtral') and all(
        type(cfg.get(k)) is int and cfg[k]>0 for k in ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads')) and not any(cfg.get(k) for k in ('kv_lora_rank','layer_types','ssm_cfg','linear_num_key_heads'))
    manual=not kind and not model.get('repo') and not model.get('path')
    gemma27=(kind=='gemma3_text' and tuple(cfg.get(k) for k in ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','head_dim'))==(62,5376,32,16,128) and not cfg.get('kv_lora_rank'))
    from .hybrid_cache import kimi_shape, qwen_delta_shape
    kimi=bool(kimi_shape(cfg))
    gdn=bool(qwen_delta_shape(cfg))
    from .mla_planning import supported as mla_supported
    mla=mla_supported(cfg)
    from .standard_attention import supported as full_history_supported
    full_history=full_history_supported(cfg)
    from .layered_cache import layers
    layered=bool(layers(cfg))
    supported=((known or manual) and not special and not matches) or gemma27 or kimi or gdn or standard_moe or mla or full_history or layered
    result = dict(status='analytic_unverified' if supported else 'adapter_required',
                analytic_supported=supported, model_type=kind,
                identity_verified=False, recipes=matches,
                reason=('MLA/DSA压缩缓存：TP复制、PP按层划分；总专家驻留，支持跨节点TP容量搜索；分块prefill、索引缓存另计，外部后端待验证。' if mla else 'Qwen Gated DeltaNet + 全注意力：分层KV/循环状态，TP容量候选；仅文本请求，视觉编码峰值、MTP、PP/EP未纳入。' if gdn else 'Kimi Linear：压缩MLA缓存+FP32 KDA状态+总MoE权重；TP容量候选，PP/EP与执行未经验证。' if kimi else 'Gemma 3 27B文本请求：全历史KV保守上界，不计滑窗回收收益；不含图片编码激活，仍需实测。' if gemma27 else '标准KV缓存/总专家权重解析估算；滑窗按全历史保守预算，路由临时张量需预算；不认证EP、后端或运行质量。' if standard_moe else '标准KV解析估算（滑窗按全历史保守预算）；仍需后端预检与压测。' if supported else
                        '未有此架构的逐卡部署适配器；通用 TP/PP 数字不构成可运行方案。'),
                cache_adapter='layered_full_history' if layered else 'mla' if mla else None,
                required_evidence=['固定 checkpoint revision 与引擎镜像摘要',
                    '逐层权重格式、缓存/循环状态布局及 TP/EP/PP 通信组',
                    '加载、prefill、decode、Graph 各阶段逐卡峰值',
                    '实际请求分布的 TTFT/ITL、错误率与故障演练'])
    if layered:
        result['reason']='逐层KV头数与K/V维度预算；滑窗全历史上界，不计跨层共享或K=V复用；等层PP取最忙阶段；仅文本、分块prefill，不认证后端或多模态峰值。'
    return result
