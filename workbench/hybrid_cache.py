"""Kimi Linear capacity adapter, not a runtime kernel implementation.

Independent formula implementation, informed by SGLang e00930c5489053f26d86b179cee0d087f846acbb
python/sglang/srt/configs/{kimi_linear,mamba_utils}.py. No upstream code copied.
One-based layer IDs must partition the decoder. MLA latent cache is replicated
across TP ranks; KDA heads are sharded. Never divide compressed MLA by TP.
"""
import math


def qwen_delta_shape(cfg):
    """Reviewed GDN family; full KV is not MLA and cannot use its TP rule.

    Shape reference: same SGLang commit as above, configs/qwen3_next.py
    mamba2_cache_params + qwen3_5.py. Independently expressed formulas.
    """
    if cfg.get('model_type') not in ('qwen3_next','qwen3_5_text','qwen3_5_moe_text'):return None
    keys=('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','head_dim',
          'linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim','linear_conv_kernel_dim')
    if not all(type(cfg.get(k)) is int and cfg[k]>0 for k in keys):return None
    layers=cfg.get('layer_types')
    if not isinstance(layers,list) or len(layers)!=cfg['num_hidden_layers'] or not set(layers)<= {'full_attention','linear_attention'}:return None
    if not all(x in layers for x in ('full_attention','linear_attention')):return None
    return dict(num_heads=cfg['linear_num_value_heads'],key_heads=cfg['linear_num_key_heads'],
                full_attn_layers=[i for i,t in enumerate(layers) if t=='full_attention'],
                kda_layers=[i for i,t in enumerate(layers) if t=='linear_attention'],kind='gdn')


def hybrid_shape(cfg):
    return kimi_shape(cfg) or qwen_delta_shape(cfg)


def kimi_shape(cfg):
    if cfg.get('model_type') != 'kimi_linear':
        return None
    a=cfg.get('linear_attn_config') or {}
    keys=('num_heads','head_dim','short_conv_kernel_size')
    if not all(type(a.get(k)) is int and a[k]>0 for k in keys):return None
    if not all(type(cfg.get(k)) is int and cfg[k]>0 for k in
               ('num_hidden_layers','hidden_size','moe_intermediate_size','kv_lora_rank','qk_rope_head_dim','num_experts','num_experts_per_token')):return None
    k,m=a.get('kda_layers',[]),a.get('full_attn_layers',[])
    if not isinstance(k,list) or not isinstance(m,list) or not k or not m or not all(type(i) is int for i in k+m):return None
    if sorted(k+m)!=list(range(1,cfg['num_hidden_layers']+1)):return None
    return a


def coefficients(cfg, tp=1):
    a=hybrid_shape(cfg)
    if not a or a['num_heads']%tp:raise ValueError('Kimi KDA heads must divide TP')
    if a.get('kind')=='gdn':
        if a['key_heads']%tp:raise ValueError('GDN key heads must divide TP')
        kh,vh=cfg['linear_num_key_heads']//tp,cfg['linear_num_value_heads']//tp
        kd,vd=cfg['linear_key_head_dim'],cfg['linear_value_head_dim']
        token=4*len(a['full_attn_layers'])*max(1,cfg['num_key_value_heads']//tp)*cfg['head_dim']
        state=len(a['kda_layers'])*(4*vh*vd*kd+2*(2*kh*kd+vh*vd)*(cfg['linear_conv_kernel_dim']-1))
        return token,state
    # BF16 compressed MLA latent + RoPE; FP32 recurrent matrix and BF16
    # short-convolution history. No prefix checkpoints/speculative state reuse.
    token=2*len(a['full_attn_layers'])*(cfg['kv_lora_rank']+cfg['qk_rope_head_dim'])
    state=len(a['kda_layers'])*(a['num_heads']//tp)*(4*a['head_dim']**2+6*a['head_dim']*(a['short_conv_kernel_size']-1))
    return token,state


def cache_gib(cfg,batch,seq,tp=1,block=16):
    token,state=coefficients(cfg,tp)
    return batch*(math.ceil(seq/block)*block*token+state)/2**30


def activation_gib(cfg,batch,seq,chunk):
    # Explicit conservative scratch proxy, NOT measured kernel liveness. Chunked
    # prefill is required for this plan. Router/logits and selected-expert buffers
    # are included per rank without assuming EP or TP scratch savings.
    tokens=min(batch*seq,chunk)
    return tokens*(12*cfg['hidden_size']+moe_scratch_bytes(cfg))/2**30


def moe_scratch_bytes(cfg):
    # ERNIE text MoE uses moe_num_experts/moe_k (ernie4.py at the fixed
    # upstream revision documented in standard_attention.py). Do not silently
    # omit its router/expert scratch just because field names differ.
    experts=cfg.get('num_experts') or cfg.get('num_local_experts') or cfg.get('n_routed_experts') or 0
    topk=cfg.get('num_experts_per_token') or cfg.get('num_experts_per_tok') or 0
    if cfg.get('model_type')=='ernie4_5_moe':
        experts=cfg.get('moe_num_experts') or 0
        topk=cfg.get('moe_k') or 0
    if cfg.get('model_type')=='gemma4_text':
        topk=cfg.get('top_k_experts') or 0
    width=cfg.get('moe_intermediate_size') or cfg.get('intermediate_size') or 0
    return 4*experts+4*topk*width
