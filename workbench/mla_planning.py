"""Independent capacity equations; NOT a serving kernel or execution approval.

Layout evidence: SGLang e00930c5489053f26d86b179cee0d087f846acbb,
srt/mem_cache/memory_pool.py (MLATokenToKVPool/DSATokenToKVPool),
srt/mem_cache/index_key_cache.py. Compressed latent+RoPE is replicated
on TP ranks, unlike GQA heads. DSA index keys/scales are additional storage.
"""
import math


def supported(cfg):
    if cfg.get('model_type') not in ('deepseek_v2','deepseek_v3','deepseek_v32','kimi_k2','glm_moe_dsa','glm4_moe_lite'):
        return False
    fields=('num_hidden_layers','hidden_size','num_attention_heads','kv_lora_rank','qk_rope_head_dim')
    if not all(type(cfg.get(k)) is int and cfg[k]>0 for k in fields):return False
    if cfg.get('model_type') in ('deepseek_v32','glm_moe_dsa'):
        if type(cfg.get('index_head_dim')) is not int or cfg['index_head_dim']<=0:return False
    return not any(cfg.get(k) for k in ('compress_ratios','ssm_cfg','layer_types','linear_attn_config'))


def token_bytes(cfg, pp=1):
    if not supported(cfg):raise ValueError('Incomplete or unsupported MLA shape')
    layers=cfg['num_hidden_layers']
    if layers%pp:raise ValueError('MLA PP layer split must be integral')
    # BF16 latent + positional key. DSA: conservatively reserve BF16 index
    # keys plus FP32 scales per 128 values (upstream uses FP8 keys).
    index=cfg.get('index_head_dim') or 0
    return layers//pp*(2*(cfg['kv_lora_rank']+cfg['qk_rope_head_dim'])+
                       (2*index+4*math.ceil(index/128) if index else 0))


def cache_gib(cfg,batch,seq,pp=1,block=64):
    # One extra page per request includes a conservative tail/page allowance.
    return batch*(math.ceil(seq/block)+1)*block*token_bytes(cfg,pp)/2**30
