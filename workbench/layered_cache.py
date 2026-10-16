"""Per-layer BF16 KV upper bound, independently implemented (no copied code).

Shape evidence: SGLang e00930c5489053f26d86b179cee0d087f846acbb,
python/sglang/srt/models/mimo_v2.py::_get_ckpt_qkv_shard_sizes.
Pattern 1 selects SWA dimensions; K and V dimensions may differ. We retain
full history for SWA: no eviction or prefix-sharing savings are promised.
Equal-layer PP uses the largest stage sum, never an average of heterogeneous
layers. Weight placement and backend PP support remain separate assumptions.
"""
import math


def layers(cfg):
    if cfg.get('model_type')=='gemma4_text':
        return gemma_layers(cfg)
    if cfg.get('model_type') not in ('mimo_v2','mimo_v2_flash'):
        return None
    count=cfg.get('num_hidden_layers')
    pattern=cfg.get('hybrid_layer_pattern')
    if type(count) is not int or count<=0 or not isinstance(pattern,list) or len(pattern)!=count:
        return None
    if any(type(x) is not int or x not in (0,1) for x in pattern):return None
    result=[]
    for kind in pattern:
        prefix='swa_' if kind else ''
        q=cfg.get(prefix+'num_attention_heads')
        kv=cfg.get(prefix+'num_key_value_heads')
        dk=cfg.get(prefix+'head_dim')
        dv=cfg.get(prefix+'v_head_dim',dk)
        if not all(type(x) is int and x>0 for x in (q,kv,dk,dv)) or q%kv:
            return None
        result.append(dict(query_heads=q,kv_heads=kv,key_dim=dk,value_dim=dv,sliding=bool(kind)))
    return result


def gemma_layers(cfg):
    # Original HF config semantics, not SGLang's renamed runtime config:
    # base fields are SWA, global_* are full attention. Evidence at fixed
    # e00930c5489053f26d86b179cee0d087f846acbb, utils/hf_transformers/config.py
    # and models/gemma4_causal.py. Independent formula, no code copied.
    # Ignore KV sharing and K==V savings deliberately: a full-history upper
    # bound for text requests, not a backend allocation prediction.
    count=cfg.get('num_hidden_layers');types=cfg.get('layer_types')
    if type(count) is not int or count<=0 or not isinstance(types,list) or len(types)!=count:return None
    result=[]
    for kind in types:
        if kind not in ('full_attention','sliding_attention'):return None
        sliding=kind=='sliding_attention'
        q=cfg.get('num_attention_heads')
        kv=cfg.get('num_key_value_heads') if sliding else cfg.get('num_global_key_value_heads') or cfg.get('num_key_value_heads')
        dim=cfg.get('head_dim') if sliding else cfg.get('global_head_dim') or cfg.get('head_dim')
        if not all(type(x) is int and x>0 for x in (q,kv,dim)) or q%kv:return None
        result.append(dict(query_heads=q,kv_heads=kv,key_dim=dim,value_dim=dim,sliding=sliding))
    return result


def token_bytes(cfg,tp=1,pp=1):
    shapes=layers(cfg)
    if not shapes or tp<1 or pp<1 or len(shapes)%pp:
        raise ValueError('Invalid layered KV shape or pipeline split')
    values=[]
    for x in shapes:
        h=x['kv_heads']
        if x['query_heads']%tp or (h%tp and tp%h):
            raise ValueError('Every layer must satisfy TP head divisibility')
        values.append(2*max(1,h//tp)*(x['key_dim']+x['value_dim']))
    width=len(values)//pp
    return max(sum(values[i:i+width]) for i in range(0,len(values),width))


def cache_gib(cfg,batch,seq,tp=1,pp=1,block=16):
    return batch*math.ceil(seq/block)*block*token_bytes(cfg,tp,pp)/2**30
