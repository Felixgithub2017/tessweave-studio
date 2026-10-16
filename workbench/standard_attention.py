"""Conservative full-history KV capacity, not execution compatibility.

The reviewed decoder families below use MHA/GQA; sliding layers retain full
history in this bound. Never apply this bound to recurrent or compressed KV.
Model config is the shape evidence; explicit head_dim overrides hidden/heads.
Multimodal *text* configs cover text requests only, not encoder activations.
"""
FAMILIES={'internlm','internlm2','internlm3','granite','olmo2','olmo3','olmoe',
          'smollm3','seed_oss','gpt_oss','glm4_moe','minimax_m2',
          'qwen2_vl','qwen2_5_vl','qwen2_5_vl_text','qwen3_vl_text','gemma3_text',
          'mimo','ernie4_5','ernie4_5_moe'}
# Shape semantics reviewed against SGLang commit
# e00930c5489053f26d86b179cee0d087f846acbb: models/mimo.py uses
# Qwen2DecoderLayer; models/ernie4.py uses ordinary GQA and explicit MoE
# routing fields. Independent capacity formulas, not copied runtime code.
# Draft/MTP layers are disabled for this base-decoder cache budget.


def supported(cfg):
    if cfg.get('model_type') not in FAMILIES:return False
    if cfg.get('model_type')=='ernie4_5_moe' and (not all(
        type(cfg.get(k)) is int and cfg[k]>0 for k in
        ('moe_num_experts','moe_k','moe_intermediate_size')) or cfg['moe_k']>cfg['moe_num_experts']):return False
    if not all(type(cfg.get(k)) is int and cfg[k]>0 for k in
               ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads')):return False
    if any(cfg.get(k) for k in ('kv_lora_rank','linear_num_key_heads','ssm_cfg','mamba_d_state','compress_ratios','linear_attn_config')):return False
    types=cfg.get('layer_types')
    if types and (len(types)!=cfg['num_hidden_layers'] or not set(types)<={'full_attention','sliding_attention'}):return False
    if cfg.get('attn_type_list') and cfg['attn_type_list']!=[1]*cfg['num_hidden_layers']:return False
    return bool(cfg.get('head_dim') or cfg['hidden_size']%cfg['num_attention_heads']==0)
