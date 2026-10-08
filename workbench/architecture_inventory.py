"""Config + complete safetensors-header inventory. Never imports model code."""
import re
from collections import Counter


def architecture_inventory(config, tensors, errors, packed):
    cfg=config.get('text_config') or config
    layers={};outside=Counter();routed=shared=total=0;dtypes=Counter()
    for t in tensors:
        name=t['name'];n=t['elements'];total+=n;dtypes[t['dtype']]+=n
        # Anchor to language-model decoder layers, not visual.blocks / MTP modules.
        match=re.search(r'^(?:model\.(?:language_model\.)?|language_model\.)layers\.(\d+)\.',name)
        expert=re.search(r'\.experts\.(\d+)\.',name)
        is_shared=bool(re.search(r'\.shared_experts?\.',name))
        if match:
            i=int(match[1]);layer=layers.setdefault(i,{'index':i,'elements':0,'routed_elements':0,'experts':{},'shared_elements':0})
            layer['elements']+=n
            if expert:
                key=expert[1];layer['experts'][key]=layer['experts'].get(key,0)+n
                layer['routed_elements']+=n;routed+=n
            if is_shared:layer['shared_elements']+=n;shared+=n
        else:
            kind='vision' if re.search(r'(?:^|\.)(?:visual|vision_model|vision_tower)\.',name) else 'mtp' if 'mtp' in name or 'nextn' in name else 'embeddings_or_other'
            outside[kind]+=n
    count=cfg.get('num_hidden_layers',0)
    auxiliary=[]
    for i in range(count,count+int(cfg.get('num_nextn_predict_layers') or 0)):
        if i in layers:
            layer=layers.pop(i);auxiliary.append(layer)
            outside['mtp']+=layer['elements']
            routed-=layer['routed_elements'];shared-=layer['shared_elements']
    n_experts=cfg.get('n_routed_experts') or cfg.get('num_experts') or cfg.get('num_local_experts') or 0
    layer_list=[layers[i] for i in sorted(layers)]
    issues=list(errors)
    if packed:issues.append('Packed/non-floating tensors cannot establish original training parameter counts.')
    if not tensors:issues.append('No complete tensor headers.')
    if set(layers)!=set(range(count)):issues.append('Decoder layer inventory does not match config layer count.')
    if n_experts and not routed:issues.append('Routed-expert tensors not recognized; fused expert layouts need a naming adapter.')
    mlp_types=cfg.get('mlp_layer_types') or []
    for layer in layer_list+auxiliary:
        if layer['experts'] and set(map(int,layer['experts']))!=set(range(n_experts)):
            issues.append('Expert IDs incomplete in layer '+str(layer['index']))
        if layer['index']<len(mlp_types) and mlp_types[layer['index']]=='sparse' and not layer['experts']:
            issues.append('Missing routed experts in configured sparse layer '+str(layer['index']))
    kinds=cfg.get('layer_types') or ['unspecified']*count
    return {'version':1,'complete':not issues,'issues':issues,
        'basis':'all safetensors headers + config; stored elements, not deduplicated trainable parameters',
        'total_elements':total,'routed_elements':routed,'shared_expert_elements':shared,
        'non_routed_elements':total-routed,'expert_parameter_fraction':routed/total if total else 0,
        'expert_count':n_experts,'expert_topk':cfg.get('num_experts_per_tok') or cfg.get('moe_router_topk'),
        'layers':layer_list,'auxiliary_layers':auxiliary,'outside_decoder':dict(outside),'dtypes':dict(dtypes),
        'attention_layers':dict(Counter(kinds)),
        'traits':{'mla_rank':cfg.get('kv_lora_rank'),'mhc_multiplier':cfg.get('hc_mult') if cfg.get('mhc') else None,
            'linear_attention':cfg.get('linear_attn_config'),'mtp_layers':cfg.get('num_nextn_predict_layers',0)},
        'config_model_type':cfg.get('model_type'),
        'non_routed_note':'Shared experts, embeddings, visual/auxiliary modules and buffers are retained; none silently dropped.'}
