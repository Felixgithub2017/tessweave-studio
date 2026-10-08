"""Weight inventory is observed; forward flow is explicitly an architecture hypothesis."""
import re
from collections import defaultdict


def category(name):
    n = name.lower()
    if any(s in n for s in ("vision", "visual", "image_encoder")):
        return "视觉组件"
    if any(s in n for s in ("audio", "speech", "vocoder")):
        return "音频组件"
    if any(s in n for s in ("embed_tokens", "word_embeddings", ".wte.")):
        return "输入嵌入"
    if any(s in n for s in ("lm_head", "output_layer")):
        return "输出投影"
    if "norm" in n:
        return "归一化"
    if any(s in n for s in ("self_attn", "attention", ".attn.")):
        return "注意力"
    if any(s in n for s in ("router", ".gate.weight")):
        return "专家路由"
    if any(s in n for s in ("experts", "expert.")):
        return "专家前馈"
    if any(s in n for s in ("mlp", "feed_forward", ".ffn.")):
        return "前馈网络"
    return "其他/待识别"


def describe(config, pipeline, tensors, packed):
    text = config.get("text_config") or config
    groups, layers, components = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for t in tensors:
        group = category(t["name"])
        amount = t["elements"]
        groups[group][0] += amount
        groups[group][1] += t.get("storage_bytes", 0)
        groups[group][2] += 1
        match = re.search(r"(?:^|\.)(layers|h|blocks|block)\.(\d+)(?:\.|$)", t["name"])
        if match:
            key = (t["component"] + ":" + t["name"][:match.end()].rstrip("."))
            layers[key][0] += amount
            layers[key][1] += 1
        components[t["component"]][0] += amount
        components[t["component"]][1] += 1
    specs = {"model_type": config.get("model_type"), "architectures": config.get("architectures"),
             "layers": text.get("num_hidden_layers"), "hidden_size": text.get("hidden_size"),
             "vocab_size": text.get("vocab_size"), "query_heads": text.get("num_attention_heads"),
             "kv_heads": text.get("num_key_value_heads"), "head_dim": text.get("head_dim"),
             "intermediate_size": text.get("intermediate_size"), "activation": text.get("hidden_act"),
             "max_position_embeddings": text.get("max_position_embeddings"), "rope_theta": text.get("rope_theta"),
             "rope_scaling": text.get("rope_scaling"), "tied_embeddings": text.get("tie_word_embeddings"),
             "routed_experts": text.get("n_routed_experts", text.get("num_local_experts", text.get("num_experts"))),
             "experts_per_token": text.get("num_experts_per_tok"), "moe_intermediate_size": text.get("moe_intermediate_size"),
             "kv_lora_rank": text.get("kv_lora_rank"), "q_lora_rank": text.get("q_lora_rank"),
             "quantization_config": config.get("quantization_config"), "layer_types": text.get("layer_types")}
    family = text.get("model_type", config.get("model_type", ""))
    known_decoder = family in {"llama", "mistral", "mixtral", "qwen2", "qwen3", "qwen3_moe", "qwen2_moe", "deepseek_v2", "deepseek_v3"}
    moe = bool(specs["routed_experts"])
    if known_decoder:
        flow = ["输入 token ID", "Embedding 查表", "重复 %s 个 Decoder Block" % (specs["layers"] or "配置未注明数量的"),
                "Norm → Attention → 残差相加", "Norm → %s → 残差相加" % ("路由 / 专家前馈" if moe else "前馈网络"),
                "最终 Norm → 输出投影 → logits"]
    else:
        flow = []
    return {"specs": specs, "flow": flow,
            "flow_basis": "按已知 Decoder 家族配置推断的主干示意，不是执行追踪；多模态连接器、逐层注意力变体与自定义 forward 需源码确认" if flow else "未推断执行顺序；下方仅展示权重组件清单，不冒充完整计算图",
            "groups": [{"name": k, "stored_elements": v[0], "storage_bytes": v[1], "tensor_count": v[2]} for k,v in groups.items()],
            "layers": [{"name": k, "stored_elements": v[0], "tensor_count": v[1]} for k,v in layers.items()],
            "components": [{"name": k, "stored_elements": v[0], "tensor_count": v[1]} for k,v in components.items()],
            "inventory_basis": "按实际 tensor 名称分类；未识别项目保留。" + ("量化存储元素不等于原参数个数。" if packed else "存储浮点元素包含可能的缓冲区，绑定权重也可能只存一份。"),
            "component_configs": {k: v for k,v in config.items() if k.endswith("_config") and isinstance(v, dict)},
            "pipeline_components": pipeline}
