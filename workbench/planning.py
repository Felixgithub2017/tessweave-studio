"""Transparent capacity ledgers, never a claim that a job is guaranteed to fit."""
import math

# Catalog capacity is a planning label; observed free memory takes precedence.
CATALOG = {"RTX 4090": 24, "RTX 5090": 32, "A100 80GB": 80,
           "H100 80GB": 80, "H200 141GB": 141, "B200 180GB": 180}
METHODS = {
    "awq": "校准激活→选择通道缩放→INT4 打包；需要质量回归与匹配内核",
    "gptq": "校准层输入→用二阶统计补偿舍入误差→低位权重；转换峰值可能大于推理",
    "fp8": "8位浮点权重/激活路径；硬件、scale 契约和算子支持决定是否提速",
    "mlx4": "Apple MLX 分组仿射权重4位；压缩表示与 Metal 内核配套",
    "structured-pruning": "删通道/头/专家要同步修改结构、恢复训练和校验；暂无通用安全执行器",
    "sparse-2of4": "每四个权重保留两个非零；需要满足布局的稀疏内核，否则零值不省算力",
}


def estimate(model, task="inference", seq=2048, batch=1, bits=16, tuner="full"):
    if task not in ("inference", "cpt", "sft", "dpo", "grpo", "ppo", "quantize"):
        raise ValueError("未知任务")
    if not 1 <= seq <= 1000000 or not 1 <= batch <= 65536 or bits not in (4, 8, 16):
        raise ValueError("seq/batch/bits 超出边界")
    p = model.get("parameter_estimate")
    cfg = model.get("config", {})
    warnings = ["下界不是最小可运行配置；未进行本模型的峰值 profiling",
                "容量标签使用 GiB 规划假设；租卡前以实测 memory.total 为准"]
    if not p:
        return {"known": False, "warnings": warnings + ["量化/缺失权重无法可靠还原参数量；禁止从打包整数元素数猜参数"],
                "weight_file_gib": model.get("weight_bytes", 0) / 2**30, "candidates": []}
    weight = p * (bits / 8 + (2 / 128 if bits == 4 else 0))
    grad, optim, frozen = 0, 0, 0
    if task not in ("inference", "quantize"):
        if tuner == "full":
            weight, grad, optim = 2*p, 2*p, 12*p
        else:
            # Not a false fixed percentage of base parameters.
            warnings.append("LoRA 只计冻结基座；adapter 参数、梯度、优化器必须在选定 target/rank 后实测补齐")
            weight = 2*p
        if task in ("dpo", "grpo"):
            frozen = 2*p
            warnings.append("参考模型保守按额外 BF16 基座计；LoRA 共享/预计算可能降低，生成池可能提高")
        if task == "ppo":
            frozen = 18*p
            warnings.append("PPO 教学容量按同规模 critic 16P + reference 2P；奖励模型尚需单独累加")
    if task == "quantize":
        weight = 2*p
        warnings.append("转换账只列高精度输入，不含校准激活、Hessian、输出暂存；不能据此保证量化够内存")
    L, d = cfg.get("num_hidden_layers"), cfg.get("hidden_size")
    hq, hkv = cfg.get("num_attention_heads"), cfg.get("num_key_value_heads", cfg.get("num_attention_heads"))
    kv = None
    if all(type(x) is int and x > 0 for x in (L, d, hq, hkv)) and not cfg.get("kv_lora_rank") and not cfg.get("layer_types"):
        head = cfg.get("head_dim", d // hq)
        kv = 2 * L * hkv * head * seq * batch * 2
    else:
        warnings.append("非普通 MHA/GQA 或配置不全：KV 需专用 MLA/混合注意力估算器")
    ledger = {"weights": weight, "gradients": grad, "adam_master_m_v": optim,
              "additional_models_assumption": frozen}
    if task == "inference" and kv is not None:
        ledger["kv_bf16"] = kv
    known_bytes = sum(ledger.values())
    candidates = []
    for name, cap in CATALOG.items():
        usable = cap * .8 * 2**30
        n = max(1, math.ceil(known_bytes / usable))
        candidates.append({"gpu": name, "capacity_gib_assumption": cap,
                           "ideal_aggregate_lower_bound": n,
                           "topology": "单机高速互联优先；跨节点须实测 RDMA/集合通信，不建议跨慢网 TP" if n > 1 else "单卡",
                           "status": "容量候选，非可执行拓扑",
                           "parallel_hint": "全参优先审计 ZeRO-3/FSDP；MoE 大规模另需 TP/PP/EP 适配" if task != "inference" else "TP 必须满足头数/后端约束；MoE EP 不按激活参数算存储"})
    return {"known": True, "parameter_estimate": p, "ledger_gib": {k: round(v/2**30, 4) for k,v in ledger.items()},
            "known_total_gib": known_bytes/2**30, "kv_gib": None if kv is None else kv/2**30,
            "unestimated": ["激活与重计算策略", "峰值 logits/attention workspace", "通信/Graph 缓冲", "加载暂存", "LoRA 具体目标层"],
            "assumptions": {"seq": seq, "batch": batch, "bits": bits, "tuner": tuner, "gpu_reserve_fraction": .2},
            "warnings": warnings, "candidates": candidates}
