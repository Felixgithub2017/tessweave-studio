"""Typed recipe builders: no arbitrary shell strings and no universal-model claims."""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .data import validate_dataset
from .planning import METHODS
from .techniques import technique_guide

TEXT = {"qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "mixtral"}
VLM = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
MODULES = {"vllm": "vllm.entrypoints.openai.api_server", "sglang": "sglang.launch_server", "mlx": "mlx_lm.server"}
REFERENCE = {
    "swift": "https://github.com/modelscope/ms-swift/tree/v4.5.3/examples",
    "vllm": "https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
    "sglang": "https://github.com/sgl-project/sglang",
    "mlx": "https://github.com/ml-explore/mlx-lm",
}


def environment(python):
    p = Path(python).expanduser()
    if not p.is_absolute() or not p.is_file() or not os.access(p, os.X_OK):
        raise ValueError("backend_python 必须是已安装 Python 的绝对路径")
    script = "import json,importlib.metadata as m; d={};\nfor n in ['torch','ms-swift','vllm','sglang','mlx-lm','deepspeed','math-verify']:\n try: d[n]=m.version(n)\n except m.PackageNotFoundError: d[n]=None\nprint(json.dumps(d))"
    r = subprocess.run([str(p), "-I", "-c", script], capture_output=True, text=True, timeout=15)
    if r.returncode:
        raise ValueError("后端 Python 无法探测版本")
    return json.loads(r.stdout)


def integer(options, name, default, low, high):
    val = options.get(name, default)
    if type(val) is not int or not low <= val <= high:
        raise ValueError(name + " 超出允许范围")
    return val


def recipe(model, request, hardware, roots, output):
    task = request.get("task", "sft")
    if task not in ("cpt", "sft", "dpo", "grpo", "ppo", "inference", "quantize"):
        raise ValueError("未知任务")
    backend = request.get("backend", "swift")
    if backend not in ("swift", "vllm", "sglang", "mlx"):
        raise ValueError("未知后端")
    python = request.get("backend_python") or sys.executable
    versions = environment(python)
    issues, warnings = list(model["errors"]), list(model["warnings"])
    mt = model["model_type"]
    gemma4_text_inference = backend == 'mlx' and task == 'inference' and mt in ('gemma4', 'gemma4_text')
    supported = mt in TEXT or (mt in VLM and task in ("sft", "dpo", "inference")) or gemma4_text_inference
    if gemma4_text_inference:
        if versions.get('mlx-lm') != '0.32.0':
            issues.append('Gemma 4 text recipe audited for mlx-lm 0.32.0; validate other versions before execution')
        warnings.append('Gemma 4: TEXT-ONLY deployment. mlx-lm extracts the language model and drops vision/audio weights; image/audio input is not supported by this recipe.')
    if not supported:
        issues.append("此架构/任务尚无本项目审计配方；已识别不等于能安全执行")
    if model["quantization"] and task != "inference":
        issues.append("量化源权重的训练/再量化尚未接通；使用原高精度包，勿静默二次量化")
    seq = integer(request, "seq", 2048, 128, 131072)
    steps = integer(request, "steps", 20, 1, 10000000)
    batch = integer(request, "batch", 1, 1, 128)
    gpus = integer(request, "gpus", 1, 1, 256)
    grad_acc = integer(request, "grad_acc", 8, 1, 100000)
    port = integer(request, "port", 8000, 1024, 65535)
    tp = integer(request, "tp", 1, 1, 256)
    tuner = request.get("tuner", "lora")
    if tuner not in ("full", "lora"):
        raise ValueError("tuner 必须为 full 或 lora")
    lr = float(request.get("learning_rate", 1e-5))
    if not 0 < lr <= .1:
        raise ValueError("learning_rate 必须在 (0,0.1]")
    argv, env, dataset = [python], {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true"}, None
    if backend != "mlx" and not hardware["gpus"]:
        issues.append("当前执行主机没有检测到 NVIDIA GPU；远端探测不代表任务会自动在远端运行")
    if backend != "mlx" and gpus > len(hardware["gpus"]):
        issues.append("请求 GPU 数超过当前主机已探测数量")
    if backend != "mlx":
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpus))
    if backend == "swift":
        if not versions.get("ms-swift"):
            issues.append("后端环境缺少 ms-swift；不是 macOS 同名 swift 编译器")
        elif not versions["ms-swift"].startswith("4.5."):
            issues.append("当前配方审计目标为 ms-swift 4.5.x；其他版本须适配参数契约")
        if task == "inference":
            raise ValueError("推理请选择 vllm / sglang / mlx")
        if task == "ppo":
            issues.append("PPO 还需独立奖励/价值模型、数据模板与角色放置审计；本版不生成不完整启动命令")
        if task == "quantize":
            method = request.get("method", "awq")
            if method not in METHODS:
                raise ValueError("未知压缩方法")
            if method not in ("awq", "gptq", "fp8"):
                issues.append("此方法暂无可执行 swift 适配器")
            argv += ["-m", "swift.cli.export", "--model", model["path"], "--quant_method", method,
                     "--output_dir", str(output)]
            if method != "fp8":
                argv += ["--quant_bits", "4"]
            if method in ("awq", "gptq"):
                dataset = validate_dataset(request.get("dataset", ""), "sft", roots)
                argv += ["--dataset", dataset["path"], "--quant_n_samples", "128", "--max_length", str(seq)]
                if dataset["rows"] < 128:
                    issues.append("AWQ/GPTQ 至少准备本配方要求的 128 条代表性校准样本")
            warnings.append("压缩结果需重新扫描并运行相同负载/质量集；小文件不代表加速")
        else:
            dataset = validate_dataset(request.get("dataset", ""), task, roots)
            module = "pt" if task == "cpt" else "sft" if task == "sft" else "rlhf"
            if gpus > 1:
                argv += ["-m", "torch.distributed.run", "--nproc_per_node", str(gpus), "--standalone", "--module", "swift.cli." + module]
                if not versions.get("deepspeed"):
                    issues.append("多卡训练配方需要 deepspeed")
            else:
                argv += ["-m", "swift.cli." + module]
            argv += ["--model", model["path"], "--dataset", dataset["path"], "--tuner_type", tuner,
                     "--torch_dtype", "bfloat16", "--max_steps", str(steps), "--max_length", str(seq),
                     "--per_device_train_batch_size", str(batch), "--gradient_accumulation_steps", str(grad_acc),
                     "--learning_rate", str(lr), "--logging_steps", "1", "--save_steps", str(min(steps, 100)),
                     "--save_total_limit", "2", "--output_dir", str(output), "--report_to", "none"]
            if gpus > 1:
                argv += ["--deepspeed", "zero3" if tuner == "full" else "zero2"]
            if tuner == "lora":
                argv += ["--lora_rank", str(integer(request, "rank", 8, 1, 256)), "--lora_alpha", "32", "--target_modules", "all-linear"]
            if task in ("dpo", "grpo", "ppo"):
                argv += ["--rlhf_type", task]
            if task == "grpo":
                generations = integer(request, "generations", 4, 2, 64)
                if batch * gpus * grad_acc % generations:
                    issues.append("全局有效 batch 必须可被 num_generations 整除")
                argv += ["--reward_funcs", "accuracy", "format", "--num_generations", str(generations),
                         "--max_completion_length", "256", "--temperature", "0.9"]
                if not versions.get("math-verify"):
                    issues.append("数学 accuracy 奖励需要 math-verify")
                warnings.append("GRPO 首版是有标准答案的数学奖励；不能泛化成任意任务的质量奖励")
    elif backend in ("vllm", "sglang"):
        if task != "inference":
            raise ValueError("该后端只处理推理任务")
        if not versions.get(backend):
            issues.append("后端环境未安装 " + backend)
        if tp != gpus:
            issues.append("首版推理仅支持单实例 TP；请令 TP 等于选择的 GPU 数")
        cfg = model["config"]
        heads = cfg.get("num_attention_heads")
        if not heads or heads % tp:
            issues.append("TP 无法按已知 Query 头数均分，需要专用适配")
        kv = cfg.get("num_key_value_heads", heads)
        if kv and kv % tp and tp % kv:
            issues.append("KV 头数与 TP 不满足均分或复制条件")
        argv += ["-m", MODULES[backend]]
        if backend == "vllm":
            argv += ["--model", model["path"], "--served-model-name", "workbench", "--host", "127.0.0.1", "--port", str(port),
                     "--tensor-parallel-size", str(tp), "--max-model-len", str(seq), "--gpu-memory-utilization", "0.80"]
        else:
            argv += ["--model-path", model["path"], "--served-model-name", "workbench", "--host", "127.0.0.1", "--port", str(port),
                     "--tp-size", str(tp), "--context-length", str(seq), "--mem-fraction-static", "0.80", "--enable-metrics"]
    else:
        if mt not in TEXT and not gemma4_text_inference:
            issues.append("mlx-lm 适配器仅用于文本语言模型；视觉语言模型需独立 mlx-vlm 适配")
        if not hardware.get("apple_unified_memory"):
            issues.append("MLX 配方仅用于 Apple Silicon")
        if not versions.get("mlx-lm"):
            issues.append("后端环境缺少 mlx-lm")
        if task == "quantize":
            argv += ["-m", "mlx_lm.convert", "--hf-path", model["path"], "--mlx-path", str(output), "-q", "--q-bits", "4", "--q-group-size", "64"]
        elif task == "inference":
            argv += ["-m", "mlx_lm.server", "--model", model["path"], "--host", "127.0.0.1", "--port", str(port)]
        else:
            raise ValueError("MLX 后端本版只接推理与量化")
    if dataset:
        issues += ["数据第 %s 行: %s" % (e["line"], e["error"]) for e in dataset["errors"]]
    if any("1070" in g["name"] for g in hardware["gpus"]):
        issues.append("GTX 1070 不支持本配方的 BF16/现代内核要求；请换受支持 GPU")
    warnings.append("配方级兼容不是硬件验收；当前项目未取得本模型 GPU smoke/质量/吞吐认证")
    explanation = None
    if task in ('inference','quantize'):
        explanation = technique_guide(backend if backend in MODULES else 'vllm',request.get('method','awq'))
        explanation['command_contract'] = {'tp':tp if task=='inference' and backend in ('vllm','sglang') else None,
           'input_quantization':model['quantization'], 'runtime_features':'自动特性不算已确认启用；以此argv、后端版本、启动日志与实际kernel为准',
           'quantization_requested':task=='quantize','measured_speedup':None}
    return {"task": task, "backend": backend, "argv": argv, "env": env, "output": str(output),
            "technique_guide":explanation,
            "backend_versions": versions, "dataset": dataset, "blockers": list(dict.fromkeys(issues)),
            "warnings": list(dict.fromkeys(warnings)), "support_level": "recipe-unverified" if not issues else "blocked",
            "source": REFERENCE[backend], "method_explanation": METHODS.get(request.get("method")),
            "profile": request.get("profile", "none")}
