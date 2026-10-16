# 模型目录与部署显存公式对照

快照日期：2026-10-16。按当前本地目录和配置分类，逐条对应计算手册；不是全球全部模型清单，也不是实机兼容认证。

共 **413 条模型仓库记录**：容量可解析 163，元数据待补 196，适配待补 50，独立上游配方 4。同架构、不同大小或量化版本各占一条；不是 413 种不同架构。

[返回计算手册](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories)

## 如何读每条记录

A—G 是计算路径，不是厂商分类。每条可解析记录列出实际缓存字段与统一小例：**TP=PP=1、B=1、总上下文4096、BF16 KV、16-token页；MLA/DSA至少64-token页并额外预留一尾页**。循环矩阵按当前适配器FP32口径。c为bytes/token，r为bytes/request。缓存GiB不包含权重、激活或运行时，不能据此直接选卡。

所有记录还需计算 P×位宽/8、量化元数据、激活与运行时，再按合法TP/PP搜索最忙rank容量。参数B是目录/缓存口径，未逐权重核验；量化记录尤其不能把packed元素数当原参数。未知参数量保持未知。A/B/C滑窗不计驱逐收益，C不计跨层共享/K=V收益。

模型类型为文本解码器时只覆盖文本路径；多模态编码、扩散、ASR、embedding或rerank任务不能由解码器缓存例推导整任务容量。R/U/M不展示伪造缓存值，需按手册的补齐路径处理。

M 表的“缺项”沿用当前规划器通用结构检查；对于编码器、扩散或音频等任务，这些缺项不表示模型应该拥有自回归KV字段，必须先选择正确任务适配器。“需补计算部分”依据已有model_type/状态字段分流，不把推测当已实现公式。

| 类别 | 数量 | 计算或补齐路径 |
| --- | ---: | --- |
| [A 标准 MHA/GQA/MQA](#category-a) | 102 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [B 标准 KV 加 MoE 权重与临时量](#category-b) | 20 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [C 逐层异构 KV](#category-c) | 13 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [D MLA 压缩缓存](#category-d) | 14 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [E MLA 加 DSA 索引](#category-e) | 6 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [F MLA 加 KDA 循环状态](#category-f) | 3 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [G 全注意力加 GDN 循环状态](#category-g) | 5 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [R 只有独立上游部署配方](#category-r) | 4 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [U 结构存在但尚未适配](#category-u) | 50 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |
| [M 配置或关键字段缺失](#category-m) | 196 | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |

<a id="category-a"></a>

## A 标准 MHA/GQA/MQA（102 条）

| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |
| --- | ---: | --- | ---: | ---: |
| 01-ai/Yi-1.5-34B · `llama` | 34.389 | L=60; Hkv=8; dh=128 | 245,760 / 0 | 0.9375 |
| 01-ai/Yi-1.5-34B-Chat · `llama` | 34.389 | L=60; Hkv=8; dh=128 | 245,760 / 0 | 0.9375 |
| 01-ai/Yi-1.5-6B · `llama` | 6.061 | L=32; Hkv=4; dh=128 | 65,536 / 0 | 0.2500 |
| 01-ai/Yi-1.5-9B · `llama` | 8.829 | L=48; Hkv=4; dh=128 | 98,304 / 0 | 0.3750 |
| 01-ai/Yi-1.5-9B-Chat · `llama` | 8.829 | L=48; Hkv=4; dh=128 | 98,304 / 0 | 0.3750 |
| 01-ai/Yi-34B · `llama` | 34.389 | L=60; Hkv=8; dh=128 | 245,760 / 0 | 0.9375 |
| 01-ai/Yi-34B-200K · `llama` | 34.389 | L=60; Hkv=8; dh=128 | 245,760 / 0 | 0.9375 |
| 01-ai/Yi-34B-Chat · `llama` | 34.389 | L=60; Hkv=8; dh=128 | 245,760 / 0 | 0.9375 |
| 01-ai/Yi-6B · `llama` | 6.061 | L=32; Hkv=4; dh=128 | 65,536 / 0 | 0.2500 |
| 01-ai/Yi-6B-200K · `llama` | 6.061 | L=32; Hkv=4; dh=128 | 65,536 / 0 | 0.2500 |
| 01-ai/Yi-6B-Chat · `llama` | 6.061 | L=32; Hkv=4; dh=128 | 65,536 / 0 | 0.2500 |
| 01-ai/Yi-Coder-9B-Chat · `llama` | 8.829 | L=48; Hkv=4; dh=128 | 98,304 / 0 | 0.3750 |
| ByteDance-Seed/Seed-Coder-8B-Instruct · `llama` | 8.250 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| ByteDance-Seed/Seed-OSS-36B-Base · `seed_oss` | 36.151 | L=64; Hkv=8; dh=128 | 262,144 / 0 | 1.0000 |
| ByteDance-Seed/Seed-OSS-36B-Instruct · `seed_oss` | 36.151 | L=64; Hkv=8; dh=128 | 262,144 / 0 | 1.0000 |
| ByteDance-Seed/Seed-X-Instruct-7B · `mistral` | 7.514 | L=32; Hkv=8; dh=128; SW全历史上界 | 131,072 / 0 | 0.5000 |
| ByteDance-Seed/Seed-X-PPO-7B · `mistral` | 7.514 | L=32; Hkv=8; dh=128; SW全历史上界 | 131,072 / 0 | 0.5000 |
| ByteDance-Seed/UI-TARS-1.5-7B · `qwen2_5_vl` | 8.292 | L=28; Hkv=4; dh=128; SW全历史上界 | 57,344 / 0 | 0.2188 |
| ByteDance-Seed/UI-TARS-2B-SFT · `qwen2_vl` | 2.442 | L=28; Hkv=2; dh=128; SW全历史上界 | 28,672 / 0 | 0.1094 |
| ByteDance/Sa2VA-Qwen2_5-VL-3B · `qwen2_5_vl_text` | 4.294 | L=36; Hkv=2; dh=128 | 36,864 / 0 | 0.1406 |
| FunAudioLLM/Fun-ASR-Nano-2512-vllm · `qwen3` | 0.985 | L=28; Hkv=8; dh=128 | 114,688 / 0 | 0.4375 |
| HuggingFaceTB/SmolLM-135M · `llama` | 0.135 | L=30; Hkv=3; dh=64 | 23,040 / 0 | 0.0879 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct · `llama` | 1.711 | L=24; Hkv=32; dh=64 | 196,608 / 0 | 0.7500 |
| HuggingFaceTB/SmolLM2-135M · `llama` | 0.135 | L=30; Hkv=3; dh=64 | 23,040 / 0 | 0.0879 |
| HuggingFaceTB/SmolLM2-135M-Instruct · `llama` | 0.135 | L=30; Hkv=3; dh=64 | 23,040 / 0 | 0.0879 |
| HuggingFaceTB/SmolLM2-360M · `llama` | 0.362 | L=32; Hkv=5; dh=64 | 40,960 / 0 | 0.1562 |
| HuggingFaceTB/SmolLM2-360M-Instruct · `llama` | 0.362 | L=32; Hkv=5; dh=64 | 40,960 / 0 | 0.1562 |
| HuggingFaceTB/SmolLM3-3B · `smollm3` | 3.075 | L=36; Hkv=4; dh=128 | 73,728 / 0 | 0.2812 |
| HuggingFaceTB/SmolLM3-3B-Base · `smollm3` | 3.075 | L=36; Hkv=4; dh=128 | 73,728 / 0 | 0.2812 |
| HuggingFaceTB/SmolVLM-256M-Instruct · `llama` | 0.256 | L=30; Hkv=3; dh=64 | 23,040 / 0 | 0.0879 |
| HuggingFaceTB/SmolVLM-500M-Instruct · `llama` | 0.507 | L=32; Hkv=5; dh=64 | 40,960 / 0 | 0.1562 |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct · `llama` | 0.507 | L=32; Hkv=5; dh=64 | 40,960 / 0 | 0.1562 |
| Lightricks/gemma-3-12b-it-qat-q4_0-unquantized · `gemma3_text` | 12.187 | L=48; Hkv=8; dh=256; SW全历史上界 | 393,216 / 0 | 1.5000 |
| OpenGVLab/InternVL3-1B-hf · `qwen2` | 0.938 | L=24; Hkv=2; dh=64 | 12,288 / 0 | 0.0469 |
| OpenGVLab/InternVL3-8B-hf · `qwen2` | 7.944 | L=28; Hkv=4; dh=128 | 57,344 / 0 | 0.2188 |
| Qwen/Qwen2.5-0.5B-Instruct · `qwen2` | 0.494 | L=24; Hkv=2; dh=64; SW全历史上界 | 12,288 / 0 | 0.0469 |
| Qwen/Qwen2.5-1.5B-Instruct · `qwen2` | 1.500 | L=28; Hkv=2; dh=128; SW全历史上界 | 28,672 / 0 | 0.1094 |
| Qwen/Qwen2.5-14B-Instruct · `qwen2` | 14.000 | L=48; Hkv=8; dh=128; SW全历史上界 | 196,608 / 0 | 0.7500 |
| Qwen/Qwen2.5-32B-Instruct · `qwen2` | 32.000 | L=64; Hkv=8; dh=128; SW全历史上界 | 262,144 / 0 | 1.0000 |
| Qwen/Qwen2.5-3B-Instruct · `qwen2` | 3.000 | L=36; Hkv=2; dh=128; SW全历史上界 | 36,864 / 0 | 0.1406 |
| Qwen/Qwen2.5-7B-Instruct · `qwen2` | 7.000 | L=28; Hkv=4; dh=128; SW全历史上界 | 57,344 / 0 | 0.2188 |
| Qwen/Qwen3-0.6B · `qwen3` | 0.600 | L=28; Hkv=8; dh=128 | 114,688 / 0 | 0.4375 |
| Qwen/Qwen3-1.7B · `qwen3` | 1.700 | L=28; Hkv=8; dh=128 | 114,688 / 0 | 0.4375 |
| Qwen/Qwen3-14B · `qwen3` | 14.000 | L=40; Hkv=8; dh=128 | 163,840 / 0 | 0.6250 |
| Qwen/Qwen3-32B · `qwen3` | 32.000 | L=64; Hkv=8; dh=128 | 262,144 / 0 | 1.0000 |
| Qwen/Qwen3-4B · `qwen3` | 4.000 | L=36; Hkv=8; dh=128 | 147,456 / 0 | 0.5625 |
| Qwen/Qwen3-8B · `qwen3` | 8.000 | L=36; Hkv=8; dh=128 | 147,456 / 0 | 0.5625 |
| Qwen/Qwen3-Embedding-0.6B · `qwen3` | 0.596 | L=28; Hkv=8; dh=128 | 114,688 / 0 | 0.4375 |
| Qwen/Qwen3-VL-8B-Instruct · `qwen3_vl_text` | 8.767 | L=36; Hkv=8; dh=128 | 147,456 / 0 | 0.5625 |
| XiaomiMiMo/MiMo-7B-Base · `mimo` | 7.833 | L=36; Hkv=8; dh=128; SW全历史上界 | 147,456 / 0 | 0.5625 |
| XiaomiMiMo/MiMo-7B-RL · `mimo` | 7.833 | L=36; Hkv=8; dh=128; SW全历史上界 | 147,456 / 0 | 0.5625 |
| allenai/OLMo-2-0425-1B · `olmo2` | 1.485 | L=16; Hkv=16; dh=128 | 131,072 / 0 | 0.5000 |
| allenai/OLMo-2-1124-7B · `olmo2` | 7.299 | L=32; Hkv=32; dh=128 | 524,288 / 0 | 2.0000 |
| allenai/Olmo-3-1025-7B · `olmo3` | 7.298 | L=32; Hkv=32; dh=128; SW全历史上界 | 524,288 / 0 | 2.0000 |
| allenai/Olmo-3-7B-Instruct · `olmo3` | 7.298 | L=32; Hkv=32; dh=128; SW全历史上界 | 524,288 / 0 | 2.0000 |
| allenai/Olmo-3-7B-Think · `olmo3` | 7.298 | L=32; Hkv=32; dh=128; SW全历史上界 | 524,288 / 0 | 2.0000 |
| allenai/wildguard · `mistral` | 7.248 | L=32; Hkv=8; dh=128; SW全历史上界 | 131,072 / 0 | 0.5000 |
| baichuan-inc/Baichuan-M2-32B · `qwen2` | 32.764 | L=64; Hkv=8; dh=128; SW全历史上界 | 262,144 / 0 | 1.0000 |
| baidu/ERNIE-4.5-0.3B-Base-PT · `ernie4_5` | 0.361 | L=18; Hkv=2; dh=128 | 18,432 / 0 | 0.0703 |
| baidu/ERNIE-4.5-0.3B-PT · `ernie4_5` | 0.361 | L=18; Hkv=2; dh=128 | 18,432 / 0 | 0.0703 |
| baidu/Qianfan-OCR · `qwen3` | 4.741 | L=36; Hkv=8; dh=128 | 147,456 / 0 | 0.5625 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B · `qwen2` | 1.777 | L=28; Hkv=2; dh=128; SW全历史上界 | 28,672 / 0 | 0.1094 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B · `qwen2` | 14.000 | L=48; Hkv=8; dh=128; SW全历史上界 | 196,608 / 0 | 0.7500 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B · `qwen2` | 32.000 | L=64; Hkv=8; dh=128; SW全历史上界 | 262,144 / 0 | 1.0000 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B · `qwen2` | 7.000 | L=28; Hkv=4; dh=128; SW全历史上界 | 57,344 / 0 | 0.2188 |
| deepseek-ai/deepseek-coder-7b-instruct-v1.5 · `llama` | 6.910 | L=30; Hkv=32; dh=128 | 491,520 / 0 | 1.8750 |
| google/embeddinggemma-300m · `gemma3_text` | 0.308 | L=24; Hkv=1; dh=256; SW全历史上界 | 24,576 / 0 | 0.0938 |
| google/gemma-3-12b-it · `gemma3_text` | 12.187 | L=48; Hkv=8; dh=240; SW全历史上界 | 368,640 / 0 | 1.4062 |
| google/gemma-3-1b-it · `gemma3_text` | 1.000 | L=26; Hkv=1; dh=256; SW全历史上界 | 26,624 / 0 | 0.1016 |
| google/gemma-3-27b-it · `gemma3_text` | 27.000 | L=62; Hkv=16; dh=128; SW全历史上界 | 507,904 / 0 | 1.9375 |
| ibm-granite/granite-3.3-2b-instruct · `granite` | 2.534 | L=40; Hkv=8; dh=64 | 81,920 / 0 | 0.3125 |
| ibm-granite/granite-4.1-30b · `granite` | 28.866 | L=64; Hkv=8; dh=128 | 262,144 / 0 | 1.0000 |
| ibm-granite/granite-4.1-3b · `granite` | 3.403 | L=40; Hkv=8; dh=64 | 81,920 / 0 | 0.3125 |
| ibm-granite/granite-docling-258M · `llama` | 0.258 | L=30; Hkv=3; dh=64 | 23,040 / 0 | 0.0879 |
| ibm-granite/granite-vision-3.3-2b · `granite` | 2.975 | L=40; Hkv=8; dh=64 | 81,920 / 0 | 0.3125 |
| internlm/Intern-S1-mini · `qwen3` | 8.539 | L=36; Hkv=8; dh=128 | 147,456 / 0 | 0.5625 |
| internlm/internlm2-1_8b-reward · `internlm2` | 1.700 | L=24; Hkv=8; dh=128 | 98,304 / 0 | 0.3750 |
| internlm/internlm2-20b · `internlm2` | 未知 | L=48; Hkv=8; dh=128 | 196,608 / 0 | 0.7500 |
| internlm/internlm2-7b · `internlm2` | 未知 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| internlm/internlm2-base-20b · `internlm2` | 未知 | L=48; Hkv=8; dh=128 | 196,608 / 0 | 0.7500 |
| internlm/internlm2-base-7b · `internlm2` | 未知 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| internlm/internlm2-chat-20b · `internlm2` | 19.861 | L=48; Hkv=8; dh=128 | 196,608 / 0 | 0.7500 |
| internlm/internlm2-chat-7b · `internlm2` | 7.738 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| internlm/internlm2_5-7b-chat · `internlm2` | 7.738 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| internlm/internlm3-8b-instruct · `internlm3` | 8.804 | L=48; Hkv=2; dh=128 | 49,152 / 0 | 0.1875 |
| microsoft/Phi-3.5-mini-instruct · `phi3` | 3.800 | L=32; Hkv=32; dh=96; SW全历史上界 | 393,216 / 0 | 1.5000 |
| microsoft/Phi-4-mini-instruct · `phi3` | 3.800 | L=32; Hkv=8; dh=128; SW全历史上界 | 131,072 / 0 | 0.5000 |
| mistralai/Mistral-7B-Instruct-v0.1 · `mistral` | 7.242 | L=32; Hkv=8; dh=128; SW全历史上界 | 131,072 / 0 | 0.5000 |
| mistralai/Mistral-7B-Instruct-v0.2 · `mistral` | 7.242 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| mistralai/Mistral-7B-Instruct-v0.3 · `mistral` | 7.000 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| mistralai/Mistral-7B-v0.1 · `mistral` | 7.242 | L=32; Hkv=8; dh=128; SW全历史上界 | 131,072 / 0 | 0.5000 |
| mistralai/Mistral-7B-v0.3 · `mistral` | 7.248 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| mistralai/Mistral-Nemo-Instruct-2407 · `mistral` | 12.248 | L=40; Hkv=8; dh=128 | 163,840 / 0 | 0.6250 |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 · `mistral` | 24.011 | L=40; Hkv=8; dh=128 | 163,840 / 0 | 0.6250 |
| mistralai/Mistral-Small-3.2-24B-Instruct-2506 · `mistral` | 24.011 | L=40; Hkv=8; dh=128 | 163,840 / 0 | 0.6250 |
| mistralai/Voxtral-Mini-3B-2507 · `llama` | 4.676 | L=30; Hkv=8; dh=128 | 122,880 / 0 | 0.4688 |
| nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 · `qwen2_5_vl_text` | 未知 | L=28; Hkv=4; dh=128 | 57,344 / 0 | 0.2188 |
| stepfun-ai/GOT-OCR-2.0-hf · `qwen2` | 0.561 | L=24; Hkv=16; dh=64 | 98,304 / 0 | 0.3750 |
| stepfun-ai/Step3-VL-10B · `qwen3` | 10.172 | L=36; Hkv=8; dh=128 | 147,456 / 0 | 0.5625 |
| tiiuae/Falcon3-10B-Instruct · `llama` | 10.306 | L=40; Hkv=4; dh=256 | 163,840 / 0 | 0.6250 |
| tiiuae/Falcon3-1B-Instruct · `llama` | 1.669 | L=18; Hkv=4; dh=256 | 73,728 / 0 | 0.2812 |
| tiiuae/Falcon3-7B-Instruct · `llama` | 7.456 | L=28; Hkv=4; dh=256 | 114,688 / 0 | 0.4375 |

<a id="category-b"></a>

## B 标准 KV 加 MoE 权重与临时量（20 条）

| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |
| --- | ---: | --- | ---: | ---: |
| MiniMaxAI/MiniMax-M2 · `minimax_m2` | 230.000 | L=62; Hkv=8; dh=128 | 253,952 / 0 | 0.9688 |
| MiniMaxAI/MiniMax-M2.1 · `minimax_m2` | 未知 | L=62; Hkv=8; dh=128 | 253,952 / 0 | 0.9688 |
| MiniMaxAI/MiniMax-M2.5 · `minimax_m2` | 未知 | L=62; Hkv=8; dh=128 | 253,952 / 0 | 0.9688 |
| MiniMaxAI/MiniMax-M2.7 · `minimax_m2` | 未知 | L=62; Hkv=8; dh=128 | 253,952 / 0 | 0.9688 |
| OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF · `gpt_oss` | 21.233 | L=24; Hkv=8; dh=64; SW全历史上界 | 49,152 / 0 | 0.1875 |
| Qwen/Qwen3-235B-A22B · `qwen3_moe` | 235.094 | L=94; Hkv=4; dh=128 | 192,512 / 0 | 0.7344 |
| Qwen/Qwen3-30B-A3B · `qwen3_moe` | 30.000 | L=48; Hkv=4; dh=128 | 98,304 / 0 | 0.3750 |
| allenai/OLMoE-1B-7B-0125-Instruct · `olmoe` | 6.919 | L=16; Hkv=16; dh=128 | 131,072 / 0 | 0.5000 |
| allenai/OLMoE-1B-7B-0924 · `olmoe` | 6.919 | L=16; Hkv=16; dh=128 | 131,072 / 0 | 0.5000 |
| baichuan-inc/Baichuan-M3-235B · `qwen3_moe` | 235.094 | L=94; Hkv=4; dh=128 | 192,512 / 0 | 0.7344 |
| baichuan-inc/Baichuan-M3-235B-GPTQ-INT4 · `qwen3_moe` | 未知 | L=94; Hkv=4; dh=128 | 192,512 / 0 | 0.7344 |
| baidu/ERNIE-4.5-21B-A3B-PT · `ernie4_5_moe` | 21.949 | L=28; Hkv=4; dh=128 | 57,344 / 0 | 0.2188 |
| baidu/ERNIE-4.5-21B-A3B-Thinking · `ernie4_5_moe` | 21.825 | L=28; Hkv=4; dh=128 | 57,344 / 0 | 0.2188 |
| baidu/ERNIE-4.5-300B-A47B-PT · `ernie4_5_moe` | 300.474 | L=54; Hkv=8; dh=128 | 221,184 / 0 | 0.8438 |
| internlm/Intern-S1 · `qwen3_moe` | 240.710 | L=94; Hkv=4; dh=128 | 192,512 / 0 | 0.7344 |
| mistralai/Mixtral-8x7B-Instruct-v0.1 · `mixtral` | 46.700 | L=32; Hkv=8; dh=128 | 131,072 / 0 | 0.5000 |
| openai/gpt-oss-120b · `gpt_oss` | 未知 | L=36; Hkv=8; dh=64; SW全历史上界 | 73,728 / 0 | 0.2812 |
| openai/gpt-oss-20b · `gpt_oss` | 未知 | L=24; Hkv=8; dh=64; SW全历史上界 | 49,152 / 0 | 0.1875 |
| zai-org/GLM-4.5 · `glm4_moe` | 358.338 | L=92; Hkv=8; dh=128 | 376,832 / 0 | 1.4375 |
| zai-org/GLM-4.5-Air · `glm4_moe` | 110.469 | L=46; Hkv=8; dh=128 | 188,416 / 0 | 0.7188 |

<a id="category-c"></a>

## C 逐层异构 KV（13 条）

| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |
| --- | ---: | --- | ---: | ---: |
| XiaomiMiMo/MiMo-V2-Flash · `mimo_v2_flash` | 未知 | 9×full(Hkv=4,dk=192,dv=128); 39×SW(Hkv=8,dk=192,dv=128) | 222,720 / 0 | 0.8496 |
| XiaomiMiMo/MiMo-V2.5 · `mimo_v2` | 未知 | 9×full(Hkv=4,dk=192,dv=128); 39×SW(Hkv=8,dk=192,dv=128) | 222,720 / 0 | 0.8496 |
| XiaomiMiMo/MiMo-V2.5-Pro · `mimo_v2` | 未知 | 10×full(Hkv=8,dk=192,dv=128); 60×SW(Hkv=8,dk=192,dv=128) | 358,400 / 0 | 1.3672 |
| XiaomiMiMo/MiMo-V2.6-Flash-MOPD · `mimo_v2` | 未知 | 9×full(Hkv=4,dk=192,dv=128); 39×SW(Hkv=8,dk=192,dv=128) | 222,720 / 0 | 0.8496 |
| XiaomiMiMo/MiMo-V2.6-Flash-RL · `mimo_v2` | 未知 | 9×full(Hkv=4,dk=192,dv=128); 39×SW(Hkv=8,dk=192,dv=128) | 222,720 / 0 | 0.8496 |
| XiaomiMiMo/MiMo-V2.6-Pro-MOPD · `mimo_v2` | 未知 | 10×full(Hkv=8,dk=192,dv=128); 60×SW(Hkv=8,dk=192,dv=128) | 358,400 / 0 | 1.3672 |
| XiaomiMiMo/MiMo-V2.6-Pro-RL · `mimo_v2` | 1020.000 | 10×full(Hkv=8,dk=192,dv=128); 60×SW(Hkv=8,dk=192,dv=128) | 358,400 / 0 | 1.3672 |
| google/gemma-4-26B-A4B-it · `gemma4_text` | 25.806 | 25×SW(Hkv=8,dk=256,dv=256); 5×full(Hkv=2,dk=512,dv=512) | 225,280 / 0 | 0.8594 |
| google/gemma-4-31B-it · `gemma4_text` | 31.000 | 50×SW(Hkv=16,dk=256,dv=256); 10×full(Hkv=4,dk=512,dv=512) | 901,120 / 0 | 3.4375 |
| google/gemma-4-E2B-it · `gemma4_text` | 5.123 | 28×SW(Hkv=1,dk=256,dv=256); 7×full(Hkv=1,dk=512,dv=512) | 43,008 / 0 | 0.1641 |
| google/gemma-4-E4B-it · `gemma4_text` | 7.996 | 35×SW(Hkv=2,dk=256,dv=256); 7×full(Hkv=2,dk=512,dv=512) | 100,352 / 0 | 0.3828 |
| nvidia/Gemma-4-26B-A4B-NVFP4 · `gemma4_text` | 未知 | 25×SW(Hkv=8,dk=256,dv=256); 5×full(Hkv=2,dk=512,dv=512) | 225,280 / 0 | 0.8594 |
| nvidia/Gemma-4-31B-IT-NVFP4 · `gemma4_text` | 未知 | 50×SW(Hkv=16,dk=256,dv=256); 10×full(Hkv=4,dk=512,dv=512) | 901,120 / 0 | 3.4375 |

<a id="category-d"></a>

## D MLA 压缩缓存（14 条）

| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |
| --- | ---: | --- | ---: | ---: |
| ByteDance-Seed/academic-ds-9B · `deepseek_v3` | 9.366 | L=16; latent=512; RoPE=64 | 18,432 / 0 | 0.0714 |
| deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct · `deepseek_v2` | 15.706 | L=27; latent=512; RoPE=64 | 31,104 / 0 | 0.1205 |
| deepseek-ai/DeepSeek-R1 · `deepseek_v3` | 未知 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| deepseek-ai/DeepSeek-V3 · `deepseek_v3` | 未知 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| deepseek-ai/DeepSeek-V3-0324 · `deepseek_v3` | 未知 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| moonshotai/Kimi-K2-Instruct · `kimi_k2` | 未知 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| moonshotai/Kimi-K2-Instruct-0905 · `kimi_k2` | 未知 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| moonshotai/Kimi-K2-Thinking · `kimi_k2` | 未知 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| moonshotai/Kimi-K2.5 · `kimi_k2` | 170.738 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| moonshotai/Kimi-K2.6 · `kimi_k2` | 170.738 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| moonshotai/Kimi-K2.7-Code · `kimi_k2` | 170.738 | L=61; latent=512; RoPE=64 | 70,272 / 0 | 0.2723 |
| moonshotai/Moonlight-16B-A3B · `deepseek_v3` | 15.960 | L=27; latent=512; RoPE=64 | 31,104 / 0 | 0.1205 |
| moonshotai/Moonlight-16B-A3B-Instruct · `deepseek_v3` | 15.960 | L=27; latent=512; RoPE=64 | 31,104 / 0 | 0.1205 |
| zai-org/GLM-4.7-Flash · `glm4_moe_lite` | 31.221 | L=47; latent=512; RoPE=64 | 54,144 / 0 | 0.2098 |

<a id="category-e"></a>

## E MLA 加 DSA 索引（6 条）

| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |
| --- | ---: | --- | ---: | ---: |
| deepseek-ai/DeepSeek-V3.2 · `deepseek_v32` | 未知 | L=61; latent=512; RoPE=64; index=128 | 86,132 / 0 | 0.3337 |
| zai-org/GLM-5 · `glm_moe_dsa` | 753.864 | L=78; latent=512; RoPE=64; index=128 | 110,136 / 0 | 0.4267 |
| zai-org/GLM-5.1 · `glm_moe_dsa` | 753.864 | L=78; latent=512; RoPE=64; index=128 | 110,136 / 0 | 0.4267 |
| zai-org/GLM-5.2 · `glm_moe_dsa` | 753.330 | L=78; latent=512; RoPE=64; index=128 | 110,136 / 0 | 0.4267 |
| zai-org/GLM-5.2-FP8 · `glm_moe_dsa` | 未知 | L=78; latent=512; RoPE=64; index=128 | 110,136 / 0 | 0.4267 |
| zai-org/GLM-5.3 · `glm_moe_dsa` | 未知 | L=78; latent=512; RoPE=64; index=128 | 110,136 / 0 | 0.4267 |

<a id="category-f"></a>

## F MLA 加 KDA 循环状态（3 条）

| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |
| --- | ---: | --- | ---: | ---: |
| moonshotai/Kimi-K3 · `kimi_linear` | 2779.932 | L=93; full=24; recurrent=69 | 27,648 / 449,372,160 | 0.5240 |
| moonshotai/Kimi-Linear-48B-A3B-Base · `kimi_linear` | 49.123 | L=27; full=7; recurrent=20 | 8,064 / 43,417,600 | 0.0712 |
| moonshotai/Kimi-Linear-48B-A3B-Instruct · `kimi_linear` | 49.123 | L=27; full=7; recurrent=20 | 8,064 / 43,417,600 | 0.0712 |

<a id="category-g"></a>

## G 全注意力加 GDN 循环状态（5 条）

| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |
| --- | ---: | --- | ---: | ---: |
| Qwen/Qwen3.5-4B · `qwen3_5_text` | 4.660 | L=32; full=8; recurrent=24 | 32,768 / 51,511,296 | 0.1730 |
| Qwen/Qwen3.5-9B · `qwen3_5_text` | 9.653 | L=32; full=8; recurrent=24 | 32,768 / 51,511,296 | 0.1730 |
| Qwen/Qwen3.8-27B · `qwen3_5_text` | 27.781 | L=64; full=16; recurrent=48 | 65,536 / 153,944,064 | 0.3934 |
| XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B · `qwen3_5_text` | 9.410 | L=32; full=8; recurrent=24 | 32,768 / 51,511,296 | 0.1730 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 · `qwen3_5_moe_text` | 未知 | L=40; full=10; recurrent=30 | 20,480 / 64,389,120 | 0.1381 |

<a id="category-r"></a>

## R 只有独立上游部署配方（4 条）

| 模型 | 配置类型 | 当前缺项 | 需补的计算部分（非已实现） |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-V4-Flash | `deepseek_v4` | 固定上游配方；不能继承当前负载/精度 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| deepseek-ai/DeepSeek-V4-Flash-0731 | `deepseek_v4` | 固定上游配方；不能继承当前负载/精度 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| deepseek-ai/DeepSeek-V4-Pro | `deepseek_v4` | 固定上游配方；不能继承当前负载/精度 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| deepseek-ai/DeepSeek-V4.1-Flash | `deepseek_v41_text` | 固定上游配方；不能继承当前负载/精度 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |

<a id="category-u"></a>

## U 结构存在但尚未适配（50 条）

| 模型 | 配置类型 | 当前缺项 | 需补的计算部分（非已实现） |
| --- | --- | --- | --- |
| ByteDance/Ouro-1.4B | `ouro` | 逐层缓存/循环状态与后端布局 | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| ByteDance/Ouro-1.4B-Thinking | `ouro` | 逐层缓存/循环状态与后端布局 | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| ByteDance/Ouro-2.6B | `ouro` | 逐层缓存/循环状态与后端布局 | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| ByteDance/Ouro-2.6B-Thinking | `ouro` | 逐层缓存/循环状态与后端布局 | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| FunAudioLLM/Fun-ASR-Nano-2512-hf | `qwen3` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| MiniMaxAI/MiniMax-M1-40k | `minimax_m1` | 逐层缓存/循环状态与后端布局 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| MiniMaxAI/MiniMax-M3 | `未知` | 逐层缓存/循环状态与后端布局 | 未知结构：先获取身份可核对的配置与任务类型 |
| MiniMaxAI/MiniMax-M3-MXFP8 | `未知` | 逐层缓存/循环状态与后端布局 | 未知结构：先获取身份可核对的配置与任务类型 |
| MiniMaxAI/MiniMax-Text-01 | `minimax_text_01` | 逐层缓存/循环状态与后端布局 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| MiniMaxAI/MiniMax-Text-01-hf | `minimax` | 逐层缓存/循环状态与后端布局 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| MiniMaxAI/MiniMax-VL-01 | `minimax_text_01` | 逐层缓存/循环状态与后端布局 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| XiaomiMiMo/MiMo-V2.5-ASR | `qwen2` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baichuan-inc/Baichuan-M1-14B-Instruct | `baichuan_m1` | 逐层缓存/循环状态与后端布局 | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| baidu/ERNIE-4.5-VL-28B-A3B-PT | `ernie4_5_moe_vl` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| baidu/ERNIE-4.5-VL-28B-A3B-Thinking | `ernie4_5_moe_vl` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| baidu/ERNIE-4.5-VL-424B-A47B-PT | `ernie4_5_moe_vl` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| baidu/Unlimited-OCR | `unlimited_ocr` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| deepseek-ai/DeepSeek-OCR | `deepseek_vl_v2` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| deepseek-ai/DeepSeek-OCR-2 | `deepseek_vl_v2` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| fishaudio/drama-3 | `fish_omni_text` | 逐层缓存/循环状态与后端布局 | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/drama-3-int4 | `fish_omni_text` | 逐层缓存/循环状态与后端布局 | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| google/gemma-4-12B-it | `gemma4_unified_text` | 逐层缓存/循环状态与后端布局 | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| microsoft/Phi-3.5-vision-instruct | `phi3_v` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| mistralai/Devstral-Small-2-24B-Instruct-2512 | `ministral3` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | `voxtral_realtime_text` | 逐层缓存/循环状态与后端布局 | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| moonshotai/Kimi-VL-A3B-Instruct | `未知` | 逐层缓存/循环状态与后端布局 | 未知结构：先获取身份可核对的配置与任务类型 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | `nemotron_h` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | `nemotron_h` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | `nemotron_h` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | `nemotron_h` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | `nemotron_h` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| openai/privacy-filter | `openai_privacy_filter` | 逐层缓存/循环状态与后端布局 | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| stepfun-ai/GOT-OCR2_0 | `GOT` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| stepfun-ai/Step-Audio-2-mini | `未知` | 逐层缓存/循环状态与后端布局 | 未知结构：先获取身份可核对的配置与任务类型 |
| stepfun-ai/Step-Audio-R1.1 | `未知` | 逐层缓存/循环状态与后端布局 | 未知结构：先获取身份可核对的配置与任务类型 |
| tencent/Hunyuan-A13B-Instruct | `hunyuan_v1_moe` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| tencent/HunyuanOCR | `hunyuan_vl_text` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| tencent/Hy-MT2-1.8B | `hunyuan_v1_dense` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| tencent/Hy3 | `hy_v3` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| tencent/Hy3-FP8 | `hy_v3` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| tencent/Hy3-preview | `hy_v3` | 逐层缓存/循环状态与后端布局 | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| tiiuae/Falcon-H1-0.5B-Base | `falcon_h1` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| tiiuae/Falcon-H1-1.5B-Base | `falcon_h1` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| tiiuae/Falcon-H1-1.5B-Deep-Instruct | `falcon_h1` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| tiiuae/Falcon-H1-3B-Instruct | `falcon_h1` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| tiiuae/Falcon-H1-7B-Instruct | `falcon_h1` | 逐层缓存/循环状态与后端布局 | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| zai-org/GLM-4.1V-9B-Thinking | `glm4v_text` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| zai-org/GLM-5.3-Flash | `glm5_next_text` | 逐层缓存/循环状态与后端布局 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| zai-org/GLM-5.3-Flash-BF16 | `glm5_next_text` | 逐层缓存/循环状态与后端布局 | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| zai-org/GLM-OCR | `glm_ocr_text` | 逐层缓存/循环状态与后端布局 | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |

<a id="category-m"></a>

## M 配置或关键字段缺失（196 条）

| 模型 | 配置类型 | 当前缺项 | 需补的计算部分（非已实现） |
| --- | --- | --- | --- |
| ByteDance-Seed/BAGEL-7B-MoT | `bagel` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 生成管线：按分辨率/帧数/步数计算编码器、去噪器及激活峰值 |
| ByteDance-Seed/SeedVR-3B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance-Seed/SeedVR2-3B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance-Seed/SeedVR2-7B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance/AnimateDiff-Lightning | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance/Hyper-SD | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance/LatentSync-1.5 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance/LatentSync-1.6 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance/SDXL-Lightning | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ByteDance/Sa2VA-4B | `sa2va_chat` | num_hidden_layers；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| ByteDance/Sa2VA-8B | `sa2va_chat` | num_hidden_layers；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| FunAudioLLM/CosyVoice2-0.5B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/Fun-ASR-MLT-Nano-2512 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/Fun-ASR-Nano-2512 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/Fun-ASR-Nano-2512-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/Fun-ASR-Nano-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/SenseVoiceSmall | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/SenseVoiceSmall-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| FunAudioLLM/fsmn-vad-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | `llama` | num_attention_heads；num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| Lightricks/LTX-2 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.3 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.3-fp8 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.3-nvfp4 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.5 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-2.5-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-Video | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| MiniMaxAI/MiniMax-H3 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| MiniMaxAI/MiniMax-Music3 | `minimax_music3` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| OpenGVLab/InternVL2-1B | `internvl_chat` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/InternVL2-2B | `internvl_chat` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/InternVL2_5-2B | `internvl_chat` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/InternVL2_5-4B-MPO | `internvl_chat` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/InternVL3-1B | `internvl_chat` | num_hidden_layers；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/InternVL3-8B | `internvl_chat` | num_hidden_layers；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/InternVL3_5-1B | `internvl_chat` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/InternVL3_5-8B | `internvl_chat` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| OpenGVLab/Mini-InternVL-Chat-2B-V1-5 | `internvl_chat` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| PaddlePaddle/PP-DocLayoutV3_safetensors | `pp_doclayout_v3` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| PaddlePaddle/PP-LCNet_x1_0_doc_ori | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PP-LCNet_x1_0_textline_ori | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PP-OCRv5_mobile_det | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PP-OCRv5_mobile_det_onnx | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PP-OCRv5_server_det | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PP-OCRv5_server_rec | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PP-OCRv6_medium_det | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PP-OCRv6_medium_rec_onnx | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/PaddleOCR-VL-1.6-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/UVDoc | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| PaddlePaddle/en_PP-OCRv5_mobile_rec | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.1-I2V-14B-480P | `i2v` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 生成管线：按分辨率/帧数/步数计算编码器、去噪器及激活峰值 |
| Wan-AI/Wan2.1-I2V-14B-480P-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.1-I2V-14B-720P-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.1-T2V-1.3B | `t2v` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 生成管线：按分辨率/帧数/步数计算编码器、去噪器及激活峰值 |
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.1-T2V-14B | `t2v` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 生成管线：按分辨率/帧数/步数计算编码器、去噪器及激活峰值 |
| Wan-AI/Wan2.1-T2V-14B-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.1-VACE-14B | `vace` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 生成管线：按分辨率/帧数/步数计算编码器、去噪器及激活峰值 |
| Wan-AI/Wan2.2-Animate-14B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.2-I2V-A14B-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| XiaomiMiMo/MiMo-Audio-Tokenizer | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| allenai/longformer-base-4096 | `longformer` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| allenai/scibert_scivocab_uncased | `bert` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| allenai/specter2_base | `bert` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| allenai/unifiedqa-t5-small | `t5` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 编码器-解码器：编码工作集、自注意力KV与交叉注意力KV分别计账 |
| baichuan-inc/Baichuan-13B-Base | `baichuan` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baichuan-inc/Baichuan-13B-Chat | `baichuan` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baichuan-inc/Baichuan-7B | `baichuan` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baichuan-inc/Baichuan-M2-32B-Q4_K_M-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| baichuan-inc/Baichuan2-13B-Base | `baichuan` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baichuan-inc/Baichuan2-13B-Chat | `baichuan` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baichuan-inc/Baichuan2-7B-Base | `baichuan` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baichuan-inc/Baichuan2-7B-Chat | `baichuan` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| baidu/ERNIE-Image | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| baidu/ERNIE-Image-Turbo | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.1-Fill-dev | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.1-Kontext-dev | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.1-dev | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.1-schnell | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-dev | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-klein-4B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-klein-4b-fp8 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-klein-9B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-klein-base-4B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-klein-base-4b-fp8 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-klein-base-9B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| black-forest-labs/FLUX.2-small-decoder | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| fishaudio/fish-agent-v0.1-3b | `dual_ar` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/fish-speech-1 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| fishaudio/fish-speech-1.2 | `dual_ar` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/fish-speech-1.2-sft | `dual_ar` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/fish-speech-1.4 | `dual_ar` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/fish-speech-1.5 | `dual_ar` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/s1-mini | `dual_ar` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/s2-pro | `fish_qwen3` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| fishaudio/speech-lm-v1 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| google/electra-base-discriminator | `electra` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| google/gemma-3-4b-it | `gemma3_text` | num_attention_heads；num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| google/owlv2-base-patch16-ensemble | `owlv2_text_model` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| google/siglip2-base-patch16-224 | `siglip_text_model` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| google/siglip2-giant-opt-patch16-384 | `siglip_text_model` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| google/vit-base-patch16-224 | `vit` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| ibm-granite/granite-4.2-30b-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ibm-granite/granite-4.2-3b-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ibm-granite/granite-4.2-8b-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| ibm-granite/granite-embedding-small-english-r2 | `modernbert` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| ibm-granite/granite-timeseries-patchtst-fm-r2 | `patchtst_fm` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 时序任务：输入窗口、通道、预测长度及前向工作集 |
| ibm-granite/granite-timeseries-tspulse-r1 | `tspulse` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 时序任务：输入窗口、通道、预测长度及前向工作集 |
| ibm-granite/granite-timeseries-ttm-r3 | `tinytimemixer` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 时序任务：输入窗口、通道、预测长度及前向工作集 |
| internlm/internlm-chat-7b | `internlm` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| meituan-longcat/LongCat-2.0 | `未知` | num_hidden_layers；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-AudioDiT-1B | `audiodit` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 生成管线：按分辨率/帧数/步数计算编码器、去噪器及激活峰值 |
| meituan-longcat/LongCat-Flash-Chat | `未知` | num_hidden_layers；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Flash-Lite | `未知` | num_hidden_layers；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Flash-Lite-FP8 | `未知` | num_hidden_layers；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Flash-Thinking-2601 | `未知` | num_hidden_layers；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Image | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Image-Dev | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Image-Edit | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Image-Edit-Turbo | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Video | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meituan-longcat/LongCat-Video-Avatar-1.5 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-2-7b-chat-hf | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-2-7b-hf | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.1-405B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.1-405B-FP8 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.1-8B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.1-8B-Instruct | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.2-1B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.2-1B-Instruct | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.2-3B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.2-3B-Instruct | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Llama-3.3-70B-Instruct | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Meta-Llama-3-8B-Instruct | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| meta-llama/Prompt-Guard-86M | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| microsoft/Florence-2-base | `未知` | hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| microsoft/TRELLIS-image-large | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| microsoft/TRELLIS.2-4B | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| microsoft/VibeVoice-ASR | `vibevoice` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| microsoft/deberta-v3-base | `deberta-v2` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| microsoft/deberta-v3-large | `deberta-v2` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| microsoft/deberta-v3-small | `deberta-v2` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| microsoft/layoutlmv3-base | `layoutlmv3` | num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| microsoft/mdeberta-v3-base | `deberta-v2` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| microsoft/table-transformer-structure-recognition | `table-transformer` | hidden_size；num_attention_heads；num_key_value_heads | 视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存 |
| microsoft/wavlm-large | `wavlm` | num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | `nemotron_h` | num_hidden_layers | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| nvidia/bigvgan_v2_22khz_80band_256x | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| nvidia/nemotron-3.5-asr-streaming-0.6b | `nemotron3_5_asr` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| openai/clip-vit-base-patch16 | `clip_text_model` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| openai/clip-vit-base-patch32 | `clip_text_model` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| openai/clip-vit-large-patch14 | `clip_text_model` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| openai/clip-vit-large-patch14-336 | `clip_text_model` | num_key_value_heads | 编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值 |
| openai/whisper-base | `whisper` | hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| openai/whisper-large-v3 | `whisper` | hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| openai/whisper-large-v3-turbo | `whisper` | hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| openai/whisper-small | `whisper` | hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| openai/whisper-tiny | `whisper` | hidden_size；num_attention_heads；num_key_value_heads | 音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值 |
| stabilityai/TripoSR | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/sd-turbo | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/sd-vae-ft-mse | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/sdxl-turbo | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/sdxl-vae | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/stable-audio-3-medium | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/stable-diffusion-3-medium-diffusers | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/stable-diffusion-3.5-large | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/stable-diffusion-3.5-medium | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/stable-diffusion-xl-base-1.0 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/stable-diffusion-xl-refiner-1.0 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stabilityai/stable-video-diffusion-img2vid-xt | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stepfun-ai/Step-3.5-Flash | `step3p5` | num_key_value_heads | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| stepfun-ai/Step-3.7-Flash | `step3p5` | num_key_value_heads | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| stepfun-ai/Step-3.7-Flash-FP8 | `step3p5` | num_key_value_heads | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| stepfun-ai/Step-3.7-Flash-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| stepfun-ai/Step-3.7-Flash-NVFP4 | `step3p5` | num_key_value_heads | 特殊逐层/复用路径：核对层表、状态复用及缓存寿命 |
| stepfun-ai/Step-Audio-EditX | `step1` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| stepfun-ai/step3 | `step3_text` | num_key_value_heads | 专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV |
| tencent/Hunyuan3D-2 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| tencent/Hunyuan3D-2.1 | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| tencent/Hunyuan3D-2mini | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| tencent/Hy-MT2-1.8B-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| tencent/Hy-MT2-30B-A3B-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| tencent/Hy-MT2-7B-GGUF | `未知` | num_hidden_layers；hidden_size；num_attention_heads；num_key_value_heads | 未知结构：先获取身份可核对的配置与任务类型 |
| tiiuae/falcon-7b | `falcon` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| tiiuae/falcon-7b-instruct | `falcon` | num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |
| tiiuae/falcon-mamba-7b | `falcon_mamba` | num_attention_heads；num_key_value_heads | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| tiiuae/falcon-mamba-tiny-dev | `falcon_mamba` | num_attention_heads；num_key_value_heads | 状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度 |
| zai-org/chatglm2-6b | `chatglm` | num_hidden_layers；num_key_value_heads | 需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性 |

¹ 参数B只帮助识别目录规模；有效计算须使用核验后的全部驻留参数，不用MoE激活参数代替。

## 更新与复核

由 `tools/build_planning_model_index.py` 离线生成，同时输出机器可读分类与示例。分类调用当前规划适配器，缓存例不启动模型。修改配置或适配器后应重新生成；历史审计清单与当前生成结果若不同，应先检查快照日期与目录版本。
