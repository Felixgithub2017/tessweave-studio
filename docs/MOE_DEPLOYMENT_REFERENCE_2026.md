# 2026 大型 MoE 部署参考与 TessWeave 规划依据

资料核对日期：2026-10-13。本文是公开资料的整理，不是 TessWeave GPU 实测报告。

## 先读这一页

“成熟部署”不能只看有没有启动命令。本文将证据分为：

1. **上游负载验证**：注明硬件、引擎和请求条件；只对这些条件负责。
2. **上游部署配方**：可作为复现起点，不代表最大并发、最低卡数或生产 SLA。
3. **冲突/待核实**：版本、参数口径、发布状态存在不一致，不能自动批准。
4. **待发布/适配资料**：不纳入已验证推理候选。

本文的 GPU 数是公开示例，不是建议用户直接租用的最低配置。所有方案目前均未由 TessWeave 本地复现；链接可能更新，正式执行前必须固定模型 revision、容器 digest、代码 commit 和命令。开放权重的许可证也须单独审核。

## 1. 模型与部署路径

| 模型 | 参数口径 | 架构重点 | 公开方案与证据等级 |
|---|---|---|---|
| Kimi K3 | 2.8T / 104B 激活 | KDA + gated MLA、视觉、MXFP4/MXFP8 | Blackwell 专用配方；页面有预发布残留说明，标记冲突待核实 |
| Qwen3.8-2.4T-A95B | 2.4T / 95B | Gated DeltaNet + gated attention，MoE | 配方容量表 BF16 24×B300、NVFP4 8×B300；不是等吞吐比较 |
| DeepSeek V4 Pro / 0813 | 1.6T / 49B | CSA/HCA、mHC、混合精度、DSpark | NVIDIA/AMD 专用配方；preview 基准不可继承到 0813 |
| MiMo V2.6 Pro RL | 1.02T / 42B | 混合注意力、多模态、存储/计算精度不同 | 8×H200 TP8 或 4×MI355X TP4 配方 |
| Tencent Hy4-preview | 770B / 49B，另有 MTP | 稀疏注意力、MoE、长上下文 | 专用配方；preview 与草稿模块分别核算 |
| GLM-5.3 | 743B / 39B | MoE、MTP | FP8 8×H200 TP8 配方；BF16 不继承此容量 |
| DeepSeek V4.1 Flash | 552B 主干 + 196B Engram | 编码器—解码器、全局 KV、CPU/UVA | Blackwell 与 NIXL P/D 配方，主存和传输不可省略 |
| Nemotron 3 Ultra | 550B / 55B | Mamba + attention + MoE | Dynamo 验证：4×B200/worker、TP4 + EP；仅对应工作负载 |
| MiniMax M3 | 模型页约 427B；引擎资料有冲突 | MSA、视觉、MXFP8 | SGLang B200 TP8、B300 TP4；部分硬件仅推断支持 |
| GLM-5.3-Flash | 标称约 320B / 18B | KDA + sparse MLA、视觉 | H200 TP8、Blackwell TP4 等配方；KV 精度支持分硬件 |
| MiMo V2.6 Flash | 309B / 15B | 混合注意力、混合精度 | 专用 H200 等配方；RL/MOPD 和页面版本须核对 |
| Step-3.7-Flash | 约 198B，含视觉部分 | 全注意力 + 滑窗、MoE | 官方部署入口；本目录未登记可复现 SLO 基准 |
| Nemotron 3 Super | 约 120B / 12B | Mamba 混合 MoE | Dynamo 验证的 worker/副本/路由方案 |
| Nemotron 3.5 Super VL | 120B 级 / 12B | 混合主干 + 视觉塔 | 查到训练适配资料；不能继承 Super 的推理成绩 |

上述总参数、激活参数是官方资料口径，不是对全部权重的重新计数。实际驻留还可能包含草稿、视觉、编码器和格式元数据。模型标称上下文也不等于指定 GPU 上能够服务的上下文。

### 1.1 Kimi K3

官方资料描述 93 层、KDA 与门控 MLA 混合、896 路由专家及共享专家。权重与激活使用不同格式。不能用普通 GQA 的 KV 公式覆盖全部层。vLLM 页面仍有 pre-release、估算容量等表述，即使同时列出 verified 硬件，也应先对齐发布日期与修订版本。

来源：[官方模型](https://huggingface.co/moonshotai/Kimi-K3)、[vLLM 配方](https://recipes.vllm.ai/moonshotai/Kimi-K3)。

### 1.2 Qwen3.8-2.4T-A95B

该开放检查点不是托管多模态 API 的同义词。混合层需分别计算序列增长的 KV 和循环状态。配方提供 BF16、FP8、NVFP4/MXFP4 路径；不能把一种格式的容量换算直接当另一种格式的吞吐。容量表不是服务压测。

来源：[官方模型](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[vLLM 配方](https://recipes.vllm.ai/Qwen/Qwen3.8-2.4T-A95B)。

### 1.3 DeepSeek V4 Pro 与 V4.1 Flash

Pro 的混合注意力、FP4 专家和其他精度模块、草稿路径均需单独预算。0813 与 preview 有不同的检查点和引擎要求，基准不能跨版本沿用。

Flash 的 552B 主干不是全部存储需求，还要处理 196B Engram。CPU/UVA、编码器缓存和 P/D 传输意味着“GPU 总显存够了”仍可能无法低延迟运行。不能给这些模型套用同一个 MLA 公式。

来源：[Pro 配方](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)、[Flash 配方](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)。

### 1.4 MiMo V2.6

Pro RL 配方约 566GB 的检查点与约 680GB 的容量估计说明：文件字节不等于 GPU 常驻字节。MXFP4 存储到 FP8 执行可能涉及转换与临时峰值。8×H200 / 4×MI355X 是复现候选，不是 SLA。

Flash 页面存在 RL/MOPD 默认检查点及示例变化。必须把仓库 revision 和启动参数共同存档，不能只保存系列名。

来源：[Pro 模型](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)、[Pro 配方](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Pro-RL)、[Flash 配方](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL)。

### 1.5 GLM-5.3 与 Flash

GLM-5.3 有 FP8 H200 配方。Flash 使用 KDA 与稀疏 MLA，其 KV 精度支持受硬件影响，不能因 Blackwell 可用 FP8 KV 就给 Hopper 相同推荐。P/D 配方还要求状态布局匹配。

启动日志根据 KV 池打印的 concurrency 只是容量比值，不能解释成满足 p99 延迟的并发；多模态预处理和 MTP 同样占资源。

来源：[GLM-5.3](https://recipes.vllm.ai/zai-org/GLM-5.3)、[Flash](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)。

### 1.6 MiniMax M3：值得模仿的部署说明方式

SGLang 明确区分直接验证与推断支持；Blackwell 使用专用 MSA 算子，Hopper 和 AMD 走各自路径。页面指定 128-token 页，提示静态内存比例过高会在长上下文/高并发 OOM，并说明 EP 会影响部分融合选项。

它还要求冷启动先处理 JIT 缓存，避免多个 rank 争用编译产物。说明中仍使用专用 dev/PR 安装路径，应固定版本。

注意：SGLang 描述约 428B/23B，而另一些官方资料口径不同。当前不批准自动取其中一个值计算。应固定检查点，区分总存储、语言主干、视觉与每 token 激活口径。

来源：[模型页](https://huggingface.co/MiniMaxAI/MiniMax-M3)、[SGLang 配方](https://docs.sglang.io/cookbook/autoregressive/MiniMax/MiniMax-M3)、[vLLM 配方](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3)。

### 1.7 NVIDIA Nemotron

Ultra 的 Dynamo 方案是每 worker 4×B200、TP4 + EP，并使用 MTP 与 KV 感知路由；其流量长度、缓存复用条件是验证的一部分。Super 的 worker 和副本配置必须一起计数：每 worker 卡数不是整个服务卡数。

3.5 Super VL 的训练覆盖不构成推理配方认证；视觉塔和混合状态均需重新核对。

来源：[Ultra 验证配方](https://docs.nvidia.com/dynamo/v1.3.0/recipes/nemotron-3-ultra)、[Super 验证配方](https://docs.nvidia.com/dynamo/dev/recipes/nemotron-3-super)、[3.5 VL 架构](https://docs.nvidia.com/nemo/automodel/model-coverage/omni/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)。

### 1.8 Tencent、Step 与待发布模型

Hy4-preview：参数计数外还需核算 MTP，不能只用激活参数算驻留。

Step-3.7-Flash：混合滑窗和视觉输入要求分层缓存和 processor 展开后的 token 预算；官方引擎入口不等于所有模态及长度完成服务验证。

Mistral Large 4 已公告，但模型页预计 2026-10-31 发布；截至本文日期不纳入已可复现部署列表。

来源：[Tencent 发布](https://www.tencent.com/tencent-releases-and-open-sources-tencent-hy4-preview/)、[Hy4 配方](https://recipes.vllm.ai/tencent/Hy4-preview)、[Step 官方模型](https://huggingface.co/stepfun-ai/Step-3.7-Flash)、[Mistral 发布状态](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)。

## 2. 规划器不能省略的计算

### 2.1 先按模块和 rank 算权重

`W(rank) = Σ(该 rank 实际驻留张量元素数 × 实际字节/元素 + scales/zeros/对齐开销)`。

路由专家、共享专家、attention、embedding、lm_head、视觉塔、MTP 分开；TP/EP 复制和分片分别计。CPU offload 同时增加主存、传输和延迟预算。文件大小不能代替加载后大小。

### 2.2 缓存依架构计算

标准 BF16 GQA 单请求缓存：`2(K,V) × 2字节 × 层数 × KV头数 × head_dim × token数`；分片后注意 KV 复制。

其他结构必须由后端适配器给出：压缩 MLA、稀疏索引、滑窗与全局层、循环/卷积状态、编码器缓存。稀疏读取不自动意味着只存稀疏 token。按后端分页向上取整，不能默认所有模型都用 16-token 页。

### 2.3 不是把所有峰值直接相加

`最危险显存 = max(执行阶段, rank)[权重 + 缓存 + 状态 + 激活 + 通信缓冲 + Graph + 工作区]`。

执行阶段至少包含加载/转换、图捕获、prefill、decode。每个阶段中实际共存的张量相加，再取最大值；不要遗漏峰值，也不要把不同时刻的峰值误称实测并发峰值。

### 2.4 三个答案分开展示

| 答案 | 含义 | 所需证据 |
|---|---|---|
| 权重容量下界 | 理论存放权重的资源 | 固定格式的张量清单 |
| 可运行配置 | 能加载并完成目标长度推理 | 后端兼容性、逐卡峰值、正确性测试 |
| 业务配置 | 满足延迟/吞吐/故障要求 | 请求分布、实测负载曲线、预热副本、故障演练 |

并发上限是满足显存、队列、TTFT、ITL、错误率等所有约束的最大值。不能拿“日请求量÷86400”代替峰值，也不能拿显存并发当 SLO 并发。

P/D 分离要分别计算两个池的权重与缓存、KV 传输时间、路由和容错。在低负载或慢网络下不一定有收益。

## 3. 本轮已实现与尚未实现

已实现：

- `deployment_recipes.py` 登记精确仓库匹配的参考配方、来源、核对日期和风险；不自动下载或执行。
- 标准解析路径仅接纳已列明的普通架构或无模型身份的手填结构假设；MoE/特殊字段和未知架构进入 `adapter_required`。
- 特殊模型不再从旧 TP/PP 公式批准最低卡数或布局；旧数学审计保留在 `unverified_capacity_audit`，方便排查。
- UI 优先呈现证据和待验证条件；不将来源配方冒充本平台认证。

尚未实现：新 MoE 的架构专用逐卡适配器、EP/通信成本求解、格式级权重扫描、自动复现配方、压测校准与生产编排。配方仓库名匹配也不证明本地文件真实性。当前 `revision=null`，意味着尚未冻结可复现工件。

下一步先选一个已公开且硬件可获得的检查点，锁定 revision/镜像，核验权重与缓存，再对比预测和实测逐卡峰值；校准后才能称该组合“已验证”。不要一次性宣布支持所有表内模型。

## 4. 租卡前的复现清单

1. 保存模型 revision、许可证、config、权重索引与量化配置。
2. 保存引擎镜像 digest、CUDA/驱动要求、专用算子和硬件型号/互联。
3. 明确每个角色的 TP/DP/EP/PP、MTP、KV dtype、分页、Graph 与 prefill 配置。
4. 测加载与短生成正确性，再测试目标上下文；记录每张 GPU 峰值和 CPU 内存。
5. 按业务输入/输出分布做并发阶梯，记录 TTFT/ITL p50/p95/p99、吞吐、错误、取消和 OOM。
6. 对量化及推测解码单独验证质量；不以一个请求输出一致代替质量评估。
7. 演练节点故障、重启预热和过载；最后生成带证据的推荐，不先承诺 SLA。
