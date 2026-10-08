# Training design rules / 训练方案设计规则

核查日期：2026-10-07。实现：`workbench/training_design.py`，版本 `2026-10-07.v1`。

这是可执行的**分析型规划器**规范，不是已经完成 GPU 认证的自动调优系统。文档的 latest 页面是核查时的快照，不能解释成所有功能均在 2026 年 10 月发布。训练算法、质量目标和总 batch 仍需研究者决定，规划器不擅自改变它们来凑性能。

## 1. 本轮落地的规则

### 分阶段长度规划（2026-10-07 补充）

计算资源页新增目标长度，默认 32,768，而非自动采用模型最大上下文。计算候选时，每个阶段独立运行资源搜索，可点击查看各阶段逐卡账本。主输入的“上下文 token 数”仍表示单次自定义方案；分阶段目标是另一项明确输入。

公开依据：[Qwen3 报告 §3.2](https://arxiv.org/html/2505.09388v1#S3.SS2) 的前两阶段都使用 4,096，后期长上下文训练使用 32,768；中期主要改变数据质量。[DeepSeek-V3 报告 §4.3](https://arxiv.org/html/2412.19437v2) 的扩展为 4K→32K→128K，两轮各 1,000 步，样本 batch 从 1,920 减为 480。它们是已公开案例，不是当前 GLM 的历史配方，也不是所有模型必须遵循的标准。

平台工程起点（占比是本平台启发式，不是官方比例）：

| 本次任务 | 前段 | 中／后段 | 默认有效 token 分配 |
|---|---|---|---|
| 从零预训练 | 4K 通用训练 | 4K 高质量巩固，然后 16K→32K | 80%／15%／2.5%／2.5% |
| 已有权重 CPT | 8K 领域适配 | 16K→32K 任务适配 | 85%／7.5%／7.5% |
| SFT | 8K 指令任务 | 16K→32K 长任务适配 | 80%／10%／10% |

目标较长时继续列出 64K、128K 等阶段，均需真实长数据与验收，不因模型支持 1M 就默认训练到 1M。目标被 config 声明上限约束；若该字段未知则提示验证，不自动改 RoPE。低于默认阶段长度时会收缩长度，但阶段目标仍保留。占比目前为预设，目标长度及本次总 token 可在页面调整；尚未自动分析数据集分布，不能声称质量最优。

只有长任务有实际需求时才接受长阶段；普通短问答可直接降低目标。先用模型 tokenizer（包含模板和特殊 token）统计样本 P50/P90/P99、截断比例，再决定长度桶与 token 占比。短样本保持短样本，packing 必须尊重文档／对话边界和监督 mask。多模态还必须用处理器展开后的视觉／音频 token，而非文本字符数；本功能不承诺完成这些数据统计。

每阶段总步数依赖实际 `DP × microbatch × accumulation × sequence × packing_efficiency`。长度增长后重新搜索显存、重计算和受支持的 TP/PP/CP/EP/ZeRO，不能只把上一阶段显存乘倍数。混合架构仍返回状态容量候选；跨阶段相同的状态账本不表示激活或通信成本相同。未实测吞吐不得承诺训练时长。

晋级门槛：损失／梯度稳定；短任务回归不退化；目标长度上的检索、跨段推理／代码或具体业务任务达标；峰值 allocated/reserved 显存、吞吐和通信已测；断点恢复验证通过。学习率与各阶段步数必须按实际训练预算重新安排，不从其他模型直接复制。

DPO/PPO/GRPO 不套用上述单模型计划：DPO 同时预算 chosen/rejected；RL 区分 prompt cap、response cap、采样组大小、actor/reference/critic 和 rollout KV cache。现有联合规划器不支持完整角色池，因此显示明确限制，而不是给出虚假的 GPU 数。

1. 先识别训练类型和架构。单模型、全参数 Adam、标准稠密注意力的预训练/CPT/SFT启用联合搜索；完整权重头清单自动提供路由专家参数占比，top-k 来自 config。MLA、混合注意力和多模态在清单完整时返回有条件的状态容量方案，不伪造激活估算；清单不完整或 RL 多角色仍返回限制说明。
2. 对选定 GPU 搜索 TP、PP、DP、CP、EP、状态分片及完整/不重计算两种策略；微批保持用户输入，梯度累积由目标 batch 和 DP 推导。搜索数量、卡数上限、淘汰计数全部返回。
3. 组合按框架分族：Megatron 候选为复制状态或 distributed optimizer；DeepSpeed 的 ZeRO-2/3 候选限定纯数据并行。不能把 ZeRO-3、任意 TP/PP 和某后端名称拼在一起就声称可运行。
4. TP 仅在用户确认的 NVLink/NVSwitch 节点内枚举。PCIe 并非绝对不能 TP，本版只是保守排除；用户可在独立测试后扩展此限制。EP 也限定节点内，当前 MoE 搜索限定 TP=CP=1，EP 嵌套 DP，专家复制组大小 DP/EP。
5. 卡数恒等式为 `N=TP×PP×CP×DP`，不是再乘 EP。CP 不增加全局样本数量。默认枚举 2 的幂及整节点规模，固定卡数时不加卡；这是有界搜索，不证明全局最优。
6. 层数、头数、维度与并行数必须满足实现中的整除限制；词表不自动 padding。CP 只在 S≥8192、S 可被 2CP 整除时枚举，这是本版搜索规则，不是硬件定律。
7. 显存先过硬门槛，再比较容量、时间、费用。没有有效算力或必要链路带宽时不输出耗时，没有单价时不输出“最省钱”。最低容量候选不等于生产吞吐推荐。

规则参考：[Megatron Bridge 并行指南](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)、[Megatron Core 组合约束](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。这些资料支持组合设计，不替代特定模型的后端预检。

## 2. 三个账本

### 状态与峰值显存

本版采用 BF16 权重、FP32 主权重和 Adam m/v；**Megatron 梯度按 FP32，DeepSpeed 候选按 BF16 假设**。这使复制状态分别为 18P 和 16P 字节。真实后端如改变梯度累积精度，必须重新计算。

`P_dense_rank = P_nonexpert/(TP×PP) + embedding_stage_allowance`

`P_expert_rank = P_routed/(PP×EP)`（本版 MoE TP=1）。

Megatron distributed optimizer 的 dense 状态分片组用 DP×CP；专家分片组 DP/EP（本版 MoE CP=1）。权重与梯度仍驻留本卡，主权重和优化器按对应组分片。DeepSpeed 按各 ZeRO 阶段分别决定权重、梯度、主权重、优化器的除数。

嵌入阶段附加项为 `2×V×d/TP`（PP>1）；由于总 P 已含 embeddings，这是**刻意重复计入的最重阶段安全余量**，不是精确层分配。所有 GPU 图暂展示最重阶段上界，不能解读为逐卡实际峰值。下一步要用 tensor→layer→stage 映射替换。

激活近似：`C×B×(S/CP)×d×(L/PP)/TP×live_microbatches`。完整重计算 C=2；不重计算 C=24，后者只是待校准经验系数，不是通用张量生命周期公式。1F1B 的存活微批上界用 `min(PP,accumulation)`。另加临时 FFN、未分块 FP32 logits、MoE dispatch、ZeRO-3 单 block 聚合、工作区、用户额外预算及安全余量。激活下限用 max 补足，不能重复加两遍。

官方 [Bridge 显存估算器](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/memory-estimator.html) 同样区分模型状态、激活与假设，并不能覆盖所有运行时分配。本项目公式不是对该工具的精确复刻；特别是完整重计算、碎片及 dispatch 必须短跑实测。

### 单步时间与通信

本版计算量代理：`6×P_active×padded_tokens + 12×B×DP×acc×L×S²×d`，完整重计算整体乘 4/3。标准 MoE 的 `P_active=P×(1−expert_fraction+expert_fraction×topk/E)`；没有把激活参数拿来算驻留权重。因果、GQA、算子实现会改变真实 FLOPs，因此该公式不能替代 profiler。

每个候选返回 TP、PP、CP、DP、EP、参数聚合的**每 rank 每步通信 GiB**。字节代理通过用户输入的有效 GiB/s 转成通信时间；多节点的节点级 NIC 带宽除以节点卡数，避免把一个 NIC 的带宽重复赠送给每张卡。跨节点组使用保守带宽，未建模机架拥塞或集合通信延迟。

流水线空泡比近似 `(PP−1)/accumulation`。时间区间为完全重叠与完全不重叠两个情景，不是统计置信区间。HBM 指标只是模型状态流量下界，未覆盖所有激活读写，不能据此宣称完整 roofline 分析。

参考：[性能调优指南](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)；[DeepSpeed 参数/聚合/卸载配置](https://deepspeed.readthedocs.io/en/latest/zero3.html)。小聚合桶、卸载或更深分片可能降低容量压力，却增加通信/等待；不默认启用这些选项。

### 全程时间、费用

`acc = floor(target_padded_tokens / (B×S×DP))`，必须在 1..4096 内。

`effective_tokens_step = B×S×DP×acc×packing_efficiency`

`steps = ceil(total_effective_training_tokens / effective_tokens_step)`

`hours = steps×step_seconds×(1+overhead_percent/100)/3600`

`cost = hours×GPU_count×GPU_hour_price`

用户输入总 token 应包含计划重复训练的轮数；SFT 掩码有效标签和实际执行 token 不是同一概念，这里 packing_efficiency 是非 padding 的执行 token 比例。若只有数据文件大小，必须先 tokenize 统计，不能按字节臆测。单位 GPU 单价统一币种；当前未计 CPU、网络、存储费。

## 3. 不同规模的可复算方案

运行：

```bash
PYTHONPATH=. python3 -B examples/training_design_cases.py
```

以下是该脚本的**合成架构分析案例，不对应官方模型配方，也未经 GPU 训练验证**。H100 80（容量暂按80GiB，需真实 memory.total 复算）、节点8卡NVLink、20%预留、激活下限8GiB、工作区4GiB、B=1、目标1048576 padded tokens/update、总训练1B有效token、packing=1、最多256卡。默认词表128256；各层宽度在脚本中公开。未输入有效TFLOPS和网络实测，所以没有虚构的小时数。

| 场景 | S | 搜索内最低卡数 | TP/PP/DP/CP/EP | 状态策略 | 重计算 | 最重卡 GiB |
|---|---:|---:|---|---|---|---:|
| 稠密0.5B |4096|1|1/1/1/1/1|复制|无|20.38|
| 稠密7B |4096|4|1/4/1/1/1|复制|完整|58.95|
| 稠密32B |4096|16|2/8/1/1/1|复制|完整|56.54|
| 稠密70B |4096|32|4/8/1/1/1|复制|完整|57.48|
| 稠密405B |4096|256|4/64/1/1/1|复制|完整|56.13|
| 稠密7B长上下文 |32768|4|1/1/4/1/1|ZeRO-3|完整|58.17|
| MoE40B，8专家/top2/90%专家参数 |4096|32|1/32/1/1/1|复制|无|57.31|
| MoE320B，128专家/top4/95%专家参数 |4096|256|1/64/4/1/4|复制|完整|62.89|

为何这些最低容量候选会出现很深的 PP？未输入性能信息时，容量目标在同卡数候选中用通信字节代理与空泡等打破平局，**不是最优吞吐策略**。必须录入实测有效算力、链路带宽、价格，再比较“预计最低成本”和“满足期限”。参考表不是建议直接租256卡，也不能作为框架支持证明。搜索目前不优化微批、VPP、offload、FP8或选择性重计算，存在未探索的更优解。

## 4. 2026 年可吸收的进展与不能照搬的结论

- [MoE 优化指南](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/moe-optimization.html) 强调存储、通信与专家计算效率一起优化。对应本版：专家总量/活跃量分开、EP通信单列、快互联域约束；DeepEP/HybridEP不能用固定倍速代替实测。
- [TorchTitan](https://github.com/pytorch/torchtitan) 是后续对接 PyTorch 原生组合训练的候选，而不是把 FSDP2 简单改名为 ZeRO-3 就算兼容。本版尚无 TorchTitan 可执行配方导出。
- 2026-09-18 的 [HyperParallel-FSDP](https://arxiv.org/abs/2609.21594) 将特定 Ascend 拓扑和状态布局联系起来。可借鉴“硬件拓扑决定分组”的思路；论文平台上的收益不能外推为 NVIDIA/消费卡提升，也未纳入本版性能常数。

## 5. 实测验收与后续边界

候选 → 后端版本和模型转换检查 → 实际 GPU/NIC 拓扑 → NCCL 对应消息大小测试 → 预热后至少20步 → 峰值allocated/reserved、稳态step time、有效tokens/s、loss → 保存恢复检查 → 重新规划。

API返回 `configuration_blueprint` 只是部分设置，`launch_ready=false`；不会自动执行命令或为用户租机。暂不自动运行校准任务，也不将手填数值标为测量证据。每次 `/api/resources` 请求仍通过现有审计机制记录输入和输出。

尚需实现：真实张量层映射、模型/后端能力注册表、自动微批与选择性重计算搜索、校准记录导入和硬件指纹、拓扑延迟/争用模型、异构卡、offload带宽、FP8稳定性验证、完整EP×TP×CP组合、RL多角色池。遇到不支持架构必须阻断，不能退回“不断加卡直到装下”当作联合推荐。
