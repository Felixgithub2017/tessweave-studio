# 资源规划与执行审计

## 从问题到计算

输入总参数（MoE 用总参数，不用激活参数）→ 选择训练或推理 → 拆静态账本 → 加运行峰值 → 枚举 GPU 数与合法切分 → 对照实际机器验证。

“静态容量下界”不含激活、工作区和聚合临时副本。“条件最低”是在本页预算、每节点卡数及切分约束下的最小候选，不是全世界所有 offload/量化/并行实现中的绝对最少卡数。默认20%安全余量，GPU铭牌容量临时按GiB输入；用实际 nvidia-smi memory.total 复算后才可租赁决策。

1. **全参训练**：每参数2字节权重、2字节梯度、4字节主权重、8字节Adam一阶/二阶状态，共16字节。这是一个明确配置，不是所有训练引擎的统一用量。
2. **LoRA**：冻结基座不保存对应Adam状态；adapter参数单独按16字节算。adapter总数需由rank/target模块或实际模型确认。
3. **QLoRA**：基座按0.5625字节/参数估含scale开销，只提供单卡预算，不假定量化参数可被任意ZeRO分片。
4. **推理**：BF16/FP16为2字节/参数，8-bit为1字节，4-bit加scale估0.5625字节；不同组大小、保留层、kernel可能不同。缓存使用BF16，不因权重4-bit就自动变成4-bit。

结构已知时，普通GQA缓存按 `4 × 并发 × token数 × 层数 × KV头数 × 头维度` 字节计算。MLA仅在配置可识别时采用压缩缓存假设；后端展开缓存或混合结构需单独核实。未知结构使用显式手填预算。

训练激活是保守代理：checkpoint边界 + 临时FFN张量 + 未分块logits；不是对真实autograd保存张量逐一计数。用户预算与代理取较大值，所有卡均保留完整该预算。实际FlashAttention、重计算、融合loss、序列并行会改变峰值。

## 切分与加卡

- 训练候选用ZeRO-3/FSDP FULL_SHARD：权重、梯度、优化器状态除以DP卡数，再加每卡激活、工作区、最大参数聚合单元。聚合单元默认按一层估，不等同于引擎真实wrap策略。没有层数时默认2GiB，必须实测修正。
- 推理枚举节点内TP与按层PP；检查头数、隐藏/FFN宽度、层数整除。TP大于KV头数时保留KV复制开销。引擎/模型能否实现该组合仍需预检，不自动启动。
- 单节点枚举1至节点容量；跨节点按整节点申请。额外卡数超过节点容量向上取整。固定卡数优先于额外卡数。加卡不保证降低延迟，也可能不符合TP/PP结构整除要求。
- NVLink/NVSwitch依赖实际整机，消费卡不假定有高速互联。跨节点给出RDMA建议，而非保证特定网络速率可达；必须跑NCCL与端到端训练/服务压测。
- DPO/PPO/GRPO多角色、MoE专家并行、异构卡、CPU/NVMe卸载、端侧统一内存不在本页自动最优化范围。不能把本页单模型预算当这些任务的最终采购单。

代码入口：`workbench/resources.py:plan_resources`。结果保留输入假设、参数口径、每卡峰值、各GPU候选与未验证告警。

## 每次点击背后发生什么

启动示例：

```bash
python3 -m workbench serve --allow-root /data/models --audit-log /data/logs/workbench.log
```

父目录需先存在。此日志为逐行JSON，追加写入；网页与原生终端进程通过文件锁协调。`operation.begin/result/error`记录API输入/输出；`command.execute`记录真正的后端argv；`terminal.output`记录脱敏后的输出；`download.progress`记录下载进度与状态。

扫描、预算等直接运行Python函数，并不为了展示而多启动一个shell。对应日志里的`equivalent_cli`可重放同一个API，须通过`WORKBENCH_SESSION_TOKEN`环境变量提供当前会话令牌；默认端口8765，非默认端口添加`--port`。脱敏后的凭据不能用于重放。

macOS选择“原生终端”时，Terminal运行每任务的`run.command`，它启动`workbench.native_runner`，再执行已确认计划。GUI停止通过取消标记通知该runner，只终止自己创建的进程组；不会按历史PID误杀别的进程。无桌面Linux应选择后台托管，桌面Linux支持gnome-terminal/xterm。

这仍是本机单用户工程预览：不支持多人权限、日志轮转、跨服务器自动部署及后台服务托管恢复。日志可能含路径、提示词、业务数据，不要直接公开上传。
# Architecture-derived capacity (2026-10-07)

Local model planning now scans every safetensors header, without loading tensor payloads or importing model Python. The inventory reconciles decoder layers, individually named routed experts, shared experts, auxiliary/MTP layers, vision modules and tensor dtypes. Missing shards, incomplete expert IDs, unrecognized/fused layouts and packed tensors cannot establish this inventory. Stored elements are conservatively budgeted as trainable; this is not deduplicated parameter counting.

For a complete hybrid/MLA/multimodal inventory, the planner returns **conditional state-capacity candidates**, not an ordinary dense-attention prediction or a validated launch recipe. It searches pure-DP ZeRO-3 and actual layer-count divisors for PP with EP nested in DP. TP/CP remain 1 pending a verified architecture-specific implementation. Non-decoder modules, including auxiliary experts, are conservatively placed together on stage 0 for PP; this can materially overestimate the minimum GPU count. Candidates are sorted by GPU count, not measured speed.

Per-rank accounting:

- ZeRO-3: BF16 weights + BF16 gradients + FP32 master/Adam m/v = `16 × stored_elements / DP` bytes, plus one largest inventoried block's BF16 gather and explicit workspace budgets. Actual wrapping/prefetch may change gather peaks.
- PP/EP distributed optimizer: let D be stage-local non-routed elements and E be this EP rank's expert elements. BF16 weights + FP32 gradients require `6(D+E)` bytes; FP32 master/m/v require `12(D/DP + E/(DP/EP))` bytes.
- Remaining budget = GPU capacity − safety reserve − accounted states/buffers. This is **not** guaranteed free memory: saved activations, transient kernels, MoE imbalance, communication overlap and pipeline liveness still require measurements. Changing context changes those unknown costs, not stored-state bytes.

The UI offers candidate selection, node paging, clickable individual GPUs, their actual layer/expert mapping, precision and arithmetic. It intentionally does not claim a minimum training configuration, time, cost or backend support from these partial results. Recompute and TP/CP optimizations require a further verified model adapter. DPO/PPO/GRPO and LoRA remain outside this full-parameter state branch.
