# TessWeave 模型部署规划计算手册

**适用范围：** Studio 部署容量规划 · **更新：** 2026-10-16 · **单位：** GiB

本文解释怎样从模型配置得到显存账本、多卡候选、长上下文增量和并发边界。

> **先记住：容量估算 ≠ 后端兼容 ≠ 实机验证。**
> 本文解释当前实现的计算，不将“数学上装得下”写成“已经能运行”。

### 阅读导航

| 你想解决的问题 | 对应章节 |
| --- | --- |
| 页面结论可信吗？输入从哪里来？ | [1. 证据等级](#evidence) · [2. 输入来源](#inputs) |
| 显存由哪些部分组成？量化省多少？ | [3. 逐卡总账](#budget) · [4. 权重与量化](#weights) |
| 不同注意力架构怎么算？ | [5. 缓存公式](#cache) · [6. 临时量](#activation) |
| 413 条模型各用哪套公式？ | [分类总览](#model-categories) · [逐模型字段与算例](MODEL_PLANNING_INDEX.zh-CN.md) |
| 要多少卡？能同时处理多少请求？ | [7. 多卡搜索](#topology) · [8. 并发与长度](#concurrency) |
| 端侧怎么算？如何逐个上机验证？ | [9. 端侧预算](#edge) · [10. 验证清单](#validation) |
| 公式对应哪段代码？依据是什么？ | [11. 实现与来源](#sources) |

**计算流程**

```text
模型身份与配置 → 权重 / 缓存 / 临时量 → 合法 TP/PP 布局
              → 最忙卡容量检查 → 并发与长度边界 → 部署验证
```

<a id="evidence"></a>

## 1. 先看结论属于哪一种证据

| 页面结果 | 实际含义 | 不代表什么 |
| --- | --- | --- |
| 容量可规划、未实机验证 | 有结构适配器，可生成条件显存候选 | 不保证启动、质量或吞吐 |
| 上游部署配方 | 固定来源中存在该检查点的部署示例 | 不自动匹配当前精度、上下文或并发 |
| 元数据缺失 | 尚缺计算必需字段 | 不是模型本身无法部署 |
| 架构适配待完成 | 配置存在，但当前计算器还不能正确解释缓存/状态 | 不允许套标准 KV 公式伪造数字 |
| 搜索范围内无候选 | 当前预算、拓扑约束没有找到可容纳方案 | 不等于所有硬件、所有 offload 策略均不可行 |

规划与运行适配分开：可以给出外部后端的候选路线，而不要求先移植其计算核。用户仍须核对精确模型、后端版本、设备和量化格式。

<a id="inputs"></a>

## 2. 输入来自哪里

配置缓存只保存元数据，不是权重。内置资料优先使用已核对身份的魔搭缓存，再使用已有其他来源；用户提供的本地/已缓存配置不应被内置旧记录覆盖。嵌套多模态模型读取 `text_config` 计算文本解码器，原始完整配置仍保留。

关键字段包括层数、隐藏维度、Q/KV 头数、显式 `head_dim`、逐层 attention 类型、MLA 低秩维度、循环状态尺寸、专家数量和上下文上限。不能总用 hidden_size / attention_heads 反推 head_dim；某些架构有显式且不同的头维度。

P 指全部驻留参数。MoE 的 A3B/A4B 是每 token 激活规模，不能当作总权重驻留量。量化文件的 packed 元素数量也不等于原始逻辑参数量。缺少可靠参数量时要求补充，而不是默认为某个规模生成貌似准确的方案。

模型声明上下文自动带入，用户可以减小工作负载。它包含输入、历史与最大生成输出；多模态还要计 processor 展开的 token。声明上限不是后端兼容或长文本质量保证。增加 GPU 不会自动突破模型声明长度。

<a id="budget"></a>

## 3. 单位与逐卡总账

内部容量使用 GiB：`1 GiB = 2^30 bytes`。GPU 目录容量目前以标称数字作为 GiB 代理，采购/上线前应换成设备实际容量。

**公式 1 · 逐卡容量判断**

$$
\begin{aligned}
M_{\mathrm{peak}} &= W + Q + K + R + A + X + E \\
M_{\mathrm{usable}} &= V(1-\rho) \\
M_{\mathrm{peak}} &\le M_{\mathrm{usable}}
\end{aligned}
$$

W：本地权重；Q：量化元数据；K：KV 缓存；R：循环状态；A：激活代理；X：workspace；E：额外预算。V 是设备容量，ρ 是预留比例（20% 写作 0.2）。此式各项统一用 GiB，后文缓存字节需先除以 2³⁰。

额外预算包括运行时、通信、CUDA Graph、其他模型和 rollout。填零代表尚未预算，不代表已经证明没有占用。预留比例不能代替遗漏组件的建模。

激活采用 `max(用户下限, 结构代理)`，不重复相加。不同组件寿命可能不重叠；当前求和是预算代理，不是逐算子存活期分析或精确峰值保证。

<a id="weights"></a>

## 4. 权重和量化

**公式 2 · 权重有效载荷**

$$
W_{\mathrm{bytes}}=\frac{P\,b_w}{8},\qquad
W_{\mathrm{GiB}}=\frac{P\,b_w}{8\times 2^{30}}
$$

P 为全部驻留参数；b_w 为权重位宽。量化元数据另计，不包含在本式中。

当前 4-bit 另加 `P × 0.0625` 字节作为 scale/zero 元数据代理，8/16-bit 没有自动加同类代理。真实 AWQ、GPTQ、GGUF、MLX、FP8 等格式的保留层、分组、scale 和解包缓冲不同，须用实际产物修正。

权重位宽不会自动改变 KV 位宽。以下缓存默认 BF16，每元素 2 bytes；FP32 循环矩阵单列。选择 INT4 容量估算不表示存在兼容权重或芯片算子，更不表示量化质量通过。

多卡权重暂按 `总权重 / (TP × PP)` 均匀分片。embedding、输出头、专家分配或视觉编码器可能使真实 rank 不均匀。未自动规划 EP 专家映射或 CPU offload。

<a id="cache"></a>

## 5. 各种缓存如何计算

### 符号速查

| 符号 | 含义 |
| --- | --- |
| P / bits | 全部驻留参数量 / 权重位宽 |
| B | 同时驻留请求数，不是日活用户数 |
| S / p | 每请求输入加最大输出 tokens / 每页 tokens |
| S̄ | 分页后的长度：`ceil(S/p) × p`，每请求独立取整 |
| L / Hkv | 层数 / KV 头数 |
| dh / dk / dv | 相同头维度 / K 维度 / V 维度 |
| TP / PP / EP | 张量并行 / 流水线并行 / 专家并行 |
| c_rank | 最忙 rank 的每 token 缓存字节数 |

以下缓存公式先得到 **bytes**，除以 `2^30` 转为 **GiB**。数学式中的 t、u 分别表示 TP、PP；c 是每 rank 每 token 的缓存字节，r 是每 rank 每请求的循环状态字节，k 是卷积核长度。

**公式 3 · 每请求分页**

$$
\bar S_i=p\left\lceil\frac{S_i}{p}\right\rceil,\qquad
M_{\mathrm{cache,GiB}}=\frac{\sum_i(\bar S_i c+r)}{2^{30}}
$$

这是一种统一记账形式：标准 KV 的 r=0；混合循环架构的 r>0；MLA/DSA 还要在每请求的分页长度上加尾页。下表中的公式编号对应后续专用展开式。

<a id="model-categories"></a>

### 413 条模型如何分配计算方法

当前目录是 **413 个仓库条目，不是 413 种架构**。相同架构的大小版本、指令版本与量化变体分别占一条。按当前代码和本地配置，163 条有容量解析路径，另有 196 条元数据待补、50 条适配待补、4 条独立上游配方。

| 类别 | 条目数 | 核心计算差异 | 代表系列或模型 | 使用公式 |
| --- | ---: | --- | --- | --- |
| A 标准 MHA/GQA/MQA | 102 | KV 随长度与请求数增长；按 KV 头 TP 切分，头数不足时复制 | 已缓存的 Qwen、Mistral、Gemma 3、ERNIE dense 等 | 2、4、10 |
| B 标准 KV + MoE | 20 | KV 同 A；权重按全部专家驻留，另计路由/专家临时量 | Qwen3 MoE、Mixtral、ERNIE MoE 等 | 2、4、10 |
| C 逐层异构 KV | 13 | 逐层头数/维度不同，PP 取最忙阶段；滑窗/共享不折扣 | MiMo V2、Gemma 4 text | 2、5、10 |
| D MLA | 14 | 缓存 latent+RoPE；TP 复制，不按 KV 头继续缩小 | DeepSeek V2/V3、Moonlight、GLM-4.7-Flash 等 | 2、6、10 |
| E MLA + DSA | 6 | 在 D 上增加索引键和 scale 预算 | DeepSeek V3.2、配置为 glm_moe_dsa 的模型 | 2、6、7、10 |
| F MLA + KDA | 3 | MLA 随 token 增长，KDA 状态随请求数增长 | Kimi Linear | 2、8、10 |
| G 全注意力 + GDN | 5 | 全注意力 KV 与线性循环状态分别计账 | 已适配的 Qwen3.5 / Qwen3-Next 配置 | 2、9、10 |
| R 上游配方 | 4 | 固定检查点/硬件/精度配方，不能推导任意负载 | DeepSeek 专用配方条目 | 不套通用缓存公式 |
| U 结构存在、待适配 | 50 | 需要核对特殊缓存、循环状态或任务结构 | 包括 Mamba 混合、尚未适配的多模态/语音等 | 先补专用布局，不能返回虚构完整显存 |
| M 配置/字段缺失 | 196 | 尚不能可靠选择上述路径 | 包括当前缺本地配置的 Llama 条目 | 先补配置，不按厂商名字猜公式 |

完整对应关系见 [413 条模型分类、字段与 4K 缓存算例](MODEL_PLANNING_INDEX.zh-CN.md)。每个仓库只出现一次；表中 c/r 和缓存例由实际适配器生成，可复算。参数规模未知时仍保留未知，不用激活参数或压缩元素数替代。

**不能合并的三种分类维度：** A—G 表示缓存计算路径；dense/MoE 决定权重与临时量；文本/视觉/语音/embedding 等任务决定额外工作集。多模态模型的文本解码器属于 A—G，不等于整条图像/音频管线已适配。厂商不是计算类别，同一厂商的新旧模型可能走不同路径。

**未完成类别怎样补：** 状态空间模型需逐层累加循环矩阵、卷积历史和注意力 KV；特殊压缩注意力需实际压缩率、索引和后端存储布局；多模态需处理器 token 展开、编码器权重/激活与可能的交叉注意力缓存；embedding/rerank 需编码前向峰值而不是持久生成 KV；扩散需按迭代步骤的激活存活期建模。这里只说明所需方法，不表示这些适配已实现。

### 5.1 标准 MHA/GQA/MQA

对于同形层、相同 K/V 维度 dh：

**公式 4 · 同形层标准 KV**

$$
\begin{aligned}
c_{\mathrm{GQA}}&=4\frac{L}{u}\max\left(1,\frac{H_{kv}}{t}\right)d_h \\
K_{\mathrm{bytes}}&=B\bar S\,c_{\mathrm{GQA}}
\end{aligned}
$$

系数 4 是 K、V 两份乘 BF16 的 2 bytes，不是 FP32 KV。TP 大于 KV 头数时会复制 KV 头，不能继续无限除以 TP。Q 头、KV 头整除/复制关系必须合法。

登记过的滑窗模型可以采用全历史 KV 上界：滑窗层也按 S 预算，不声称后端一定如此分配。可能高估；上界超预算不代表模型绝对不可部署。

2026-10-16 起标准路径候选账本也逐请求分页，与并发包络一致。4097 tokens、16-token 页实际预留 4112 tokens；不能遗漏最后未填满的一页。缓存分页不要求把激活计算的工作 token 数也扩大到分页长度。

MiMo 7B 与 ERNIE 4.5 文本 dense/MoE 已接入标准 GQA 容量适配。ERNIE 0.3B 的显式 head_dim=128，不能用 hidden_size/heads=64 代替；ERNIE MoE 的 `moe_num_experts`、`moe_k`、`moe_intermediate_size` 用于路由/专家临时代理。按全部驻留参数计权重，禁用 MTP/投机生成。不是将其视觉或语音同名变体自动视为兼容。

### 5.2 MiMo：逐层全注意力/滑窗，K/V 可不等维

`hybrid_layer_pattern` 的 0 使用普通 attention 字段，1 使用 `swa_*` 字段。每层读取自己的 Q/KV 头数、K 维度和 V 维度：

**公式 5 · 逐层 KV 与最忙 PP 阶段**

$$
\begin{aligned}
c_\ell&=2\max\left(1,\frac{H_{kv,\ell}}{t}\right)(d_{k,\ell}+d_{v,\ell}) \\
c_{\mathrm{rank}}&=\max_j\sum_{\ell\in\mathcal L_j}c_\ell \\
K_{\mathrm{bytes}}&=B\bar S\,c_{\mathrm{rank}}
\end{aligned}
$$

𝓛ⱼ 是第 j 个 PP 阶段分配到的连续层集合。

等层 PP 将连续层分给不同阶段，取 `c_rank = max_stage Σ c_l`，再计算 `B × S̄ × c_rank`。不能把全部缓存平均除以 PP：层分布不均可能使某个阶段更大。每层均须满足 TP 头数约束。

**计算例：** 4 层，前两层 KV 头数 2、后两层 4，dk=64、dv=32。

| TP | PP | 最忙 rank 的每 token 缓存 | 说明 |
| --- | --- | --- | --- |
| 1 | 1 | `2 × (2+2+4+4) × 96 = 2304 bytes` | 全部层在同一阶段 |
| 1 | 2 | `2 × (4+4) × 96 = 1536 bytes` | 取较大阶段，不是平均值 1152 |
| 2 | 2 | `768 bytes` | 本例 KV 头可按 TP 分片 |

这是计算回归夹具，不是真实模型性能。

MiMo V2/V2 Flash 适配只计算文本请求，滑窗按全历史，不计 MTP、前缀共享、图像/音频编码峰值。包含编码器权重不等于包含编码器激活。

### 5.3 MLA 与 DSA

GLM-4.7-Flash 这类 MLA 的 hidden_size 不必整除 Q 头数：缓存来自显式 latent/RoPE 维度，不能被标准 GQA 的隐式 head_dim 检查拒绝。2026-10-16 已修复此错误门槛，并加入主流系列试算回归。

缓存低秩 latent 和 RoPE，不缓存展开后的全部 KV 头：

**公式 6 · MLA 压缩缓存**

$$
c_{\mathrm{MLA}}=\frac{L}{u}\,2(d_{\mathrm{latent}}+d_{\mathrm{RoPE}}),\qquad
K_{\mathrm{bytes}}=B(\bar S+p)c_{\mathrm{MLA}}
$$

d_latent 对应 `kv_lora_rank`，d_RoPE 对应 `qk_rope_head_dim`。专用混合 KDA 路径见公式 8，不继承此处独立 MLA 适配器的额外尾页策略。

当前 MLA 预算在 TP rank 间复制，不除以 TP。DSA 每层另加索引键代理 `2 × index_head_dim + 4 × ceil(index_head_dim/128)` bytes/token，使用 BF16 保守预算，不冒充上游 FP8 实际占用。

**公式 7 · DSA 索引增量**

$$
c_{\mathrm{DSA}}=c_{\mathrm{MLA}}+
\frac{L}{u}\left(2d_{\mathrm{index}}+4\left\lceil\frac{d_{\mathrm{index}}}{128}\right\rceil\right)
$$

将公式 6 中的 c_MLA 替换为 c_DSA，再计请求与分页；d_index 对应 `index_head_dim`。

页至少 64 tokens；每请求额外一个尾页：`KV = B × (S̄+p) × c`。后端若展开缓存、改变索引布局或压缩存储，须对应版本独立建模。

### 5.4 Kimi Linear：MLA + KDA

MLA 层按压缩维度随 token 增长。KDA 按请求保存状态：

**公式 8 · KDA 每请求状态与混合总量**

$$
\begin{aligned}
r_{\mathrm{KDA}}&=L_{\mathrm{KDA}}\frac{H}{t}
\left[4d_h^2+6d_h(k-1)\right] \\
c_{\mathrm{hybrid}}&=2L_{\mathrm{MLA}}(d_{\mathrm{latent}}+d_{\mathrm{RoPE}}) \\
M_{\mathrm{cache,bytes}}&=B(\bar S c_{\mathrm{hybrid}}+r_{\mathrm{KDA}})
\end{aligned}
$$

前项是 FP32 矩阵，后项是 BF16 卷积历史代理。总量为 `B × (S̄ × MLA每token字节 + KDA每请求字节)`。循环状态不再乘上下文长度。当前仅 TP、PP=EP=1，不含前缀状态快照/投机状态。

### 5.5 Qwen Gated DeltaNet + 全注意力

仅全注意力层按标准 KV 计算。线性层每请求状态为：

**公式 9 · GDN 矩阵与卷积历史**

$$
\begin{aligned}
r_{\mathrm{matrix}}&=4\frac{H_v}{t}d_vd_k \\
r_{\mathrm{conv}}&=2\left(2\frac{H_k}{t}d_k+\frac{H_v}{t}d_v\right)(k-1) \\
r_{\mathrm{GDN}}&=L_{\mathrm{linear}}(r_{\mathrm{matrix}}+r_{\mathrm{conv}}) \\
M_{\mathrm{cache,bytes}}&=B(\bar S c_{\mathrm{full}}+r_{\mathrm{GDN}})
\end{aligned}
$$

H_k/H_v 是线性层 key/value 头数，不是全注意力的 Hkv。c_full 用公式 4，但 L 只计全注意力层；当前 u=1。

同样是 FP32 循环矩阵加 BF16 卷积历史。全注意力部分随 token 增长，循环部分随驻留请求数增长。当前 PP=EP=1，文本模式；不能据此推断所有状态空间架构。

### 5.6 Gemma 4：全注意力与滑窗的不同维度

Gemma 4 的 `gemma4_text` 逐层适配（2026-10-16）使用以下原始配置字段：

| 层类型 | 头维度 | KV 头数 |
| --- | --- | --- |
| 滑窗层 | `head_dim` | `num_key_value_heads` |
| 全注意力层 | `global_head_dim` | `num_global_key_value_heads` |

全注意力的 global 字段为空时回退普通字段。采用第 5.2 节逐层累加公式，不扣除跨层 KV 共享或 K=V 复用，故为保守缓存上界。26B-A4B 的 `top_k_experts` 另计入 MoE 临时代理。

> **覆盖边界：** 已对 E2B、E4B、26B-A4B、31B 文本请求试算；视觉/音频激活及 unified 架构未随之批准。

字段依据为固定 SGLang commit `e00930c5489053f26d86b179cee0d087f846acbb` 的 `utils/hf_transformers/config.py` 与 `models/gemma4_causal.py`，独立实现、没有复制模型代码。

<a id="activation"></a>

## 6. Prefill、MoE 与临时量

标准推理使用 `12 × B × S × hidden` bytes 临时张量代理。专用 MLA、KDA/GDN、MiMo 要求分块 prefill，`T=min(B×S, chunk_tokens)`，临时量为：

**公式 10 · 分块 prefill 临时量代理**

$$
\begin{aligned}
T&=\min(BS,C_{\mathrm{chunk}}) \\
A_{\mathrm{proxy,bytes}}&=T(12d_{\mathrm{hidden}}+4N_{\mathrm{experts}}+4k_{\mathrm{top}}d_{\mathrm{expert}}) \\
A_{\mathrm{GiB}}&=\max\left(A_{\mathrm{floor}},\frac{A_{\mathrm{proxy,bytes}}}{2^{30}}\right)
\end{aligned}
$$

标准 A/B 路径当前采用 T=BS，不自动假设分块收益；专用 C—G 采用上述分块上限。k_top 为路由 top-k，与循环卷积核长度 k 不同。无 MoE 时专家项为零。

系数 12 是张量合计代理，不是 dtype。该式不保证覆盖全部 attention、通信或工作区峰值；须配置 workspace 并实测各阶段。减小 chunk 不会消除已经驻留的全上下文 KV。

MoE 全专家权重仍驻留；top-k 仅影响临时计算代理。不同字段须适配，不能用激活参数量替代总参数。

<a id="topology"></a>

## 7. 多卡候选怎么搜索

自动卡数先枚举节点内数量，再枚举整节点数量，最多 4096 卡。每个卡数枚举 TP/PP，检查层数、头数、隐藏/FFN 维度整除关系，检查最忙卡峰值。MLA 允许特定 2 的幂跨节点 TP；其他路径以节点内 TP 加等层 PP 为主，循环混合路径仅 TP。

这是当前搜索及预算下的条件最小卡数，不是所有系统中的最优卡数。固定卡数不会擅自扩卡。DP 副本提高服务容量，不等同于将一份权重拆小。互联延迟、EP、非均匀 PP 和 offload 不在当前最优搜索中。

<a id="concurrency"></a>

## 8. 单请求、多请求、长短混合

### 8.1 固定长度的并发上限

B 是同时驻留序列数。权重共享一次，KV/循环状态按每请求增加；B=1 就是单请求容量，不是全天用户数量。

固定 GPU、TP/PP 后，整数二分找最大 B，使 `峰值(B,S) ≤ 可用预算`，搜索上限 65536。上下文边界同理搜索并受模型声明限制。达到搜索上限是截断结果，不是精确极限。

### 8.2 长短请求混合

混合长度 `2048:8,8192:2` 表示 8 个 2048-token 和 2 个 8192-token 请求：

**公式 11 · 按长度组累加**

$$
\begin{aligned}
K_{\mathrm{bytes}}&=c_{\mathrm{rank}}\sum_g B_g p\left\lceil\frac{S_g}{p}\right\rceil \\
R_{\mathrm{bytes}}&=r_{\mathrm{rank}}\sum_g B_g
\end{aligned}
$$

g 表示长度分组，B_g 是这一组的请求数。权重和固定运行时预算只计一次，不乘请求数。

MLA 逐请求再加尾页。不能用平均长度取整代替逐请求取整。

### 8.3 长上下文的单独成本

长上下文增量在同一 TP/PP 布局下与 2048-token 比较，分开列 KV 与激活增量。重新优化卡数会改变权重分摊，掩盖长度单独成本，因此不混算。

内存并发不是 GPU 利用率、token/s、TTFT 或 SLO 并发。服务副本规划需要相同模型、精度、拓扑和请求分布的联合实测 RPS、p95/p99 与错误率，见 [推理服务副本规划](SERVING_PLANNER.md)。

<a id="edge"></a>

## 9. 端侧、Apple Silicon、座舱和机器人

端侧可用量为 `设备/统一内存 × (1−reserve) − 系统预算`；驻留量为权重、量化元数据、缓存和用户运行时预算。运行时预算须覆盖真实临时峰值；不从 TOPS 自动推算 token/s。端侧混合长度尚未接入，会明确提示。

能装下不等于 NPU/GPU 算子支持。MLX、llama.cpp、QNN、RKLLM 等路线的转换格式、芯片 SKU、驱动/BSP、上下文和算子边界须独立验证。座舱/机器人候选不进入安全关键控制闭环。本轮没有真实端侧芯片运行验证。

<a id="validation"></a>

## 10. 如何验证方案

### 10.1 通用验证顺序

1. 固定模型 revision、tokenizer、chat template、权重格式和后端版本；保存原始配置及来源。
2. 核对总参数、逐层缓存字段、KV dtype、分页和拓扑，以及 MTP/共享/Graph 开关。
3. 从小上下文单请求验证加载与输出，再逐步增加长度和并发。
4. 逐卡采集加载、prefill、decode、Graph、通信峰值，修正预算。
5. 测混合长度、取消、超时、过载和恢复；同时检查输出质量与延迟。
6. 保存成功和失败证据，只升级已经验证的精确组合，不推广到整个家族。

### 10.2 低于 50B 的优先验证清单

清单更新：2026-10-16。以下仅为本轮优先模型，不是完整支持目录。

| 模型 | 首轮负载 | 随后扩大 | 特别检查 |
| --- | --- | --- | --- |
| baidu/ERNIE-4.5-0.3B-PT、0.3B-Base-PT | BF16、单请求、2048 总 tokens | 4096→32768，再到声明上限；并发 1→2→4 | 显式 128 维 attention head；Base 不默认 chat 模板 |
| XiaomiMiMo/MiMo-7B-Base、MiMo-7B-RL | BF16、单请求、2048 总 tokens | 4096→32768；并发 1→2→4 | 不启用 MTP；分别核对模板和输出行为 |
| baidu/ERNIE-4.5-21B-A3B-PT、21B-A3B-Thinking | BF16、单请求、2048 总 tokens | 固定设备逐步扩长度/并发；超预算重新规划多卡 | 不是仅驻留 3B；router/专家缓冲及共享专家峰值须实测 |

这些是验证顺序，不是可直接执行的通用启动命令：先在规划器选择真实设备和预算，再核对后端对精确模型的支持。容量足够才进入下一档；量化单独验证实际产物和质量，不能继承 BF16 结论。尚未进行本轮权重下载或实机测试。

### 10.3 每次测试记录什么

| 记录类别 | 必填内容 |
| --- | --- |
| 身份与环境 | 模型 revision、后端版本、芯片与显存 |
| 配置与负载 | 权重/KV 格式、TP/PP、总上下文、并发 |
| 容量与性能 | 峰值显存、TTFT/ITL |
| 正确性与故障 | 输出正确性、失败日志、取消/超时回收结果 |

保持同一拓扑做长短请求对照。规划器测试覆盖 4/8/16 位及并发 1/4 的非整页长度，不意味着这些量化格式都存在。

<a id="sources"></a>

## 11. 实现与可复核来源

### 11.1 代码入口

| 模块（位于 `workbench/`） | 职责 |
| --- | --- |
| `resources.py` | 多卡枚举与逐卡账本 |
| `deployment_envelope.py` | 并发、上下文与混合长度边界 |
| `edge_planning.py` | 端侧容量预算 |
| `hybrid_cache.py` | KDA / GDN 混合状态 |
| `mla_planning.py` | MLA / DSA 缓存 |
| `layered_cache.py` | MiMo / Gemma 4 逐层缓存 |

### 11.2 来源与实现归属

本轮标准结构语义参考同一 SGLang 固定 commit 的 `models/mimo.py`（Qwen2DecoderLayer）、`models/ernie4.py`（LlamaAttention 与 ERNIE router 字段）。属于思路/字段语义参考，没有复制运行时代码；原生推理核支持与容量规划分开记账。

MiMo 字段语义核对自 SGLang commit `e00930c5489053f26d86b179cee0d087f846acbb` 的 `python/sglang/srt/models/mimo_v2.py`、函数 `_get_ckpt_qkv_shard_sizes`。这是配置语义参考与独立公式实现，未复制模型代码，也未将外部后端称为自研核。可复核[固定版本源码](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/models/mimo_v2.py)。其他专用公式的来源约束随模块注释保存。

### 11.3 回归与覆盖清单

`tests/test_layered_cache.py` 检查异构层、非等维 K/V、最忙 PP 阶段、TP 复制、非法布局与混合请求对账。`tools/audit_planning_coverage.py` 遍历目录试算 H100 容量组合；不是全硬件执行测试，缺元数据不算通过。最新覆盖清单见 [2026-10-16 规划覆盖审计](planning-coverage-20261016.json)；[2026-10-15 清单](planning-coverage-20261015.json) 保留作历史对照。

尚需补齐未适配的状态空间/压缩架构、精确后端布局、非均匀权重放置、真实量化缓冲和设备实测。本文解释当前可复核的计算，不承诺任意新模型可直接套用。
