# Training-design evidence library / 训练规划证据库

Review cutoff: **2026-10-07**. This is a curated research snapshot, not a claim that every vendor has released a complete 2026 training recipe.

Start with the [clickable local archive index](INDEX.md): 36 verified sources, including 21 PDFs; six unavailable entries are explicitly listed. This includes 16 model/system reports and five CS336 slide decks, plus two lecture source files and the course page.

## Archive and distribution

- `../workbench/knowledge/sources.json`: official URLs, year, document type, scope and reviewed license information.
- `../workbench/knowledge/rules.json`: reviewed, bilingual planning rules and release-specific references. These ship with the Python package.
- `receipts.jsonl`: append-only retrieval history, including failures, resolved URLs, byte counts and SHA-256.
- `../.knowledge-cache/<source-id>/<SHA256>.pdf`: permanent local full-text archive; corresponding `.txt` files contain page-delimited searchable extraction. HTML and code are stored as reference material, never executed. No automatic expiry or cleanup exists.
- `../tools/knowledge_status.py`: checks actual archived bytes, not merely successful HTTP status.

Run from the project directory:

```sh
python3 tools/knowledge_status.py
python3 tools/archive_knowledge.py --transport curl --id mimo26
python3 tools/archive_knowledge.py --transport curl
```

The last command refreshes all manifest entries and keeps previous content-addressed versions. Both transports disable explicit proxy use; neither controls VPN/transparent routing. PDF extraction is optional and uses `pypdf` if already installed. A failed refresh does not invalidate an older verified copy. A failed download is **not** counted as an archive. Dynamic dashboards may yield only an HTML shell, not their telemetry.

**Open-source publication:** project-authored summaries, rules, scripts and source metadata are distributed with the project. Third-party full texts do not inherit the project's MIT license. `.knowledge-cache` is excluded from Git pending document-by-document permission review; it remains on this machine and must be included in the user's backups. Re-distributable copies, when explicitly reviewed, belong under `redistributable/` with attribution and original license. Model-weight licenses, repository code licenses and report licenses are separate. “Permanent” here means no expiry, not protection against disk loss.

## Vendor coverage and what the evidence actually establishes

Use the status command for current download/integrity results. An entry below is not itself proof that retrieval succeeded.

| Vendor | Manifest entries | Planning takeaway and limitation |
| --- | --- | --- |
| Z.ai / GLM | `glm5`, `glm53-card` | GLM-5 report describes sparse attention, mid-training and decoupled RL. Its 4K→200K schedule is not GLM-5.3-Flash's undisclosed schedule. |
| DeepSeek | `deepseek4` | §4.2.2: 4K→16K→64K→1M, mixed Muon/AdamW and a change in attention regime. A length change may require different kernels and optimizer accounting, not just a new integer. |
| NVIDIA | `nemotron3-ultra` | §2.5: GB200, CP32/TP8/EP128/PP2; 1M and 4K iterations alternate. EP nesting and rack-scale fast domains must be understood before calculating GPU count. Hybrid layers and NVFP4 require their own adapters. |
| Mistral | `mistral-voxtral`, `mistral-small4`, `mistral-models` | Voxtral TTS is a speech report. Small 4 announcement and model catalog are not full pretraining configurations for Small/Large 4. |
| Google / Gemma | `gemma4`, `diffusiongemma` | Distinguish dense/MoE variants, distillation, multimodal components and diffusion objectives. Neither parameter size nor a family name proves the causal-decoder estimator applies. |
| Meta / Llama | `llama3`, `llama4-card` | Historical 2024 report and 2025 model card. No verified 2026 Llama training report located in this review; do not relabel predecessors as new releases. |
| Qwen | `qwen38`, `qwen35-omni`, `qwen3` | Mainline repository, 2026 Omni report and explicitly historical Qwen3 report are separate. Omni S1 freezes LLM, S2 unfreezes all at 32,768, S3 uses 262,144; S1 length remains unknown. |
| Xiaomi / MiMo | `mimo26`, `mimo26-announcement`, `mimo26-live`, `mimo25-inference`, `mimo2-flash` | V2.6 report and public RL resources are primary references. Official ModelScope PDF is verified against HF's published SHA-256. V2.5 inference results do not establish V2.6 pretraining hardware. V2-Flash is a historical predecessor. |
| MiniMax | `minimax-m2`, `minimax-h3`, `minimax-m3` | M2 report: 8K→32K→192K, with short and long data in the decay phase. H3/M3 announcements cannot substitute for their own full technical reports. H3 is not a generic causal-LLM training job. |
| Ai2 | `olmo-hybrid`, `olmocore3`, `olmocore3-blog`, `olmo3` | Hybrid architecture and new systems experiments; distinguish completed model flows from capacity/throughput benchmarks. Olmo 3 is explicitly historical. |
| AMD | `instella-moe` | Fully open 16B-total/2.8B-active model flow: data mixtures, configurations, code, checkpoints. Primus/Megatron on ROCm for training; Miles/Primus learner with SGLang rollout for RL. |

## 从报告到规划器：不是照抄厂商卡数

1. **确认对象。** 先读取 config 和全部权重头，分清总存储元素、唯一逻辑参数、可训练参数、每 token 激活参数。量化打包权重的形状不能直接等于原参数数。精确仓库身份才关联已核查版本，目录名 `source-bf16` 不能决定训练历史。
2. **确认任务。** 从头预训练、CPT、SFT、DPO、GRPO 的目标不同。发布一个已训练 checkpoint 不意味着用户需要重做几十万亿 token。先确定本次 token 预算、数据长度分布、训练掩码与质量门槛。
3. **建立逐卡账本。** Adam 常见布局含权重、梯度、主副本、两个动量；精度与分片分开计算。加入激活、重计算临时量、logits、专家 dispatch、通信工作区和运行时余量。`16P` 只对应指定布局，不是所有优化器的定律。冻结模块不需要完整可训练状态；没有对应适配器就不能伪造节省量。
4. **组合并行。** TP 切算子、PP 切层、CP 切上下文、DP 复制数据工作，ZeRO/分布式优化器切状态；EP 在当前实现中嵌套 DP。不能把所有轴独立相乘，也不能把全部显存除以总卡数。先满足形状整除、实际互联、框架支持和 batch 约束。
5. **同时看三堵墙。** 容量墙看每张卡峰值；带宽墙看 HBM 字节／有效带宽；通信墙看集合操作消息量／实测链路。通信重叠、低精度转换、流水线气泡与路由偏斜可能抵消理论收益。少卡、低成本、最快完成是三个不同目标。
6. **每阶段重新求解。** 长度增长会改变激活、注意力计算和可用微批量；不能沿用上阶段配置。当前工程默认 token 比例仍标记为启发式，不冒充论文配比。官方历史显示在独立参考区，不自动替换 CPT/SFT 方案。
7. **小跑校准再放大。** 先执行小规模 scaling ladder；预热后至少收集 20 个稳定 step 的峰值显存、耗时、有效 token、loss/梯度、通信。20 步只是本项目烟测门槛，不证明收敛。验证保存／恢复，再用实测值重算成本和期限。

### MoE 与 RL 的新增检查

对固定 token batch，`floor(batch_tokens / sequence_length)` 只是 packed sequence 数上界，不等于独立文档数。长阶段要测逐专家 token、最大／平均负载、丢弃率和 all-to-all，而非只看 balance loss。

MiMo-V2.6 报告的训练配置含 1,568 prompts × 16 rollouts、异步 partial rollout 与 staleness=4；路由冻结是该运行的稳定性措施，不应自动强加给所有模型。预算必须覆盖 actor、reference、grader/reward、rollout KV、轨迹传输以及角色同时驻留时段。公开 RL 资源也不等于完整预训练语料全部公开。

### 已落实与仍未实现

已落实：离线规则加载及哈希溯源、精确版本参考、优化器不匹配阻断、分阶段独立资源搜索、MoE 阶段验证门槛、packed sequence 上界、多模态/RL 校验要求、界面来源展示。

仍需专项实现与硬件验证：Muon/NVFP4 实际状态适配、任意混合注意力激活生命周期、跨节点 NVLink 快速域搜索、训练掩码精确优化、自动多角色 RL 编排。报告存在不会把这些能力自动变成可运行支持。

## Public training projects: degrees of openness

| Project | What is exposed | What this archive does not claim |
| --- | --- | --- |
| [Marin](https://marin.community/) | Code/data/experiments, public Hero run issue and 32B retrospective | Ongoing 535B plan is not a completed recipe; archived issue does not contain every comment, dashboard or checkpoint. TPU/JAX layout is not a drop-in NVIDIA plan. |
| [MiMo-V2.6](https://mimo.mi.com/docs/en-US/news/latest/v2-6) | Report, weights and RL research resources; public RL demonstration | Announcement and dashboard shell do not archive every trajectory or prove all pretraining data is public. |
| [Olmo 3](https://arxiv.org/abs/2512.13961) | Historical open model flow with data/checkpoints/code | Availability of linked assets is not a claim this project downloaded all of them. |
| [Olmo-core 3](https://huggingface.co/blog/allenai/olmocore3) | Training systems, design discussion and benchmark experiments | Random-routing trillion-parameter benchmark is not a converged trained model; short capacity test is not sustained throughput. |
| [Instella-MoE](https://arxiv.org/abs/2609.00791) | Report explicitly releases weights, configs, data mixtures and training code | ROCm/MI300X/MI325X results cannot be substituted for measured NVIDIA results. |
| SmolLM3 / training playbook | Pinned training README and stage-1 YAML; links to logs, nanotron/datatrove/lighteval | Playbook retrieval failed; the recipe is not a full-playbook archive. Historical reference, not a new 2026 model release. |

## CS336 reading route

Official course: <https://cs336.stanford.edu/>. Lecture source files are archived as inert text from commit `de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e`; slides use the official course site and are content-hashed because that site can update in place.

| Source ID | Study question | Workbench consequence |
| --- | --- | --- |
| `cs336-02` | Where do FLOPs, bytes and arithmetic intensity come from? | Explain each ledger term and distinguish compute from bandwidth. |
| `cs336-07`, `cs336-08` | What is partitioned, replicated and communicated? | Candidate divisibility/topology checks, not `total memory / GPU count`. |
| `cs336-09`, `cs336-11` | How do we allocate compute and fit scaling behavior? | Small measured ladder; no universal best token/parameter ratio. |
| `cs336-15`, `cs336-16` | How do mid/post-training and RL change objectives and roles? | Stage-specific budgets and separate rollout/optimizer accounting. |

Only the selected related materials are archived; this is not a claim to have mirrored every video, assignment, linked paper or student solution. The previously mentioned local `lectures-main` was not located in the searched project locations; this archive uses the official source instead.

## Maintenance contract

When adding a rule, record its source, exact release scope, section/page where applicable, and implementation boundary. Add a regression test for nonmatching releases. Recompute the manifest hashes automatically through the planner. Before release, run the archive integrity check and report unavailable entries. Never turn a publication's throughput claim into a GPU-table default without a compatible local measurement.
