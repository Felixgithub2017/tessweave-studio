"""Explain the actual recipe contract, not an invented performance certification."""
import copy

VLLM = 'https://docs.vllm.ai/en/stable/configuration/engine_args/'
SGLANG = 'https://docs.sglang.io/docs/advanced_features/server_arguments'
MLX = 'https://github.com/ml-explore/mlx-lm'
SWIFT = 'https://github.com/modelscope/ms-swift/tree/v4.5.3/examples'


def item(name, status, flow, principle, execution, benefit, limits, verify, source):
    return dict(name=name,status=status,flow=flow,principle=principle,execution=execution,
                benefit=benefit,limits=limits,verify=verify,source=source)


SCHEDULING = item('连续批处理：把空出来的位置及时补上','引擎机制；实际调度由版本与负载决定',
    'A/B一起生成 → A结束 → 下一轮补入C → B/C继续',
    '普通逐请求循环常等待前一个请求全部完成；连续批处理按生成轮次调度，不必等整个静态批次结束。原生Transformers也可用高效注意力/批处理，不能把它等同于最慢实现。',
    '当前启动引擎服务，调度器在引擎内部；没有独立的“连续批处理倍速”开关。',
    '主要提高并发吞吐与GPU忙碌比例。教学例：同一时间完成20条代替10条是2×吞吐，不代表每条延迟减半。',
    '单请求可能没有收益；排队、批次变大也可能增加尾延迟。',
    '同一模型、相同精度与提示集，分别并发1/4/16；比较输出token/s、失败率与p95首增量延迟。',VLLM)

INFERENCE = {
 'vllm':[
    SCHEDULING,
    item('PagedAttention / 分块KV管理','引擎机制；不是量化',
      '生成token → 写入KV块 → 块表映射 → 后续注意力读取',
      '像分配固定大小的小抽屉，序列增长时再领新块，减少预留连续大空间产生的浪费。不会减少一个已缓存token自身的K/V数学内容。',
      '当前命令指定 --gpu-memory-utilization 0.80；这是引擎显存预算，不是80%加速率，也不是KV精度设置。',
      '收益首先是更多并发/更少碎片，不给固定速度倍数。若每块16token，17token需2块、15个槽位未用（教学示例）。',
      '仍有块尾浪费和图执行工作区；不能解决所有OOM。',
      '长短请求混合压测，记录可驻留请求数、显存峰值、抢占/重算和吞吐。',VLLM),
    item('前缀缓存','后端默认/自动；本配方未显式锁定开关',
      '公共提示前缀 → 首次计算KV → 后续命中直接复用 → 只算新增后缀',
      '缓存的是已计算的前缀KV，不是把问题的答案直接抄给下一个请求。',
      '当前argv没有 --enable-prefix-caching；具体版本是否默认启用、是否命中，查看启动日志和指标。',
      '4000个公共token+1000个新token：命中后可跳过公共部分的prefill计算。不是端到端固定5×，也不直接减少逐token decode。',
      '冷启动/不同前缀/缓存被驱逐时收益小；多租户需审计缓存隔离。',
      '同样请求分别冷缓存与热缓存测试，单列prefill/首token指标，不混报平均值。','https://docs.vllm.ai/en/v0.12.0/features/automatic_prefix_caching/'),
    item('融合注意力与CUDA Graph','后端自动选择；实际内核未验证',
      '算子序列 → 融合减少显存读写 / 捕获可复用执行图 → 后续重放',
      '融合减少中间张量搬运；Graph减少CPU反复提交小算子的开销，两者优化的是不同成本。',
      '当前命令没有强制指定attention backend或Graph配置；必须结合GPU、dtype、shape及启动日志确认。',
      '内存IO或launch开销占比高时有机会获益；不能将后端名称换算成固定倍数。',
      '动态shape、unsupported算子、额外图缓存会改变收益与峰值。',
      'Nsight Systems看CPU launch空隙；Nsight Compute看实际kernel读写/耗时；剖析与正式跑分分开。',VLLM)
 ],
 'sglang':[
    dict(SCHEDULING,source=SGLANG),
    item('RadixAttention / 前缀树缓存','后端自动；启动日志确认实际缓存策略',
      'token前缀 → 在前缀树找共同路径 → 复用对应KV → 为新后缀分配缓存',
      '例如两次请求都有同一段系统提示，只重算新问题部分；它是KV复用与管理，不是训练出新的注意力参数。',
      '本配方未指定 --disable-radix-cache；不能据此宣称每个请求都会命中。',
      '多轮会话/公共长前缀可减少重复prefill；无共享前缀时不应期待缓存收益。',
      '真实收益受前缀完全匹配、驱逐、可用内存和引擎版本影响。',
      '冷/热缓存分组对比，监控命中情况、首token延迟及内存。',SGLANG),
    item('分块prefill、融合内核与CUDA Graph','由后端版本/模型自动决定；未显式调参',
      '长prompt分段 → 与decode请求协调调度 → 优化内核执行',
      '切小一次prefill的工作量，降低长请求霸占计算的影响；Graph和融合分别针对提交开销与张量搬运。',
      '当前仅设 --context-length 和 --mem-fraction-static 0.80；未设置chunked-prefill-size或Graph专用参数。',
      '混合长短请求可能改善尾延迟；不是对所有prefill都更快。',
      '分块更小可能增加调度开销，Graph需要额外显存；需实测找折中。',
      '长prompt混入短请求，对比p95延迟、吞吐、显存峰值并检查kernel trace。',SGLANG)
 ],
 'mlx':[
    item('MLX / Metal 与统一内存','当前选择的Apple执行路径',
      '本地权重 → MLX数组 → Metal GPU算子 → KV缓存 → 输出token',
      '使用Apple Silicon上的MLX运行时和Metal实现；统一内存避免把CPU内存与独立显存当两份地址空间，但仍有带宽、同步和工作区成本。',
      '本配方运行 python -m mlx_lm.server --model 路径；不会叠加vLLM或SGLang。',
      '可利用Mac GPU；对比CPU基线可能提速，但对比其它Metal引擎没有预设胜负。',
      '48GB是操作系统与应用共享总内存，不是48GB全部可放权重。',
      '测首增量延迟、token/s、内存压力与swap；发生换页时不能只看权重文件大小。',MLX),
    item('加载MLX分组量化权重','仅当选中的权重已转换',
      '量化包中的整数+scale/bias → 匹配的量化矩阵乘内核 → 输出',
      '更少的权重字节降低带宽压力；不是先永久解压一份完整FP16模型。',
      '先在压缩页选MLX4导出新目录，再扫描该目录并用于MLX推理；选MLX引擎本身不会自动量化。',
      '权重主体约缩至原FP16的1/4，再加scale/bias与未量化层；速度需实测。',
      '量化不会自动压缩KV或激活；适配范围以mlx-lm为准。',
      '同一模型量化前后、相同上下文与并发，比较内存和质量。',MLX)
 ]
}

COMPRESSION = {
 'awq':item('AWQ · 激活感知INT4权重量化','已有导出配方；尚未GPU验收',
    '高精度模型+128条校准数据 → 收集激活 → 搜索通道缩放 → 舍入/打包INT4 → 保存新目录',
    '并非所有通道都同样敏感。利用代表性输入观察激活，选择缩放以减轻低位表示对重要通道的误差；不等于把重要权重全部保留FP16。',
    'swift.cli.export --quant_method awq --quant_bits 4 --dataset … --quant_n_samples 128 --max_length …；当前校准数据沿用SFT JSONL校验。',
    '原始权重16→4bit，理想主体减少75%。加组scale、零点及保留层后略大；端到端速度不承诺4×。',
    '需要代表性校准集、转换内存与匹配AWQ内核；分组/保留层使用后端配置，必须读输出量化配置确认。',
    '重新扫描输出，核对quantization_config、字节数；同负载测质量与速度。','https://arxiv.org/abs/2306.00978'),
 'gptq':item('GPTQ · 二阶误差补偿INT4','已有导出配方；尚未GPU验收',
    '高精度模型+校准输入 → 统计层输入相关性 → 逐步量化 → 补偿剩余权重 → 打包输出',
    '把某些权重舍入后，会改变层输出；用输入统计估计误差并调整尚未量化部分。它不是重新做一次完整预训练。',
    'swift.cli.export --quant_method gptq --quant_bits 4 --dataset … --quant_n_samples 128 --max_length …。',
    'INT4权重主体理想约4×缩小；比AWQ更快或更准都不能预先保证。',
    '二阶统计与校准有额外内存开销；group size、act-order及内核兼容以导出配置为准。',
    '校准集与验收集分开；观察长上下文、代码/数学等敏感任务退化。','https://arxiv.org/abs/2210.17323'),
 'fp8':item('FP8 · 8位浮点导出','已有导出配方；W8A8执行未认证',
    '高精度权重 → 选择浮点缩放/表示 → 存储FP8权重与scale → 后端加载',
    'FP8仍有指数和尾数，用更少位表示浮点范围；具体格式、scale粒度以及激活是否FP8是独立契约。',
    'swift.cli.export --quant_method fp8；本配方没有传校准集，也没有强制KV精度或激活W8A8路径。',
    '权重主体16→8bit约减半；仅在支持的低精度kernel上才可能加速计算。',
    '不能把FP8文件等同于硬件原生FP8运算，更不能宣称KV也减半；输出配置与GPU代际须核实。',
    '检查输出dtype/scales、后端加载日志和实际kernel；对比质量与吞吐。',SWIFT),
 'mlx4':item('MLX4 · 每64个权重一组','已有Apple导出配方；待真实模型验收',
    '高精度线性层 → 每64个值共用scale/bias → 4bit整数打包 → MLX目录',
    '组内用少量刻度和偏移还原近似值。例：scale=0.1、偏移=0，0.26舍入到整数3，再还原0.30，误差0.04；真实scale由该组决定。',
    'mlx_lm.convert --hf-path … --mlx-path 新目录 -q --q-bits 4 --q-group-size 64；不使用AWQ/GPTQ校准流程。',
    '若scale/bias各16bit：每参数4+32/64=4.5bit，主体约3.56×缩小；未量化层另计。',
    '这里的4.5bit是指定元数据位宽的教学估算，不代替读取实际文件；转换峰值仍可能需要高精度模型。',
    '用同一Mac对比MLX BF16与MLX4，记录内存压力、token/s与质量。',MLX),
 'structured-pruning':item('结构化剪枝','未接执行器；禁止当作已实现',
    '评估重要性 → 删除通道/头/专家 → 修改结构 → 恢复训练 → 验收',
    '真正减小矩阵维度才可能减少计算；仅把权重填零不等于删掉矩阵行列。',
    '当前只能阅读说明，计划会阻断，不会运行通用删除脚本。',
    '取决于删去结构与恢复质量，不提供没有模型依据的压缩率或倍速。',
    '残差宽度、头数、路由结构和配置必须一致；可能需要恢复训练。',
    '结构合法性、恢复后的质量、实际kernel形状及端到端延迟。',SWIFT),
 'sparse-2of4':item('2:4结构稀疏','未接执行器；禁止当作已实现',
    '每4个值选2个非零 → 压缩值与索引 → 匹配稀疏kernel',
    '非零数量减半，但若仍调用密集矩阵乘，GPU仍会计算零值。',
    '当前计划阻断；没有稀疏导出、恢复训练与部署适配器。',
    '50%零值不等于总显存减少50%或整模型2×加速；索引与非稀疏部分仍占资源。',
    '必须同时满足硬件、布局、dtype和kernel支持。',
    '核查是否真正使用稀疏kernel，而非只比较零值比例。',SWIFT)
}


def technique_guide(backend='vllm',method='awq'):
    if backend not in INFERENCE or method not in COMPRESSION:
        raise ValueError('未知推理引擎或压缩方式')
    inference=copy.deepcopy(INFERENCE[backend])
    if backend in ('vllm','sglang'):
        inference.append(item('TP张量并行与量化包加载','TP由界面参数配置；量化取决于所选权重',
            '选择权重目录 → 读取量化配置 → 按TP切矩阵 → 各卡计算并通信归并',
            'TP把同一层分给多卡，不是每张卡放完整模型。量化减少权重字节；两者可组合，但增加通信或反量化开销。',
            ('命令 --tensor-parallel-size' if backend=='vllm' else '命令 --tp-size')+' 等于所选TP，TP=1表示不做多卡切分。未请求在线量化；已有量化包须由引擎识别格式并选择支持的kernel。',
            '主要让模型装下，并有机会提高计算吞吐。权重理想均分不代表每卡总显存严格除以TP；速度也不随卡数线性增加。',
            '头数整除、KV复制、互联带宽、格式支持都会影响结果。两个引擎不能用串联服务的方式叠加加速。',
            '核对生成计划TP/设备数、输入quantization_config、启动日志中的量化kernel；在同等质量下跑端到端评测。',VLLM if backend=='vllm' else SGLANG))
    return copy.deepcopy({'backend':backend,'method':method,'inference':inference,
      'compression':COMPRESSION[method],
      'relationship':'vLLM / SGLang / MLX 是本项目互选的执行引擎，不是依次串起来提速。量化是权重表示，可与支持该格式的引擎搭配；不保证所有导出格式都跨引擎兼容。',
      'not_enabled':'本配方未接入MTP/投机解码、KV量化、PD分离、专家并行及CPU卸载。模型包含MTP权重也不代表服务已经使用它。',
      'measurement':'当前没有本模型实测提升。固定模型、提示集、token预算、采样方式、硬件和并发；冷/热缓存分组；用同一质量集对比。网页探针报告首个SSE增量，不保证就是首个可见token。'})
