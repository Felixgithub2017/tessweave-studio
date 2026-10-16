"""Evidence-linked scenario triage, not an execution/price certification."""
from .resources import number

MODELS = [
    dict(id='Qwen/Qwen3-4B',modality='text',parameters_b=4.0,source='https://huggingface.co/Qwen/Qwen3-4B',
         why='小型文本基线候选；先测任务质量、检索与工具调用，再决定是否后训练。'),
    dict(id='Qwen/Qwen2.5-VL-7B-Instruct',modality='vision',parameters_b=None,source='https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct',
         why='图像／文档理解候选；需把视觉编码器、分辨率与图像 token 纳入账本，不能只按名称中的 7B 估算。'),
    dict(id='lerobot/smolvla_base',modality='action',parameters_b=None,source='https://huggingface.co/lerobot/smolvla_base',
         why='视觉—语言—动作策略候选；核对机器人形态、动作维度、控制频率与训练数据匹配。'),
    dict(id='Physical-Intelligence/openpi',modality='action',parameters_b=None,source='https://github.com/Physical-Intelligence/openpi',
         why='动作策略训练／适配路线参考；仓库不是单一权重，须进一步选择具体 checkpoint 与许可证。'),
]


def scenario_plan(body):
    scene=body.get('scene','assistant');target=body.get('target','cuda')
    if scene not in ('assistant','documents','cockpit','robot'):raise ValueError('未知场景')
    if target not in ('cuda','mac','qualcomm','jetson','rockchip','ascend','musa'):raise ValueError('未知设备路线')
    modality=body.get('modality','text')
    if modality not in ('text','vision','action'):raise ValueError('未知模态')
    memory=number(body,'memory_gib',0,0,2048)
    budget=number(body,'budget_cny',0,0,1e9)
    rate=number(body,'hour_price_cny',0,0,1e6)
    fixed=number(body,'fixed_cost_cny',0,0,1e9)
    rows=[]
    for model in MODELS:
        if model['modality']!=modality:continue
        p=model['parameters_b'];weight=None if p is None else p*1e9*2/2**30
        rows.append(dict(model,weight_only_bf16_gib=weight,
            capacity_status='unknown' if not memory or weight is None else 'weights_alone_exceed_budget' if weight>memory else 'weights_only_not_total_fit',
            evidence='official_source_candidate_not_studio_certified'))
    blockers=[]
    if not memory:blockers.append('未输入模型可用内存；统一内存须扣除 OS 和其他进程。')
    if target in ('qualcomm','jetson','rockchip','ascend','musa'):blockers.append('Studio 尚未认证此模型—芯片—运行时组合；当前只输出集成路线，不生成启动命令。')
    if modality=='action':blockers.append('动作策略不可提交聊天训练配方；必须独立审计动作头、归一化、时序窗口及机器人驱动。')
    if scene=='robot' and modality!='action':blockers.append('当前只规划机器人语言／感知模块，不包含动作控制器。')
    if scene=='cockpit':blockers.append('座舱语言助手不等于驾驶控制系统；不允许直接驱动车辆控制，需权限隔离和安全认证。')
    if not budget or not rate:blockers.append('缺少预算或实报价，无法给出可负担机时；也不能从参数量猜训练耗时。')
    route={
        'cuda':'审计 CUDA／驱动与框架 → 模型算子预检 → 训练／量化 → 质量回归 → vLLM/SGLang 等候选压测。',
        'mac':'核对 MLX 模型支持与统一内存 → 匹配权重格式 → 本机实测；不把 CUDA 训练命令复制到 macOS。',
        'qualcomm':'选定具体 SoC／OS → 检查 AI Hub 与 QAIRT/QNN 支持 → 代表性校准数据 → 转换／编译 → 芯片实测。手机、PC、座舱不能互相代替认证。',
        'jetson':'选定 Jetson／JetPack → Arm 与 CUDA/TensorRT 版本匹配 → 算子／模型验证 → 持续负载与温控测试。',
        'rockchip':'选定芯片／BSP → 核对 RKLLM 支持架构 → 转换量化 → 匹配板端 runtime；视觉组件可能使用独立 RKNN 路径。',
        'ascend':'选定 Ascend SKU／CANN／框架 → 专用算子和分布式适配 → 数值验证 → 目标设备压测。',
        'musa':'选定 MTT SKU／MUSA → 核对训练／服务分支 → 数值与算子验证 → 实测容量与吞吐。',
    }[target]
    training='先做零训练基线；知识更新优先评估检索，行为／格式问题再做 SFT；偏好对可用 DPO；有可靠奖励才考虑 GRPO。'
    data='分离训练／验证／测试，去重、版权与隐私审核；文本使用模型模板，视觉数据保存图像及问答对应关系和预处理版本。'
    evaluation='独立任务质量 + 峰值内存 + p50/p95 TTFT/TPOT + 并发吞吐 + 稳态功耗；固定输入输出长度分布。'
    if modality=='action':
        training='优先基于示教轨迹做行为克隆／所选策略的动作目标微调。具体可能是动作回归、flow matching 或其他策略损失；不能统一写成文本交叉熵或直接沿用聊天 GRPO。'
        data='记录同步相机、关节／末端状态、动作、任务、episode 边界与时间戳；校准坐标系、单位、动作范围、频率，按任务和场景划分验证集。'
        evaluation='先离线与仿真，再限速限力的受控实机；测任务成功率、干预率、动作延迟/抖动和安全停止。独立 watchdog、急停、动作边界必须先就绪。'
    return dict(status='advisory_only',scene=scene,target=target,modality=modality,candidates=rows,blockers=blockers,
                training=training,data=data,deployment=route,evaluation=evaluation,
                economics=dict(budget_cny=budget,fixed_cost_cny=fixed,hour_price_cny=rate,
                               affordable_hours=max(0,budget-fixed)/rate if budget and rate else None,
                               formula='max(0, 总预算 − 非机时费用) / 整套实例每小时报价；不是训练耗时或完成保证'),
                missing=['具体模型 revision／许可证','设备 SKU、OS、驱动与运行时','数据规模与质量门槛','上下文／图像分辨率／动作频率','并发、延迟目标、联网与隐私限制','实测吞吐、训练步数与质量验证'],
                sources=['https://aihub.qualcomm.com/','https://github.com/airockchip/rknn-llm','https://github.com/huggingface/lerobot'],
                note='首批人工策划参考，不是全面模型榜单；容量只显示 BF16 权重下界，不是训练／部署总显存。没有自动安装、下载、租机或执行。')
