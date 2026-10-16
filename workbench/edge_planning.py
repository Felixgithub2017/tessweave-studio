"""Single-device edge estimates. Runtime pointers are not compatibility approvals."""
import math

CHECKED = '2026-10-14'
NVIDIA = 'https://developer.nvidia.com/embedded/jetson-modules'
ROCKCHIP = 'https://github.com/airockchip/rknn-llm'
QUALCOMM = 'https://dev.aihub.qualcomm.com/docs/hub/faq.html'
ASCEND = 'https://www.hiascend.com/doc_center/source/zh/Atlas%20200I%20A2/23.0.0/Har/hardwareguide/Atlas%20200I%20A2%20%E5%8A%A0%E9%80%9F%E6%A8%A1%E5%9D%97%20%E7%A1%AC%E4%BB%B6%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97.pdf'
EDGE_DEVICES = {
    'orin_nano': dict(name='Jetson Orin Nano', runtime='JetPack / CUDA；核对具体推理引擎、量化内核和 JetPack 版本', source=NVIDIA),
    'orin_nx': dict(name='Jetson Orin NX', runtime='JetPack / CUDA；机器人视觉编码器与语言模型须一起计入内存', source=NVIDIA),
    'rk3588': dict(name='Rockchip RK3588', runtime='RKLLM-Toolkit → RKLLM Runtime / RKNPU；转换格式与驱动必须匹配', source=ROCKCHIP),
    'rk3576': dict(name='Rockchip RK3576', runtime='RKLLM-Toolkit → RKLLM Runtime / RKNPU；核对具体模型与算子支持', source=ROCKCHIP),
    'snapdragon': dict(name='Qualcomm Snapdragon（手机 / AI PC，需明确 SKU）', runtime='AI Hub / QNN；按实际芯片、OS 与量化模型配方核验，不跨芯片继承支持', source=QUALCOMM),
    'qcs6490': dict(name='Qualcomm QCS6490（端侧 / 机器人）', runtime='AI Hub / QNN；平台被登记不代表任意 LLM 可运行，核对编译及内存限制', source=QUALCOMM),
    'atlas200ia2': dict(name='Ascend Atlas 200I A2', runtime='昇腾软件栈；硬件指南不等于 LLM / INT4 支持，需专用模型转换与算子验证', source=ASCEND),
    'apple_silicon': dict(name='Apple Silicon（统一内存，需明确 M 系列）', runtime='MLX-LM / Metal；MLX 量化格式需转换核验，不视为 Apple Neural Engine 路径', source='https://github.com/ml-explore/mlx-lm'),
}


from .engine_deployment import engine as _catalog_engine
try:
    _catalog=_catalog_engine().catalog()
    for _h in _catalog['hardware']:
        if _h['category']!='gpu':
            EDGE_DEVICES.setdefault(_h['id'],dict(name=_h['name'],runtime=_catalog['backends'][_h['backend']]['name']+'；模型/SDK兼容待验证',source=_h['source']))
except ValueError:
    pass


def plan_edge(body, model=None):
    from .resources import number
    from .deployment_recipes import assessment
    key=body['deployment_target']
    if key not in EDGE_DEVICES:raise ValueError('未知端侧设备')
    bits=number(body,'bits',16,4,16,True)
    if bits not in (4,8,16):raise ValueError('权重精度必须为4/8/16位')
    model=model or {};cfg=model.get('config') or {}
    p=model.get('parameter_estimate')
    if p is None:p=number(body,'parameters_b',7,.000001,100000)*1e9
    if not isinstance(p,(int,float)) or isinstance(p,bool) or not math.isfinite(p) or p<=0:raise ValueError('参数量无效')
    seq=number(body,'seq',4096,128,10000000,True)
    batch=number(body,'batch',1,1,65536,True)
    memory=number(body,'edge_memory_gib',0,0,4096)
    system=number(body,'edge_system_gib',2,0,4096)
    runtime=number(body,'edge_runtime_gib',1,0,4096)
    reserve=number(body,'reserve_percent',20,5,50)/100
    dims=[number({'x':cfg.get(k) or body.get(b) or 0},'x',0,0,1000000,True)
          for k,b in [('num_hidden_layers','layers'),('hidden_size','hidden'),('num_attention_heads','heads'),('num_key_value_heads','kv_heads')]]
    layers,hidden,heads,kvheads=dims
    head=cfg.get('head_dim') or (hidden//heads if heads and hidden%heads==0 else 0)
    valid=all(dims) and head and assessment(model)['analytic_supported']
    # Weight-only quantization does NOT shrink KV. No TOPS-to-token/s conversion.
    weights=p*bits/8/2**30
    metadata=p*.0625/2**30 if bits==4 else 0
    block=number(body,'kv_block_tokens',16,1,1024,True)
    unit=2*2*layers*kvheads*head/2**30 if valid else None
    kv=batch*math.ceil(seq/block)*block*unit if valid else None
    delta=batch*(math.ceil(seq/block)-math.ceil(2048/block))*block*unit if valid else None
    from .hybrid_cache import hybrid_shape, cache_gib
    hybrid=hybrid_shape(cfg)
    if hybrid and valid:
        kv=cache_gib(cfg,batch,seq,block=block)
        delta=kv-cache_gib(cfg,batch,2048,block=block)
    from .mla_planning import supported as mla_supported, cache_gib as mla_cache
    if mla_supported(cfg) and valid:
        block=max(64,block)
        kv=mla_cache(cfg,batch,seq,block=block)
        delta=kv-mla_cache(cfg,batch,2048,block=block)
    from .layered_cache import layers as cache_layers, cache_gib as layered_cache
    if valid and cache_layers(cfg):
        kv=layered_cache(cfg,batch,seq,block=block)
        delta=kv-layered_cache(cfg,batch,2048,block=block)
    per_request=kv/batch if kv is not None else None
    total=weights+metadata+runtime+kv if kv is not None else None
    available=max(0,memory*(1-reserve)-system) if memory else None
    return dict(edge_plan=dict(device=EDGE_DEVICES[key],checked=CHECKED,weight_bits=bits,
        weights_gib=weights,quant_metadata_gib=metadata,kv_gib=kv,context_delta_gib=delta,
        context_tokens=seq,concurrency=batch,baseline_tokens=2048,kv_precision='BF16 / 2 bytes',
        memory_gib=memory or None,system_gib=system,runtime_gib=runtime,reserve_gib=memory*reserve,
        usable_gib=available,estimated_gib=total,
        memory_concurrency=max(0,math.floor((available-weights-metadata-runtime)/per_request)) if per_request and available is not None else None,
        request_mix_note='端侧尚未实现混合长度容量适配，不使用此输入；当前图仍为等长并发。' if body.get('request_mix') else None,
        fits_budget=total<=available if total is not None and available is not None else None,
        deployment_verified=False,ttft=None,tokens_per_second=None,
        notes=['单设备容量草案；内存填写实际板卡可见 GiB，不是芯片固定容量，也不是独占 NPU 显存。',
               '系统预留默认2 GiB、运行时/激活预算默认1 GiB仅为手填起点；多模态编码器、摄像头及并行应用需追加。',
               '4-bit元数据按0.0625字节/参数粗估；8-bit未计格式元数据。混合精度保留层、解包/编译临时副本可能增加峰值。',
               '缓存按已适配结构计算：标准全注意力或混合KV+FP32循环状态；无前缀共享。容量不证明NPU/Metal后端支持；缺结构显示未知。权重量化不改变此缓存精度假设。',
               '必须核验模型revision、量化格式/分组/校准集、运行时/驱动版本、算子覆盖、上下文限制和任务质量。',
               '容量满足不代表可部署；尚无实机性能认证。需测加载/prefill/decode峰值、TTFT、温度/功耗及长时间稳定性。']))
