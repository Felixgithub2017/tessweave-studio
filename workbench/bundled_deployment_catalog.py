"""Offline <50B deployment catalog shared by Engine and Studio.

Independent implementation, not transplanted vendor code. Recipes describe
external runtimes; candidate means reviewable, never tested or ready-to-launch.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import shlex

DATA = Path(__file__).with_name('knowledge')/'deployment-models-20261015.json'
BUILD = 'https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md'
BACKENDS = {
    'cuda': dict(name='llama.cpp / CUDA',format='GGUF',source=BUILD,flag='GGML_CUDA'),
    'hip': dict(name='llama.cpp / HIP',format='GGUF',source=BUILD,flag='GGML_HIP'),
    'sycl': dict(name='llama.cpp / SYCL',format='GGUF',source=BUILD,flag='GGML_SYCL'),
    'vulkan': dict(name='llama.cpp / Vulkan',format='GGUF',source=BUILD,flag='GGML_VULKAN'),
    'mlx': dict(name='MLX-LM / Metal',format='MLX directory',source='https://github.com/ml-explore/mlx-lm'),
    'rkllm': dict(name='RKLLM Toolkit / Runtime',format='.rkllm',source='https://github.com/airockchip/rknn-llm'),
    'qnn': dict(name='Qualcomm QNN / Genie recipe',format='SoC-specific context + tokenizer',source='https://aihub.qualcomm.com/models'),
    'ascend': dict(name='Ascend SDK recipe required',format='SDK-specific compiled model',source='https://www.hiascend.com/'),
    'mediatek': dict(name='MediaTek vendor/LiteRT recipe required',format='SDK-specific model',source='https://ai.google.dev/edge/litert/android/npu'),
}


def _hardware():
    # Memory presets are editable planning hints, NOT probed available memory.
    rows=[
        ('rtx3060','RTX 3060 12GB',12,'cuda','gpu'),('rtx3090','RTX 3090 24GB',24,'cuda','gpu'),
        ('rtx4080','RTX 4080 16GB',16,'cuda','gpu'),('4090','RTX 4090 24GB',24,'cuda','gpu'),
        ('5090','RTX 5090 32GB',32,'cuda','gpu'),('rtxpro6000','RTX PRO 6000 Blackwell 96GB',96,'cuda','gpu'),
        ('t4','T4 16GB',16,'cuda','gpu'),('a10','A10 24GB',24,'cuda','gpu'),
        ('l4','L4 24GB',24,'cuda','gpu'),('l40s','L40S 48GB',48,'cuda','gpu'),
        ('a6000','RTX A6000 48GB',48,'cuda','gpu'),('a10040','A100 40GB',40,'cuda','gpu'),
        ('a100','A100 80GB',80,'cuda','gpu'),('h100','H100 80GB',80,'cuda','gpu'),
        ('h200','H200 141GB',141,'cuda','gpu'),('b200','B200 180GB',180,'cuda','gpu'),
        ('rx7900xtx','Radeon RX 7900 XTX 24GB',24,'hip','gpu'),
        ('mi300x','Instinct MI300X 192GB',192,'hip','gpu'),
        ('arc_a770','Intel Arc A770 16GB',16,'sycl','gpu'),
        ('vulkan_custom','Other Vulkan GPU — specify memory',None,'vulkan','gpu'),
        ('apple_silicon','Apple Silicon M-series — specify SKU/memory',None,'mlx','unified'),
        ('orin_nano','Jetson Orin Nano — specify board memory',None,'cuda','robot'),
        ('orin_nx','Jetson Orin NX — specify board memory',None,'cuda','robot'),
        ('agx_orin','Jetson AGX Orin — specify board memory',None,'cuda','robot'),
        ('jetson_thor','Jetson Thor — verify JetPack/backend',None,'cuda','robot'),
        ('rk3588','Rockchip RK3588',None,'rkllm','edge'),
        ('rk3576','Rockchip RK3576',None,'rkllm','edge'),
        ('snapdragon','Snapdragon mobile / AI PC — exact SKU required',None,'qnn','edge'),
        ('qcs6490','Qualcomm QCS6490',None,'qnn','robot'),
        ('sa8295','Qualcomm SA8295P cockpit — BSP/SDK access required',None,'qnn','cockpit'),
        ('sa8155','Qualcomm SA8155P cockpit — recipe unverified',None,'qnn','cockpit'),
        ('atlas200ia2','Ascend Atlas 200I A2 — LLM recipe unverified',None,'ascend','robot'),
        ('dimensity','MediaTek Dimensity — exact SKU/SDK required',None,'mediatek','edge'),
    ]
    return [dict(id=k,name=n,memory_gib=g,backend=b,category=c,
                 memory_basis='editable_capacity_preset_not_probe',source=BACKENDS[b]['source']) for k,n,g,b,c in rows]


@lru_cache(maxsize=1)
def _models():
    rows=json.loads(DATA.read_text())
    references=json.loads(DATA.with_name('deployment-architecture-references.json').read_text())
    for row in rows:
        reference=references.get(row['id'])
        if reference and not row['config']:
            row.update(config=reference['config'],config_source=reference['source'],
                metadata_status='public_architecture_reference',architecture_reference=reference)
    return rows


def catalog():
    return dict(schema='tessweave.deployment-catalog.v1',checked='2026-10-15',
        scope='curated_under_50B_not_exhaustive',models=[
            dict(id=m['id'],nominal_b=m['nominal_b'],revision=m['revision'],
                 metadata_status=m['metadata_status'],source=m['source'],
                 declared_context=(m['config'].get('text_config') or m['config']).get('max_position_embeddings'))
            for m in _models()],hardware=_hardware(),backends=BACKENDS)


def _number(value,name,low,high,integer=False):
    if isinstance(value,bool):raise ValueError(name+' must be numeric')
    try: n=float(value)
    except (ValueError,TypeError):raise ValueError(name+' must be numeric') from None
    if not math.isfinite(n) or not low<=n<=high or (integer and n!=int(n)):
        raise ValueError(name+' out of bounds')
    return int(n) if integer else n


def deployment_plan(model_id,hardware_id,*,bits=4,context=None,concurrency=1,
                    memory_gib=None,devices=1,artifact=None):
    models={m['id']:m for m in _models()};hardware={h['id']:h for h in _hardware()}
    if model_id not in models or hardware_id not in hardware:raise ValueError('Unknown catalog model/device')
    m=models[model_id];h=hardware[hardware_id];backend=BACKENDS[h['backend']]
    bits=_number(bits,'bits',4,16,True)
    if bits not in (4,8,16):raise ValueError('Choose 4, 8 or 16 weight bits')
    concurrency=_number(concurrency,'concurrency',1,256,True)
    devices=_number(devices,'devices',1,8,True)
    if devices>1 and (h['category']!='gpu' or h['backend'] not in ('cuda','hip')):
        raise ValueError('Multi-device recipe currently limited to homogeneous CUDA/HIP GPUs')
    cfg=m['config'].get('text_config') or m['config']
    declared=cfg.get('max_position_embeddings')
    if not isinstance(declared,int) or isinstance(declared,bool) or declared<=0:declared=None
    context=_number(context if context is not None else declared or 4096,'context',128,10000000,True)
    memory=_number(memory_gib if memory_gib is not None else h['memory_gib'] or 0,'memory_gib',0,4096)
    if artifact is not None and (not isinstance(artifact,str) or not Path(artifact).is_absolute()):
        raise ValueError('Artifact must be an absolute local path')
    nominal=m['nominal_b']
    params=m.get('tensor_count') if not m['config'].get('quantization_config') else None
    params=params or nominal*1e9
    if nominal>=50 or params>=50e9:raise ValueError('Outside current <50B total-parameter scope')
    weight=params*bits/8/2**30
    # MoE resident weights use TOTAL parameters, never active parameters.
    quant=params*(.0625 if bits==4 else .015625 if bits==8 else 0)/2**30
    layers=cfg.get('num_hidden_layers');hidden=cfg.get('hidden_size')
    heads=cfg.get('num_attention_heads');kvheads=cfg.get('num_key_value_heads') or heads
    head=cfg.get('head_dim') or (hidden//heads if hidden and heads and hidden%heads==0 else None)
    kv=None;kv_basis='unknown';warnings=[]
    if all(isinstance(x,int) and x>0 for x in (layers,hidden,heads,kvheads,head)):
        if cfg.get('kv_lora_rank'):
            kv_basis='MLA requires backend cache layout'
        else:
            # Conservative mixed/sliding estimate retains full history for ALL
            # layers at maximum declared widths, not assumed backend reclamation.
            mixed=bool(cfg.get('layer_types') or cfg.get('sliding_window'))
            head=max(head,cfg.get('global_head_dim') or head)
            kvheads=max(kvheads,cfg.get('num_global_key_value_heads') or kvheads)
            kv=2*2*layers*kvheads*head*math.ceil(context/16)*16*concurrency/2**30
            kv_basis='full_history_upper_bound' if mixed else 'standard_full_attention'
    activation=max(1,12*concurrency*context*hidden/2**30) if hidden else None
    runtime=2*devices
    system=2 if h['category']!='gpu' else 0
    usable=max(0,memory*.8-system)*devices if memory else None
    total=weight+quant+kv+activation+runtime if kv is not None and activation is not None else None
    fits=total<=usable if total is not None and usable is not None else None
    if kv_basis=='full_history_upper_bound':warnings.append('混合/滑窗KV按完整历史保守估算；并非后端实际回收策略。')
    if m['config'].get('vision_config') or m['config'].get('audio_config'):
        warnings.append('本方案仅文本请求；未计图像/音频编码激活和展开token，不适用于多模态请求。')
    warnings.extend(['量化位宽仅容量假设，不证明已有该格式产物；4/8位元数据是代理预算。',
                     '激活/运行时为代理预算；加载临时副本、内核workspace和系统负载需实测。',
                     '厂商标称容量为可编辑预设，必须填写系统实际可用GiB。',
                     '显存容纳不等于芯片/模型/运行时兼容，也不代表吞吐、延迟或质量。'])
    if devices>1:warnings.append('多卡仅总容量及均分代理；层分配、主卡输出层、KV落点和互联可能造成不均衡。')
    vendor=h['backend'] in ('rkllm','qnn','ascend','mediatek')
    state=('vendor_recipe_required' if vendor else
           'metadata_required' if kv is None or declared is None else
           'context_exceeded' if context>declared else
           'memory_required' if not memory else 'over_budget' if not fits else 'candidate')
    default_artifact='/models/'+model_id.split('/')[-1]+('.gguf' if 'GGUF'==backend['format'] else '')
    artifact=artifact or default_artifact
    argv=None
    if state=='candidate':
        if h['backend']=='mlx':
            argv=['python','-m','mlx_lm.server','--model',artifact,'--host','127.0.0.1','--port','8080']
            warnings.append('MLX模板不强制上下文/并发；须在所安装服务中另外设置准入。')
        else:
            argv=['llama-server','--model',artifact,'--host','127.0.0.1','--port','8080',
                  '--ctx-size',str(context*concurrency),'--parallel',str(concurrency),'--n-gpu-layers','99']
            if devices>1:argv+=['--split-mode','layer','--tensor-split',','.join(['1']*devices)]
    steps=[
        '核对模型许可证、固定revision、tokenizer和chat_template；不自动下载权重。',
        '准备已有本地'+backend['format']+'产物，记录转换工具commit、量化配方与sha256。',
        '核对精确芯片SKU、OS/BSP、驱动、运行时版本；查官方兼容列表。',
        '先4K上下文/单路小输出验证，再逐级升至模型声明长度与目标并发；超预算不执行。',
        '比较参考输出/任务质量，测加载、prefill、decode内存、TTFT与持续功耗温度。',
        '仅验证通过后接业务服务；机器人/座舱不接安全关键控制回路。']
    if vendor:steps.insert(3,'需厂商编译/校准配方和设备侧SDK桥接；目前未提供可执行转换/启动命令。')
    elif backend.get('flag'):steps.insert(3,'在匹配工具链中按官方build文档启用 '+backend['flag']+'，检查实际GPU卸载日志。')
    return dict(schema='tessweave.deployment-plan.v1',model=model_id,model_revision=m['revision'],
        model_source=m.get('config_source',m['source']),hardware=h,backend=backend,
        bits=bits,context=context,declared_context=declared,concurrency=concurrency,devices=devices,
        memory_gib=memory or None,state=state,execution_owner='external_backend',
        device_verified=False,quality_verified=False,backend_model_compatibility_verified=False,artifact=artifact,launch_argv=argv,
        budget=dict(weight_gib=weight,quant_metadata_gib=quant,kv_gib=kv,kv_basis=kv_basis,
            activation_proxy_gib=activation,runtime_proxy_gib=runtime,total_gib=total,
            usable_total_gib=usable,fits_budget=fits,per_device_average_gib=total/devices if total is not None else None),
        steps=steps,warnings=warnings,tutorial_id=model_id.replace('/','--'))


def matrix():
    return [dict(model=m['id'],hardware=h['id'],bits=b,
                 **{k:p[k] for k in ('state','context','declared_context','device_verified','budget')})
            for m in _models() for h in _hardware() for b in (4,8,16)
            for p in [deployment_plan(m['id'],h['id'],bits=b)]]


def tutorial(model_id):
    m=next((x for x in _models() if x['id']==model_id),None)
    if m is None:raise ValueError('Unknown model')
    default=deployment_plan(model_id,'h100')
    fence=chr(96)*3
    lines=['# '+model_id+' 部署与逐项验收',
        '', '状态：**待实机验证**。教程/条件预算，不是兼容认证；总参数低于50B。',
        '',f"- 固定revision：{m['revision'] or '未取得，需有权限后补齐'}",
        f"- 官方配置：{m.get('config_source',m['source'])}",
        f"- 标称总参数：{m['nominal_b']}B；MoE不能以激活参数量替代权重驻留量。",
        f"- 声明上下文：{default['declared_context'] or '未知'} tokens；含输入+输出，不额外乘RoPE。",
        '', '## 1. 准备与产物', '',
        '确认模型使用许可并准备已有权重。不要自动下载或覆盖原权重。记录原模型revision、tokenizer/chat_template、转换工具commit、量化格式/校准集以及产物sha256。',
        'GGUF路线按已锁定llama.cpp版本的convert_hf_to_gguf.py --help执行转换；量化使用其llama-quantize。MLX路线使用已锁定mlx_lm.convert --help。不要把GGUF、MLX、RKLLM、QNN二进制互换。',
        '', '## 2. 生成硬件方案', '',
        '以下只输出JSON，不启动服务。先4K单路，再使用配置默认长上下文；共享内存平台必须给实际可用容量。',
        '', fence+'sh',
        f"python -m flow_engine.deployment_catalog plan --model {shlex.quote(model_id)} --hardware h100 --bits 4 --context 4096",
        f"python -m flow_engine.deployment_catalog plan --model {shlex.quote(model_id)} --hardware apple_silicon --memory-gib 48 --bits 4 --context 4096",
        f"python -m flow_engine.deployment_catalog plan --model {shlex.quote(model_id)} --hardware rk3588 --memory-gib 8 --bits 4 --context 4096",
        fence, '', '仅candidate返回供审阅的launch_argv；绝不自动执行。替换本地artifact路径，确认实际文件/服务版本和预算后再手工启动。NPU/座舱需厂商配方，不会生成虚构命令。',
        '', '## 3. 逐硬件验收表', '',
        '表内按4K、单路、4位估算；共享内存设备未指定板卡容量。完整默认上下文×16/8/4位矩阵由matrix命令提供。',
        '', '| 硬件 | 外部运行时 | 状态 | 总需求GiB（代理） |','|---|---|---|---|']
    for h in _hardware():
        p=deployment_plan(model_id,h['id'],context=4096)
        total=p['budget']['total_gib']
        lines.append(f"| {h['name']} | {p['backend']['name']} | {p['state']} | {round(total,2) if total is not None else '未知'} |")
    lines+=['','## 4. 上下文、并发、质量与服务测试','',
        '每一设备分别测试：加载；4K单路；8K/32K/声明上限（仅在预算内）；并发1/2/4。记录真实token数，不用字符数替代。若OOM先收缩配置，不把CPU回退算GPU成功。',
        '固定任务集比较原始精度与量化的质量；验收输出完整性、重复稳定性、取消/超时回收。记录TTFT、逐token延迟、加载/prefill/decode峰值、功耗和热降频。host总耗时不能当decode速度。',
        '通过已启动的标准文本服务可使用Engine edge.generate契约；厂商SDK桥接尚需实现。不得在机器人运动/座舱安全控制链直接接入未经验证模型。',
        '', '## 5. 验证记录模板','',
        fence+'json',json.dumps(dict(model=model_id,model_revision=m['revision'],artifact_sha256=None,
            hardware_sku=None,os_bsp=None,driver_version=None,runtime_commit=None,quantization=None,
            context_tokens=None,concurrency=None,quality_passed=False,peak_memory_gib=None,
            ttft_ms=None,decode_tokens_per_second=None,power_w=None,thermal_soak_minutes=None,
            raw_log_path=None,device_verified=False),ensure_ascii=False,indent=2),fence,
        '', '每个模型/芯片/量化/运行时组合独立记账，空字段不能填成0或通过。',
        '', '## 原理与限制','']
    lines+=['- '+x for x in default['warnings']]
    lines+=['','权重预算按完整参数×位宽/8；BF16 KV按2(K/V)×2字节×层数×KV头×头宽×页对齐上下文×并发。滑窗/混合结构使用完整历史保守上界，不是假称已有精确适配。多模态模型本轮只接受文本需求。',
        '', '官方工具链参考：']+['- '+v['name']+'：'+v['source'] for v in BACKENDS.values()]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline curated <50B deployment recipes; no execution')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list');sub.add_parser('matrix')
    doc=sub.add_parser('tutorial');doc.add_argument('--model',required=True)
    p=sub.add_parser('plan');p.add_argument('--model',required=True);p.add_argument('--hardware',required=True)
    for flag,typ,default in [('bits',int,4),('context',int,None),('concurrency',int,1),('memory-gib',float,None),('devices',int,1)]:
        p.add_argument('--'+flag,type=typ,default=default)
    p.add_argument('--artifact')
    a=parser.parse_args(argv)
    try:
        if a.command=='tutorial':print(tutorial(a.model),end='');return
        result=catalog() if a.command=='list' else matrix() if a.command=='matrix' else deployment_plan(
            a.model,a.hardware,bits=a.bits,context=a.context,concurrency=a.concurrency,memory_gib=a.memory_gib,devices=a.devices,artifact=a.artifact)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except ValueError as e:parser.exit(2,str(e)+'\n')


if __name__=='__main__':main()
