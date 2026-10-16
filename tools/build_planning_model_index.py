"""Offline documentation index. No inference, downloads, or guessed shapes.

Every catalog row appears once. Cache examples are TP=PP=1, B=1, S=4096,
not total VRAM or deployment approval. Use the same adapters as the planner.
"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workbench.featured_models import featured_models
from workbench.engine_deployment import models
from workbench.model_metadata import with_bundled_metadata
from workbench.deployment_recipes import coverage
from workbench.layered_cache import layers, token_bytes as layered_bytes
from workbench.mla_planning import supported as is_mla, token_bytes as mla_bytes
from workbench.hybrid_cache import hybrid_shape, coefficients

TITLES={
 'A':'标准 MHA/GQA/MQA', 'B':'标准 KV 加 MoE 权重与临时量',
 'C':'逐层异构 KV', 'D':'MLA 压缩缓存', 'E':'MLA 加 DSA 索引',
 'F':'MLA 加 KDA 循环状态', 'G':'全注意力加 GDN 循环状态',
 'R':'只有独立上游部署配方', 'U':'结构存在但尚未适配', 'M':'配置或关键字段缺失'}


def pending_work(cfg):
    """Calculation work implied by observed config, not a support claim."""
    kind=cfg.get('model_type')
    if not kind:return '未知结构：先获取身份可核对的配置与任务类型'
    if kind in ('nemotron_h','falcon_h1','falcon_mamba') or cfg.get('ssm_cfg') or cfg.get('mamba_d_state'):
        return '状态空间/混合：逐层循环矩阵、卷积历史及注意力KV；核对状态精度'
    if kind in ('deepseek_v4','deepseek_v41_text','glm5_next_text','minimax_text_01','minimax_m1','minimax','step3_text'):
        return '专用压缩/混合路径：核对压缩、索引或状态布局及附加模块，不能沿用普通KV'
    if kind in ('step3p5','baichuan_m1','gemma4_unified_text','ouro'):
        return '特殊逐层/复用路径：核对层表、状态复用及缓存寿命'
    if kind in ('bert','deberta-v2','electra','modernbert','longformer','openai_privacy_filter','clip_text_model','siglip_text_model','owlv2_text_model','vit'):
        return '编码/分类类：前向激活与输出张量；不能按自回归持久KV估算任务峰值'
    if kind in ('t2v','i2v','vace','bagel','audiodit'):
        return '生成管线：按分辨率/帧数/步数计算编码器、去噪器及激活峰值'
    if kind in ('patchtst_fm','tspulse','tinytimemixer'):
        return '时序任务：输入窗口、通道、预测长度及前向工作集'
    if kind in ('whisper','dual_ar','fish_omni_text','fish_qwen3','voxtral_realtime_text','nemotron3_5_asr','vibevoice','wavlm','minimax_music3'):
        return '音频管线：时长/采样率、编码器、解码器/声码器及各阶段峰值'
    if kind in ('internvl_chat','sa2va_chat','ernie4_5_moe_vl','deepseek_vl_v2','unlimited_ocr','phi3_v','hunyuan_vl_text','glm4v_text','glm_ocr_text','GOT','layoutlmv3','table-transformer','pp_doclayout_v3'):
        return '视觉/文档管线：图像尺寸与展开tokens、编码器、解码器及交叉缓存'
    if kind=='t5':return '编码器-解码器：编码工作集、自注意力KV与交叉注意力KV分别计账'
    return '需核对该model_type的头维度、层表、权重与任务峰值；不按名称猜兼容性'


def classify(model):
    cfg=model.get('config') or {};cfg=cfg.get('text_config') or cfg
    status=coverage(model)
    result=dict(repo=model['repo'],model_type=cfg.get('model_type','未知'),status=status['status'],
                missing=status.get('missing',[]),category=None,example=None,
                parameters=model.get('parameter_estimate') or (model.get('nominal_parameters_b') or 0)*1e9 or None)
    if status['status']!='analytic_unverified':
        result['category']={'upstream_recipe':'R','metadata_required':'M','adapter_required':'U'}[status['status']]
        result['required_calculation']=pending_work(cfg)
        return result
    shape=hybrid_shape(cfg);state=0;tail=0;detail=''
    if shape:
        group='G' if shape.get('kind')=='gdn' else 'F'
        c,state=coefficients(cfg)
        detail=f"L={cfg['num_hidden_layers']}; full={len(shape['full_attn_layers'])}; recurrent={len(shape['kda_layers'])}"
    elif is_mla(cfg):
        group='E' if cfg.get('index_head_dim') else 'D';c=mla_bytes(cfg);tail=64
        detail=f"L={cfg['num_hidden_layers']}; latent={cfg['kv_lora_rank']}; RoPE={cfg['qk_rope_head_dim']}"
        if group=='E':detail+=f"; index={cfg['index_head_dim']}"
    elif layers(cfg):
        group='C';c=layered_bytes(cfg)
        counts=Counter((x['sliding'],x['kv_heads'],x['key_dim'],x['value_dim']) for x in layers(cfg))
        detail='; '.join(f"{n}×{'SW' if sw else 'full'}(Hkv={h},dk={dk},dv={dv})" for (sw,h,dk,dv),n in counts.items())
    else:
        moe=any(cfg.get(k) for k in ('num_experts','num_local_experts','n_routed_experts','moe_num_experts'))
        group='B' if moe else 'A'
        h=cfg['num_key_value_heads'];d=cfg.get('head_dim') or cfg['hidden_size']//cfg['num_attention_heads']
        c=4*cfg['num_hidden_layers']*h*d
        detail=f"L={cfg['num_hidden_layers']}; Hkv={h}; dh={d}"
        if cfg.get('sliding_window') or 'sliding_attention' in (cfg.get('layer_types') or []):detail+='; SW全历史上界'
    result.update(category=group,inputs=detail,example=dict(token_bytes=c,state_bytes=state,
        extra_tail_tokens=tail,cache_gib=((4096+tail)*c+state)/2**30))
    return result


def records():
    rows={m['repo']:m for m in featured_models()}
    rows.update({m['repo']:m for m in models()})
    return [classify(with_bundled_metadata(m)) for _,m in sorted(rows.items())]


def cell(value):
    return str(value).replace('|','\\|').replace('\n',' ')


def render(rows):
    counts=Counter(x['category'] for x in rows);states=Counter(x['status'] for x in rows)
    lines=['# 模型目录与部署显存公式对照', '',
      '快照日期：2026-10-16。按当前本地目录和配置分类，逐条对应计算手册；不是全球全部模型清单，也不是实机兼容认证。', '',
      f'共 **{len(rows)} 条模型仓库记录**：容量可解析 {states["analytic_unverified"]}，元数据待补 {states["metadata_required"]}，适配待补 {states["adapter_required"]}，独立上游配方 {states["upstream_recipe"]}。同架构、不同大小或量化版本各占一条；不是 413 种不同架构。', '',
      '[返回计算手册](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories)', '',
      '## 如何读每条记录', '',
      'A—G 是计算路径，不是厂商分类。每条可解析记录列出实际缓存字段与统一小例：**TP=PP=1、B=1、总上下文4096、BF16 KV、16-token页；MLA/DSA至少64-token页并额外预留一尾页**。循环矩阵按当前适配器FP32口径。c为bytes/token，r为bytes/request。缓存GiB不包含权重、激活或运行时，不能据此直接选卡。', '',
      '所有记录还需计算 P×位宽/8、量化元数据、激活与运行时，再按合法TP/PP搜索最忙rank容量。参数B是目录/缓存口径，未逐权重核验；量化记录尤其不能把packed元素数当原参数。未知参数量保持未知。A/B/C滑窗不计驱逐收益，C不计跨层共享/K=V收益。', '',
      '模型类型为文本解码器时只覆盖文本路径；多模态编码、扩散、ASR、embedding或rerank任务不能由解码器缓存例推导整任务容量。R/U/M不展示伪造缓存值，需按手册的补齐路径处理。', '',
      'M 表的“缺项”沿用当前规划器通用结构检查；对于编码器、扩散或音频等任务，这些缺项不表示模型应该拥有自回归KV字段，必须先选择正确任务适配器。“需补计算部分”依据已有model_type/状态字段分流，不把推测当已实现公式。', '',
      '| 类别 | 数量 | 计算或补齐路径 |','| --- | ---: | --- |']
    for key,title in TITLES.items():
        lines.append(f'| [{key} {title}](#category-{key.lower()}) | {counts[key]} | [手册说明](DEPLOYMENT_PLANNING_MATH.zh-CN.md#model-categories) |')
    for group,title in TITLES.items():
        lines+=['',f'<a id="category-{group.lower()}"></a>','',f'## {group} {title}（{counts[group]} 条）','']
        if group in 'ABCDEFG':
            lines+=['| 模型与配置类型 | 参数B¹ | 缓存输入 | c / r 字节 | 4K单请求缓存GiB |',
                    '| --- | ---: | --- | ---: | ---: |']
        else:
            lines+=['| 模型 | 配置类型 | 当前缺项 | 需补的计算部分（非已实现） |','| --- | --- | --- | --- |']
        for x in rows:
            if x['category']!=group:continue
            if x['example']:
                e=x['example'];p=f'{x["parameters"]/1e9:.3f}' if x['parameters'] else '未知'
                lines.append(f'| {cell(x["repo"])} · `{cell(x["model_type"])}` | {p} | {cell(x["inputs"])} | {e["token_bytes"]:,} / {e["state_bytes"]:,} | {e["cache_gib"]:.4f} |')
            else:
                gap='固定上游配方；不能继承当前负载/精度' if group=='R' else '；'.join(x['missing']) or '架构缓存/状态布局待核对'
                lines.append(f'| {cell(x["repo"])} | `{cell(x["model_type"])}` | {cell(gap)} | {cell(x["required_calculation"])} |')
    lines+=['','¹ 参数B只帮助识别目录规模；有效计算须使用核验后的全部驻留参数，不用MoE激活参数代替。', '',
      '## 更新与复核', '',
      '由 `tools/build_planning_model_index.py` 离线生成，同时输出机器可读分类与示例。分类调用当前规划适配器，缓存例不启动模型。修改配置或适配器后应重新生成；历史审计清单与当前生成结果若不同，应先检查快照日期与目录版本。', '']
    return '\n'.join(lines)


if __name__=='__main__':
    rows=records()
    assert len({x['repo'] for x in rows})==len(rows)
    (ROOT/'docs/MODEL_PLANNING_INDEX.zh-CN.md').write_text(render(rows))
    (ROOT/'docs/model-planning-index-20261016.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(total=len(rows),categories=dict(Counter(x['category'] for x in rows))),ensure_ascii=False))
