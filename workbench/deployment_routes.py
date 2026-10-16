"""Separate deployability evidence from our capacity-adapter implementation."""
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def recipes():
    return json.loads((Path(__file__).parent/'knowledge/deepseek-recipes-20261015.json').read_text())


def route(body,model,support):
    model=model or {};repo=model.get('repo');cfg=model.get('config') or {}
    available=[dict(r) for r in recipes() if r['repo']==repo]
    hardware=body.get('gpu','h100')
    # A recipe is tied to its checkpoint, precision and hardware. It never
    # inherits the current analytical bit-width/context/concurrency inputs.
    for r in available:
        r['matches_selected_hardware']=r['hardware']==hardware and body.get('deployment_target','gpu')=='gpu'
        r['matches_requested_workload']=False
        if repo=='deepseek-ai/DeepSeek-V4.1-Flash':
            r['runtime']='SGLang preview: lmsysorg/sglang:dev-dsv41-mi35x' if r['hardware']=='mi350x' else 'SGLang preview: lmsysorg/sglang:dev-dsv41'
        else:r['runtime']='SGLang：使用固定配方指定的对应镜像/依赖；不可默认安装通用latest'
    available.sort(key=lambda r:(not r['matches_selected_hardware'],not r['upstream_verified'],r['gpus']))
    status='analytic_candidate' if support['analytic_supported'] else 'upstream_recipe' if available or support['recipes'] else 'needs_validation'
    return dict(status=status,repo=repo,recipes=available,execution_verified=False,
        label={'analytic_candidate':'容量候选 + 部署验证步骤','upstream_recipe':'上游部署配方 · 非当前输入的容量认证','needs_validation':'部署调研方案 · 尚无已核实容量或启动配方'}[status],
        warnings=['上游卡数是配方配置，不是最低卡数；max-running-requests不是给定上下文的显存并发或SLO保证。',
                  'FP4/FP8混合检查点不等于统一4/8/16位假设；必须采用配方对应产物和引擎版本，不能跨硬件套用。',
                  'V4.1 Engram、视觉编码器与草稿模块不能漏计；主存卸载会改变主存/PCIe和时延预算。'] if available else [],
        steps=['固定当前模型仓库、revision、许可证、tokenizer及实际权重格式；不自动下载或执行。',
               '选择精确匹配的外部后端/硬件配方；无精确配方时先确认架构支持，不能由参数量推断兼容。',
               '先以短上下文单请求验证加载、输出和逐卡峰值，再逐级测试模型声明上下文。',
               '按业务长短请求分布增加并发，测prefill/decode/Graph峰值、TTFT/ITL与错误率；再确定准入上限。',
               '只有验证通过才开放业务；缺少证据的数字保持未知，不把规划当自动部署。'])
