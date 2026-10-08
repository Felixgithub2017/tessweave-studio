"""Explicit heuristic curricula, distinct from a checkpoint's training history."""
from .resources import number

REFERENCES=[
    {'title':'Qwen3: 4K general, 4K reasoning, 32K long-context pretraining',
     'url':'https://arxiv.org/html/2505.09388v1#S3.SS2'},
    {'title':'DeepSeek-V3: 4K to 32K to 128K; extension batch 1920 to 480',
     'url':'https://arxiv.org/html/2412.19437v2'},
]


def length_curriculum(body, model=None):
    cfg=(model or {}).get('config') or {}
    cfg=cfg.get('text_config') or cfg
    limit=cfg.get('max_position_embeddings')
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<128:limit=None
    phase=body.get('training_phase','cpt')
    from .knowledge_base import planning_evidence
    evidence=planning_evidence(body,model)
    goal=number(body,'curriculum_target_seq',32768,128,1000000,True)
    total=number(body,'train_tokens_b',0,0,1e18)*1e9
    cap=min(goal,limit) if limit else goal
    out={'basis':'engineering starting point, not official training history or measured optimum',
         'phase':phase,'declared_context_limit':limit,'official_training_length':None,
         'requested_target':goal,'effective_target':cap,'stages':[], 'references':REFERENCES,
         'warnings':['Measure tokenized sample lengths and truncation before accepting this curriculum.',
                     'Changing length requires re-tuning microbatch, accumulation, recomputation and supported parallelism.',
                     'No automatic RoPE modification. Declared context support is not training recipe or backend certification.',
                     'Token shares below are engineering defaults, not ratios from the cited papers.']}
    if limit and goal>limit:out['warnings'].append('Target capped at config limit; extension beyond it requires a separately validated recipe.')
    out['published_reference']=evidence['published_reference']
    out['knowledge_version']=evidence['version']
    if not limit:out['warnings'].append('Context limit unknown: validate every proposed length against the model and backend.')
    if phase not in ('pretrain','cpt','sft'):
        out['status']='role_plan_required'
        out['warnings'].append('DPO counts both chosen/rejected sequences. PPO/GRPO need separate prompt/response caps, group size, actor/reference/critic and rollout KV budgets; no single-sequence curriculum is issued.')
        return out
    # Shares apply to this job only; SFT/CPT do not replay historical pretraining.
    if phase=='pretrain':
        spec=[('general','General acquisition','通用预训练',4096,80),
              ('quality','High-quality consolidation','高质量中期训练',4096,15)]
        tail_share=5
    elif phase=='cpt':
        spec=[('adapt','Domain adaptation','领域继续训练',8192,85)]
        tail_share=15
    else:
        spec=[('instruction','Instruction/task learning','指令／任务微调',8192,80)]
        tail_share=20
    # Longer stages are opt-in through the target, not inferred from model size.
    lengths=[x for x in (16384,32768,65536,131072,262144,524288) if spec[-1][3]<x<cap]+[cap]
    lengths=sorted(set(max(min(spec[-1][3],cap),x) for x in lengths))
    for n,l in enumerate(lengths):
        spec.append((f'long{n}','Long-context task validation','长上下文任务适配',l,tail_share/len(lengths)))
    for key,en,zh,seq,share in spec:
        seq=min(seq,cap)
        out['stages'].append({'id':key,'name_en':en,'name_zh':zh,'seq':seq,'token_share_percent':share,
            'tokens':round(total*share/100) if total else None,
            'advance_gate_en':'Stable loss/gradients; short-task retention; target-length task scores; measured peak VRAM and throughput; checkpoint resume passes.',
            'advance_gate_zh':'损失／梯度稳定、短任务能力保持、目标长度任务评测达标；实测峰值显存与吞吐，验证断点恢复。',
            'parameter_patch':{'seq':seq,'train_tokens_b':total*share/100/1e9 if total else 0}})
        if any(r['id']=='routing-gates' for r in evidence['rules']):
            out['stages'][-1]['advance_gate_en']+=' Measure expert max/mean load, dropped tokens and all-to-all at this sequence length.'
            out['stages'][-1]['advance_gate_zh']+=' 实测当前长度的专家最大／平均负载、token 丢弃率与 all-to-all 耗时。'
    out['status']='heuristic'
    return out
