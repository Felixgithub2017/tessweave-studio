"""Reproducible analytical scenarios, NOT official recipes or GPU benchmarks.

From project root: PYTHONPATH=. python3 -B examples/training_design_cases.py
"""
import json
from workbench.resources import plan_resources

CASES = [
    ('dense-0.5B',.5,24,1024,2816,16,4096,{}),
    ('dense-7B',7,32,4096,11008,32,4096,{}),
    ('dense-32B',32,64,5120,27648,40,4096,{}),
    ('dense-70B',70,80,8192,28672,64,4096,{}),
    ('dense-405B',405,128,16384,53248,128,4096,{}),
    ('dense-7B-long',7,32,4096,11008,32,32768,{}),
    ('moe-40B',40,32,4096,14336,32,4096,
     {'expert_count':8,'expert_topk':2,'expert_parameter_fraction':.9}),
    ('moe-320B',320,64,6144,16384,48,4096,
     {'expert_count':128,'expert_topk':4,'expert_parameter_fraction':.95}),
]

if __name__=='__main__':
    for label,p,L,d,f,h,s,extra in CASES:
        inputs=dict(parameters_b=p,layers=L,hidden=d,ffn=f,heads=h,vocab=128256,
            seq=s,node_fabric='nvlink',max_search_gpus=256,gpu='h100',
            train_tokens_b=1,global_batch_tokens=1048576,**extra)
        result=plan_resources(inputs)['training_design']
        candidate=result['recommendations'].get('minimum_capacity')
        print(json.dumps({'scenario':label,'inputs':inputs,'blockers':result['blockers'],
            'minimum':{k:candidate[k] for k in ('gpus','tp','pp','dp','cp','ep','state_strategy','checkpoint','gradient_accumulation','peak_gib','steps')} if candidate else None},ensure_ascii=False))
