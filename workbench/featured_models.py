"""Curated entry points, not a live popularity ranking or scanned parameter counts."""
MODELS=[
    ('deepseek-ai/DeepSeek-V4-Pro',1600,'https://arxiv.org/abs/2606.19348v1'),
    ('XiaomiMiMo/MiMo-V2.6-Pro-RL',1020,'https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL'),
    ('zai-org/GLM-5',744,'https://arxiv.org/abs/2602.15763'),
    ('meta-llama/Llama-3.1-405B',405,'https://arxiv.org/abs/2407.21783'),
    ('zai-org/GLM-5.3-Flash-BF16',320,'https://huggingface.co/zai-org/GLM-5.3-Flash-BF16'),
    ('Qwen/Qwen3-235B-A22B',235,'https://arxiv.org/abs/2505.09388v1'),
    ('MiniMaxAI/MiniMax-M2',230,'https://arxiv.org/abs/2605.26494'),
    ('Qwen/Qwen3-32B',32,'https://arxiv.org/abs/2505.09388v1'),
    ('google/gemma-4-31B-it',31,'https://arxiv.org/abs/2607.02770v2'),
    ('Qwen/Qwen3-8B',8,'https://arxiv.org/abs/2505.09388v1'),
    ('Qwen/Qwen3-0.6B',.6,'https://arxiv.org/abs/2505.09388v1'),
]


def featured_models():
    import json
    from pathlib import Path
    path=Path(__file__).parent/'knowledge'/'model-browser.json'
    snapshot=json.loads(path.read_text()) if path.exists() else {'models':[]}
    rows={ 'huggingface:'+m['id']:dict(key='huggingface:'+m['id'],repo=m['id'],name=m['id'],kind='remote',source='huggingface',featured=True,parameter_estimate=None,config={},pipeline_tag=m.get('pipeline_tag'),license=m.get('license'),gated=m.get('gated'),reported_parameters=m.get('parameters'),metadata_status='catalog_only') for m in snapshot['models']}
    for row in [dict(key='huggingface:'+repo,repo=repo,name=repo,kind='remote',source='huggingface',
                 featured=True,nominal_parameters_b=b,parameter_estimate=None,config={},
                 parameter_source=url,reviewed='2026-10-07') for repo,b,url in MODELS]:
        rows[row['key']]={**rows.get(row['key'],{}),**row}
    return list(rows.values())
