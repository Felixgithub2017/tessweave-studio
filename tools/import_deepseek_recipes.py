"""Extract declarative evidence without executing upstream JavaScript.

Only quoted match fields, flag/env arrays and literal booleans are accepted.
This is a source-data transcription with fixed commit provenance, not a port
of SGLang code. Recipe commands remain documentation, never auto-executed.
"""
import json
import re
import subprocess
from pathlib import Path
import sys

root=Path(sys.argv[1])
revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
rows=[]
for filename in ('deepseek-v4.jsx','deepseek-v4_1.jsx'):
    relative='docs/src/snippets/configs/deepseek-ai/'+filename
    source=(root/relative).read_text()
    names=source.split('modelNames: {',1)[1].split('},',1)[0]
    mappings=dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"',names))
    default=re.search(r'default:\s*"([^"]+)"',names)
    cells=source.split('cells: [',1)[1]
    starts=list(re.finditer(r'match:\s*\{([^}]+)\}',cells))
    for i,start in enumerate(starts):
        match=dict(re.findall(r'(\w+):\s*"([^"]+)"',start.group(1)))
        text=cells[start.end():starts[i+1].start() if i+1<len(starts) else len(cells)]
        def strings(key):
            found=re.search(key+r':\s*\[([^\]]*)\]',text)
            return [json.loads(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"',found.group(1))] if found else []
        flags,env=strings('flags'),strings('env')
        tp=next((int(f.split()[-1]) for f in flags if re.fullmatch(r'--tp \d+',f)),None)
        if not tp:continue
        key='|'.join(match.get(k,'') for k in ('variant','quant'))
        repo=mappings.get(match.get('hw','')+'|'+key) or mappings.get(key) or (default.group(1) if default else None)
        if not repo:continue
        nodes=int(match.get('nodes','single').replace('multi-','')) if match.get('nodes','single')!='single' else 1
        flag_nodes=next((int(f.split()[-1]) for f in flags if re.fullmatch(r'--nnodes \d+',f)),None)
        if flag_nodes:nodes=flag_nodes
        rows.append(dict(repo=repo,hardware=match.get('hw'),strategy=match.get('strategy'),
            precision=match.get('quant','checkpoint_mixed_fp8_fp4'),gpus=tp,nodes=nodes,
            flags=flags,env=env,upstream_verified=bool(re.search(r'verified:\s*true',text)),
            locally_verified=False,executable=False,source_commit=revision,
            source='https://github.com/sgl-project/sglang/blob/'+revision+'/'+relative,
            checked='2026-10-15'))
target=Path(__file__).resolve().parents[1]/'workbench/knowledge/deepseek-recipes-20261015.json'
target.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(len(rows),'source recipe cells')
