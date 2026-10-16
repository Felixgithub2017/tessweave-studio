"""Explicit, bounded public metadata audit; no model code or weights executed."""
import concurrent.futures
import json
from pathlib import Path
import sys
import urllib.request
import urllib.error

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workbench.featured_models import featured_models


def read(repo):
    root='https://huggingface.co/'
    row=dict(repo=repo,checked='2026-10-15',errors=[])
    def get(url):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TessWeave-public-planning-audit'}),timeout=12) as response:
                raw=response.read(2_000_001)
                if len(raw)>2_000_000:raise ValueError('metadata exceeds 2MB limit')
                return json.loads(raw)
        except (urllib.error.URLError,ValueError,TimeoutError) as e:
            row['errors'].append(dict(url=url,error=type(e).__name__,status=getattr(e,'code',None)))
            return {}
    info=get(root+'api/models/'+repo)
    row['revision']=info.get('sha')
    row['parameter_estimate']=(info.get('safetensors') or {}).get('total')
    row['pipeline_tag']=info.get('pipeline_tag')
    # Pin successful reads. Never run remote auto_map implementations.
    if row['revision']:
        row['config_url']=root+repo+'/resolve/'+row['revision']+'/config.json'
        row['config']=get(row['config_url'])
    return row


if __name__=='__main__':
    path=Path(sys.argv[1]);rows=[]
    repos=sorted({m['repo'] for m in featured_models()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(read,repos):
            rows.append(row)
            if len(rows)%25==0:print(f'{len(rows)}/{len(repos)} public metadata entries inspected',flush=True)
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(f'Complete: {len(rows)} records; {sum(bool(r.get("config")) for r in rows)} configs',flush=True)
