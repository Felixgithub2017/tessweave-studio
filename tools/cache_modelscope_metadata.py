"""ModelScope-first public config cache. Strict file allowlist; NEVER weights.

Exact repo identity must be confirmed by the provider's Path/Name. Different
publisher IDs are not guessed. Preserve alternate HF evidence independently;
do not mix a count from one revision with another provider's config.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import urllib.error
from urllib.parse import urlencode

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workbench.featured_models import featured_models
from workbench.engine_deployment import models

FILES={'config.json','generation_config.json','tokenizer_config.json','preprocessor_config.json',
       'processor_config.json','quantization_config.json','model_index.json'}


def fetch(repo):
    out=dict(repo=repo,source='modelscope',checked='2026-10-15',errors=[],files={})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base='https://www.modelscope.cn/api/v1/models/'+repo
    def get(url):
        try:
            with opener.open(url,timeout=10) as response:
                raw=response.read(2_000_001)
                if len(raw)>2_000_000:raise ValueError('metadata exceeds 2MB limit')
                return json.loads(raw),raw
        except (urllib.error.URLError,ValueError,TimeoutError) as e:
            out['errors'].append(dict(url=url,error=type(e).__name__,status=getattr(e,'code',None)))
            return {},b''
    meta,_=get(base);info=meta.get('Data') or {}
    if meta.get('Code')!=200 or info.get('Path','')+'/'+info.get('Name','')!=repo:
        out['status']='no_verified_mapping_or_access'
        return out
    out['provider_identity']=dict(path=info['Path'],name=info['Name'],id=info.get('Id'))
    out['parameter_estimate']=((info.get('ModelInfos') or {}).get('safetensor') or {}).get('model_size')
    listing,_=get(base+'/repo/files?Revision=master&Recursive=true')
    for f in (listing.get('Data') or {}).get('Files',[]):
        name=f.get('Path');revision=f.get('Revision')
        if name not in FILES or not revision:continue
        if f.get('Size',0)>2_000_000:
            out['errors'].append(dict(file=name,error='metadata exceeds 2MB limit'));continue
        url=base+'/repo?'+urlencode(dict(Revision=revision,FilePath=name))
        data,raw=get(url)
        if not raw:continue
        sha=hashlib.sha256(raw).hexdigest()
        if f.get('Sha256') and f['Sha256']!=sha:
            out['errors'].append(dict(file=name,error='sha256 mismatch'));continue
        out['files'][name]=dict(url=url,revision=revision,sha256=sha,data=data)
        if name=='config.json':
            out.update(config=data,config_url=url,revision=revision)
    out['status']='cached' if out.get('config') else 'configuration_missing'
    return out


if __name__=='__main__':
    path=Path(sys.argv[1]);old={}
    if path.exists():old={r['repo']:r for r in json.loads(path.read_text()) if r.get('status')=='cached'}
    repos=sorted({m['repo'] for m in featured_models()}|{m['repo'] for m in models()})
    rows=[]
    def cached(repo):return old.get(repo) or fetch(repo)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(cached,repos):
            rows.append(row)
            if len(rows)%25==0:print(f'{len(rows)}/{len(repos)} ModelScope repositories inspected',flush=True)
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(f'Cached configs: {sum(bool(r.get("config")) for r in rows)}/{len(rows)}',flush=True)
