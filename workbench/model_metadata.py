"""Best-effort public metadata resolution. No weight downloads or remote code execution."""
import concurrent.futures
import json
import time
from pathlib import Path
from . import hub


def modality(config):
    return 'multimodal' if any(config.get(k) for k in ('vision_config','audio_config')) else 'text'


def resolve(library, selected, refresh=False, source=None, offline=False):
    selected=dict(selected)
    if selected.get('metadata_resolved_at') and not refresh:
        return dict(selected, metadata_origin='cache')
    # Reviewed raw config snapshots work even on the first offline launch.
    for path in (Path(__file__).parent/'knowledge'/'model_snapshots').glob('*.json'):
        snapshot=json.loads(path.read_text())
        if snapshot['repo']==selected['repo'] and not selected.get('config'):
            config=snapshot['config']
            selected.update(config=config.get('text_config') or config,full_config=config,
                            modality=modality(config),
                            model_type=config.get('model_type'),metadata_origin='bundled_snapshot',
                            metadata_source=snapshot['source'],config_url=snapshot['config_url'],
                            metadata_retrieved_at=snapshot['retrieved_at'])
    if selected.get('metadata_origin')=='bundled_snapshot' and not refresh:
        selected.update(metadata_status='partial',metadata_resolved_at=time.time(),metadata_errors=[])
        library.remember_metadata(selected)
        return selected
    if offline:
        return dict(selected,metadata_status='partial' if selected.get('config') else 'unavailable')
    preferred=source or selected['source']
    hub.source(preferred)  # Validate before constructing any public URLs.
    # HF mirror preserves repository IDs. ModelScope IDs must not be guessed from HF IDs.
    sources=[preferred]
    if preferred in ('huggingface','hf-mirror'):
        sources.append('hf-mirror' if preferred=='huggingface' else 'huggingface')
    errors=[]
    repo=hub.repo_id(selected['repo'])
    for name in sources:
        base=hub.source(name)
        config_url=(base+'/api/v1/models/'+repo+'/repo?Revision=master&FilePath=config.json'
                    if name=='modelscope' else base+'/'+repo+'/resolve/main/config.json')
        info_url=base+('/api/v1/models/' if name=='modelscope' else '/api/models/')+repo
        def read(url):
            try:
                value=hub.get_json(url,timeout=8)
                if not isinstance(value,dict):raise ValueError('Expected metadata object')
                return value,None
            except Exception as exc:
                return {},hub.public_error(exc)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(read,[info_url,config_url]))
        (info,info_error),(config,config_error)=results
        for url,error in [(info_url,info_error),(config_url,config_error)]:
            if error:errors.append({'source':name,'url':url,'error':error})
        if not config.get('model_type') and not config.get('text_config'):
            if not config_error:errors.append({'source':name,'error':'config.json lacks model architecture'})
            continue
        # Keep metadata from one source together; never mix a stale count with a new config.
        total=(info.get('safetensors') or {}).get('total') if name!='modelscope' else None
        selected.update(config=config.get('text_config') or config,full_config=config,
                        modality=modality(config),
                        model_type=config.get('model_type'),
                        parameter_estimate=total if type(total) is int and total>0 and not config.get('quantization_config') else None,
                        metadata_source=name,config_url=config_url,metadata_origin='network',
                        metadata_retrieved_at=time.time(),metadata_resolved_at=time.time())
        selected['metadata_status']='ready' if selected['parameter_estimate'] else 'partial'
        selected['metadata_errors']=errors
        library.remember_metadata(selected)
        return selected
    # A refresh failure must not destroy a previously usable snapshot.
    selected.update(metadata_status='partial' if selected.get('config') else 'unavailable',metadata_errors=errors)
    return selected
