"""Persistent selection metadata; execution always re-inspects local artifacts."""
import json
import threading
from pathlib import Path
from .downloads import atomic_json


class ModelLibrary:
    def __init__(self,state):
        self.path=state/'model-library.json'
        self.lock=threading.RLock()
        self.records=json.loads(self.path.read_text()) if self.path.exists() else {}

    def local(self,models):
        with self.lock:
            for m in models:
                if not m.get('fingerprint'):continue
                key='local:'+m['path']
                self.records[key]={'key':key,'kind':'local',**{k:m.get(k) for k in
                    ('path','name','model_card_title','model_type','parameter_estimate','config','fingerprint','modality')}}
            self.save()

    def remembered_path(self,path):
        if not isinstance(path,str) or not Path(path).is_absolute():return
        with self.lock:
            key='local:'+path
            if key not in self.records:
                self.records[key]={'key':key,'kind':'local','path':path,'name':Path(path).name,
                    'parameter_estimate':None,'config':{},'note':'Recovered recent path; selection will re-inspect weights'}
                self.save()

    def catalog(self,result):
        with self.lock:
            for m in result.get('models',[]):
                key=result['source']+':'+m['id']
                self.records[key]={'key':key,'kind':'remote','source':result['source'],'repo':m['id'],
                    'name':m['id'],'parameter_estimate':None,'config':{},'pipeline_tag':m.get('pipeline_tag'),
                    **self.records.get(key,{})}
            self.save()

    def remember_metadata(self, model):
        with self.lock:
            self.records[model['key']]=dict(model)
            self.save()

    def save(self):
        while len(self.records)>500:self.records.pop(next(iter(self.records)))
        atomic_json(self.path,self.records)

    def list(self,include_featured=False):
        from .featured_models import featured_models
        with self.lock:
            rows={r['key']:r for r in featured_models()} if include_featured else {}
            for key,r in self.records.items():rows[key]={**rows.get(key,{}),**r}
            return sorted(rows.values(),key=lambda r:-(r.get('parameter_estimate') or (r.get('nominal_parameters_b') or 0)*1e9))

    def get(self,key):
        with self.lock:
            if key not in self.records:
                from .featured_models import featured_models
                match=next((r for r in featured_models() if r['key']==key),None)
                if match:return match
                raise ValueError('Model selection expired; refresh the model list')
            from .featured_models import featured_models
            base=next((r for r in featured_models() if r['key']==key),{})
            return {**base,**self.records[key]}
