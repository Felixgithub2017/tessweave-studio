"""Bounded, loopback-only service registry and streamed text conversations.

Cloud engines are reached via an explicit SSH tunnel. No endpoint credentials,
redirects, arbitrary remote URLs, or model code are accepted here.
"""
import json
import math
import threading
import time
import uuid
import re
import base64
import urllib.request
from pathlib import Path
from .benchmark import endpoint, NoRedirect
from .downloads import atomic_json
from .telemetry import sample_host, sample_remote, sample_engine


class Services:
    def __init__(self, state, audit):
        self.path=Path(state)/'services.json'
        self.audit=audit
        self.lock=threading.RLock()
        self.records=json.loads(self.path.read_text()) if self.path.exists() else {}
        self.chats={}
        self.recording_dir=Path(state)/'chat-recordings'
        self.recording_dir.mkdir(exist_ok=True)

    def recordings(self):
        return [{'id':p.stem,'bytes':p.stat().st_size,'updated_at':p.stat().st_mtime}
                for p in sorted(self.recording_dir.glob('*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)[:100]]

    def replay(self,jid):
        if not re.fullmatch(r'[a-f0-9]{32}',str(jid)):raise ValueError('Invalid recording ID')
        p=self.recording_dir/(jid+'.jsonl')
        if not p.is_file() or p.stat().st_size>8*1024*1024:raise ValueError('Recording missing or too large')
        with self.lock:
            events=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return {'id':jid,'events':events,'log_path':str(p)}

    def register(self, body):
        url=endpoint(body['url'])
        alias=str(body.get('telemetry_ssh_alias') or '').strip()
        if alias and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',alias):raise ValueError('Invalid SSH alias')
        model=str(body.get('model','workbench')).strip()
        if not model or len(model)>200:
            raise ValueError('Model identifier must contain 1–200 characters')
        name=str(body.get('name') or model)[:120]
        with self.lock:
            same=next((r for r in self.records.values() if r['url']==url and r['model']==model),None)
            jid=same['id'] if same else uuid.uuid4().hex
            if not same and len(self.records)>=100:
                raise ValueError('Service registry limit reached (100)')
            r={'id':jid,'name':name,'url':url,'model':model,'status':'unchecked','checked_at':None,
               'telemetry_ssh_alias':alias,
               'job_id':body.get('job_id'),'supports_images':body.get('supports_images') is True,
               'created_at':same['created_at'] if same else time.time()}
            self.records[jid]=r;atomic_json(self.path,self.records)
            return dict(r)

    def get(self,jid):
        with self.lock:
            if jid not in self.records:raise ValueError('Unknown service')
            return dict(self.records[jid])

    def list(self):
        with self.lock:return [dict(r) for r in self.records.values()]

    def readiness(self,jid):
        r=self.get(jid)
        with self.lock:
            if r.get('readiness_chat') in self.chats and self.chats[r['readiness_chat']]['status']=='running':
                return {'id':r['readiness_chat'],'status':'running'}
        result=self.start_chat({'service_id':jid,'messages':[{'role':'user','content':'Reply only OK.'}],
                               'max_tokens':32,'temperature':0,'thinking':'off'})
        with self.lock:
            self.records[jid].update(readiness_chat=result['id'],status='checking-generation')
            chat=self.chats[result['id']]
            if chat['status']!='running':
                self.records[jid]['status']='generation-verified' if chat['status']=='succeeded' and chat['metrics'].get('content_chars') else 'generation-unverified'
            atomic_json(self.path,self.records)
        return result

    def set_status(self,jid,status):
        with self.lock:
            self.records[jid].update(status=status,checked_at=time.time())
            atomic_json(self.path,self.records)

    def validate_content(self,content,role,service):
        if isinstance(content,str) and len(content)<=100000:return
        if role!='user' or not service.get('supports_images') or not isinstance(content,list) or not 1<=len(content)<=4:
            raise ValueError('This service accepts text only, or the content format is invalid')
        for part in content:
            if not isinstance(part,dict):raise ValueError('Invalid content part')
            if part.get('type')=='text' and set(part)=={'type','text'} and isinstance(part['text'],str) and len(part['text'])<=100000:continue
            if part.get('type')!='image_url' or set(part)!={'type','image_url'} or not isinstance(part['image_url'],dict):raise ValueError('Only text and inline PNG/JPEG images are supported')
            url=part['image_url'].get('url','')
            match=re.fullmatch(r'data:image/(png|jpeg);base64,([A-Za-z0-9+/=]+)',url) if isinstance(url,str) else None
            if not match:raise ValueError('Use an inline PNG/JPEG; external image URLs are not accepted')
            raw=base64.b64decode(match[2],validate=True)
            if len(raw)>256*1024 or not (raw.startswith(b'\x89PNG\r\n\x1a\n') if match[1]=='png' else raw.startswith(b'\xff\xd8\xff')):raise ValueError('Invalid image or image exceeds 256 KiB')

    def check(self,jid):
        r=self.get(jid)
        try:
            req=urllib.request.Request(endpoint(r['url'])+'/v1/models')
            with urllib.request.build_opener(NoRedirect()).open(req,timeout=5) as response:
                raw=response.read(1024*1024+1)
            if len(raw)>1024*1024:raise ValueError('Model list exceeds limit')
            models=json.loads(raw).get('data',[])
            ids=[m.get('id') for m in models if isinstance(m,dict)]
            status='api-reachable' if r['model'] in ids else 'model-not-listed'
            result={'status':status,'models':ids[:100],'error':None}
        except Exception as exc:
            result={'status':'unreachable','models':[],'error':str(exc)[:300]}
        with self.lock:
            self.records[jid].update(result,checked_at=time.time())
            atomic_json(self.path,self.records)
        return self.get(jid)

    def profile(self,body):
        service=self.get(body['service_id'])
        action=body.get('action');backend=body.get('backend')
        if action not in ('start','stop') or backend not in ('vllm','sglang'):raise ValueError('Unsupported profiler operation')
        if body.get('confirm')!=service['id']:raise ValueError('Explicit service confirmation required')
        payload={'num_steps':10,'activities':['CPU','GPU']} if backend=='sglang' and action=='start' else {}
        req=urllib.request.Request(endpoint(service['url'])+'/'+action+'_profile',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open(req,timeout=30) as response:
            message=response.read(16384).decode('utf-8',errors='replace')
        self.audit.emit('profiler.control',service=service['id'],action=action,backend=backend,response=message)
        return {'status':'engine-accepted','action':action,'response':message,
                'note':'Engine-wide capture; stop vLLM manually. Files are on the engine host. Timeout does not prove capture stopped.'}

    def start_chat(self,body):
        service=self.get(body['service_id'])
        messages=body.get('messages')
        if not isinstance(messages,list) or not 1<=len(messages)<=100:
            raise ValueError('Provide 1–100 text messages')
        for m in messages:
            if not isinstance(m,dict) or set(m)!={'role','content'} or m['role'] not in ('system','user','assistant'):
                raise ValueError('Only bounded system/user/assistant text messages are supported')
            self.validate_content(m['content'],m['role'],service)
        if len(json.dumps(messages).encode())>512*1024:
            raise ValueError('Conversation exceeds 512 KiB')
        temp=body.get('temperature',.7);max_tokens=body.get('max_tokens',512)
        thinking=body.get('thinking','auto')
        if thinking not in ('auto','on','off'):raise ValueError('thinking must be auto, on or off')
        if isinstance(temp,bool) or not isinstance(temp,(int,float)) or not math.isfinite(temp) or not 0<=temp<=2:
            raise ValueError('temperature must be finite and between 0 and 2')
        if type(max_tokens) is not int or not 1<=max_tokens<=8192:
            raise ValueError('max_tokens must be 1–8192')
        with self.lock:
            if sum(c['status']=='running' for c in self.chats.values())>=4:
                raise ValueError('At most four active conversations are allowed')
            while len(self.chats)>=32:
                old=next((k for k,v in self.chats.items() if v['status']!='running'),None)
                if not old:raise ValueError('Conversation limit reached')
                del self.chats[old]
            jid=uuid.uuid4().hex
            self.chats[jid]={'id':jid,'status':'running','events':[],'stop':threading.Event(),'service_id':service['id'],
                'created_at':time.time(),'metrics':{},'size':0,'telemetry_done':threading.Event()}
        payload={'model':service['model'],'messages':messages,'temperature':temp,'max_tokens':max_tokens,
                 'stream':True,'stream_options':{'include_usage':True}}
        if thinking!='auto':payload['chat_template_kwargs']={'enable_thinking':thinking=='on'}
        self._emit(jid,{'type':'request','elapsed_s':0,'model':service['model'],'service_id':service['id'],
                        'request_bytes':len(json.dumps(payload).encode()),'messages':messages,'thinking':thinking})
        threading.Thread(target=self._chat,args=(jid,service,payload),daemon=True).start()
        return {'id':jid,'status':'running','log_path':str(self.recording_dir/(jid+'.jsonl'))}

    def _emit(self,jid,event):
        with self.lock:
            c=self.chats[jid]
            c['size']+=len(json.dumps(event).encode())
            if c['size']>2*1024*1024 or len(c['events'])>=20000:
                raise ValueError('Stream buffer limit reached')
            c['events'].append(event)
            with (self.recording_dir/(jid+'.jsonl')).open('a',encoding='utf-8') as log:
                log.write(json.dumps(event,ensure_ascii=False)+'\n')
        self.audit.emit('playground.event',chat=jid,event=event)

    def _chat(self,jid,service,payload):
        start=time.monotonic();first=None;visible=None;usage={};done=False
        received=0;finish_reason=None;content_chars=0;reasoning_chars=0
        finished=self.chats[jid]['telemetry_done']
        def monitor():
            while not finished.is_set():
                try:
                    try:sample=sample_remote(service['telemetry_ssh_alias']) if service.get('telemetry_ssh_alias') else sample_host()
                    except Exception as exc:sample={'host':service.get('telemetry_ssh_alias'),'scope':'ssh-host, unavailable','gpus':[],'error':str(exc)[:200]}
                    self._emit(jid,{'type':'telemetry','elapsed_s':time.monotonic()-start,'sample':sample})
                    if finished.is_set():break
                    engine=sample_engine(service['url'])
                    if not finished.is_set():self._emit(jid,{'type':'engine_metrics','elapsed_s':time.monotonic()-start,'sample':engine})
                except Exception:break
                if finished.wait(2):break
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        stop=self.chats[jid]['stop']
        status='failed';error=None
        try:
            req=urllib.request.Request(endpoint(service['url'])+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.build_opener(NoRedirect()).open(req,timeout=15) as response:
                while not stop.is_set():
                    if time.monotonic()-start>300:raise ValueError('Conversation exceeded 300 seconds')
                    raw=response.readline(1024*1024+1)
                    received+=len(raw)
                    if len(raw)>1024*1024:raise ValueError('SSE event exceeds limit')
                    if not raw:break
                    if not raw.startswith(b'data:'):continue
                    data=raw[5:].strip()
                    if data==b'[DONE]':done=True;break
                    obj=json.loads(data)
                    if obj.get('error'):raise ValueError('Engine returned a streaming error')
                    if isinstance(obj.get('usage'),dict):usage=obj['usage']
                    for choice in obj.get('choices',[]):
                        if choice.get('finish_reason'):finish_reason=choice['finish_reason']
                        delta=choice.get('delta',{})
                        for kind,text in (('reasoning',delta.get('reasoning_content') or delta.get('reasoning')),('content',delta.get('content'))):
                            if not isinstance(text,str) or not text:continue
                            if kind=='content':content_chars+=len(text)
                            else:reasoning_chars+=len(text)
                            elapsed=time.monotonic()-start
                            if first is None:first=elapsed
                            if kind=='content' and visible is None:visible=elapsed
                            self._emit(jid,{'type':kind,'text':text,'elapsed_s':elapsed,'response_bytes':received})
            status='cancelled' if stop.is_set() else 'succeeded' if done else 'failed'
            if status=='failed':error='Stream ended without [DONE]'
        except Exception as exc:
            status='cancelled' if stop.is_set() else 'failed';error=None if stop.is_set() else str(exc)[:500]
        elapsed=time.monotonic()-start
        finished.set();monitor_thread.join(timeout=5)
        output=usage.get('completion_tokens');inputs=usage.get('prompt_tokens')
        output=output if type(output) is int and output>=0 else None
        inputs=inputs if type(inputs) is int and inputs>=0 else None
        metrics={'duration_s':elapsed,'first_delta_s':first,'first_visible_delta_s':visible,
                 'finish_reason':finish_reason,'content_chars':content_chars,'reasoning_chars':reasoning_chars,
                 'response_bytes':received,'request_bytes':len(json.dumps(payload).encode()),
                 'input_tokens':inputs,'output_tokens':output,
                 'end_to_end_output_tokens_per_s':output/elapsed if output is not None and status=='succeeded' else None,
                 'note':'Token counts come from engine usage. SSE chunks are not tokens; no per-token ITL is inferred.'}
        try:self._emit(jid,{'type':'completed','elapsed_s':elapsed,'status':status,'metrics':metrics,'error':error,'response_bytes':received})
        except ValueError:pass
        with self.lock:self.chats[jid].update(status=status,error=error,metrics=metrics)
        with self.lock:
            record=self.records.get(service['id'],{})
            if record.get('readiness_chat')==jid and record.get('status')=='checking-generation':
                record.update(status='generation-verified' if status=='succeeded' and content_chars else 'generation-unverified',checked_at=time.time(),readiness_error=error)
                atomic_json(self.path,self.records)
        self.audit.emit('playground.completed',chat=jid,status=status,error=error,metrics=metrics)

    def events(self,jid,cursor=0):
        if type(cursor) is not int or cursor<0:raise ValueError('Invalid stream cursor')
        with self.lock:
            if jid not in self.chats:raise ValueError('Conversation expired or unknown')
            c=self.chats[jid];end=min(len(c['events']),cursor+200)
            if cursor>len(c['events']):raise ValueError('Stream cursor out of bounds')
            return {'id':jid,'status':c['status'],'events':c['events'][cursor:end],'cursor':end,
                    'has_more':end<len(c['events']),'metrics':dict(c['metrics']),'error':c.get('error')}

    def stop(self,jid):
        with self.lock:
            if jid not in self.chats:raise ValueError('Conversation expired or unknown')
            self.chats[jid]['stop'].set()
        return {'id':jid,'status':'cancellation-requested','note':'Upstream read may take up to 15 seconds to unblock.'}

    def shutdown(self):
        with self.lock:
            for c in self.chats.values():c['stop'].set()
