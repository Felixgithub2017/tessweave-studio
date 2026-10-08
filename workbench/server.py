"""Loopback-only single-owner control plane. Not an Internet-facing multi-tenant server."""
import hmac
import json
import os
import secrets
import shlex
import shutil
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from . import __version__
from .adapters import recipe
from .benchmark import run_benchmark
from .data import EXAMPLES, validate_dataset, dataset_guide
from .model_library import ModelLibrary
from .model_reference import model_identity
from .hardware import local_probe, remote_probe
from .inspection import inspect_model, discover, browse_directories, scan_directory, create_directory
from .techniques import technique_guide
from .jobs import Jobs
from .downloads import Downloads, atomic_json
from . import hub
from .planning import METHODS, estimate
from .audit import Audit, clean
from .resources import GPUS, plan_resources
from .services import Services
from .profiler import read_trace
from .environments import environment_plan, conda_installations
from .environment_inventory import scan as scan_environments, assess as assess_environments
from .cloud import cloud_plan


class App:
    def __init__(self, state, roots, audit_log=None):
        self.jobs = Jobs(state)
        self.audit = Audit(audit_log or self.jobs.state / "operations.log")
        self.jobs.audit = self.audit
        self.roots = [str(Path(p).expanduser().resolve(strict=True)) for p in roots]
        self.token = secrets.token_urlsafe(32)
        self.downloads = Downloads(state, self.roots)
        self.downloads.audit = self.audit
        self.services = Services(self.jobs.state,self.audit)
        self.evaluation_lock = threading.Lock()
        self.evaluations = {}
        self.hub_manifests = {}
        self.hardware = None
        self.history = []
        self.lock = threading.RLock()
        self.model_library = ModelLibrary(self.jobs.state)
        # Best-effort upgrade: recover only the bounded recent audit window.
        for event in self.audit.tail()['events']:
            if event.get('kind')=='operation.begin' and event.get('operation') in ('/api/scan','/api/resources','/api/data/guide'):
                inputs=event.get('inputs',{})
                self.model_library.remembered_path(inputs.get('model_path') or (inputs.get('path') if event['operation']=='/api/scan' else None))
            if event.get('kind')=='operation.result' and event.get('operation') in ('/api/scan','/api/discover'):
                result=event.get('result',{})
                models=result if isinstance(result,list) else result.get('models',[result]) if isinstance(result,dict) else []
                self.model_library.local(models)
        catalog_cache=self.jobs.state/'catalog.json'
        if catalog_cache.exists():self.model_library.catalog(json.loads(catalog_cache.read_text()))

    def probe(self):
        with self.lock:
            if self.hardware and time.time() - self.hardware["timestamp"] < 3:
                return self.hardware
            self.hardware = local_probe()
            self.history.append(self.hardware)
            self.history = self.history[-120:]
            return self.hardware

    def plan(self, body):
        allowed = {"model_path", "backend_python", "dataset", "task", "backend", "method", "tuner", "seq", "batch", "steps", "grad_acc", "learning_rate", "gpus", "tp", "port", "profile", "rank", "generations"}
        if set(body) - allowed:
            raise ValueError("计划含未识别字段；不接受凭据、任意命令或隐式执行参数")
        model = inspect_model(body["model_path"], self.roots)
        jid = uuid.uuid4().hex
        output = self.jobs.state / jid / "artifacts"
        plan = recipe(model, body, self.probe(), self.roots, output)
        profile = plan["profile"]
        if profile not in ("none", "nsys"):
            raise ValueError("只支持 none/nsys；ncu 全量剖析会强烈扰动，需专门实验")
        if profile == "nsys":
            if not shutil.which("nsys"):
                plan["blockers"].append("未安装 nsys")
            else:
                plan["argv"] = [shutil.which("nsys"), "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--output=" + str(self.jobs.state/jid/"profile")] + plan["argv"]
                plan["warnings"].append("nsys 会扰动运行；不要把剖析结果直接作为公平跑分")
        plan.update(id=jid, model=model, request=body, created_at=time.time(),
                    resource_estimate=estimate(model, body.get("task", "sft"), body.get("seq", 2048), body.get("batch", 1), tuner=body.get("tuner", "lora")))
        if model["modality"] != "text":
            plan["warnings"].append("多模态 media token 预算尚未做processor实测；规划不能只用文本seq")
        plan["support_level"] = "blocked" if plan["blockers"] else "recipe-unverified"
        return self.jobs.create(plan)

    def start(self, body):
        jid = body["id"]
        if body.get("confirm") != jid:
            raise ValueError("必须使用当前计划ID确认执行")
        plan = self.jobs.get(jid)["plan"]
        if plan.get('task') in ('environment','cloud-workspace'):
            return self.jobs.launch(jid,native=body.get('native',False))
        fresh = inspect_model(plan["model"]["path"], self.roots)
        if fresh["fingerprint"] != plan["model"]["fingerprint"]:
            raise ValueError("模型在生成计划后已变化；请重新扫描规划")
        if plan["dataset"]:
            d = plan["dataset"]
            now = validate_dataset(d["path"], d["task"], self.roots)
            if not now["valid"] or now["sha256"] != d["sha256"]:
                raise ValueError("数据在规划后已变化，请重建计划")
        if Path(plan["output"]).exists():
            raise ValueError("输出已存在，拒绝覆盖")
        launched=self.jobs.launch(jid, native=body.get("native", False))
        if plan['task']=='inference':
            self.services.register({'url':'http://127.0.0.1:'+str(plan['request'].get('port',8000)),
                'model':plan['model']['path'] if plan['backend']=='mlx' else 'workbench',
                'supports_images':plan['backend'] in ('vllm','sglang') and plan['model'].get('modality')=='vision-language',
                'name':plan['model']['name']+' / '+plan['backend'],'job_id':jid})
        return launched

    def api(self, path, body, query):
        if path == "/api/audit":
            return self.audit.tail()
        request_id = uuid.uuid4().hex
        entry = {"/api/resources":"workbench.resources.plan_resources", "/api/scan":"workbench.inspection.inspect_model/discover",
                 "/api/hub/manifest":"workbench.hub.manifest", "/api/hub/speed":"workbench.hub.speed_test",
                 "/api/catalog/sync":"workbench.hub.catalog", "/api/plan":"workbench.adapters.recipe",
                 "/api/start":"workbench.jobs.Jobs.launch", "/api/benchmark":"workbench.benchmark.run_benchmark",
                 "/api/remote":"workbench.hardware.remote_probe"}.get(path,"workbench.server.App.dispatch")
        command = shlex.join([sys.executable, "-m", "workbench", "api-call", "--operation", path,
                              "--payload", json.dumps(clean(body), ensure_ascii=False), "--query", json.dumps(clean(query))])
        self.audit.emit("operation.begin", request_id=request_id, operation=path, inputs=body,
                        code=entry, equivalent_cli=command)
        try:
            result = self.dispatch(path, body, query)
            if path == '/api/scan':self.model_library.local(result if isinstance(result,list) else [result])
            if path == '/api/discover':self.model_library.local(result['models'])
            self.audit.emit("operation.result", request_id=request_id, operation=path, result=result)
            return result
        except Exception as exc:
            self.audit.emit("operation.error", request_id=request_id, operation=path, error=str(exc))
            raise

    def dispatch(self, path, body, query):
        if path == '/api/modelchoices':
            return self.model_library.list(body.get('include_featured',False))
        if path == '/api/modelchoices/select':
            selected=self.model_library.get(body['key'])
            if selected['kind']=='local':
                model=inspect_model(selected['path'],self.roots)
                self.model_library.local([model])
                return dict(model,kind='local',identity=model_identity(model))
            from .model_metadata import resolve
            selected=resolve(self.model_library,selected,refresh=body.get('refresh',False),source=body.get('source'))
            selected['note']='Remote metadata only; no weights downloaded. Missing parameter counts must be supplied manually. Remote revision may change.'
            selected['identity']=model_identity(selected)
            return selected
        if path == '/api/services':
            rows=self.services.list()
            for row in rows:
                if row.get('job_id'):
                    try:
                        row['process_status']=self.jobs.get(row['job_id'])['status']
                        if row['process_status'] in ('cancelled','failed','succeeded','interrupted'):
                            row['status']='process-exited' if row['process_status']!='interrupted' else 'ownership-lost'
                    except (ValueError,KeyError):row['process_status']='unknown'
            return rows
        if path == '/api/services/register':
            return self.services.register(body)
        if path == '/api/services/check':
            return self.services.check(body['id'])
        if path == '/api/services/ready':
            return self.services.readiness(body['id'])
        if path == '/api/services/unload':
            service=self.services.get(body['id'])
            if body.get('confirm')!=service['id']:raise ValueError('Confirm this service before unloading')
            if not service.get('job_id'):raise ValueError('External service: stop it in its owning terminal; no unknown process is killed')
            job=self.jobs.get(service['job_id']);plan=job['plan']
            if plan.get('task')!='inference' or service['url']!='http://127.0.0.1:'+str(plan['request'].get('port',8000)):
                raise ValueError('Service ownership does not match the deployment')
            with self.services.lock:
                for chat in self.services.chats.values():
                    if chat['service_id']==service['id'] and chat['status']=='running':chat['stop'].set()
            self.jobs.stop(service['job_id'])
            self.services.set_status(service['id'],'stopping')
            return {'status':'stopping','job_id':service['job_id'],'note':'Wait for process exit; weights on disk are not deleted.'}
        if path == '/api/services/profile':
            return self.services.profile(body)
        if path == '/api/chat/start':
            return self.services.start_chat(body)
        if path == '/api/chat/events':
            return self.services.events(body['id'],body.get('cursor',0))
        if path == '/api/chat/stop':
            return self.services.stop(body['id'])
        if path == '/api/chat/recordings':
            return self.services.recordings()
        if path == '/api/chat/replay':
            return self.services.replay(body['id'])
        if path == '/api/profiler/import':
            trace=read_trace(body['path'],self.roots)
            jid=uuid.uuid4().hex
            event={'type':'profiler','elapsed_s':0,'trace':trace}
            target=self.services.recording_dir/(jid+'.jsonl')
            target.write_text(json.dumps(event,ensure_ascii=False)+'\n',encoding='utf-8')
            return {'id':jid,'events':[event],'log_path':str(target)}
        if path == '/api/evaluations/start':
            service=self.services.get(body['service_id'])
            prompts=body.get('prompts');concurrency=body.get('concurrency',1);max_tokens=body.get('max_tokens',128)
            if not isinstance(prompts,list) or not 1<=len(prompts)<=200 or any(not isinstance(p,dict) or not isinstance(p.get('prompt'),str) or not p['prompt'].strip() for p in prompts):
                raise ValueError('Provide 1–200 non-empty prompts')
            if type(concurrency) is not int or not 1<=concurrency<=16 or type(max_tokens) is not int or not 1<=max_tokens<=8192:
                raise ValueError('Concurrency must be 1–16; max_tokens must be 1–8192')
            with self.evaluation_lock:
                if any(x['status']=='running' for x in self.evaluations.values()):
                    raise ValueError('Wait for the current benchmark to finish')
                jid=uuid.uuid4().hex
                if len(self.evaluations)>=50:self.evaluations.pop(next(iter(self.evaluations)))
                self.evaluations[jid]={'id':jid,'status':'running','created_at':time.time()}
            def evaluate():
                try:
                    report=run_benchmark(service['url'],service['model'],prompts,concurrency,max_tokens)
                    report.update(id=jid,label=str(body.get('label','benchmark'))[:100],timestamp=time.time(),service=service,request=body)
                    folder=self.jobs.state/'benchmarks';folder.mkdir(exist_ok=True)
                    atomic_json(folder/(jid+'.json'),report)
                    result={'id':jid,'status':'completed','report':report}
                except Exception as exc:result={'id':jid,'status':'failed','error':str(exc)[:500]}
                with self.evaluation_lock:self.evaluations[jid]=result
                self.audit.emit('evaluation.completed',result=result)
            threading.Thread(target=evaluate,daemon=True).start()
            return {'id':jid,'status':'running'}
        if path == '/api/evaluations/status':
            with self.evaluation_lock:
                if body['id'] not in self.evaluations:raise ValueError('Unknown evaluation; inspect saved reports after a restart')
                return dict(self.evaluations[body['id']])
        if path == "/api/folders":
            return browse_directories(body.get("path") or self.roots[0],self.roots)
        if path == '/api/folders/create':
            return create_directory(body['parent'],body['name'],self.roots)
        if path == "/api/discover":
            return scan_directory(body["path"],self.roots)
        if path == "/api/techniques":
            return technique_guide(body.get("backend","vllm"),body.get("method","awq"))
        if path == "/api/resources":
            model = inspect_model(body["model_path"], self.roots) if body.get("model_path") else None
            if model is None and body.get('model_key'):
                from .model_metadata import resolve
                model=resolve(self.model_library,self.model_library.get(body['model_key']),offline=True)
            return plan_resources(body, model)
        if path == '/api/data/guide':
            model = inspect_model(body['model_path'], self.roots) if body.get('model_path') else None
            return dataset_guide(body.get('task','sft'),model,body.get('language','en'))
        if path == '/api/environment/plan':
            return self.jobs.create(environment_plan(body,self.roots,self.jobs.state))
        if path == '/api/environment/scan':
            paths=body.get('paths',[])
            if not isinstance(paths,list) or len(paths)>8:raise ValueError('At most 8 environment search directories')
            from .inspection import permitted
            paths=[str(permitted(p,self.roots)) for p in paths]
            return assess_environments(scan_environments(body.get('alias',''),paths),body.get('recipe','download'))
        if path == '/api/cloud/plan':
            return self.jobs.create(cloud_plan(body,self.jobs.state))
        if path == '/api/cloud/access':
            job=self.jobs.get(body['id'])
            if job['plan'].get('task')!='cloud-workspace' or job['status']!='running':raise ValueError('Cloud tunnel is not running')
            access=self.jobs.state/job['id']/'cloud-access.json'
            if not access.exists():raise ValueError('Cloud workspace is still connecting; inspect task log')
            return json.loads(access.read_text())
        if path == "/api/info":
            return {"version": __version__, "hostname":__import__('platform').node(), "roots": self.roots, "state": str(self.jobs.state),
                    "gpu_catalog": GPUS, "audit_path": str(self.audit.path), "native_terminal": sys.platform == "darwin",
                    "apple_silicon": sys.platform == 'darwin' and __import__('platform').machine() == 'arm64',
                    "conda_installations": conda_installations(),
                    "python": sys.executable, "examples": EXAMPLES, "methods": METHODS,
                    "execution_host": "控制台运行主机；不是远端探测所选服务器",
                    "download_sources": hub.SOURCES, "model_authors": hub.AUTHORS,"model_vendors":hub.VENDORS}
        if path == "/api/catalog":
            cache = self.jobs.state/"catalog.json"
            return json.loads(cache.read_text()) if cache.exists() else {"models": [], "errors": [], "synced_at": None}
        if path == "/api/catalog/sync":
            result = hub.catalog(body["source"], body.get("author", "Qwen"), body.get("search", ""), body.get("sort", "downloads"),body,body.get('limit',50))
            with self.lock:
                atomic_json(self.jobs.state/"catalog.json", result)
            self.model_library.catalog(result)
            return result
        if path == "/api/hub/detail":
            try:
                return hub.model_detail(body["source"], body["repo"], body.get("revision", "main"))
            except Exception as exc:
                raise ValueError(hub.public_error(exc)) from None
        if path == "/api/hub/manifest":
            try:
                result = hub.manifest(body["source"], body["repo"], body.get("revision", "main"))
            except Exception as exc:
                raise ValueError(hub.public_error(exc)) from None
            result["id"] = uuid.uuid4().hex
            with self.lock:
                # Bound in-memory previews. Existing download jobs have independent manifests.
                if len(self.hub_manifests) >= 50:
                    self.hub_manifests.pop(next(iter(self.hub_manifests)))
                self.hub_manifests[result["id"]] = result
            return result
        if path == "/api/hub/speed":
            return hub.speed_test(self.hub_manifests[body["manifest_id"]], body["source"])
        if path == "/api/downloads":
            return self.downloads.list()
        if path == "/api/downloads/remove":
            return self.downloads.remove(body['id'])
        if path == "/api/downloads/prepare":
            return self.downloads.prepare(self.hub_manifests[body["manifest_id"]], body["destination"], body["files"])
        if path == "/api/downloads/start":
            if body.get("confirm") != body["id"]:
                raise ValueError("请确认下载计划 ID")
            return self.downloads.start(body["id"], body.get("source"))
        if path == "/api/downloads/pause":
            return self.downloads.pause(body["id"])
        if path == "/api/tensors":
            model = inspect_model(body["path"], self.roots, all_tensors=True)
            offset = body.get("offset", 0)
            if type(offset) is not int or offset < 0:
                raise ValueError("非法分页偏移")
            term = str(body.get("search", ""))[:200].lower()
            selected = [t for t in model["tensors"] if term in (t["component"]+":"+t["name"]).lower()]
            return {"total": len(selected), "offset": offset, "tensors": selected[offset:offset+100], "fingerprint": model["fingerprint"]}
        if path == "/api/hardware":
            return {"current": self.probe(), "history": self.history}
        if path == "/api/remote":
            return remote_probe(body["alias"])
        if path == "/api/scan":
            return discover(body["path"], self.roots) if body.get("recursive") else inspect_model(body["path"], self.roots)
        if path == "/api/data":
            return validate_dataset(body["path"], body["task"], self.roots)
        if path == "/api/estimate":
            model = inspect_model(body["model_path"], self.roots)
            return estimate(model, body.get("task", "inference"), body.get("seq", 2048), body.get("batch", 1), body.get("bits", 16), body.get("tuner", "full"))
        if path == "/api/plan":
            return self.plan(body)
        if path == "/api/start":
            return self.start(body)
        if path == "/api/stop":
            self.jobs.stop(body["id"])
            return {"ok": True}
        if path == "/api/jobs":
            return self.jobs.list()
        if path == "/api/job":
            return self.jobs.get(query["id"][0])
        if path == "/api/logs":
            return self.jobs.logs(query["id"][0])
        if path == "/api/benchmark":
            report = run_benchmark(body["url"], body.get("model", "workbench"), body["prompts"], body.get("concurrency", 1), body.get("max_tokens", 128))
            report["id"] = uuid.uuid4().hex
            report["label"] = str(body.get("label", "unlabeled"))[:100]
            report["timestamp"] = time.time()
            folder = self.jobs.state / "benchmarks"
            folder.mkdir(exist_ok=True)
            (folder/(report["id"]+".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2))
            return report
        if path == "/api/reports":
            return [json.loads(f.read_text()) for f in sorted((self.jobs.state/"benchmarks").glob("*.json"))[-50:]]
        raise ValueError("未知API")


def make_server(app, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # no URL/session token logging

        def send(self, status, data, mime="application/json"):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if mime == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def handle_request(self):
            actual_port = self.server.server_address[1]
            hosts = {"127.0.0.1:"+str(actual_port), "localhost:"+str(actual_port)}
            if self.headers.get("Host") not in hosts:
                return self.send(403, {"error": "Host rejected"})
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://"+h for h in hosts}:
                return self.send(403, {"error": "Origin rejected"})
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                token = self.headers.get("X-Workbench-Token", "")
                if not hmac.compare_digest(token, app.token):
                    return self.send(401, {"error": "打开启动命令给出的带会话令牌链接"})
                body = {}
                if self.command == "POST":
                    if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                        return self.send(415, {"error": "JSON required"})
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 1024*1024:
                        return self.send(413, {"error": "body too large/empty"})
                    body = json.loads(self.rfile.read(size))
                elif parsed.path not in ("/api/info", "/api/hardware", "/api/jobs", "/api/job", "/api/logs", "/api/reports", "/api/downloads", "/api/catalog", "/api/audit", "/api/services"):
                    return self.send(405, {"error": "POST required"})
                return self.send(200, app.api(parsed.path, body, parse_qs(parsed.query)))
            static = {"/marked.js":"marked.js", "/purify.js":"purify.js", "/": "workspace.html", "/classic":"index.html", "/workspace.js":"workspace.js", "/workspace.css":"workspace.css", "/locales.js":"locales.js", "/terminal": "terminal.html", "/app.js": "app.js", "/style.css": "style.css", "/terminal.js": "terminal.js", "/models.js": "models.js", "/resources.js": "resources.js", "/compact.css": "compact.css", "/guides.js":"guides.js"}
            static["/tessweave.svg"] = "tessweave.svg"
            if parsed.path not in static or self.command != "GET":
                return self.send(404, {"error": "not found"})
            f = Path(__file__).parent/"static"/static[parsed.path]
            mime = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}[f.suffix]
            self.send(200, f.read_bytes(), mime)

        def do_GET(self):
            try:
                self.handle_request()
            except (ValueError, KeyError, TypeError, OSError) as e:
                self.send(400, {"error": str(e)[:1000]})
            except Exception:
                self.send(500, {"error": "内部错误；请在CLI检查配置，不向浏览器泄漏调用栈"})

        do_POST = do_GET
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
