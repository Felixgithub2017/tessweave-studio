"""Public HF-compatible catalogs and bounded, credential-free HTTP transfers."""
import concurrent.futures
import hashlib
import ipaddress
import json
import re
import socket
import time
import math
from datetime import datetime
import urllib.error
import urllib.request
from pathlib import PurePosixPath
from urllib.parse import quote, urlencode, urlparse

SOURCES = {"huggingface": {"name": "Hugging Face 官方", "url": "https://huggingface.co"},
           "modelscope": {"name": "魔搭 ModelScope（原生 API）", "url": "https://www.modelscope.cn"},
           "hf-mirror": {"name": "HF Mirror（第三方，请自行评估信任）", "url": "https://hf-mirror.com"}}
AUTHORS = ["Qwen", "deepseek-ai", "zai-org", "MiniMaxAI", "meta-llama", "google", "mistralai",
           "microsoft", "nvidia", "openai", "HuggingFaceTB", "black-forest-labs", "Lightricks", "Wan-AI"]
VENDORS = {"Qwen":"Alibaba · Qwen", "deepseek-ai":"DeepSeek", "zai-org":"Z.ai · GLM", "THUDM":"THUDM · legacy GLM",
    "MiniMaxAI":"MiniMax", "moonshotai":"Moonshot · Kimi", "XiaomiMiMo":"Xiaomi · MiMo", "tencent":"Tencent",
    "baidu":"Baidu", "PaddlePaddle":"Baidu · PaddlePaddle", "ByteDance":"ByteDance", "ByteDance-Seed":"ByteDance · Seed",
    "stepfun-ai":"StepFun", "internlm":"Shanghai AI Lab · InternLM", "OpenGVLab":"OpenGVLab · InternVL",
    "01-ai":"01.AI · Yi", "baichuan-inc":"Baichuan", "meituan-longcat":"Meituan · LongCat",
    "meta-llama":"Meta · Llama", "google":"Google · Gemma", "mistralai":"Mistral", "microsoft":"Microsoft · Phi",
    "nvidia":"NVIDIA", "openai":"OpenAI · open weights only", "CohereForAI":"Cohere For AI", "allenai":"AI2 · OLMo",
    "tiiuae":"TII · Falcon", "ibm-granite":"IBM · Granite", "HuggingFaceTB":"Hugging Face · SmolLM",
    "stabilityai":"Stability AI", "black-forest-labs":"Black Forest Labs", "Lightricks":"Lightricks · LTX",
    "Wan-AI":"Alibaba · Wan", "FunAudioLLM":"FunAudioLLM", "fishaudio":"Fish Audio"}
AUTHORS=list(VENDORS)
MAX_JSON = 32 * 1024**2


def source(key):
    if key not in SOURCES:
        raise ValueError("未知下载源；仅允许内置的 HTTPS HF 兼容站点")
    return SOURCES[key]["url"]


def repo_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", value, re.ASCII) or ".." in value:
        raise ValueError("模型 ID 应为组织/仓库，例如 Qwen/Qwen2.5-0.5B-Instruct")
    return value


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 32 for c in name):
        raise ValueError("非法仓库文件名")
    parts = name.split("/")
    if any(p in ("", ".", "..") or p.startswith(".workbench") or ":" in p for p in parts):
        raise ValueError("仓库路径穿越或保留名称")
    if PurePosixPath(name).is_absolute():
        raise ValueError("绝对仓库路径被拒绝")
    return name


def check_url(url):
    p = urlparse(url)
    host = p.hostname or ""
    suffixes = ("huggingface.co", "hf.co", "hf-mirror.com", "amazonaws.com", "cloudfront.net", "modelscope.cn", "aliyuncs.com")
    if p.scheme != "https" or p.username or p.password or p.port not in (None, 443):
        raise ValueError("下载只允许 HTTPS 公网站点")
    if not any(host == s or host.endswith("." + s) for s in suffixes):
        raise ValueError("下载重定向目的地未获允许")
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("拒绝下载源解析到非公网地址")


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        # No authorization/cookie headers are ever accepted by this downloader.
        return super().redirect_request(req, fp, 307 if code == 308 else code, msg, headers, newurl)

    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302


def open_url(url, headers=None, data=None, method=None, timeout=20):
    check_url(url)
    request = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "ModelWorkbench/0.1", "Accept-Encoding": "identity", **(headers or {})})
    # Explicitly disable environment AND macOS system proxy discovery. Redirects
    # stay on this same opener; failure must never fall back to a paid proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirect()).open(request, timeout=timeout)


def get_json(url, payload=None, timeout=20):
    with open_url(url, {"Content-Type":"application/json"} if payload else None,
                  json.dumps(payload).encode() if payload else None, "PUT" if payload else None, timeout=timeout) as response:
        data = response.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise ValueError("远端元数据超过限制")
    return json.loads(data)


def public_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return "源站要求授权/许可或拒绝访问；当前仅支持公开免登录模型，不绕过访问限制"
        return "源站 HTTP %s；请检查版本、镜像同步状态或稍后重试" % exc.code
    # Signed CDN URLs and headers must not enter persistent reports.
    return "网络/格式/校验失败：" + (str(exc)[:180] if isinstance(exc, ValueError) else type(exc).__name__)


def filter_catalog(models, filters):
    filters=filters or {}; low=filters.get('min_b'); high=filters.get('max_b')
    def bound(v):
        if v in (None,''):return None
        f=float(v)
        if not math.isfinite(f) or f<0:raise ValueError('Parameter range must be nonnegative')
        return f
    low,high=bound(low),bound(high)
    if low is not None and high is not None and low>high:raise ValueError('Minimum parameters exceed maximum')
    after=filters.get('after') or '';before=filters.get('before') or ''
    for date in (after,before):
        if date:datetime.strptime(date,'%Y-%m-%d')
    if after and before and after>before:raise ValueError('Date range reversed')
    modality=filters.get('modality') or ''
    def match(m):
        count=m.get('parameters')
        if low is not None or high is not None:
            if count is None:return False
            if low is not None and count<low*1e9:return False
            if high is not None and count>high*1e9:return False
        date=str(m.get('created_at') or '')[:10]
        if (after or before) and not date:return False
        if after and date<after:return False
        if before and date>before:return False
        return not modality or m.get('pipeline_tag')==modality
    return [m for m in models if match(m)]


def catalog(source_key, author="Qwen", search="", sort="downloads", filters=None, limit=50):
    if type(limit) is not int or not 1<=limit<=100:raise ValueError('Per-publisher limit must be 1–100')
    filter_catalog([],filters)
    if sort not in ("downloads", "lastModified"):
        raise ValueError("不支持的排序")
    authors = AUTHORS if author == "all" else [author]
    if any(not re.fullmatch(r"[\w.-]{1,80}", a, re.ASCII) for a in authors) or len(search) > 120:
        raise ValueError("组织或检索词非法")
    base = source(source_key)

    def fetch(a):
        if source_key == "modelscope":
            try:
                data = get_json(base+"/api/v1/models", {"Path":a,"PageNumber":1,"PageSize":limit})
                if data.get("Code") != 200:
                    raise ValueError("魔搭目录查询失败")
                entries = data.get("Data", {}).get("Models", [])
                out=[]
                for m in entries:
                    rid = repo_id(m.get("Path", a)+"/"+m["Name"])
                    if search.lower() not in rid.lower():
                        continue
                    out.append({"id":rid,"author":a,"downloads":m.get("Downloads"),"likes":m.get("Likes"),
                                "pipeline_tag":m.get("Task"),"last_modified":m.get("LastUpdatedTime"),
                                "license":m.get("License"),"gated":False,"url":base+"/models/"+rid,
                                "parameters":None,"created_at":None})
                return out,None
            except Exception as exc:
                return [], {"author":a,"error":public_error(exc)}
        query = {"author": a, "sort": sort, "direction": -1, "limit": limit,
                 "expand": ["safetensors", "createdAt", "lastModified", "pipeline_tag", "cardData", "downloads", "likes"]}
        if (filters or {}).get('modality'):query['pipeline_tag']=filters['modality']
        if search.strip():
            query["search"] = search.strip()
        try:
            values = get_json(base + "/api/models?" + urlencode(query, doseq=True))
            if not isinstance(values, list):
                raise ValueError("模型目录不是列表")
            out = []
            for m in values:
                rid = repo_id(m.get("id", ""))
                card = m.get("cardData") or {}
                count=(m.get('safetensors') or {}).get('total')
                count=count if type(count) is int and count>0 else None
                out.append({"id": rid, "author": a, "downloads": m.get("downloads"), "likes": m.get("likes"),
                            "pipeline_tag": m.get("pipeline_tag"), "library_name": m.get("library_name"),
                            "last_modified": m.get("lastModified", m.get("last_modified")), "gated": m.get("gated", False),
                            "license": card.get("license"), "url": base + "/" + rid,
                            "parameters":count,"parameter_basis":"Hub-reported safetensors elements, not inferred from name",
                            "created_at":m.get('createdAt',m.get('created_at'))})
            return out, None
        except Exception as exc:
            return [], {"author": a, "error": public_error(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, authors))
    models=[m for values,_ in results for m in values]
    return {"source": source_key, "author": author, "search": search, "sort": sort, "synced_at": time.time(),
            "models":filter_catalog(models,filters),"fetched_count":len(models),"filters":filters,
            "unknown_parameters":sum(m.get('parameters') is None for m in models),
            "unknown_dates":sum(not m.get('created_at') for m in models),"errors":[e for _,e in results if e],
            "scope":f"每个组织最多{limit}项，不是全站完整结果；参数/日期在该批结果中过滤，缺失字段不猜测且不匹配相应筛选。日期是仓库创建时间，不保证等于模型正式发布时间；魔搭暂不提供参数/创建时间筛选元数据，组织ID可能与HF不同。"}


def model_detail(source_key, repo, revision="main"):
    """Read metadata only; never load weights or execute repository Python/HTML."""
    base, repo = source(source_key), repo_id(repo)
    if not isinstance(revision, str) or not re.fullmatch(r"[\w./-]{1,160}", revision, re.ASCII) or ".." in revision:
        raise ValueError("Invalid revision")
    errors = []
    metadata = {}
    if source_key == "modelscope":
        if revision == "main":
            revision = "master"
        try:
            response = get_json(base + "/api/v1/models/" + repo)
            if response.get("Code") != 200:
                raise ValueError("ModelScope metadata request failed")
            info = response.get("Data") or {}
            tensors = (info.get("ModelInfos") or {}).get("safetensor") or {}
            tasks = info.get("Tasks") or []
            metadata = {"safetensors": {"total": tensors.get("model_size")},
                        "cardData": {"license": info.get("License")},
                        "pipeline_tag": tasks[0].get("Name") if tasks else None,
                        "library_name": ", ".join(info.get("Libraries") or []),
                        "downloads": info.get("Downloads"), "likes": info.get("Stars"),
                        "createdAt": info.get("CreatedTime"), "lastModified": info.get("LastUpdatedTime")}
        except Exception as exc:
            errors.append({"file": "model metadata", "error": public_error(exc)})
    else:
        metadata = get_json(base + "/api/models/" + repo + "/revision/" + quote(revision, safe=""))
        if not isinstance(metadata, dict):
            raise ValueError("Invalid model metadata")
    resolved = metadata.get("sha") or revision
    def file_url(name):
        if source_key == "modelscope":
            return base + "/api/v1/models/" + repo + "/repo?" + urlencode({"Revision": resolved, "FilePath": name})
        return base + "/" + repo + "/resolve/" + quote(resolved, safe="") + "/" + name
    def read_file(name):
        try:
            with open_url(file_url(name)) as response:
                data = response.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ValueError("Preview exceeds 1 MiB")
            return json.loads(data) if name.endswith('.json') else data.decode('utf-8')
        except Exception as exc:
            return {"preview_error": public_error(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        config, readme = list(pool.map(read_file, ["config.json", "README.md"]))
    for name, value in [("config.json", config), ("README.md", readme)]:
        if isinstance(value, dict) and "preview_error" in value:
            errors.append({"file": name, "error": value["preview_error"]})
    config = config if isinstance(config, dict) and "preview_error" not in config else {}
    readme = readme if isinstance(readme, str) else ""
    card = metadata.get("cardData") or {}
    count = (metadata.get("safetensors") or {}).get("total")
    count = count if type(count) is int and count > 0 else None
    # Preserve explicitly labelled publisher claims verbatim, without guessing from names.
    parameter_claims = [line.strip() for line in readme.splitlines()
                        if re.match(r"^\s*[-*| ]*(?:Number of Parameters|Total Parameters|Active Parameters|总参数量|激活参数量)\s*[:：|]", line, re.I)][:8]
    return {"source": source_key, "repo": repo, "revision": resolved,
            "parameter_claims": parameter_claims,
            "url": base + ("/models/" if source_key == "modelscope" else "/") + repo,
            "parameters": count, "parameter_basis": "hub_safetensors_elements" if count else "unknown",
            "pipeline_tag": metadata.get("pipeline_tag"), "library_name": metadata.get("library_name"),
            "license": card.get("license"), "downloads": metadata.get("downloads"), "likes": metadata.get("likes"),
            "created_at": metadata.get("createdAt"), "last_modified": metadata.get("lastModified"),
            "gated": metadata.get("gated"), "tags": metadata.get("tags", []),
            "config": config, "card": card, "readme": readme, "errors": errors}


def manifest(source_key, repo, revision="main"):
    repo = repo_id(repo)
    if not isinstance(revision, str) or not re.fullmatch(r"[\w./-]{1,160}", revision, re.ASCII) or ".." in revision:
        raise ValueError("非法 revision")
    if source_key == "modelscope":
        revision = "master" if revision == "main" else revision
        raw = get_json(source(source_key)+"/api/v1/models/"+repo+"/repo/files?"+urlencode({"Revision":revision,"Recursive":"true"}))
        if raw.get("Code") != 200:
            raise ValueError("魔搭文件清单失败")
        files=[]
        for item in raw.get("Data",{}).get("Files",[]):
            if item.get("Type") == "tree":
                continue
            name= safe_name(item["Path"])
            digest,size,rev=item.get("Sha256"),item.get("Size"),item.get("Revision","")
            if type(size) is not int or size < 0 or not re.fullmatch(r"[a-f0-9]{64}",digest or "") or not re.fullmatch(r"[a-f0-9]{40,64}",rev):
                raise ValueError("魔搭文件缺少固定版本或 SHA256："+name)
            files.append({"name":name,"size":size,"digest":digest,"algorithm":"sha256","file_revision":rev})
        if not files or len(files)>20000 or len({f["name"] for f in files})!=len(files):
            raise ValueError("魔搭清单为空、重复或过大")
        return {"source":source_key,"provider":"modelscope","repo":repo,"revision":revision,
                "commit":hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
                "version_basis":"文件集合指纹，每个文件锁定自身 Revision；不是 HF commit",
                "files":files,"total_bytes":sum(f["size"] for f in files),"gated":False,"license":None,
                "url":source(source_key)+"/models/"+repo,"fetched_at":time.time()}
    raw = get_json(source(source_key) + "/api/models/" + repo + "/revision/" + quote(revision, safe="") + "?blobs=true")
    commit = raw.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40,64}", commit):
        raise ValueError("源站未提供不可变 commit；拒绝不稳定的断点下载")
    files = []
    for item in raw.get("siblings", []):
        name = safe_name(item["rfilename"])
        lfs = item.get("lfs") or {}
        size = lfs.get("size", item.get("size"))
        digest = lfs.get("sha256") or item.get("blobId")
        algorithm = "sha256" if lfs.get("sha256") else "git-sha1"
        if type(size) is not int or size < 0 or not re.fullmatch(r"[a-f0-9]{%d}" % (64 if algorithm == "sha256" else 40), digest or ""):
            raise ValueError("文件缺少可靠大小或内容哈希：" + name)
        files.append({"name": name, "size": size, "digest": digest, "algorithm": algorithm})
    if not files or len(files) > 20000 or len({f["name"] for f in files}) != len(files):
        raise ValueError("文件清单为空、重复或过大")
    card = raw.get("cardData") or {}
    return {"source": source_key, "repo": repo, "commit": commit, "files": files,
            "total_bytes": sum(f["size"] for f in files), "gated": raw.get("gated", False),
            "license": card.get("license"), "url": source(source_key) + "/" + repo, "fetched_at": time.time()}


def file_url(m, f, source_key=None):
    if (source_key or m["source"]) == "modelscope":
        return source("modelscope")+"/api/v1/models/"+repo_id(m["repo"])+"/repo?"+urlencode({"Revision":f["file_revision"],"FilePath":safe_name(f["name"])})
    return source(source_key or m["source"]) + "/" + repo_id(m["repo"]) + "/resolve/" + m["commit"] + "/" + quote(safe_name(f["name"]), safe="/") + "?download=true"


def resolve_source(m, key):
    if m.get("provider") == "modelscope" or key == "modelscope":
        if m.get("provider") != "modelscope" or key != "modelscope":
            raise ValueError("魔搭与HF仓库版本体系不同；请重新读取目标源清单、建立独立下载任务，不按同名仓库盲目换源")
        return m  # Immutable per-file revisions + hashes survive branch updates.
    return manifest(key, m["repo"], m["commit"])


def range_start(response, offset, size):
    encoding = response.headers.get("Content-Encoding", "identity")
    if encoding != "identity":
        raise ValueError("拒绝压缩传输，避免 Range 字节偏移失效")
    if response.status == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or int(match[1]) != offset or int(match[3]) != size or not offset <= int(match[2]) < size:
            raise ValueError("服务端 Content-Range 与文件清单不一致")
        return offset
    if response.status == 200:
        return 0  # Source ignored Range: never append a complete file to a partial file.
    raise ValueError("下载响应状态异常")


def speed_test(m, source_key, limit=2*1024**2):
    source(source_key)
    start = time.monotonic()
    try:
        # Same immutable revision/content for an honest source comparison.
        remote = resolve_source(m, source_key)
        original = max(m["files"], key=lambda f: f["size"])
        match = next((f for f in remote["files"] if f == original), None)
        if not match or not match["size"]:
            raise ValueError("所选源尚未同步相同文件版本")
        n = min(limit, match["size"])
        transfer = time.monotonic()
        with open_url(file_url(remote, match), {"Range": "bytes=0-%d" % (n-1)}) as response:
            range_start(response, 0, match["size"])
            first = time.monotonic() - transfer
            data = response.read(n)
        if len(data) != n:
            raise ValueError("测速响应提前结束；样本不完整，请重试")
        duration = time.monotonic() - transfer
        return {"source": source_key, "ok": True, "commit": m["commit"], "file": match["name"], "bytes": len(data),
                "seconds": duration, "first_response_s": first, "mib_per_s": len(data)/1024**2/max(duration, .000001),
                "total_seconds": time.monotonic()-start, "note": "最多读取2MiB（含连接开销）；不是整仓持续速度承诺，不落盘、不自动选择源"}
    except Exception as exc:
        return {"source": source_key, "ok": False, "error": public_error(exc)}
