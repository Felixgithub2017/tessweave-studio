"""OpenAI-compatible SSE probe; no fabricated token counts or per-event ITL."""
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request
from urllib.parse import urlparse


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("禁止服务重定向；请明确指定本地受控端点")


def endpoint(url):
    p = urlparse(url)
    if p.scheme != "http" or p.hostname not in ("localhost", "127.0.0.1", "::1") or p.username or p.password or p.query or p.fragment:
        raise ValueError("仅探测 loopback HTTP；云服务请先 SSH 转发到本机")
    if not p.port or not 1024 <= p.port <= 65535:
        raise ValueError("端口必须在 1024..65535")
    return url.rstrip("/")


def one(url, payload, expected=None):
    start = time.monotonic()
    first, events, usage, content = None, 0, {}, ""
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(endpoint(url) + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    done = False
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=180) as r:
            for raw in r:
                if len(raw) > 1024*1024:
                    raise ValueError("SSE 单事件超限")
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True; break
                obj = json.loads(data)
                if obj.get("error"):
                    raise ValueError("服务返回 error 事件")
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    fragment = delta.get("content") or delta.get("reasoning_content") or ""
                    if fragment:
                        first = first if first is not None else time.monotonic() - start
                        events += 1
                        content += delta.get("content") or ""
                        if len(content) > 2*1024*1024:
                            raise ValueError("输出超限")
        duration = time.monotonic() - start
        tokens = usage.get("completion_tokens")
        if type(tokens) is not int or tokens < 0:
            tokens = None
        return {"ok": done, "error": None if done else "流未以 DONE 结束", "duration_s": duration,
                "first_delta_s": first, "delta_events": events, "completion_tokens": tokens,
                "output_tokens_per_s": tokens/duration if isinstance(tokens, int) and duration else None,
                "exact_match": content.strip() == expected.strip() if isinstance(expected, str) else None,
                "output_sha256": hashlib.sha256(content.encode()).hexdigest()}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "duration_s": time.monotonic()-start}


def run_benchmark(url, model, prompts, concurrency=1, max_tokens=128):
    endpoint(url)
    if not 1 <= concurrency <= 16 or not 1 <= len(prompts) <= 200 or not 1 <= max_tokens <= 8192:
        raise ValueError("探针限制：并发1..16、请求1..200、输出1..8192")
    if any(not isinstance(p, dict) or not isinstance(p.get("prompt"), str) or not p["prompt"].strip() for p in prompts):
        raise ValueError("每条请求需要非空 prompt")
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, url, {"model": model, "messages": [{"role": "user", "content": p["prompt"]}], "temperature": 0, "max_tokens": max_tokens}, p.get("expected")) for p in prompts]
        results = [f.result() for f in futures]
    elapsed = time.monotonic()-start
    successful = [r for r in results if r["ok"]]
    tokens = [r.get("completion_tokens") for r in successful]
    firsts = sorted(r["first_delta_s"] for r in successful if r.get("first_delta_s") is not None)
    return {"model": model, "endpoint": url, "concurrency": concurrency, "max_tokens": max_tokens,
            "results": results, "elapsed_s": elapsed, "successes": len(successful), "requests": len(results),
            "aggregate_output_tokens_per_s": sum(tokens)/elapsed if len(successful) == len(results) and all(type(t) is int for t in tokens) else None,
            "p95_first_delta_s": firsts[math.ceil(.95*len(firsts))-1] if len(firsts) >= 20 else None,
            "workload_sha256": hashlib.sha256(json.dumps({"prompts": prompts, "max_tokens": max_tokens, "concurrency": concurrency}, sort_keys=True).encode()).hexdigest(),
            "note": "闭环探针，不是固定到达率压测；SSE事件可能含多个token，不报告伪造ITL；小于20样本不报告p95；exact_match仅为明确提供的答案检查"}
