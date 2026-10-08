import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path
from .hardware import local_probe, remote_probe
from .inspection import inspect_model
from .server import App, make_server


def main():
    p = argparse.ArgumentParser(description="Model Workbench 本地模型工作台")
    sub = p.add_subparsers(dest="action", required=True)
    ui = sub.add_parser("serve")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--state", default=".workbench")
    ui.add_argument("--allow-root", action="append", default=[])
    ui.add_argument("--open", action="store_true")
    ui.add_argument("--audit-log", help="统一 JSONL 日志绝对路径；父目录须已存在")
    call = sub.add_parser("api-call", help="重放 GUI 操作；认证令牌从 WORKBENCH_SESSION_TOKEN 环境变量读取")
    call.add_argument("--operation", required=True)
    call.add_argument("--payload", default="{}")
    call.add_argument("--query", default="{}")
    call.add_argument("--port", type=int, default=8765)
    scan = sub.add_parser("inspect")
    scan.add_argument("path")
    hw = sub.add_parser("hardware")
    hw.add_argument("--ssh-alias")
    attach = sub.add_parser("attach")
    attach.add_argument("id")
    attach.add_argument("--state", default=".workbench")
    a = p.parse_args()
    if a.action == "api-call":
        import os
        from urllib.request import Request, urlopen
        from urllib.parse import urlencode
        if not a.operation.startswith("/api/") or "?" in a.operation or "#" in a.operation:
            p.error("operation must be an /api/ path")
        url = "http://127.0.0.1:%s%s?%s" % (a.port, a.operation, urlencode(json.loads(a.query), doseq=True))
        req = Request(url, data=json.dumps(json.loads(a.payload)).encode(), headers={"Content-Type":"application/json", "X-Workbench-Token":os.environ.get("WORKBENCH_SESSION_TOKEN", "")})
        with urlopen(req, timeout=120) as response:
            print(response.read().decode())
    elif a.action == "hardware":
        print(json.dumps(remote_probe(a.ssh_alias) if a.ssh_alias else local_probe(), ensure_ascii=False, indent=2))
    elif a.action == "inspect":
        print(json.dumps(inspect_model(a.path, [Path(a.path).expanduser().resolve()]), ensure_ascii=False, indent=2))
    elif a.action == "attach":
        import re
        if not re.fullmatch(r"[a-f0-9]{32}", a.id):
            p.error("invalid job id")
        log = Path(a.state).resolve()/a.id/"terminal.log"
        offset = 0
        try:
            while True:
                if log.exists():
                    with log.open() as f:
                        f.seek(offset); print(f.read(), end="", flush=True); offset=f.tell()
                time.sleep(.5)
        except KeyboardInterrupt:
            pass
    else:
        if not 1024 <= a.port <= 65535:
            p.error("port must be 1024..65535")
        app = App(a.state, a.allow_root or [str(Path.cwd())], a.audit_log)
        server = make_server(app, a.port)
        url = "http://127.0.0.1:%s/#token=%s" % (a.port, app.token)
        print("本地单用户工作台（不要分享会话链接）：\n" + url, flush=True)
        print("执行主机是此 Python 所在主机。已授权目录：", app.roots, flush=True)
        if a.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.services.shutdown()
            app.downloads.shutdown()
            app.jobs.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
