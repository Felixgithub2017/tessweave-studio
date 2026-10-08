"""Entry point executed INSIDE a real terminal window. No arbitrary API commands."""
import argparse
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from .jobs import Jobs
from .audit import Audit


def run(state,jid,audit_path=None):
    if not re.fullmatch(r"[a-f0-9]{32}",jid):
        raise ValueError("非法任务ID")
    jobs=Jobs(state,recover=False,audit=Audit(audit_path) if audit_path else None)
    row=jobs.get(jid); plan=row["plan"]
    if plan["blockers"]:
        raise ValueError("计划存在阻断")
    with jobs.connect() as c:
        updated=c.execute("UPDATE jobs SET status='running',updated=? WHERE id=? AND status='starting'",(time.time(),jid))
        if updated.rowcount!=1:
            raise ValueError("任务不在等待原生终端状态；拒绝重复执行")
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN","SECRET","PASSWORD","API_KEY"))}
    env.update(plan["env"],PYTHONUNBUFFERED="1")
    print("Model Workbench · 原生终端执行\n",plan["argv"],flush=True)
    proc=None
    try:
        if (jobs.state/jid/"cancel.request").exists():
            jobs.update(jid,"cancelled"); return
        if jobs.audit:
            jobs.audit.emit("command.execute",job=jid,argv=plan["argv"],env=plan["env"],code="workbench.native_runner.run",cwd=str(jobs.state/jid))
        proc=subprocess.Popen(plan["argv"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                              cwd=str(jobs.state/jid),env=env,start_new_session=True,bufsize=0)
        jobs.processes[jid]=proc
        jobs.echo=True
        def stop(*unused):
            if proc.poll() is None:
                jobs.stop(jid)
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            signal.signal(sig,stop)
        def cancel_watch():
            while proc.poll() is None:
                if (jobs.state/jid/"cancel.request").exists():
                    stop(); return
                time.sleep(.25)
        threading.Thread(target=cancel_watch,daemon=True).start()
        jobs._watch(jid,proc)
        print("\n任务结束：",jobs.get(jid)["status"],flush=True)
    except Exception:
        jobs.update(jid,"failed")
        raise
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--state",required=True);p.add_argument("--id",required=True);p.add_argument("--audit-log")
    a=p.parse_args();run(a.state,a.id,a.audit_log)
