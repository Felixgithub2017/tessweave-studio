"""Durable local job state, owned process groups, append logs and structured metrics."""
import ast
import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
import shlex
import shutil
import sys
from pathlib import Path


def redact(line):
    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
    line = re.sub(r"\b(hf_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{12,})\b", "[REDACTED]", line)
    return re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", line)


def metrics(line):
    text = line.strip()
    # Recognize a JSON or Python-literal log record without eval.
    if "{" not in text or "}" not in text:
        return {}
    text = text[text.index("{"):text.rindex("}")+1]
    if len(text) > 8192:
        return {}
    try:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = ast.literal_eval(text)
        keys = ("loss", "learning_rate", "epoch", "grad_norm", "reward", "rewards/accuracy/mean", "kl", "global_step", "step", "train_runtime", "train_samples_per_second")
        return {k: v for k, v in obj.items() if k in keys and type(v) in (float, int)} if isinstance(obj, dict) else {}
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return {}


class Jobs:
    def __init__(self, state, recover=True, audit=None):
        self.state = Path(state).resolve()
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state, 0o700)
        self.db = self.state / "jobs.sqlite3"
        self.lock, self.processes = threading.RLock(), {}
        self.external = set()
        self.audit = audit
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, created REAL, updated REAL, plan TEXT, returncode INTEGER)")
            c.execute("CREATE TABLE IF NOT EXISTS events (job TEXT, ts REAL, data TEXT)")
            if recover:
                c.execute("UPDATE jobs SET status='interrupted', updated=? WHERE status IN ('running','starting','stopping')", (time.time(),))
        os.chmod(self.db, 0o600)

    def connect(self):
        c = sqlite3.connect(self.db, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def create(self, plan):
        jid = plan["id"]
        with self.connect() as c:
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,NULL)", (jid, "planned", time.time(), time.time(), json.dumps(plan)))
        folder = self.state / jid
        folder.mkdir(mode=0o700)
        (folder / "manifest.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        return self.get(jid)

    def get(self, jid):
        with self.connect() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not r:
            raise ValueError("任务不存在")
        result = dict(r)
        result["plan"] = json.loads(result["plan"])
        return result

    def list(self):
        with self.connect() as c:
            rows = [dict(x) for x in c.execute("SELECT id,status,created,updated,returncode,plan FROM jobs ORDER BY created DESC LIMIT 100")]
        for row in rows:
            plan = json.loads(row.pop('plan'))
            row['task'] = plan.get('task', plan.get('request', {}).get('task'))
        return rows

    def update(self, jid, status, returncode=None):
        with self.connect() as c:
            c.execute("UPDATE jobs SET status=?,updated=?,returncode=? WHERE id=?", (status, time.time(), returncode, jid))
        if self.audit:
            self.audit.emit("job.status",job=jid,status=status,returncode=returncode)

    def launch(self, jid, native=False):
        with self.lock:
            row = self.get(jid)
            if row["status"] != "planned":
                raise ValueError("任务已启动或已结束，不允许重复提交")
            self.external={j for j in self.external if self.get(j)["status"] in ("starting","running","stopping")}
            busy=[j for j in list(self.processes)+list(self.external) if self.get(j)['plan'].get('task')!='cloud-workspace']
            if busy and row['plan'].get('task')!='cloud-workspace':
                raise ValueError("本地安全策略一次一个任务；先停止已有服务/训练，避免抢显存")
            plan = row["plan"]
            if plan["blockers"]:
                raise ValueError("存在预检阻断项，不能执行")
            if native:
                return self._native(jid, plan)
            self.update(jid, "starting")
            if self.audit:
                self.audit.emit("command.execute",job=jid,argv=plan["argv"],env=plan["env"],code="workbench.jobs.Jobs.launch",cwd=str(self.state/jid))
            env = {k: v for k,v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))}
            env.update(plan["env"], PYTHONUNBUFFERED="1")
            try:
                proc = subprocess.Popen(plan["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, env=env, cwd=str(self.state / jid),
                                        start_new_session=True, bufsize=0)
            except OSError:
                self.update(jid, "failed")
                raise
            self.processes[jid] = proc
            self.update(jid, "running")
            threading.Thread(target=self._watch, args=(jid, proc), daemon=True).start()
            return self.get(jid)

    def _watch(self, jid, proc):
        log = self.state / jid / "terminal.log"
        count, pending = 0, b""
        with log.open("w", encoding="utf-8") as f:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                pending += chunk.replace(b"\r", b"\n")
                lines = pending.split(b"\n")
                pending = lines.pop()
                if len(pending) > 32768:
                    lines.append(pending); pending = b""
                for raw in lines:
                    line = redact(raw.decode("utf-8", errors="replace"))
                    f.write(line + "\n"); f.flush(); count += len(line.encode())
                    if self.audit:
                        self.audit.emit("terminal.output",job=jid,text=line)
                    if getattr(self,"echo",False):
                        print(line,flush=True)
                    m = metrics(line)
                    if m:
                        with self.connect() as c:
                            c.execute("INSERT INTO events VALUES (?,?,?)", (jid, time.time(), json.dumps(m)))
            if pending:
                f.write(redact(pending.decode("utf-8", errors="replace")))
                if self.audit:
                    self.audit.emit("terminal.output",job=jid,text=redact(pending.decode("utf-8",errors="replace")))
        proc.stdout.close()
        rc = proc.wait()
        with self.lock:
            stopped = self.get(jid)["status"] == "stopping"
            self.processes.pop(jid, None)
            self.update(jid, "cancelled" if stopped else "succeeded" if rc == 0 else "failed", rc)

    def stop(self, jid):
        with self.lock:
            if jid in self.external:
                (self.state/jid/"cancel.request").touch()
                self.update(jid,"stopping")
                return
            proc = self.processes.get(jid)
            if proc is None:
                raise ValueError("没有属于本控制台的活动进程，不按历史 PID 杀进程")
            self.update(jid, "stopping")
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            def kill_later():
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    with self.lock:
                        if self.processes.get(jid) is proc and proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
            threading.Thread(target=kill_later, daemon=True).start()

    def _native(self,jid,plan):
        """Terminal executes a runner; GUI cancellation uses a file, never a historical PID."""
        package_root=Path(__file__).resolve().parent.parent
        argv=[sys.executable,"-m","workbench.native_runner","--state",str(self.state),"--id",jid]
        if self.audit:
            argv += ["--audit-log",str(self.audit.path)]
        script=self.state/jid/"run.command"
        script.write_text("#!/bin/sh\ncd "+shlex.quote(str(package_root))+" || exit 1\nexec "+shlex.join(argv)+"\n")
        script.chmod(0o700)
        if sys.platform=="darwin":
            opener=["open","-a","Terminal",str(script)]
        elif sys.platform.startswith("linux"):
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise ValueError("服务器无桌面会话；请选择托管终端，并在SSH终端用attach查看，不冒充已打开桌面窗口")
            if shutil.which("gnome-terminal"):
                opener=["gnome-terminal","--",str(script)]
            elif shutil.which("xterm"):
                opener=["xterm","-e",str(script)]
            else:
                raise ValueError("未发现受支持的桌面终端")
        else:
            raise ValueError("原生终端执行目前支持macOS/Linux；Windows请使用WSL工作台")
        self.update(jid,"starting")
        if self.audit:
            self.audit.emit("terminal.open",job=jid,argv=opener,runner=argv,backend=plan["argv"],script=str(script))
        try:
            subprocess.run(opener,check=True,capture_output=True,timeout=15)
        except (OSError,subprocess.SubprocessError):
            self.update(jid,"failed")
            raise ValueError("终端未成功打开；请检查桌面权限") from None
        self.external.add(jid)
        def timeout():
            time.sleep(30)
            with self.connect() as c:
                c.execute("UPDATE jobs SET status='failed',updated=? WHERE id=? AND status='starting'",(time.time(),jid))
        threading.Thread(target=timeout,daemon=True).start()
        return self.get(jid)

    def logs(self, jid):
        self.get(jid)
        p = self.state / jid / "terminal.log"
        text = ""
        if p.exists():
            with p.open("rb") as f:
                f.seek(max(0, p.stat().st_size - 64000))
                text = f.read().decode("utf-8", errors="replace")
        with self.connect() as c:
            data = c.execute("SELECT ts,data FROM events WHERE job=? ORDER BY ts DESC LIMIT 1000", (jid,)).fetchall()
        return {"text": text, "metrics": [{"ts": x["ts"], **json.loads(x["data"])} for x in reversed(data)]}

    def shutdown(self):
        for jid in self.external:
            if self.get(jid)["status"] in ("starting","running","stopping"):
                (self.state/jid/"cancel.request").touch()
        deadline=time.monotonic()+17
        while self.external and time.monotonic()<deadline:
            active=[jid for jid in self.external if self.get(jid)['status'] in ('starting','running','stopping')]
            if not active:
                break
            time.sleep(.1)
        owned = list(self.processes.values())
        for jid in list(self.processes):
            try:
                self.stop(jid)
            except (ValueError, ProcessLookupError):
                pass
        for proc in owned:
            try:
                proc.wait(timeout=16)
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
