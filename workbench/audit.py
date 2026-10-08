"""Append-only JSONL trace shared by the web process and native terminal runners."""
import json
import os
import re
import threading
import time
from pathlib import Path
from .jobs import redact


def clean(value):
    if isinstance(value,dict):
        return {k:("[REDACTED]" if re.search(r"token|password|secret|authorization|api.key",k,re.I) and k not in ("max_tokens","completion_tokens","tokens","token_count") else clean(v)) for k,v in value.items()}
    if isinstance(value,list):
        return [clean(x) for x in value]
    if isinstance(value,str):
        return redact(value)
    return value


class Audit:
    def __init__(self,path):
        self.path=Path(path).expanduser().absolute()
        if self.path.is_symlink() or not self.path.parent.is_dir():
            raise ValueError("日志路径不可为符号链接，父目录必须已存在")
        self.lock=threading.RLock()
        fd=os.open(str(self.path),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        os.close(fd)

    def emit(self,kind,**fields):
        event=clean({"time":time.time(),"kind":kind,**fields})
        data=(json.dumps(event,ensure_ascii=False,default=str)+"\n").encode()
        with self.lock:
            with self.path.open("ab",buffering=0) as f:
                # POSIX flock also serializes independent native terminal processes.
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX)
                f.write(data)
                fcntl.flock(f,fcntl.LOCK_UN)

    def tail(self):
        with self.path.open("rb") as f:
            offset=max(0,self.path.stat().st_size-128*1024)
            f.seek(offset)
            if offset:
                f.readline()
            raw=f.read()
        events=[]
        for line in raw.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        return {"path":str(self.path),"events":events[-50:],"bytes":self.path.stat().st_size}
