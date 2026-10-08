"""One managed download at a time; immutable manifests, verified files, resumable parts."""
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from . import hub
from .inspection import permitted


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(str(temporary), str(path))


def verify(path, entry, stop=None):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    h = hashlib.sha256() if entry["algorithm"] == "sha256" else hashlib.sha1()
    if entry["algorithm"] == "git-sha1":
        h.update(("blob %d\0" % entry["size"]).encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            if stop and stop.is_set():
                raise InterruptedError("校验暂停")
            h.update(chunk)
    return h.hexdigest() == entry["digest"]


class Downloads:
    def __init__(self, state, roots):
        self.folder = Path(state).resolve()/"downloads"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.roots = roots
        self.lock = threading.RLock()
        self.records, self.stops, self.threads = {}, {}, {}
        for path in self.folder.glob("*.json"):
            record = json.loads(path.read_text())
            if record["status"] in ("downloading", "pausing", "verifying"):
                record["status"] = "paused"
                record["error"] = "控制台中断；可恢复并重新验证已完成文件"
            self.records[record["id"]] = record

    def save(self, record):
        record["updated_at"] = time.time()
        atomic_json(self.folder/(record["id"]+".json"), record)
        if getattr(self,"audit",None):
            self.audit.emit("download.progress",job=record['id'],status=record['status'],
                            downloaded_bytes=record['downloaded_bytes'],total_bytes=record['total_bytes'],
                            source=record['source'],current_file=record['current_file'],error=record['error'])

    def get(self, jid):
        with self.lock:
            if jid not in self.records:
                raise ValueError("下载任务不存在")
            return json.loads(json.dumps(self.records[jid]))

    def list(self):
        with self.lock:
            return [{k: v for k, v in r.items() if k not in ("manifest", "completed")}
                    for r in sorted(self.records.values(), key=lambda r: r["created_at"], reverse=True)[:100]]

    def remove(self, jid):
        """Archive a stopped record, preserving all downloaded files and partials."""
        with self.lock:
            record = self.get(jid)
            if (jid in self.threads and self.threads[jid].is_alive()) or record['status'] in ('downloading','verifying','pausing'):
                raise ValueError('Pause and wait before removing this record')
            archive = self.folder/'archived'
            archive.mkdir(exist_ok=True)
            os.replace(str(self.folder/(jid+'.json')), str(archive/(jid+'.json')))
            del self.records[jid]
            return {'removed':jid,'files_preserved':True,'archive':str(archive/(jid+'.json'))}

    def prepare(self, m, destination, names):
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            raise ValueError("请明确选择文件，且不得重复")
        chosen = [f for f in m["files"] if f["name"] in names]
        if len(chosen) != len(names):
            raise ValueError("所选文件不在服务端清单中")
        if m.get("gated"):
            raise ValueError("受限模型请先通过官方许可流程；本版不管理登录令牌")
        dest = Path(destination).expanduser()
        if not dest.is_absolute() or dest.name in ("", ".", ".."):
            raise ValueError("请填写绝对目标路径，其父目录必须已存在")
        parent = permitted(dest.parent, self.roots)
        dest = parent / dest.name
        if dest.exists() or dest.is_symlink():
            raise ValueError("目标必须是新目录，不覆盖已有模型；续传请使用原任务的继续按钮")
        total = sum(f["size"] for f in chosen)
        if shutil.disk_usage(parent).free < total + 64*1024**2:
            raise ValueError("目标磁盘剩余空间不足（保留至少64MiB）")
        with self.lock:
            if any(r["destination"] == str(dest) and r["status"] != "completed" for r in self.records.values()):
                raise ValueError("此目录已有下载计划，请继续该计划")
            jid = uuid.uuid4().hex
            record = {"id": jid, "repo": m["repo"], "commit": m["commit"], "source": m["source"],
                      "destination": str(dest), "manifest": dict(m, files=chosen), "total_bytes": total,
                      "file_count": len(chosen), "completed": [], "downloaded_bytes": 0, "current_file": None,
                      "status": "planned", "created_at": time.time(), "error": None, "bytes_per_s": 0}
            self.records[jid] = record
            self.save(record)
            return self.get(jid)

    def start(self, jid, source_key=None):
        with self.lock:
            r = self.get(jid)
            if any(t.is_alive() for t in self.threads.values()):
                raise ValueError("已有下载或暂停清理进行中，请等它完成")
            if r["status"] not in ("planned", "paused", "failed"):
                raise ValueError("当前下载状态不能继续")
            key = source_key or r["source"]
            hub.source(key)
            dest = Path(r["destination"])
            permitted(dest.parent, self.roots)
            marker = dest/".workbench-download.json"
            if dest.exists():
                if dest.is_symlink() or not marker.is_file() or marker.is_symlink() or json.loads(marker.read_text()).get("id") != jid:
                    raise ValueError("目标目录不是本任务管理的目录，拒绝写入")
            else:
                if r["status"] != "planned":
                    raise ValueError("下载目录被移走，请恢复目录后重试")
                dest.mkdir()
                atomic_json(marker, {"id": jid, "repo": r["repo"], "commit": r["commit"]})
            r = self.records[jid]
            r.update(status="downloading", source=key, error=None, bytes_per_s=0)
            self.save(r)
            stop = threading.Event()
            thread = threading.Thread(target=self.run, args=(jid, stop), daemon=True)
            self.stops[jid], self.threads[jid] = stop, thread
            thread.start()
            return self.get(jid)

    def update(self, jid, **fields):
        with self.lock:
            self.records[jid].update(fields)
            self.save(self.records[jid])

    def pause(self, jid):
        with self.lock:
            if jid in self.stops and self.threads[jid].is_alive():
                self.stops[jid].set()
                self.update(jid, status="pausing")
            return self.get(jid)

    def run(self, jid, stop):
        r = self.get(jid)
        dest = Path(r["destination"])
        try:
            # Verify all selected files at the alternate source before reusing partials.
            remote = hub.resolve_source(r["manifest"], r["source"])
            if any(f not in remote["files"] for f in r["manifest"]["files"]) or remote["commit"] != r["commit"]:
                raise ValueError("下载源版本/文件指纹不一致，拒绝混合权重")
            parts = dest/".workbench-parts"
            if parts.is_symlink():
                raise ValueError("临时文件目录不能是符号链接")
            parts.mkdir(exist_ok=True)
            completed, total_done = [], 0
            for f in r["manifest"]["files"]:
                if stop.is_set():
                    break
                name = hub.safe_name(f["name"])
                target = dest.joinpath(*name.split("/"))
                current = dest
                for segment in name.split("/")[:-1]:
                    current = current/segment
                    if current.is_symlink():
                        raise ValueError("下载子路径是符号链接")
                    current.mkdir(exist_ok=True)
                if target.is_symlink():
                    raise ValueError("下载目标是符号链接")
                part = parts/(hashlib.sha256(name.encode()).hexdigest()+".part")
                if part.is_symlink():
                    raise ValueError("续传文件是符号链接")
                self.update(jid, current_file=name, status="verifying", bytes_per_s=0)
                if target.exists():
                    if not verify(target, f, stop):
                        raise ValueError("已完成文件校验失败；保留现场，不覆盖：" + name)
                else:
                    offset = part.stat().st_size if part.exists() else 0
                    if offset > f["size"]:
                        raise ValueError("续传文件大于清单大小")
                    self.update(jid, status="downloading", downloaded_bytes=total_done+offset)
                    if offset < f["size"]:
                        with hub.open_url(hub.file_url(remote, f), {"Range": "bytes=%d-" % offset}) as response:
                            start_at = hub.range_start(response, offset, f["size"])
                            offset = start_at
                            timer, previous, checkpoint = time.monotonic(), offset, time.monotonic()
                            with part.open("ab" if start_at else "wb") as stream:
                                while not stop.is_set():
                                    chunk = response.read(min(1024**2, f["size"]-offset+1))
                                    if not chunk:
                                        break
                                    if offset + len(chunk) > f["size"]:
                                        raise ValueError("远端内容大于锁定文件大小")
                                    stream.write(chunk); offset += len(chunk)
                                    now = time.monotonic()
                                    if now-checkpoint >= .5:
                                        stream.flush()
                                        self.update(jid, downloaded_bytes=total_done+offset,
                                                    bytes_per_s=(offset-previous)/max(now-timer, .001))
                                        checkpoint = now
                                        timer, previous = now, offset
                                stream.flush(); os.fsync(stream.fileno())
                    elif not part.exists():
                        part.touch()  # Empty files still require their content hash.
                    self.update(jid, downloaded_bytes=total_done+offset, status="verifying", bytes_per_s=0)
                    if stop.is_set():
                        break
                    if not verify(part, f, stop):
                        # Bad bytes cannot be resumed. Preserve them for diagnosis instead of deleting.
                        if part.stat().st_size == f["size"]:
                            part.rename(part.with_name(part.name+".invalid-"+uuid.uuid4().hex[:8]))
                        raise ValueError("文件长度或内容哈希校验失败：" + name)
                    os.replace(str(part), str(target))
                completed.append(name); total_done += f["size"]
                self.update(jid, completed=completed, downloaded_bytes=total_done)
            if stop.is_set():
                self.update(jid, status="paused", bytes_per_s=0)
            else:
                atomic_json(dest/".workbench-download.json", {"id": jid, "repo": r["repo"], "commit": r["commit"],
                            "source": r["source"], "files": r["manifest"]["files"], "verified_at": time.time()})
                self.update(jid, status="completed", current_file=None, bytes_per_s=0)
        except Exception as exc:
            self.update(jid, status="paused" if stop.is_set() else "failed", bytes_per_s=0, error=hub.public_error(exc))

    def shutdown(self):
        for event in self.stops.values():
            event.set()
        deadline = time.monotonic()+25
        for thread in self.threads.values():
            thread.join(timeout=max(0, deadline-time.monotonic()))
