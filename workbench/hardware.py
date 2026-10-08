"""Non-privileged, bounded inventory. Same source can run over an approved SSH alias."""
import csv
import io
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path


def command(args, timeout=8):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "output": p.stdout[:100000], "error": p.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "output": "", "error": str(e)}


def local_probe():
    gpus = []
    query = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"])
    if query["ok"]:
        for row in csv.reader(io.StringIO(query["output"])):
            if len(row) == 8:
                def num(x):
                    try:
                        return float(x.strip())
                    except ValueError:
                        return None
                gpus.append(dict(zip(("index", "name", "uuid", "total_mib", "used_mib", "utilization", "temperature", "power_w"),
                                     [row[0].strip(), row[1].strip(), row[2].strip()] + [num(x) for x in row[3:]])))
    mem = None
    if platform.system() == "Darwin":
        r = command(["sysctl", "-n", "hw.memsize"])
        if r["ok"]:
            mem = int(r["output"].strip())
    elif Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem = int(line.split()[1]) * 1024
    net = {}
    if Path("/proc/net/dev").exists():
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, values = line.split(":", 1)
            vals = values.split()
            net[name.strip()] = {"rx_bytes": int(vals[0]), "tx_bytes": int(vals[8])}
    ib = command(["ibstat"])
    return {"timestamp": time.time(), "hostname": platform.node(), "os": platform.system(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "ram_bytes": mem,
            "apple_unified_memory": platform.system() == "Darwin" and platform.machine() == "arm64",
            "gpus": gpus, "gpu_probe": query, "topology": command(["nvidia-smi", "topo", "-m"]),
            "rdma": ib, "network_counters": net, "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
            "tools": {t: shutil.which(t) for t in ("nvidia-smi", "nsys", "ncu", "dcgmi", "rocminfo", "rocm-smi", "npu-smi", "ibstat", "ssh")},
            "note": "网卡字节计数不是 NCCL 吞吐；拓扑不是带宽实测；Apple CPU/GPU 共用内存不能相加"}


def remote_probe(alias):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", alias):
        raise ValueError("仅接受 ~/.ssh/config 中的简单主机别名，不接收密码或任意命令")
    source = Path(__file__).read_text()
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                        "-o", "ConnectTimeout=8", alias, "python3 -"], input=source,
                       capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise ValueError("SSH 探测失败；先在终端确认密钥与 known_hosts。" + p.stderr[:500])
    result = json.loads(p.stdout)
    result["ssh_alias"] = alias
    return result


if __name__ == "__main__":
    print(json.dumps(local_probe()))
