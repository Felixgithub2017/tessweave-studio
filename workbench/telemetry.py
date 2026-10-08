"""Unprivileged host samples. Never attribute host totals to a remote model."""
import csv
import io
import os
import platform
import time
import json
import re
import math
import subprocess
from pathlib import Path
from .hardware import command


def sample_host():
    sample = {'timestamp':time.time(),'host':platform.node(),'scope':'workbench-host, not per-model',
              'load1':os.getloadavg()[0] if hasattr(os,'getloadavg') else None,
              'cpu_count':os.cpu_count(),'gpus':[], 'memory_available_bytes':None}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):sample['memory_available_bytes']=int(line.split()[1])*1024
    if platform.system()=='Darwin':
        r=command(['ioreg','-r','-c','IOAccelerator','-l'],timeout=1)
        for line in r['output'].splitlines():
            if '"PerformanceStatistics" =' not in line:continue
            stats=dict((k,float(v)) for k,v in re.findall(r'"([^"]+)"\s*=\s*([0-9.]+)',line))
            if 'Device Utilization %' in stats:
                sample['gpus'].append({'index':str(len(sample['gpus'])), 'utilization':stats['Device Utilization %'],
                    'used_mib':None,'total_mib':None,'power_w':None,
                    'driver_in_use_mib':stats.get('In use system memory',0)/1048576,
                    'source':'IOAccelerator PerformanceStatistics (undocumented, driver-wide)'})
        sample['gpu_note']='Apple driver counters are best-effort, system-wide; driver memory is NOT model memory or dedicated VRAM.'
    else:
        r=command(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],timeout=1)
        if r['ok']:
            for row in csv.reader(io.StringIO(r['output'])):
                if len(row)!=5:continue
                def number(x):
                    try:return float(x.strip())
                    except ValueError:return None
                sample['gpus'].append(dict(zip(['index','utilization','used_mib','total_mib','power_w'],[row[0].strip()]+[number(v) for v in row[1:]])))
        else:sample['gpu_note']='GPU telemetry unavailable on this host'
    sample['network_counters']={}
    if Path('/proc/net/dev').exists():
        for line in Path('/proc/net/dev').read_text().splitlines()[2:]:
            name,values=line.split(':',1);v=values.split()
            sample['network_counters'][name.strip()]={'rx_bytes':int(v[0]),'tx_bytes':int(v[8])}
    return sample


def sample_remote(alias):
    """Explicit SSH alias only. Fixed read-only collector; no shell interpolation."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',alias):raise ValueError('Invalid SSH alias')
    source=Path(__file__).read_text().replace('from .hardware import command', '''def command(args, timeout=1):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
        return {'ok':p.returncode==0,'output':p.stdout[:100000]}
    except (OSError,subprocess.TimeoutExpired):return {'ok':False,'output':''}''')
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=2',alias,'python3 -'],
                     input=source+'\nprint(json.dumps(sample_host()))\n',capture_output=True,text=True,timeout=4)
    if p.returncode:raise ValueError('SSH collector failed; check keys and known_hosts in terminal')
    result=json.loads(p.stdout);result['scope']='ssh-host, not per-model';result['ssh_alias']=alias
    return result


METRICS={'vllm:kv_cache_usage_perc','vllm:gpu_cache_usage_perc','vllm:num_requests_running',
         'vllm:num_requests_waiting','sglang:token_usage','sglang:num_running_reqs','sglang:num_queue_reqs',
         'sglang:num_used_tokens'}


def parse_metrics(text):
    """Keep series separate: TP/DP gauge values must not be blindly summed."""
    rows=[]
    for line in text.splitlines():
        m=re.fullmatch(r'([\w:]+)(\{.*\})?\s+([-+\w.eE]+)(?:\s+\S+)?',line.strip())
        if not m or m[1] not in METRICS:continue
        try:value=float(m[3])
        except ValueError:continue
        if not math.isfinite(value):continue
        rows.append({'name':m[1],'labels':(m[2] or '')[:1000],'value':value})
        if len(rows)>=100:break
    return rows


def sample_engine(url):
    import urllib.request
    from .benchmark import endpoint, NoRedirect
    source=endpoint(url)+'/metrics'
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open(source,timeout=1) as response:
            data=response.read(1024*1024+1)
        if len(data)>1024*1024:raise ValueError('Metrics exceed 1 MiB')
        rows=parse_metrics(data.decode('utf-8'))
        return {'source':source,'scope':'engine-wide, not per-request','series':rows,
                'status':'ok' if rows else 'no-supported-series'}
    except Exception as exc:return {'source':source,'status':'unavailable','series':[],'error':str(exc)[:200]}
