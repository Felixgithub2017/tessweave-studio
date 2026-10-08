"""Bounded Python metadata discovery; never installs packages or imports models."""
import concurrent.futures
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

PROBE = """import sys,json,importlib.metadata as m
d={}
for n in ['torch','ms-swift','vllm','sglang','mlx','mlx-lm','mlx-vlm','mflux','diffusers','transformers','huggingface-hub','modelscope']:
 try:d[n]=m.version(n)
 except m.PackageNotFoundError:d[n]=None
print(json.dumps(dict(python=sys.executable,version=list(sys.version_info[:3]),prefix=sys.prefix,base_prefix=sys.base_prefix,packages=d)))
"""

def inventory(paths=()):
    home=Path.home()
    candidates=[Path(sys.executable)]
    for name in ('python3','python'):
        if shutil.which(name):candidates.append(Path(shutil.which(name)))
    for key in ('VIRTUAL_ENV','CONDA_PREFIX'):
        if os.environ.get(key):candidates.append(Path(os.environ[key])/'bin/python')
    roots=[Path.cwd(),home/'miniforge3',home/'miniconda3',home/'anaconda3',home/'.virtualenvs',home/'.venvs',home/'miniconda3/envs',
           home/'anaconda3/envs',home/'miniforge3/envs',home/'.conda/envs']+[Path(x).expanduser() for x in paths]
    for root in roots:
        candidates.extend([root/'bin/python',root/'.venv/bin/python',root/'venv/bin/python'])
        try:
            for child in sorted(root.iterdir())[:100]:
                if child.is_dir() and not child.is_symlink():candidates.append(child/'bin/python')
        except OSError:pass
    # Do not resolve symlinks: different venvs often point at the same base Python.
    unique=list(dict.fromkeys(str(p.absolute()) for p in candidates if p.is_file() and os.access(p,os.X_OK)))
    def probe(path):
        argv=[path,'-I','-c',PROBE]
        try:
            p=subprocess.run(argv,capture_output=True,text=True,timeout=5)
            data=json.loads(p.stdout) if p.returncode==0 else {}
            return dict(data,path=path,command=argv,returncode=p.returncode,stdout=p.stdout[:16000],stderr=p.stderr[:2000])
        except (OSError,ValueError,subprocess.TimeoutExpired) as exc:return dict(path=path,error=str(exc)[:500],command=argv)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(probe,unique[:24]))
    return dict(hostname=platform.node(),os=platform.system(),machine=platform.machine(),environments=rows,
                truncated=len(unique)>24,scope=[str(x) for x in roots],
                note='Metadata candidates only; no GPU/model certification. Bounded scan, not every folder on disk.')

def scan(alias='',paths=()):
    if not alias:return inventory(paths)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',alias):raise ValueError('Use an SSH config alias')
    if paths:raise ValueError('Custom paths must be scanned in the cloud workspace')
    source=Path(__file__).read_text()+ '\nprint(json.dumps(inventory()))\n'
    argv=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',alias,'python3 -']
    p=subprocess.run(argv,input=source,capture_output=True,text=True,timeout=55)
    if p.returncode:raise ValueError('SSH environment scan failed: '+p.stderr[:500])
    result=json.loads(p.stdout);result.update(ssh_alias=alias,command=argv)
    return result

def assess(report,recipe):
    required={'mlx':['mlx-lm'],'vllm':['vllm'],'sglang':['sglang'],'train':['ms-swift','torch'],
              'download':['huggingface-hub','modelscope']}
    if recipe not in required:raise ValueError('Unknown environment recipe')
    for row in report['environments']:
        reasons=[]
        if row.get('error') or row.get('returncode')!=0:reasons.append('Python probe failed')
        for name in required[recipe]:
            if not row.get('packages',{}).get(name):reasons.append('Missing '+name)
        if recipe=='mlx' and (report['os']!='Darwin' or report['machine']!='arm64'):reasons.append('MLX recipe requires Apple Silicon')
        if recipe in ('vllm','sglang','train') and report['os']!='Linux':reasons.append('This GPU recipe requires Linux; GPU/driver check still required')
        if recipe=='train' and not (row.get('packages',{}).get('ms-swift') or '').startswith('4.5.'):reasons.append('Current training adapter requires ms-swift 4.5.x')
        row.update(candidate=not reasons,reasons=reasons)
    report['recipe']=recipe
    return report
