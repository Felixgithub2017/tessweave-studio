"""SSH bootstrap and foreground tunnel. The remote UI owns its jobs and logs."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

BOOTSTRAP=r'''
import sys,os,json,io,zipfile,subprocess,time,re,urllib.request
from pathlib import Path
os.umask(0o077)
root=Path(ROOT);root.mkdir(parents=True,exist_ok=True)
if sys.version_info<(3,9):raise RuntimeError('Python 3.9+ required')
archive=sys.stdin.buffer.read(8*1024*1024+1)
if len(archive)>8*1024*1024:raise ValueError('Bundle too large')
import hashlib
if hashlib.sha256(archive).hexdigest()[:16]!=DIGEST:raise ValueError('Bundle checksum mismatch')
appdir=root/('app-'+DIGEST)
if not appdir.exists():
    appdir.mkdir(mode=0o700)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for n in z.namelist():
            if not n.startswith('workbench/') or '..' in Path(n).parts:raise ValueError('Invalid bundle path')
        z.extractall(appdir)
record=root/'access.json'
info=None
if record.exists():
    old=json.loads(record.read_text())
    if old.get('port')!=PORT:raise RuntimeError('Workspace exists on a different port; use its port or a new directory')
    try:
        req=urllib.request.Request('http://127.0.0.1:%d/api/info'%PORT,headers={'X-Workbench-Token':old['token']})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=2) as response:response.read(1048576)
        info=old
    except Exception:pass
if info is None:
    state=root/'state';log=root/'server.log'
    argv=[sys.executable,'-m','workbench','serve','--port',str(PORT),'--state',str(state),'--allow-root',str(root)]
    with (root/'bootstrap-commands.jsonl').open('a') as f:f.write(json.dumps({'argv':argv,'cwd':str(appdir),'time':time.time()})+'\n')
    env=dict(os.environ,PYTHONPATH=str(appdir),PYTHONUNBUFFERED='1')
    with log.open('w') as out:
        proc=subprocess.Popen(argv,cwd=appdir,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(60):
        if proc.poll() is not None:raise RuntimeError('Remote server failed; inspect '+str(log))
        text=log.read_text(errors='replace')
        m=re.search(r'#token=([A-Za-z0-9_-]+)',text)
        if m:
            info={'token':m[1],'port':PORT,'pid':proc.pid,'root':str(root),'app':str(appdir)}
            record.write_text(json.dumps(info));record.chmod(0o600);break
        time.sleep(.1)
    if info is None:raise RuntimeError('Remote startup timed out; inspect '+str(log))
print(json.dumps(info))
'''


def run(manifest):
    plan=json.loads(Path(manifest).read_text());r=plan['request'];cwd=Path(manifest).parent
    package=Path(__file__).parent;buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(package.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.js','.css','.html','.txt'):
                archive.write(p,'workbench/'+str(p.relative_to(package)))
    data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()[:16]
    source='ROOT='+repr(r['root'])+'\nPORT='+str(r['remote_port'])+'\nDIGEST='+repr(digest)+'\n'+BOOTSTRAP
    sourcefile=cwd/'bootstrap.py';sourcefile.write_text(source)
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    argv=ssh+[r['alias'],shlex.join(['python3','-c',source])]
    print(json.dumps({'command':argv,'stdin_bundle_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}),flush=True)
    result=subprocess.run(argv,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
    print(result.stderr.decode(errors='replace'),flush=True)
    print('bootstrap returncode='+str(result.returncode),flush=True)
    if result.returncode:raise RuntimeError('Cloud bootstrap failed; see preceding diagnostics')
    info=json.loads(result.stdout)
    print(json.dumps({k:v for k,v in info.items() if k!='token'}),flush=True)
    access=cwd/'cloud-access.json';access.write_text(json.dumps({'token':info['token'],'port':r['local_port'],'root':r['root']}));access.chmod(0o600)
    tunnel=ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f"127.0.0.1:{r['local_port']}:127.0.0.1:{r['remote_port']}",r['alias']]
    print(json.dumps({'command':tunnel}),flush=True)
    print('Cloud link ready after tunnel connects. Stop this job closes tunnel only; remote workspace stays running.',flush=True)
    # exec retains the Jobs-owned process group: cancel stops only this tunnel.
    os.execvp(tunnel[0],tunnel)


if __name__=='__main__':run(sys.argv[1])
