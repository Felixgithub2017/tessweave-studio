"""Audited subprocess sequence, launched by Jobs in native or managed terminal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(manifest):
    plan=json.loads(Path(manifest).read_text());r=plan['request'];target=Path(r['path'])
    if target.exists():raise ValueError('Refusing to overwrite an existing environment')
    cwd=Path(manifest).parent
    env=dict(os.environ)
    for key in list(env):
        if 'proxy' in key.lower() or key.startswith(('PIP_','UV_')):env.pop(key,None)
    env.update(NO_PROXY='*',no_proxy='*',PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONUNBUFFERED='1')
    log=cwd/'environment-commands.jsonl'
    def command(argv):
        event={'time':time.time(),'argv':argv,'cwd':str(cwd),'network':'proxy environment cleared; conda-forge / PyPI (Conda config may apply)'}
        with log.open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps({'command':argv}),flush=True)
        p=subprocess.run(argv,cwd=cwd,env=env)
        with log.open('a') as f:f.write(json.dumps({'time':time.time(),'returncode':p.returncode})+'\n')
        if p.returncode:raise RuntimeError('Command failed: '+str(p.returncode))
    command(r['create_argv'] if 'create_argv' in r else [r['python'],'-m','venv',str(target)])
    py=r.get('env_python') or str(target/'bin/python')
    command([py,'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:','--upgrade','pip'])
    command([py,'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:',
             '--dry-run','--report',str(cwd/'resolve.json'),*r['packages']])
    report=json.loads((cwd/'resolve.json').read_text());locks=[]
    import re
    for item in report['install']:
        name=item['metadata']['name'];version=item['metadata']['version']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or not re.fullmatch(r'[A-Za-z0-9.+!_-]+',version):raise ValueError('Invalid package metadata')
        locks.append(name+'=='+version)
    lock=cwd/'resolved-requirements.txt';lock.write_text('\n'.join(locks)+'\n')
    command([py,'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:','-r',str(lock)])
    command([py,'-m','pip','check'])
    command([py,'-m','pip','freeze'])
    smoke="import sys,platform,importlib.util; print({'python':sys.version,'host':platform.node()}); "
    if r['recipe'] in ('vllm','sglang','train'):
        smoke+="import torch; print({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_build':torch.version.cuda,'gpu_count':torch.cuda.device_count()})"
    elif r['recipe']=='mlx':smoke+="import mlx.core as mx; print({'metal_available':mx.metal.is_available()})"
    else:smoke+="import huggingface_hub,modelscope; print('download SDK imports OK')"
    command([py,'-c',smoke])
    (cwd/'environment-result.json').write_text(json.dumps({'python':py,'recipe':r['recipe'],'status':'installed','lock':str(lock)},indent=2))
    print('ENVIRONMENT_READY '+py,flush=True)


if __name__=='__main__':run(sys.argv[1])
