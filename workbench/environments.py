"""Explicit, isolated environment plans. Run on the host owning this console."""
import os
import re
import sys
import uuid
import platform
import shutil
from pathlib import Path
from .inspection import permitted

PACKAGES={'download':['huggingface-hub','modelscope'], 'vllm':['vllm'],
          'sglang':['sglang'], 'train':['ms-swift','torch'], 'mlx':['mlx-lm']}


def conda_installations():
    """Discover executables without activating shells or modifying Conda config."""
    candidates=[]
    for name in ('miniforge3','Miniforge3','miniconda3','Miniconda3','anaconda3'):
        base=Path.home()/name
        candidates.append(base/('Scripts/conda.exe' if os.name=='nt' else 'bin/conda'))
    for value in (os.environ.get('CONDA_EXE'),shutil.which('conda')):
        if value:candidates.append(Path(value))
    rows=[];seen=set()
    for p in candidates:
        if not p.is_file() or not os.access(p,os.X_OK):continue
        p=p.resolve()
        base=p.parent.parent
        identity=(base.stat().st_dev,base.stat().st_ino)
        if identity in seen:continue
        seen.add(identity)
        rows.append({'executable':str(p),'root':str(base),'envs_dir':str(base/'envs')})
    return rows


def environment_plan(body, roots, state):
    recipe=body.get('recipe','download')
    if recipe not in PACKAGES:raise ValueError('Unknown environment recipe')
    manager=body.get('manager','conda')
    if manager not in ('conda','venv'):raise ValueError('Choose conda or venv')
    if not str(body.get('path','')).strip():raise ValueError('Choose an absolute target folder for the new environment / 请填写新环境的绝对路径')
    target=Path(body['path']).expanduser()
    if not target.is_absolute():raise ValueError('Use an absolute environment path')
    parent=permitted(target.parent,roots);target=parent/target.name
    if target.name in ('','.','..'):raise ValueError('Invalid environment name')
    version=str(body.get('version','')).strip() or ('4.5.3' if recipe=='train' else '')
    if version and not re.fullmatch(r'\d+(?:\.\d+){1,3}(?:(?:a|b|rc)\d+)?',version):raise ValueError('Version must be an exact release number')
    python=None
    if manager=='venv':
        python=permitted(body.get('python') or sys.executable, list(roots)+[str(Path(sys.executable).resolve().parent)])
        if not python.is_file() or not os.access(python,os.X_OK):raise ValueError('Python must be an executable file')
        create=[str(python),'-m','venv',str(target)]
    else:
        installs=conda_installations()
        executable=body.get('conda') or (installs[0]['executable'] if installs else '')
        if not executable:raise ValueError('Miniforge/Miniconda not found. Install it explicitly or choose venv; no automatic installation.')
        conda=permitted(executable,roots)
        if not conda.is_file() or not os.access(conda,os.X_OK):raise ValueError('Conda executable is unavailable')
        py_version=str(body.get('python_version','3.11'))
        if not re.fullmatch(r'3\.(?:10|11|12|13)(?:\.\d+)?',py_version):raise ValueError('Choose Python 3.10–3.13 (backend compatibility still needs validation)')
        create=[str(conda),'create','--yes','--prefix',str(target),'--override-channels','--channel','https://conda.anaconda.org/conda-forge','--no-default-packages','python='+py_version,'pip']
    packages=list(PACKAGES[recipe]);packages[0]+= ('=='+version) if version else ''
    jid=uuid.uuid4().hex;folder=Path(state)/jid
    env_python=str(target/('python.exe' if manager=='conda' and os.name=='nt' else 'Scripts/python.exe' if os.name=='nt' else 'bin/python'))
    request={'recipe':recipe,'path':str(target),'python':str(python) if python else None,'packages':packages,
             'manager':manager,'create_argv':create,'env_python':env_python}
    blockers=['Environment already exists; choose a new directory'] if target.exists() else []
    if recipe=='mlx' and (platform.system()!='Darwin' or platform.machine()!='arm64'):
        blockers.append('MLX recipe requires Apple Silicon; choose the correct execution host')
    if recipe in ('vllm','sglang','train') and platform.system()!='Linux':
        blockers.append('This GPU environment recipe requires Linux; open the cloud Linux workspace')
    return {'id':jid,'task':'environment','request':request,'output':None,
            'argv':[sys.executable,str(Path(__file__).with_name('environment_runner.py')),str(folder/'manifest.json')],
            'env':{},'blockers':blockers,
            'steps':[create,
                     [env_python,'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:','--upgrade','pip'],
                     [env_python,'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:','--dry-run','--report',str(folder/'resolve.json'),*packages],
                     [env_python,'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:','-r',str(folder/'resolved-requirements.txt')],
                     [env_python,'-m','pip','check'],
                     [env_python,'-m','pip','freeze']],
            'warnings':['Fresh environment only. No sudo, drivers, base-environment or shell-init changes.',
                        'Conda installs Python/pip from conda-forge; model packages use PyPI wheels. Existing Conda configuration may affect network access.',
                        'Resolver selects compatible versions at execution; exact resolved versions and artifacts are logged.',
                        'Successful install is not GPU/model certification. Inspect smoke-test output.'],
            'created_at':__import__('time').time()}
