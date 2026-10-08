"""Plan a private cloud workspace, not an arbitrary SSH command console."""
import re
import sys
import time
import uuid
from pathlib import Path, PurePosixPath


def cloud_plan(body,state):
    alias=str(body.get('alias',''))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',alias):raise ValueError('Use an existing SSH config alias')
    root=str(body.get('root',''))
    path=PurePosixPath(root)
    if not path.is_absolute() or '..' in path.parts or len(path.parts)<3 or any(c in root for c in '\n\r\x00'):
        raise ValueError('Use a dedicated absolute remote directory, e.g. /home/user/model-workbench')
    ports={k:body.get(k,d) for k,d in [('local_port',8876),('remote_port',8876)]}
    if any(type(v) is not int or not 1024<=v<=65535 for v in ports.values()):raise ValueError('Port must be 1024–65535')
    jid=uuid.uuid4().hex
    request=dict(alias=alias,root=root,**ports)
    return {'id':jid,'task':'cloud-workspace','request':request,'output':None,'env':{},'blockers':[],
            'argv':[sys.executable,str(Path(__file__).with_name('cloud_runner.py')),str(Path(state)/jid/'manifest.json')],
            'steps':['SSH read-only Python check; upload only Workbench source (never weights/data/credentials)',
                     'Start/reuse loopback cloud workspace in the chosen directory',
                     f"SSH 127.0.0.1:{ports['local_port']} → remote 127.0.0.1:{ports['remote_port']}",
                     'Open cloud workspace: downloads, environments, training and deployment execute there'],
            'warnings':['Remote account requires Python >=3.9, SSH keys and known_hosts. No sudo.',
                        'Cloud server survives tunnel disconnect; stopping this local task stops the tunnel only.',
                        'All cloud task logs remain on cloud disk; model weights never transit this Mac.',
                        'Source bootstrap is experimental; no automatic driver installation or unattended upgrades.'],
            'created_at':time.time()}
