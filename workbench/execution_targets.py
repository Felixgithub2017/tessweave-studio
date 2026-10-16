"""Host identity and selectable, owned SSH workspaces. Never arbitrary API proxying."""
import json
import os
import platform
from pathlib import Path


def identity(state):
    return {'kind': 'remote' if os.environ.get('TESSWEAVE_REMOTE_WORKSPACE') == '1' else 'local',
            'hostname': platform.node(), 'state': str(Path(state).resolve()),
            'scope': ['models', 'downloads', 'environments', 'training', 'deployment', 'evaluation', 'logs']}


def targets(jobs):
    result = []
    for row in jobs.list():
        job = jobs.get(row['id'])
        if job['plan'].get('task') != 'cloud-workspace':
            continue
        r = job['plan']['request']
        access = jobs.state/job['id']/'cloud-access.json'
        # Expose no credentials in the inventory. The existing access endpoint
        # retrieves the token only after an explicit selection.
        ready = False
        if row['status'] == 'running' and access.is_file():
            try:
                data = json.loads(access.read_text())
                ready = type(data.get('port')) is int and data['port'] == r['local_port']
            except (OSError, ValueError):
                pass
        result.append({'id': row['id'], 'alias': r['alias'], 'root': r['root'],
                       'port': r['local_port'], 'status': row['status'], 'selectable': ready})
    return {'current': identity(jobs.state), 'targets': result}
