"""Two real loopback workspaces for browser QA, not a real SSH/GPU test."""
import json
from pathlib import Path
import tempfile
import threading
from workbench.server import App, make_server
from workbench.cloud import cloud_plan


class LocalFixture(App):
    def dispatch(self,path,body,query):
        result=super().dispatch(path,body,query)
        if path=='/api/info':
            result['hostname']='studio-demo.local'
            result['execution_target']['hostname']='studio-demo.local'
        if path=='/api/execution-targets':result['current']['hostname']='studio-demo.local'
        return result


class RemoteFixture(App):
    def dispatch(self,path,body,query):
        result=super().dispatch(path,body,query)
        if path=='/api/info':
            result['hostname']='remote-gpu-fixture'
            result['execution_target'].update(kind='remote',hostname='remote-gpu-fixture')
        if path=='/api/execution-targets':result['current'].update(kind='remote',hostname='remote-gpu-fixture')
        return result


with tempfile.TemporaryDirectory() as root:
    local=LocalFixture(root+'/local',[root+'/local']);remote=RemoteFixture(root+'/remote',[root+'/remote'])
    from tests.test_kimi_planning import model
    local.model_library.remember_metadata(dict(model(),key='qa:kimi',kind='hub',source='huggingface',name='Kimi Linear QA',metadata_resolved_at=1))
    a=make_server(local,0);b=make_server(remote,0)
    port=b.server_address[1]
    job=local.jobs.create(cloud_plan({'alias':'qa-cloud','root':'/root/qa-workspace','local_port':port},local.jobs.state))
    with local.jobs.connect() as c:c.execute("UPDATE jobs SET status='running' WHERE id=?",(job['id'],))
    (local.jobs.state/job['id']/'cloud-access.json').write_text(json.dumps(dict(port=port,token=remote.token,root='/root/qa-workspace')))
    for server in (a,b):threading.Thread(target=server.serve_forever,daemon=True).start()
    print(json.dumps({'local_url':f'http://127.0.0.1:{a.server_address[1]}/#token={local.token}','remote_port':port,'job':job['id']}),flush=True)
    try:input()
    finally:a.shutdown();b.shutdown();a.server_close();b.server_close()
