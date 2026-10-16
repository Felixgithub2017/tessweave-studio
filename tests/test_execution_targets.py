import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workbench.server import App
from workbench.cloud import cloud_plan
from workbench.server import make_server


class ExecutionTargetsTests(unittest.TestCase):
    def test_inventory_available_over_authenticated_get(self):
        import threading
        from urllib.request import build_opener, ProxyHandler, Request
        with tempfile.TemporaryDirectory() as root:
            app=App(root+'/state',[root]);server=make_server(app,0)
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            try:
                req=Request(f'http://127.0.0.1:{server.server_address[1]}/api/execution-targets',headers={'X-Workbench-Token':app.token})
                with build_opener(ProxyHandler({})).open(req,timeout=3) as response:
                    self.assertEqual(json.load(response)['targets'],[])
            finally:server.shutdown();server.server_close();worker.join()

    def test_inventory_excludes_token_and_unready_tunnel(self):
        with tempfile.TemporaryDirectory() as root:
            app=App(root+'/state',[root])
            job=app.jobs.create(cloud_plan({'alias':'gpu','root':'/root/workspace'},app.jobs.state))
            data=app.dispatch('/api/execution-targets',{}, {})
            self.assertFalse(data['targets'][0]['selectable'])
            with app.jobs.connect() as c:c.execute("UPDATE jobs SET status='running' WHERE id=?",(job['id'],))
            path=app.jobs.state/job['id']/'cloud-access.json'
            path.write_text(json.dumps({'port':8876,'token':'secret-value'}))
            data=app.dispatch('/api/execution-targets',{}, {})
            self.assertTrue(data['targets'][0]['selectable'])
            self.assertNotIn('secret-value',json.dumps(data))
            path.write_text('broken')
            self.assertFalse(app.dispatch('/api/execution-targets',{}, {})['targets'][0]['selectable'])

    def test_host_scoped_state_and_remote_identity(self):
        with tempfile.TemporaryDirectory() as root:
            local=App(root+'/local',[root]);remote=App(root+'/remote',[root])
            local.jobs.create(cloud_plan({'alias':'gpu','root':'/root/workspace'},local.jobs.state))
            self.assertEqual(remote.jobs.list(),[])
            with patch.dict(os.environ,{'TESSWEAVE_REMOTE_WORKSPACE':'1'}):
                identity=remote.dispatch('/api/info',{}, {})['execution_target']
                self.assertEqual(identity['kind'],'remote')
                self.assertIn('downloads',identity['scope'])
                self.assertIn('training',identity['scope'])
                self.assertEqual(identity['state'],str((Path(root)/'remote').resolve()))
