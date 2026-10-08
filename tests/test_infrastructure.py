import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from workbench.environments import environment_plan
from workbench.cloud import cloud_plan
from workbench.cloud_runner import BOOTSTRAP
from workbench.server import App


class InfrastructureTests(unittest.TestCase):
    def test_cloud_rejects_injection_and_broad_roots(self):
        for alias in ['-oProxyCommand=evil','host;id','user@host']:
            with self.assertRaises(ValueError):cloud_plan({'alias':alias,'root':'/home/me/wb'},'/tmp')
        for root in ['/','/home','/home/me/../other','relative']:
            with self.assertRaises(ValueError):cloud_plan({'alias':'gpu','root':root},'/tmp')
        with self.assertRaises(ValueError):cloud_plan({'alias':'gpu','root':'/home/me/wb','local_port':True},'/tmp')

    def test_cloud_bootstrap_is_valid_python(self):
        compile("ROOT='/home/me/wb'\nPORT=8876\nDIGEST='abc'\n"+BOOTSTRAP,'bootstrap','exec')
        plan=cloud_plan({'alias':'gpu','root':'/home/me/wb'},'/tmp')
        self.assertEqual(plan['task'],'cloud-workspace')
        self.assertEqual(plan['request']['local_port'],8876)

    def test_environment_paths_and_version(self):
        with tempfile.TemporaryDirectory() as root:
            body={'path':root+'/venv','recipe':'download','manager':'venv'}
            plan=environment_plan(body,[root],root)
            self.assertEqual(plan['request']['packages'],['huggingface-hub','modelscope'])
            self.assertEqual(plan['blockers'],[])
            Path(body['path']).mkdir()
            self.assertTrue(environment_plan(body,[root],root)['blockers'])
            for version in ['1.0;id','latest','--index-url']:
                with self.assertRaises(ValueError):environment_plan(dict(body,version=version),[root],root)
            with self.assertRaises(ValueError):environment_plan(dict(body,path='/etc/newenv'),[root],root)

    def test_environment_api_does_not_install_until_confirmed(self):
        with tempfile.TemporaryDirectory() as root:
            app=App(root+'/state',[root])
            job=app.dispatch('/api/environment/plan',{'path':root+'/env','recipe':'mlx','manager':'venv'},{})
            self.assertFalse(Path(root+'/env').exists())
            with self.assertRaises(ValueError):app.start({'id':job['id']})
            with patch.object(app.jobs,'launch',return_value={'ok':True}) as launch:
                app.start({'id':job['id'],'confirm':job['id']})
                launch.assert_called_once()

    def test_runner_failure_stops_sequence(self):
        from workbench.environment_runner import run
        from subprocess import CompletedProcess
        with tempfile.TemporaryDirectory() as root:
            plan=environment_plan({'path':root+'/env','recipe':'download','manager':'venv'},[root],root)
            manifest=Path(root)/'manifest.json';manifest.write_text(json.dumps(plan))
            with patch('workbench.environment_runner.subprocess.run',return_value=CompletedProcess([],1)) as call:
                with self.assertRaises(RuntimeError):run(manifest)
                self.assertEqual(call.call_count,1)
            self.assertTrue((Path(root)/'environment-commands.jsonl').exists())

    def test_cloud_bootstrap_local_transport_fixture(self):
        """Exercise real bootstrap on an isolated loopback host, not an SSH claim."""
        import io,zipfile,hashlib,socket,subprocess,os,signal
        import workbench
        buffer=io.BytesIO();package=Path(workbench.__file__).parent
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for p in package.glob('*.py'):z.write(p,'workbench/'+p.name)
        data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()[:16]
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as root:
            code='ROOT='+repr(root)+'\nPORT='+str(port)+'\nDIGEST='+repr(digest)+'\n'+BOOTSTRAP
            pid=None
            try:
                result=subprocess.run([sys.executable,'-c',code],input=data,capture_output=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr.decode())
                first=json.loads(result.stdout);pid=first['pid']
                again=subprocess.run([sys.executable,'-c',code],input=data,capture_output=True,timeout=15)
                self.assertEqual(again.returncode,0,again.stderr.decode())
                self.assertEqual(json.loads(again.stdout)['pid'],pid)
                self.assertEqual((Path(root)/'access.json').stat().st_mode&0o777,0o600)
            finally:
                if pid:os.kill(pid,signal.SIGTERM)
