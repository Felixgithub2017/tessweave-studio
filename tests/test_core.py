import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from workbench.inspection import inspect_model, permitted, safetensors_header
from workbench.data import validate_dataset
from workbench.planning import estimate
from workbench.adapters import recipe
from workbench.jobs import Jobs, metrics, redact
from workbench.benchmark import endpoint, run_benchmark
from workbench.server import App, make_server


def fixture(root, model_type="qwen2", quant=False):
    root.mkdir(parents=True)
    config = {"model_type": model_type, "architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 2, "hidden_size": 4, "num_attention_heads": 2, "num_key_value_heads": 1}
    if quant:
        config["quantization_config"] = {"quant_method": "awq"}
    (root/"config.json").write_text(json.dumps(config))
    header = json.dumps({"layer.weight": {"dtype": "F16", "shape": [4,4], "data_offsets": [0,32]}}).encode()
    (root/"model.safetensors").write_bytes(struct.pack("<Q", len(header))+header+b"\0"*32)
    return root


class Core(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = fixture(self.root/"model")

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_no_import(self):
        (self.model/"evil.py").write_text("raise RuntimeError('NEVER IMPORT')")
        m = inspect_model(self.model, [self.root])
        self.assertEqual(m["parameter_estimate"], 16)
        self.assertFalse(m["errors"])

    def test_quant_does_not_guess_parameter_count(self):
        q = fixture(self.root/"quant", quant=True)
        self.assertIsNone(inspect_model(q, [self.root])["parameter_estimate"])

    def test_outside_root(self):
        with self.assertRaises(ValueError):
            permitted(self.root, [self.model])

    def test_symlink_escape(self):
        (self.model/"escape").symlink_to(self.root)
        with self.assertRaises(ValueError):
            permitted(self.model/"escape", [self.model])

    def test_index_traversal(self):
        (self.model/"model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../bad.safetensors"}}))
        with self.assertRaises(ValueError):
            inspect_model(self.model, [self.root])

    def test_header_limit(self):
        p = self.root/"bad.safetensors"
        p.write_bytes(struct.pack("<Q", 2**50))
        with self.assertRaises(ValueError):
            safetensors_header(p)

    def test_missing_shard(self):
        (self.model/"model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
        self.assertTrue(inspect_model(self.model, [self.root])["errors"])

    def test_custom_code_blocks(self):
        p=self.model/"config.json";v=json.loads(p.read_text());v["auto_map"]={"x":"evil.py"};p.write_text(json.dumps(v))
        self.assertTrue(inspect_model(self.model,[self.root])["errors"])

    def test_data(self):
        p=self.root/"sft.jsonl"
        p.write_text(json.dumps({"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]})+'\n')
        self.assertTrue(validate_dataset(p,"sft",[self.root])["valid"])
        self.assertFalse(validate_dataset(p,"dpo",[self.root])["valid"])
        self.assertFalse(validate_dataset(p,"grpo",[self.root])["valid"])

    def test_data_empty(self):
        p=self.root/"empty.jsonl";p.write_text('')
        self.assertFalse(validate_dataset(p,"cpt",[self.root])["valid"])

    def test_resource_ledger(self):
        m=inspect_model(self.model,[self.root]);r=estimate(m,"sft",tuner="full")
        self.assertAlmostEqual(r["known_total_gib"],16*16/2**30)
        self.assertEqual(estimate(m,"inference",seq=3)["kv_gib"],2*2*1*2*3*1*2/2**30)

    def test_unknown_capacity(self):
        self.assertFalse(estimate({"parameter_estimate":None})["known"])

    def test_recipe_arguments_not_shell(self):
        p=self.root/"train; touch evil.jsonl";p.write_text('{"text":"hello"}\n')
        m=inspect_model(self.model,[self.root])
        with patch('workbench.adapters.environment',return_value={"ms-swift":"4.5.3"}):
            r=recipe(m,{"task":"cpt","backend":"swift","dataset":str(p)},
                     {"gpus":[{"name":"H100"}]},[self.root],self.root/'out')
        self.assertIn(str(p.resolve()),r["argv"])
        self.assertFalse(r["blockers"])

    def test_unknown_model_blocks(self):
        m=inspect_model(self.model,[self.root]);m['model_type']='new_custom'
        with patch('workbench.adapters.environment',return_value={"vllm":"x"}):
            r=recipe(m,{"task":"inference","backend":"vllm"},{"gpus":[{"name":"H100"}]},[self.root],self.root/'out')
        self.assertTrue(r["blockers"])

    def test_mlx_service_registration_uses_local_model_id(self):
        app=App(str(self.root/'service-state'),[str(self.root)])
        m=inspect_model(self.model,[self.root])
        plan={'id':'mlx-test','task':'inference','model':m,'dataset':None,
              'output':str(self.root/'unused-output'),'backend':'mlx','request':{'port':18082}}
        with patch.object(app.jobs,'get',return_value={'plan':plan}), patch.object(app.jobs,'launch',return_value={}), patch.object(app.services,'register') as register:
            app.start({'id':'mlx-test','confirm':'mlx-test'})
            self.assertEqual(register.call_args[0][0]['model'],m['path'])

    def test_unload_requires_confirmation_and_owned_job(self):
        app=App(str(self.root/'unload-state'),[str(self.root)])
        service=app.services.register({'url':'http://127.0.0.1:18082','model':'fixture'})
        with self.assertRaises(ValueError):app.dispatch('/api/services/unload',{'id':service['id']},{})
        with self.assertRaises(ValueError):app.dispatch('/api/services/unload',{'id':service['id'],'confirm':service['id']},{})
        service=app.services.register({'url':'http://127.0.0.1:18082','model':'fixture','job_id':'owned'})
        with patch.object(app.jobs,'get',return_value={'plan':{'task':'inference','request':{'port':18082}}}), patch.object(app.jobs,'stop') as stop:
            result=app.dispatch('/api/services/unload',{'id':service['id'],'confirm':service['id']},{})
            stop.assert_called_once_with('owned')
            self.assertEqual(result['status'],'stopping')

    def test_gemma4_mlx_text_recipe_is_scoped(self):
        m=inspect_model(self.model,[self.root]);m['model_type']='gemma4'
        hw={'gpus':[],'apple_unified_memory':True}
        with patch('workbench.adapters.environment',return_value={'mlx-lm':'0.32.0'}):
            r=recipe(m,{'task':'inference','backend':'mlx'},hw,[self.root],self.root/'out')
            self.assertFalse(r['blockers'])
            self.assertTrue(any('TEXT-ONLY' in w for w in r['warnings']))
            q=recipe(m,{'task':'quantize','backend':'mlx'},hw,[self.root],self.root/'out')
            self.assertTrue(q['blockers'])
        with patch('workbench.adapters.environment',return_value={'mlx-lm':'0.25.0'}):
            r=recipe(m,{'task':'inference','backend':'mlx'},hw,[self.root],self.root/'out')
            self.assertTrue(r['blockers'])

    def test_parse_metrics(self):
        self.assertEqual(metrics("INFO {'loss': 1.2, 'epoch': 0.3}")["loss"],1.2)
        self.assertEqual(metrics("__import__('os').system('bad')"),{})
        self.assertNotIn('hf_abcdefghijklmnop',redact('hf_abcdefghijklmnop'))

    def test_loopback_only(self):
        for bad in ['https://example.com','http://169.254.169.254:8000','http://x@localhost:8000','http://localhost:80']:
            with self.assertRaises(ValueError):endpoint(bad)

    def test_real_subprocess_lifecycle(self):
        j=Jobs(self.root/'state');jid='a'*32
        j.create({"id":jid,"blockers":[],"argv":[sys.executable,'-c',"print({'loss':1.25})"],"env":{}})
        j.launch(jid)
        for _ in range(100):
            if j.get(jid)['status'] not in ('running','starting'):break
            time.sleep(.02)
        self.assertEqual(j.get(jid)['status'],'succeeded')
        self.assertEqual(j.logs(jid)['metrics'][0]['loss'],1.25)
        with self.assertRaises(ValueError):j.launch(jid)

    def test_cancel(self):
        j=Jobs(self.root/'state');jid='b'*32
        j.create({"id":jid,"blockers":[],"argv":[sys.executable,'-c','import time;time.sleep(30)'],"env":{}})
        j.launch(jid);j.stop(jid)
        for _ in range(100):
            if j.get(jid)['status']=='cancelled':break
            time.sleep(.02)
        self.assertEqual(j.get(jid)['status'],'cancelled')


class HTTP(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.model=fixture(self.root/'model')
        self.app=App(self.root/'state',[self.root]);self.server=make_server(self.app,0)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.url='http://127.0.0.1:'+str(self.server.server_address[1])

    def tearDown(self):
        self.app.jobs.shutdown();self.server.shutdown();self.server.server_close();self.tmp.cleanup()

    def request(self,path,body=None,token=True,extra=None):
        headers={'X-Workbench-Token':self.app.token} if token else {}
        if body is not None:headers['Content-Type']='application/json'
        headers.update(extra or {})
        return urllib.request.urlopen(urllib.request.Request(self.url+path,headers=headers,data=json.dumps(body).encode() if body is not None else None))

    def test_api_auth(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.request('/api/info',token=False)
        self.assertEqual(e.exception.code,401)

    def test_cross_origin(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.request('/api/info',extra={'Origin':'https://attacker.test'})
        self.assertEqual(e.exception.code,403)

    def test_get_cannot_start(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.request('/api/start')
        self.assertEqual(e.exception.code,405)

    def test_ui_and_scan(self):
        with self.request('/',token=False) as r:self.assertIn(b'MODEL',r.read())
        with self.request('/api/scan',{'path':str(self.model)}) as r:self.assertEqual(json.load(r)['model_type'],'qwen2')

    def test_brand_asset_is_served_as_svg(self):
        with self.request('/', token=False) as r:
            self.assertIn(b'TessWeave Studio', r.read())
        with self.request('/tessweave.svg', token=False) as r:
            self.assertTrue(r.headers['Content-Type'].startswith('image/svg+xml'))
            self.assertIn(b'<svg', r.read())

    def test_secrets_rejected(self):
        with self.assertRaises(urllib.error.HTTPError):self.request('/api/plan',{'password':'never-store'})


if __name__=='__main__':unittest.main()
