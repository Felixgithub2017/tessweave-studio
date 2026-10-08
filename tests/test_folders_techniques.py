import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_core import fixture
from workbench.inspection import browse_directories,scan_directory,inspect_model
from workbench.techniques import technique_guide,COMPRESSION
from workbench.adapters import recipe


class Folders(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.models=self.root/'models';self.models.mkdir()
        fixture(self.models/'one');fixture(self.models/'two')
    def tearDown(self):self.temp.cleanup()
    def test_browse_and_discover(self):
        r=browse_directories(self.models,[self.models])
        self.assertIsNone(r['parent'])
        self.assertEqual(len(r['directories']),2)
        report=scan_directory(self.models,[self.models])
        self.assertEqual(len(report['models']),2)
        self.assertFalse(report['limit_reached'])
    def test_root_is_model(self):
        self.assertEqual(len(scan_directory(self.models/'one',[self.models])['models']),1)
    def test_outside_symlink_not_browsable(self):
        (self.models/'escape').symlink_to(self.root)
        r=browse_directories(self.models,[self.models])
        self.assertEqual(r['skipped'],1)
        with self.assertRaises(ValueError):browse_directories(self.models/'escape',[self.models])
    def test_limits_and_file_rejection(self):
        self.assertTrue(scan_directory(self.models,[self.models],limit=1)['limit_reached'])
        self.assertTrue(scan_directory(self.models,[self.models],max_directories=1)['limit_reached'])
        with self.assertRaises(ValueError):scan_directory(self.models/'one'/'config.json',[self.models])
    def test_empty_folder(self):
        empty=self.models/'empty';empty.mkdir()
        self.assertEqual(scan_directory(empty,[self.models])['models'],[])

    def test_appledouble_sidecars_are_not_weights_or_indexes(self):
        model=self.models/'one'
        baseline=inspect_model(model,[self.models])
        for name in ('._model.safetensors','._model.safetensors.index.json','._config.json'):
            (model/name).write_bytes(b'\x00\x05\x16\x07\xb0'+b'\x00'*64)
        result=inspect_model(model,[self.models])
        self.assertEqual(result['fingerprint'],baseline['fingerprint'])
        self.assertEqual(result['weight_bytes'],baseline['weight_bytes'])
        self.assertEqual(result['tensor_count'],baseline['tensor_count'])

    def test_invalid_real_index_reports_filename(self):
        model=self.models/'one'
        (model/'model.safetensors.index.json').write_bytes(b'\xb0')
        with self.assertRaisesRegex(ValueError,'model.safetensors.index.json'):
            inspect_model(model,[self.models])


class Guides(unittest.TestCase):
    def test_all_routes_explained(self):
        for backend in ('vllm','sglang','mlx'):
            for method in COMPRESSION:
                r=technique_guide(backend,method)
                for t in r['inference']+[r['compression']]:
                    for key in ('flow','principle','execution','benefit','limits','verify','source','status'):
                        self.assertTrue(t[key])
                self.assertIn('MTP',r['not_enabled'])
    def test_unknown_and_no_shared_mutation(self):
        with self.assertRaises(ValueError):technique_guide('unknown','awq')
        r=technique_guide();r['inference'][0]['name']='changed'
        self.assertNotEqual(technique_guide()['inference'][0]['name'],'changed')
    def test_plan_includes_contract_not_fake_speed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m=inspect_model(fixture(root/'model'),[root])
            with patch('workbench.adapters.environment',return_value={'vllm':'test-version'}):
                p=recipe(m,{'task':'inference','backend':'vllm'}, {'gpus':[{'name':'H100'}]},[root],root/'output')
            self.assertIsNone(p['technique_guide']['command_contract']['measured_speedup'])
            self.assertFalse(p['technique_guide']['command_contract']['quantization_requested'])
            self.assertNotIn('--enable-prefix-caching',p['argv'])
