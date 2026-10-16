import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench.server import App


class DeploymentCatalogTests(unittest.TestCase):
    def test_standalone_snapshot_without_sibling_checkout(self):
        import sys
        from workbench.engine_deployment import engine
        engine.cache_clear()
        try:
            with patch.dict(sys.modules, {'flow_engine': None}), patch('pathlib.Path.is_file', return_value=False):
                catalog = engine()
                self.assertEqual(catalog.__name__, 'workbench.bundled_deployment_catalog')
                self.assertEqual(len(catalog.catalog()['models']), 25)
                self.assertIn('sha256', catalog.tutorial('Qwen/Qwen3-8B'))
        finally:
            engine.cache_clear()

    def test_shared_catalog_plan_and_tutorial_without_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=App(Path(tmp)/'state',[tmp])
            catalog=app.dispatch('/api/deployment/catalog',{}, {})
            self.assertEqual(len(catalog['models']),25)
            self.assertEqual(len(catalog['hardware']),33)
            choices=app.dispatch('/api/modelchoices',{'include_featured':True},{})
            self.assertEqual(sum(x['key'].startswith('deployment:') for x in choices),25)
            key='deployment:Qwen/Qwen3-8B'
            selected=app.dispatch('/api/modelchoices/select',{'key':key},{})
            self.assertTrue(selected['config']['max_position_embeddings'])
            integrated=app.dispatch('/api/resources',dict(task='inference',model_key=key,parameters_b=8,gpu='h100',gpu_count=1,seq=8192,batch=3,request_mix='2049:2,8192:1'),{})
            self.assertEqual(integrated['deployment_envelope']['mixed_requests']['reserved_tokens'],12320)
            self.assertEqual(integrated['deployment_recipe']['model'],'Qwen/Qwen3-8B')
            result=app.dispatch('/api/deployment/plan',dict(model='Qwen/Qwen3-8B',hardware='apple_silicon',memory_gib=48,context=4096),{})
            self.assertEqual(result['state'],'candidate')
            self.assertFalse(result['device_verified'])
            self.assertFalse(result['backend_model_compatibility_verified'])
            doc=app.dispatch('/api/deployment/tutorial',dict(model='Qwen/Qwen3-8B'),{})
            self.assertIn('sha256',doc['markdown'])
            with self.assertRaises(ValueError):
                app.dispatch('/api/deployment/plan',dict(model='unknown',hardware='apple_silicon'),{})


if __name__=='__main__':unittest.main()
