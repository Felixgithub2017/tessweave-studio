import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench.server import App


class Gemma3PlanningTests(unittest.TestCase):
    def test_screenshot_configuration_and_default_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=App(Path(tmp)/'state',[tmp]);key='deployment:google/gemma-3-27b-it'
            m=app.dispatch('/api/modelchoices/select',{'key':key},{})
            self.assertEqual(m['config']['max_position_embeddings'],131072)
            r=app.dispatch('/api/resources',dict(task='inference',model_key=key,parameters_b=27,
                gpu='h100',gpu_count=0,bits=16,seq=4096,batch=1,activation_gib=8,workspace_gib=4),{})
            self.assertEqual(r['planning_coverage']['status'],'estimated')
            e=r['deployment_envelope']
            self.assertTrue(e['target']['fits_memory'])
            self.assertEqual(e['allocated_gpus'],2)
            blocked=app.dispatch('/api/resources',dict(task='inference',model_key=key,parameters_b=27,gpu='h100',gpu_count=1,bits=16,seq=4096,batch=1,activation_gib=8,workspace_gib=4),{})
            self.assertIn('未找到可容纳布局',' '.join(blocked['deployment_envelope']['blockers']))
            self.assertIn('保守',r['kv_basis'])
            self.assertIsNotNone(e['memory_concurrency'])
            one=app.dispatch('/api/resources',dict(task='inference',model_key=key,parameters_b=27,gpu='h100',gpu_count=1,bits=16,seq=4096,batch=1,activation_gib=2,workspace_gib=4),{})
            self.assertEqual(one['planning_coverage']['status'],'estimated')
            self.assertTrue(one['deployment_envelope']['target']['fits_memory'])
            self.assertAlmostEqual(one['deployment_envelope']['target']['peak_gib'],58.22891902923584,places=6)

    def test_refresh_uses_resolver_and_cached_result_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=App(Path(tmp)/'state',[tmp]);key='deployment:google/gemma-3-27b-it'
            m=app.dispatch('/api/modelchoices/select',{'key':key},{})
            def fake(lib,selected,**kwargs):
                self.assertTrue(kwargs['refresh']);self.assertEqual(kwargs['source'],'huggingface')
                selected={**selected,'config':{**m['config'],'max_position_embeddings':65536},'metadata_resolved_at':1}
                lib.remember_metadata(selected);return selected
            with patch('workbench.model_metadata.resolve',side_effect=fake) as resolve:
                app.dispatch('/api/modelchoices/select',{'key':key,'refresh':True,'source':'huggingface'},{})
                self.assertEqual(resolve.call_count,1)
            self.assertEqual(app.dispatch('/api/modelchoices/select',{'key':key},{})['config']['max_position_embeddings'],65536)
