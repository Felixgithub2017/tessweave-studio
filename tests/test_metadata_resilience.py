import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from workbench.model_library import ModelLibrary
from workbench.model_metadata import resolve
from workbench.server import App


class MetadataResilienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.lib=ModelLibrary(self.root)
        self.lib.catalog({'source':'huggingface','models':[{'id':'Test/model'}]})
        self.entry=self.lib.get('huggingface:Test/model')

    def test_timeout_returns_structured_status(self):
        with patch('workbench.hub.get_json',side_effect=URLError('timed out')) as fetch:
            result=resolve(self.lib,self.entry)
        self.assertEqual(result['metadata_status'],'unavailable')
        self.assertEqual(len(result['metadata_errors']),4)
        self.assertEqual(fetch.call_count,4)
        self.assertIsNone(result['parameter_estimate'])

    def test_mirror_fallback_and_persistent_cache(self):
        def fetch(url,**kwargs):
            if 'huggingface.co' in url:raise URLError('timed out')
            return {'model_type':'llama','hidden_size':64} if 'config.json' in url else {'safetensors':{'total':123}}
        with patch('workbench.hub.get_json',side_effect=fetch):
            result=resolve(self.lib,self.entry)
        self.assertEqual(result['metadata_source'],'hf-mirror')
        self.assertEqual(result['parameter_estimate'],123)
        self.lib.catalog({'source':'huggingface','models':[{'id':'Test/model'}]})
        lib=ModelLibrary(self.root)
        with patch('workbench.hub.get_json',side_effect=AssertionError('Must use cache')) as fetch:
            cached=resolve(lib,lib.get(self.entry['key']))
            fetch.assert_not_called()
        self.assertEqual(cached['config']['hidden_size'],64)
        self.assertEqual(cached['metadata_origin'],'cache')

    def test_config_survives_optional_info_failure(self):
        def fetch(url,**kwargs):
            if 'config.json' in url:return {'model_type':'llama'}
            raise URLError('timed out')
        with patch('workbench.hub.get_json',side_effect=fetch):
            result=resolve(self.lib,self.entry)
        self.assertEqual(result['metadata_status'],'partial')
        self.assertEqual(result['config']['model_type'],'llama')

    def test_mimo_snapshot_and_calculation_need_no_network(self):
        app=App(self.root/'state',[self.root])
        key='huggingface:XiaomiMiMo/MiMo-V2.6-Pro-RL'
        with patch('workbench.hub.get_json',side_effect=AssertionError('Offline')) as fetch:
            m=app.dispatch('/api/modelchoices/select',{'key':key},{})
            r=app.dispatch('/api/resources',{'model_key':key,'parameters_b':1020,'max_search_gpus':8},{})
            fetch.assert_not_called()
        self.assertEqual(m['config']['num_hidden_layers'],70)
        self.assertEqual(m['config']['n_routed_experts'],384)
        self.assertEqual(m['modality'],'multimodal')
        self.assertIsNone(m['parameter_estimate'])
        self.assertTrue(r['training_design']['blockers'])

    def test_failed_refresh_retains_snapshot(self):
        entry=self.lib.get('huggingface:XiaomiMiMo/MiMo-V2.6-Pro-RL')
        old=resolve(self.lib,entry)
        with patch('workbench.hub.get_json',side_effect=URLError('timed out')):
            result=resolve(self.lib,old,refresh=True)
        self.assertEqual(result['config'],old['config'])
        self.assertTrue(result['metadata_errors'])

    def test_resource_call_does_not_fetch_unknown_model(self):
        app=App(self.root/'state',[self.root])
        app.model_library.catalog({'source':'huggingface','models':[{'id':'Test/model'}]})
        with patch('workbench.hub.get_json') as fetch:
            app.dispatch('/api/resources',{'model_key':self.entry['key'],'parameters_b':1,'max_search_gpus':8},{})
            fetch.assert_not_called()
