import io
import unittest
from unittest.mock import patch
from workbench import hub


class ModelDetailTests(unittest.TestCase):
    def test_modelscope_uses_master_and_native_metadata(self):
        urls=[]
        def read(url):
            urls.append(url)
            return io.BytesIO(b'{}' if 'config.json' in url else b'# Model card')
        meta={'Code':200,'Data':{'ModelInfos':{'safetensor':{'model_size':42}},'Tasks':[{'Name':'text-generation'}],'License':'MIT'}}
        with patch.object(hub,'get_json',return_value=meta),patch.object(hub,'open_url',side_effect=read):
            result=hub.model_detail('modelscope','org/model')
        self.assertEqual(result['revision'],'master')
        self.assertEqual(result['parameters'],42)
        self.assertEqual(result['license'],'MIT')
        self.assertTrue(all('Revision=master' in url for url in urls))
        self.assertEqual(result['readme'],'# Model card')

    def test_transport_explicitly_disables_proxy_discovery(self):
        with patch.object(hub,'check_url'),patch.object(hub.urllib.request,'build_opener') as build:
            hub.open_url('https://www.modelscope.cn/test')
        handlers=build.call_args.args
        self.assertIsInstance(handlers[0],hub.urllib.request.ProxyHandler)
        self.assertEqual(handlers[0].proxies,{})
        self.assertIsInstance(handlers[1],hub.Redirect)

    def test_metadata_and_pinned_safe_preview(self):
        urls = []
        def read(url):
            urls.append(url)
            return io.BytesIO(b'{"model_type":"qwen3","hidden_size":1024}' if url.endswith('config.json') else b'# Model\n- Number of Parameters: 0.6B\n<script>unsafe</script>')
        with patch.object(hub, 'get_json', return_value={'sha':'abc123', 'safetensors':{'total':751632384}, 'cardData':{'license':'apache-2.0'}}), patch.object(hub, 'open_url', side_effect=read):
            result = hub.model_detail('huggingface','Qwen/Qwen3-0.6B')
        self.assertEqual(result['parameters'],751632384)
        self.assertEqual(result['parameter_claims'],['- Number of Parameters: 0.6B'])
        self.assertEqual(result['config']['hidden_size'],1024)
        self.assertIn('<script>',result['readme'])  # UI renders via textContent, not HTML.
        self.assertTrue(all('/resolve/abc123/' in u for u in urls))

    def test_missing_files_does_not_destroy_metadata(self):
        with patch.object(hub,'get_json',return_value={'safetensors':{'total':0}}), patch.object(hub,'open_url',side_effect=ValueError('missing')):
            result=hub.model_detail('huggingface','org/70B')
        self.assertIsNone(result['parameters'])
        self.assertEqual(len(result['errors']),2)
        self.assertEqual(result['config'],{})

    def test_input_validation(self):
        for repo, revision in [('org/../x','main'),('org/model','../secret')]:
            with self.assertRaises(ValueError):
                hub.model_detail('huggingface',repo,revision)
