import unittest
from unittest.mock import patch
from workbench.model_metadata import bundled_records, with_bundled_metadata
from tools.cache_modelscope_metadata import FILES


class MetadataCacheTests(unittest.TestCase):
    def test_modelscope_priority_without_network(self):
        with patch('workbench.hub.get_json',side_effect=AssertionError('offline')):
            row=with_bundled_metadata({'repo':'Qwen/Qwen3.5-9B'})
        self.assertEqual(row['metadata_source'],'modelscope')
        self.assertEqual(row['config']['max_position_embeddings'],262144)
        self.assertEqual(row['parameter_estimate'],9653104368)
        self.assertIn('Revision=',row['config_url'])

    def test_user_config_preserved(self):
        row=with_bundled_metadata({'repo':'Qwen/Qwen3.5-9B','config':{'custom':True}})
        self.assertEqual(row['config'],{'custom':True})

    def test_allowlist_never_weights_or_executable_code(self):
        self.assertEqual(FILES,{'config.json','generation_config.json','tokenizer_config.json',
                               'preprocessor_config.json','processor_config.json',
                               'quantization_config.json','model_index.json'})
        self.assertGreater(len(bundled_records()),300)
