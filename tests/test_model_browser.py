import json
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench.featured_models import featured_models
from workbench.hub import catalog


class ModelBrowserTests(unittest.TestCase):
    def test_offline_snapshot_and_types(self):
        data=json.loads((Path(__file__).parents[1]/'workbench/knowledge/model-browser.json').read_text())
        self.assertGreaterEqual(len(data['models']),350)
        tags={m.get('pipeline_tag') for m in data['models']}
        self.assertIn('text-generation',tags)
        self.assertIn('image-text-to-text',tags)
        rows=featured_models()
        self.assertEqual(len({r['key'] for r in rows}),len(rows))
        self.assertTrue(all(r['parameter_estimate'] is None for r in rows))

    def test_global_text_search_no_author_constraint(self):
        with patch('workbench.hub.get_json',return_value=[]) as get:
            catalog('huggingface',author='',search='some-new-model',limit=100)
            self.assertNotIn('author=',get.call_args[0][0])
            self.assertIn('search=some-new-model',get.call_args[0][0])
        with self.assertRaises(ValueError):catalog('huggingface',author='',search='')
