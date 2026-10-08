import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench.model_library import ModelLibrary
from workbench.server import App
from test_core import fixture


class LibraryTests(unittest.TestCase):
    def test_scan_persist_and_select(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model=fixture(root/'model');app=App(root/'state',[root])
            app.api('/api/scan',{'path':str(model)},{})
            rows=ModelLibrary(root/'state').list()
            self.assertEqual(len(rows),1)
            r=app.dispatch('/api/modelchoices/select',{'key':rows[0]['key']},{})
            self.assertEqual(r['path'],str(model.resolve()))
            self.assertGreater(r['parameter_estimate'],0)

    def test_remote_selection_not_name_guess(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);app=App(root/'state',[root])
            app.model_library.catalog({'source':'huggingface','models':[{'id':'Org/777B'}]})
            with patch('workbench.hub.get_json',side_effect=[{}, {'model_type':'llama'}]):
                r=app.dispatch('/api/modelchoices/select',{'key':'huggingface:Org/777B'},{})
            self.assertIsNone(r['parameter_estimate'])
            with patch('workbench.hub.get_json',side_effect=[{'safetensors':{'total':123}}, {'model_type':'llama'}]):
                r=app.dispatch('/api/modelchoices/select',{'key':'huggingface:Org/777B','refresh':True},{})
            self.assertEqual(r['parameter_estimate'],123)
