import tempfile
import unittest
from pathlib import Path
from test_core import fixture
from workbench.inspection import inspect_model
from workbench.model_reference import model_identity
from workbench.resources import plan_resources


class ModelReferenceTests(unittest.TestCase):
    def test_release_metadata_and_exact_count_are_separate(self):
        model={'name':'source-bf16','model_card_title':'GLM-5.3-Flash-BF16',
               'model_type':'glm5_next','parameter_estimate':321323031390}
        identity=model_identity(model)
        self.assertEqual(identity['official_name'],'GLM-5.3-Flash-BF16')
        self.assertEqual(identity['stored_parameter_count'],321323031390)
        self.assertEqual(identity['official']['total_parameters'],320000000000)
        self.assertIsNone(identity['official']['training']['gpu_count'])
        self.assertFalse(identity['checkpoint_verified'])
        identity['official']['training']['gpu_count']=123
        self.assertIsNone(model_identity(model)['official']['training']['gpu_count'])

    def test_never_infer_identity_from_folder_or_size(self):
        self.assertIsNone(model_identity({'name':'GLM-5.3-Flash-BF16',
            'parameter_estimate':320000000000})['official'])
        self.assertIsNone(model_identity({'model_card_title':'GLM-5.3-Flash-BF16',
            'model_type':'llama'})['official'])

    def test_readme_title_and_planner(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder=fixture(root/'model')
            (folder/'README.md').write_text('---\nlanguage: en\n---\n# Example model\n',encoding='utf-8')
            model=inspect_model(folder,[root])
            self.assertEqual(model['model_card_title'],'Example model')
            r=plan_resources({'parameters_b':1},model)
            self.assertEqual(r['model_identity']['display_name'],'Example model')
            self.assertIsNone(r['model_identity']['official'])

    def test_moe_is_not_production_recommendation(self):
        r=plan_resources({'parameters_b':1},{'config':{'n_routed_experts':8}})
        self.assertTrue(r['planning_scope']['limited'])
        self.assertFalse(r['planning_scope']['official_equivalent'])
