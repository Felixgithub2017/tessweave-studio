import unittest
from workbench.deployment_recipes import assessment, RECIPES
from workbench.resources import plan_resources


class RecipeTests(unittest.TestCase):
    def test_identity_is_exact_not_family_or_folder(self):
        repo='XiaomiMiMo/MiMo-V2.6-Pro-RL'
        self.assertEqual(len(assessment({'repo':repo})['recipes']),1)
        for model in ({'path':'/models/'+repo}, {'repo':repo+'-other'}, {'name':repo}):
            self.assertFalse(assessment(model)['recipes'])

    def test_unknown_and_moe_require_adapters(self):
        for config in ({'model_type':'future_moe'},
                       {'model_type':'qwen3','num_experts':128},
                       {'model_type':'llama','vision_config':{'hidden_size':12}}):
            self.assertFalse(assessment({'config':config})['analytic_supported'])
        self.assertTrue(assessment({'config':{'model_type':'llama'}})['analytic_supported'])

    def test_reference_not_executable_or_locally_verified(self):
        for recipe in RECIPES:
            self.assertFalse(recipe['locally_verified'])
            self.assertFalse(recipe['executable'])
            self.assertTrue(recipe['source'].startswith('https://'))

    def test_no_legacy_recommendation_for_moe(self):
        model={'repo':'zai-org/GLM-5.3','config':{
            'model_type':'new_moe','num_hidden_layers':32,'hidden_size':4096,
            'num_attention_heads':32,'num_key_value_heads':8}}
        report=plan_resources({'task':'inference','parameters_b':7},model)
        self.assertIsNone(report['deployment_envelope']['minimum_gpus_target'])
        self.assertIsNone(report['selected']['topology'])
        self.assertIsNone(report['placement'])
        self.assertTrue(report['unverified_capacity_audit'])
        for candidate in report['candidates']:
            self.assertFalse(candidate['feasible_under_assumptions'])
            self.assertIsNone(candidate['conditional_min_gpus'])

    def test_training_not_changed(self):
        report=plan_resources({'task':'train','parameters_b':7})
        self.assertNotIn('deployment_support', report)
