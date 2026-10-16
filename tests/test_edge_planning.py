import unittest
from workbench.resources import plan_resources


class EdgePlanningTests(unittest.TestCase):
    def plan(self, **kw):
        return plan_resources(dict(task='inference', deployment_target='rk3588', parameters_b=3,
                                   edge_memory_gib=8, layers=32, hidden=4096, heads=32,
                                   kv_heads=8, **kw))['edge_plan']

    def test_quantization_changes_weights_not_kv(self):
        a,b=self.plan(bits=16),self.plan(bits=4)
        self.assertEqual(a['kv_gib'], b['kv_gib'])
        self.assertGreater(a['weights_gib'], b['weights_gib'])
        self.assertFalse(b['deployment_verified'])

    def test_missing_structure_and_memory_are_unknown(self):
        r=plan_resources(dict(task='inference',deployment_target='rk3588',parameters_b=3))
        self.assertIsNone(r['edge_plan']['fits_budget'])
        self.assertNotIn('selected',r)

    def test_invalid_target_and_bits(self):
        for body in ({'deployment_target':'madeup'},{'bits':3}):
            with self.assertRaises(ValueError):
                plan_resources(dict(task='inference',deployment_target='rk3588')|body)

    def test_special_architecture_no_kv_guess(self):
        r=plan_resources(dict(task='inference',deployment_target='rk3588'),
                         {'config':{'kv_lora_rank':512}})
        self.assertIsNone(r['edge_plan']['kv_gib'])
