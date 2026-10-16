import unittest
from workbench.layered_cache import layers, token_bytes
from workbench.model_metadata import with_bundled_metadata
from workbench.resources import plan_resources
from workbench.hybrid_cache import moe_scratch_bytes


class Gemma4Planning(unittest.TestCase):
    def model(self,name):
        return with_bundled_metadata(dict(repo='google/'+name))

    def test_global_and_sliding_shapes(self):
        cfg=self.model('gemma-4-26B-A4B-it')['config']
        shapes=layers(cfg)
        self.assertEqual(shapes[0]['kv_heads'],8)
        self.assertEqual(shapes[0]['key_dim'],256)
        self.assertEqual(shapes[5]['kv_heads'],2)
        self.assertEqual(shapes[5]['key_dim'],512)
        self.assertEqual(token_bytes(cfg),25*4*8*256+5*4*2*512)
        self.assertEqual(moe_scratch_bytes(cfg),4*128+4*8*704)

    def test_four_checkpoints_generate_reconciled_multicard_plans(self):
        for name in ('gemma-4-E2B-it','gemma-4-E4B-it','gemma-4-26B-A4B-it','gemma-4-31B-it'):
            for bits in (4,8,16):
                with self.subTest(name=name,bits=bits):
                    m=self.model(name)
                    r=plan_resources(dict(task='inference',gpu='h100',seq=4097,batch=4,
                        bits=bits,parameters_b=31 if '31B' in name else m['parameter_estimate']/1e9),m)
                    self.assertEqual(r['planning_coverage']['status'],'estimated')
                    self.assertTrue(r['selected']['feasible_under_assumptions'])
                    self.assertAlmostEqual(r['detailed_budget']['peak_gib'],r['deployment_envelope']['target']['peak_gib'])

    def test_shared_kv_not_discounted_and_invalid_layers_rejected(self):
        cfg=self.model('gemma-4-E2B-it')['config']
        self.assertEqual(len(layers(cfg)),35)
        self.assertEqual(token_bytes(cfg),28*4*256+7*4*512)
        cfg=dict(cfg,layer_types=['unknown']*35)
        self.assertIsNone(layers(cfg))
