import unittest
from workbench.model_metadata import with_bundled_metadata
from workbench.resources import plan_resources
from workbench.hybrid_cache import moe_scratch_bytes


class SmallModelAdapters(unittest.TestCase):
    REPOS=('XiaomiMiMo/MiMo-7B-Base','XiaomiMiMo/MiMo-7B-RL',
           'baidu/ERNIE-4.5-0.3B-PT','baidu/ERNIE-4.5-0.3B-Base-PT',
           'baidu/ERNIE-4.5-21B-A3B-PT','baidu/ERNIE-4.5-21B-A3B-Thinking')

    def test_cached_models_precision_concurrency_and_ragged_pages(self):
        for repo in self.REPOS:
            m=with_bundled_metadata(dict(repo=repo))
            self.assertLess(m['parameter_estimate'],50e9)
            for bits in (4,8,16):
                for batch in (1,4):
                    with self.subTest(repo=repo,bits=bits,batch=batch):
                        r=plan_resources(dict(task='inference',gpu='h100',bits=bits,
                                              batch=batch,seq=4097),m)
                        self.assertEqual(r['planning_coverage']['status'],'estimated')
                        self.assertTrue(r['selected']['feasible_under_assumptions'])
                        self.assertAlmostEqual(r['detailed_budget']['peak_gib'],
                                               r['deployment_envelope']['target']['peak_gib'])

    def test_ernie_router_fields_are_not_ignored(self):
        m=with_bundled_metadata(dict(repo=self.REPOS[-2]))
        self.assertEqual(moe_scratch_bytes(m['config']),4*64+4*6*1536)

    def test_future_special_cache_is_not_whitelisted(self):
        from workbench.standard_attention import supported
        m=with_bundled_metadata(dict(repo=self.REPOS[0]))
        cfg=dict(m['config'],ssm_cfg={'state':128})
        self.assertFalse(supported(cfg))
