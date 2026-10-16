import unittest
from workbench.resources import plan_resources


class DeploymentEnvelopeTests(unittest.TestCase):
    def test_mixed_lengths_round_each_request_and_share_fixed(self):
        e=self.plan(seq=8192,batch=3,request_mix='2049:2,8192:1')
        m=e['mixed_requests']
        self.assertEqual(m['concurrency'],3)
        self.assertEqual(m['reserved_tokens'],2*2064+8192)
        self.assertAlmostEqual(m['kv_gib'],m['reserved_tokens']*e['kv_bytes_per_token_per_rank']/2**30)
        self.assertLess(m['peak_gib'],e['target']['peak_gib'])
        self.assertEqual(m['fixed_gib'],e['concurrency_memory'][0]['fixed_gib'])
        self.assertGreaterEqual(m['maximum_whole_groups'],1)
        self.assertFalse(self.plan(request_mix='65536:1')['mixed_requests']['within_context_limit'])
        for mix in ('0:2','4096:-1','4096:1.5','oops','4096:65537'):
            with self.assertRaises(ValueError):self.plan(request_mix=mix)

    def test_declared_default_and_concurrency_ledger(self):
        e = self.plan()
        self.assertEqual(e['target']['context_tokens'], 32768)
        self.assertEqual(e['declared_context']['context_tokens'], 32768)
        rows = e['concurrency_memory']
        one = next(x for x in rows if x['concurrency'] == 1)
        two = next(x for x in rows if x['concurrency'] == 2)
        self.assertAlmostEqual(two['kv_gib'], 2 * one['kv_gib'])
        self.assertEqual(two['fixed_gib'], one['fixed_gib'])
        self.assertAlmostEqual(two['extra_vs_one_gib'], two['peak_gib'] - one['peak_gib'])
        self.assertEqual(self.plan(seq=4096)['target']['context_tokens'], 4096)
        self.assertEqual(self.plan(seq=4096)['declared_context']['context_tokens'], 32768)

    def test_context_delta_holds_layout_and_concurrency_fixed(self):
        e = self.plan(seq=8193, batch=2)
        x = e['context_memory_delta']
        self.assertEqual(x['baseline_tokens'], 2048)
        self.assertAlmostEqual(x['kv_delta_gib'], 2*(8208-2048)*e['kv_bytes_per_token_per_rank']/2**30)
        self.assertAlmostEqual(x['peak_delta_gib'], x['kv_delta_gib']+x['activation_delta_gib'])
        self.assertEqual(self.plan(seq=2048)['context_memory_delta']['peak_delta_gib'], 0)
        self.assertLess(self.plan(seq=512)['context_memory_delta']['kv_delta_gib'], 0)

    def plan(self, cfg=None, **kw):
        config = dict(num_hidden_layers=32, hidden_size=4096,
                      num_attention_heads=32, num_key_value_heads=8,
                      max_position_embeddings=32768)
        config.update(cfg or {})
        return plan_resources(dict(task='inference', parameters_b=7, gpu_count=1,
                                   gpu='h100', **kw), {'config':config})['deployment_envelope']

    def test_monotonic_and_boundaries(self):
        e = self.plan()
        self.assertFalse(e['blockers'])
        self.assertIsNone(e['slo_concurrency'])
        counts = [x['memory_concurrency'] for x in e['curve']]
        self.assertEqual(counts, sorted(counts, reverse=True))
        c = e['memory_concurrency']
        self.assertTrue(self.plan(batch=c)['target']['fits_memory'])
        self.assertFalse(self.plan(batch=c+1)['target']['fits_memory'])

    def test_context_not_infinite_and_unknown(self):
        self.assertTrue(self.plan(seq=65536)['blockers'])
        self.assertIsNone(self.plan(context_limit=0)['context_at_target_concurrency'])
        self.assertLessEqual(self.plan()['context_at_target_concurrency'],32768)

    def test_special_architecture_not_guessed(self):
        for cfg in ({'kv_lora_rank':512}, {'layer_types':['linear_attention']}, {'vision_config':{'x':1}}):
            e = self.plan(cfg)
            self.assertIsNone(e['memory_concurrency'])
            self.assertIsNone(e['minimum_gpus_target'])
            self.assertTrue(e['blockers'])

    def test_formula_and_oom(self):
        e = self.plan()
        self.assertEqual(e['kv_bytes_per_token_per_rank'], 2*2*32*8*128)
        self.assertAlmostEqual(sum(x['gib'] for x in e['ledger']), e['target']['peak_gib'])
        self.assertTrue(self.plan(other_models_gib=100)['blockers'])

    def test_page_padding_and_invalid(self):
        a, b = self.plan(seq=4096), self.plan(seq=4097)
        self.assertGreater(b['ledger'][1]['gib'], a['ledger'][1]['gib'])
        for key, value in [('kv_block_tokens',0), ('context_limit',float('nan'))]:
            with self.assertRaises(ValueError): self.plan(**{key:value})
