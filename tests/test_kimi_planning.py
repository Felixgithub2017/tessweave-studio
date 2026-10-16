import unittest
from workbench.resources import plan_resources


def model():
    return dict(repo='moonshotai/Kimi-Linear-48B-A3B-Base', parameter_estimate=49122681728,
        config=dict(model_type='kimi_linear', hidden_size=2304, intermediate_size=9216,
            num_hidden_layers=27, num_attention_heads=32, num_key_value_heads=32,
            kv_lora_rank=512, qk_rope_head_dim=64, model_max_length=1048576,
            moe_intermediate_size=1024, num_experts=256, num_experts_per_token=8,
            linear_attn_config=dict(num_heads=32, head_dim=128, short_conv_kernel_size=4,
                full_attn_layers=[4,8,12,16,20,24,27],
                kda_layers=[i for i in range(1,28) if i not in [4,8,12,16,20,24,27]])))


class KimiPlanningTests(unittest.TestCase):
    def test_auto_multigpu_and_ledger(self):
        r=plan_resources(dict(task='inference',gpu='h100',seq=4096),model())
        self.assertEqual(r['planning_coverage']['status'],'estimated')
        self.assertEqual(r['selected']['allocated_gpus'],2)
        self.assertEqual(r['selected']['topology']['pp'],1)
        self.assertAlmostEqual(sum(x['per_gpu_gib'] or 0 for x in r['detailed_budget']['rows']),r['deployment_envelope']['target']['peak_gib'])
        self.assertGreater(r['deployment_envelope']['memory_concurrency'],1)

    def test_context_alias_and_mixed_residents(self):
        r=plan_resources(dict(task='inference',gpu='h100',request_mix='2048:2,8192:1'),model())
        e=r['deployment_envelope']
        self.assertEqual(e['target']['context_tokens'],1048576)
        self.assertEqual(e['context_limit'],1048576)
        self.assertTrue(e['target']['fits_memory'])
        self.assertGreater(e['mixed_requests']['kv_gib'],0)

    def test_no_fake_pp_or_malformed_support(self):
        r=plan_resources(dict(task='inference',gpu='h100',seq=4096,gpu_count=1),model())
        self.assertFalse(r['selected']['feasible_under_assumptions'])
        self.assertEqual(r['selected']['conditional_min_gpus'],2)
        m=model(); m['config']['linear_attn_config']['full_attn_layers']=[4]
        self.assertFalse(plan_resources(dict(task='inference',seq=4096),m)['deployment_support']['analytic_supported'])

    def test_cache_arithmetic_and_tp_replication(self):
        from workbench.hybrid_cache import coefficients
        c=model()['config']
        token,state=coefficients(c,1)
        self.assertEqual(token,7*2*(512+64))
        self.assertEqual(state,20*32*(4*128*128+6*128*3))
        self.assertEqual(coefficients(c,4),(token,state//4))
        r=plan_resources(dict(task='inference',gpu='h100',seq=4097,kv_block_tokens=32),model())
        self.assertAlmostEqual(r['detailed_budget']['peak_gib'],r['deployment_envelope']['target']['peak_gib'])
