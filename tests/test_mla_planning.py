import unittest
from workbench.model_metadata import with_bundled_metadata
from workbench.resources import plan_resources
from workbench.mla_planning import cache_gib,token_bytes


class MLAPlanning(unittest.TestCase):
    def model(self,repo):
        return with_bundled_metadata(dict(repo=repo))

    def test_deepseek_multi_node_and_ledger(self):
        m=self.model('deepseek-ai/DeepSeek-V3')
        for bits in (4,8,16):
            r=plan_resources(dict(task='inference',parameters_b=685,bits=bits,gpu='h100'),m)
            self.assertEqual(r['planning_coverage']['status'],'estimated')
            self.assertTrue(r['selected']['feasible_under_assumptions'])
            self.assertAlmostEqual(r['detailed_budget']['peak_gib'],r['deployment_envelope']['target']['peak_gib'])
            if bits==16:self.assertGreater(r['selected']['nodes'],1)

    def test_mla_not_divided_by_tensor_parallel(self):
        m=self.model('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct')
        cfg=m['config']
        self.assertEqual(token_bytes(cfg),27*2*(512+64))
        rows=[]
        for n in (1,2):
            r=plan_resources(dict(task='inference',gpu_count=n,seq=4096,gpu='h100'),m)
            rows.append(next(x['per_gpu_gib'] for x in r['detailed_budget']['rows'] if x['item']=='kv'))
        self.assertEqual(rows[0],rows[1])
        self.assertAlmostEqual(rows[0],cache_gib(cfg,1,4096))

    def test_dsa_index_and_mixed_requests(self):
        m=self.model('deepseek-ai/DeepSeek-V3.2');cfg=m['config']
        self.assertEqual(token_bytes(cfg),61*(1152+260))
        r=plan_resources(dict(task='inference',parameters_b=686,seq=4096,bits=8,
                             request_mix='2048:2,8192:1'),m)
        mix=r['deployment_envelope']['mixed_requests']
        self.assertAlmostEqual(mix['kv_gib'],cache_gib(cfg,2,2048)+cache_gib(cfg,1,8192))

    def test_unknown_is_not_mla(self):
        from workbench.mla_planning import supported
        cfg=dict(self.model('deepseek-ai/DeepSeek-V3')['config'],model_type='future')
        self.assertFalse(supported(cfg))

    def test_standard_head_dim_and_sliding_upper_bound(self):
        from workbench.deployment_recipes import assessment
        cfg=dict(model_type='gpt_oss',num_hidden_layers=24,hidden_size=2880,
                 num_attention_heads=64,num_key_value_heads=8,head_dim=64,
                 num_local_experts=32,num_experts_per_tok=4,intermediate_size=2880,
                 layer_types=['sliding_attention','full_attention']*12)
        m=dict(config=cfg,parameter_estimate=21_000_000_000)
        self.assertTrue(assessment(m)['analytic_supported'])
        r=plan_resources(dict(task='inference',seq=4096,gpu='h100'),m)
        kv=next(x['per_gpu_gib'] for x in r['detailed_budget']['rows'] if x['item']=='kv')
        self.assertEqual(kv,4*4096*24*8*64/2**30)
        self.assertAlmostEqual(r['detailed_budget']['peak_gib'],r['deployment_envelope']['target']['peak_gib'])
        cfg['layer_types'][0]='linear_attention'
        self.assertFalse(assessment(m)['analytic_supported'])
