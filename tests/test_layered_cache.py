import unittest
from workbench.resources import plan_resources
from workbench.layered_cache import layers, token_bytes, cache_gib


class LayeredCacheTests(unittest.TestCase):
    def config(self):
        return dict(model_type='mimo_v2',num_hidden_layers=4,hidden_size=512,
                    intermediate_size=1024,num_attention_heads=8,num_key_value_heads=2,
                    head_dim=64,v_head_dim=32,swa_num_attention_heads=8,
                    swa_num_key_value_heads=4,swa_head_dim=64,swa_v_head_dim=32,
                    hybrid_layer_pattern=[0,0,1,1],max_position_embeddings=32768)

    def test_asymmetric_kv_and_worst_pipeline_stage(self):
        cfg=self.config()
        self.assertEqual(token_bytes(cfg),2*(2*2+2*4)*(64+32))
        self.assertEqual(token_bytes(cfg,tp=2,pp=2),2*2*2*96)
        # PP must use the busiest stage, not the average layer coefficient.
        self.assertGreater(token_bytes(cfg,pp=2),token_bytes(cfg)/2)
        self.assertEqual(token_bytes(cfg,tp=8),4*2*96)

    def test_invalid_pattern_and_layout_fail_closed(self):
        cfg=self.config();cfg['hybrid_layer_pattern']=[0,1]
        self.assertIsNone(layers(cfg))
        cfg=self.config();cfg['hybrid_layer_pattern'][0]=2
        self.assertIsNone(layers(cfg))
        with self.assertRaises(ValueError):token_bytes(self.config(),tp=3)

    def test_cached_mimo_configs_and_multicard_reconciliation(self):
        from workbench.model_metadata import bundled_records
        count=0
        for repo,record in bundled_records().items():
            cfg=record.get('config') or {};cfg=cfg.get('text_config') or cfg
            if cfg.get('model_type') not in ('mimo_v2','mimo_v2_flash'):continue
            with self.subTest(repo=repo):
                self.assertIsNotNone(layers(cfg))
                # Explicit synthetic total weight, NOT a claim of checkpoint size.
                model=dict(repo=repo,config=cfg)
                r=plan_resources(dict(task='inference',parameters_b=310,seq=4097,
                                      bits=16,gpu='h100'),model)
                self.assertTrue(r['selected']['feasible_under_assumptions'])
                self.assertGreater(r['selected']['allocated_gpus'],1)
                self.assertAlmostEqual(r['detailed_budget']['peak_gib'],
                                       r['deployment_envelope']['target']['peak_gib'])
                count+=1
        self.assertGreaterEqual(count,7)

    def test_ragged_context_and_mixed_lengths_reconcile(self):
        cfg=self.config();model=dict(config=cfg,parameter_estimate=1_000_000_000)
        report=plan_resources(dict(task='inference',seq=4097,batch=3,gpu='h100',
                                   request_mix='4097:2,8193:1'),model)
        self.assertEqual(report['planning_coverage']['status'],'estimated')
        env=report['deployment_envelope']
        self.assertAlmostEqual(report['detailed_budget']['peak_gib'],env['target']['peak_gib'])
        self.assertAlmostEqual(env['mixed_requests']['kv_gib'],
                               cache_gib(cfg,2,4097)+cache_gib(cfg,1,8193))
        self.assertEqual(env['cache_adapter'],'layered_full_history')
