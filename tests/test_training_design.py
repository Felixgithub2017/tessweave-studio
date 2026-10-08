import unittest
from workbench.resources import plan_resources


BASE=dict(parameters_b=7,layers=32,hidden=4096,ffn=11008,heads=32,
          kv_heads=8,vocab=32000,node_fabric='nvlink',max_search_gpus=64,
          activation_gib=2,train_tokens_b=1,effective_tflops=250,
          intra_gib_s=200,inter_gib_s=40,gpu_hour_price=2)


def design(**kw):
    return plan_resources(dict(BASE,**kw))['training_design']


class TrainingDesignTests(unittest.TestCase):
    def test_joint_constraints_and_reconciled_ledger(self):
        r=design()
        self.assertEqual(r['status'],'estimated')
        self.assertGreater(r['feasible_count'],10)
        for c in r['candidates']:
            self.assertEqual(c['gpus'],c['tp']*c['pp']*c['cp']*c['dp'])
            self.assertAlmostEqual(c['peak_gib'],sum(x['per_gpu_gib'] for x in c['rows']))
            self.assertLessEqual(c['peak_gib'],64)
            self.assertEqual(c['effective_batch_tokens'],c['microbatch']*4096*c['dp']*c['gradient_accumulation'])
            self.assertFalse(c['launch_ready'])
            if c['backend_family']=='deepspeed':
                self.assertEqual(c['tp']*c['pp']*c['cp']*c['ep'],1)
            if c['backend_family']=='megatron':
                grad=next(x for x in c['rows'] if x['item']=='gradients')
                self.assertEqual(grad['precision'],'FP32 · 4 bytes')

    def test_data_size_changes_work_not_memory(self):
        a=design(train_tokens_b=1)['recommendations']['minimum_capacity']
        b=design(train_tokens_b=10)['recommendations']['minimum_capacity']
        self.assertEqual(a['id'],b['id'])
        self.assertEqual(a['peak_gib'],b['peak_gib'])
        self.assertGreater(b['hours_range'][1],a['hours_range'][1]*9.9)

    def test_missing_timing_not_false_speed_claim(self):
        r=design(effective_tflops=0)
        self.assertIsNone(r['recommendations']['lowest_estimated_cost'])
        self.assertIsNone(r['recommendations']['minimum_capacity']['hours_range'])
        self.assertIsNone(design(gpu_hour_price=0)['recommendations']['lowest_estimated_cost'])

    def test_pcie_no_tensor_parallel_and_fixed_hardware(self):
        r=design(node_fabric='pcie',gpu_count=8)
        self.assertTrue(r['candidates'])
        for c in r['candidates']:
            self.assertEqual(c['tp'],1)
            self.assertEqual(c['gpus'],8)

    def test_cp_not_extra_data_batch(self):
        r=design(seq=32768,gpu_count=32,global_batch_tokens=1048576)
        cp=[x for x in r['candidates'] if x['cp']>1]
        self.assertTrue(cp)
        for c in cp:self.assertLessEqual(c['effective_batch_tokens'],1048576)

    def test_moe_nesting(self):
        r=design(parameters_b=40,expert_count=8,expert_topk=2,expert_parameter_fraction=.9,gpu_count=32)
        self.assertEqual(r['status'],'estimated')
        for c in r['candidates']:
            self.assertEqual(c['dp']%c['ep'],0)
            self.assertEqual(c['tp'],1)
            self.assertEqual(c['cp'],1)
            self.assertEqual(c['gpus'],c['pp']*c['dp'])

    def test_fail_closed_missing_moe_hybrid_or_roles(self):
        self.assertEqual(design(expert_count=8)['status'],'blocked')
        for phase in ('dpo','ppo','grpo'):self.assertEqual(design(training_phase=phase)['status'],'blocked')
        r=plan_resources(BASE,{'config':{'layer_types':['linear_attention']}})
        self.assertEqual(r['training_design']['status'],'blocked')

    def test_deadline_and_no_feasible(self):
        self.assertIsNone(design(deadline_hours=.0001)['recommendations']['deadline'])
        self.assertIsNotNone(design(deadline_hours=1e6)['recommendations']['deadline'])
        self.assertEqual(design(parameters_b=405,gpu_count=1)['status'],'blocked')

    def test_invalid_measurements(self):
        for field in ('train_tokens_b','effective_tflops','intra_gib_s','gpu_hour_price'):
            with self.assertRaises(ValueError):design(**{field:float('nan')})
        with self.assertRaises(ValueError):design(packing_efficiency=2)

    def test_small_model_does_not_require_zero_sharding(self):
        c=design(parameters_b=.5)['recommendations']['minimum_capacity']
        self.assertEqual(c['gpus'],1)
        self.assertEqual(c['zero_stage'],0)
