import unittest
from workbench.architecture_inventory import architecture_inventory
from workbench.resources import plan_resources


class InventoryTests(unittest.TestCase):
    def fixture(self):
        cfg=dict(num_hidden_layers=2,hidden_size=64,num_attention_heads=4,
                 intermediate_size=128,vocab_size=100,n_routed_experts=2,
                 num_experts_per_tok=1,num_nextn_predict_layers=1,
                 layer_types=['linear_attention','sparse_attention'])
        ts=[]
        for layer in range(3):
            for e in range(2):
                ts.append(dict(name=f'model.language_model.layers.{layer}.mlp.experts.{e}.weight',elements=1000,dtype='BF16'))
            ts.append(dict(name=f'model.language_model.layers.{layer}.mlp.shared_experts.weight',elements=100,dtype='BF16'))
        ts.append(dict(name='model.visual.patch.weight',elements=200,dtype='BF16'))
        return cfg,ts

    def test_conservation_and_auxiliary(self):
        cfg,ts=self.fixture();i=architecture_inventory(cfg,ts,[],False)
        self.assertTrue(i['complete']);self.assertEqual(i['total_elements'],6500)
        self.assertEqual(i['routed_elements'],4000);self.assertEqual(i['shared_expert_elements'],200)
        self.assertEqual(i['outside_decoder'],dict(mtp=2100,vision=200))
        self.assertEqual(sum(x['elements'] for x in i['layers'])+sum(i['outside_decoder'].values()),6500)

    def test_incomplete_and_packed_fail_closed(self):
        cfg,ts=self.fixture()
        for items,errors,packed in [(ts[1:],[],False),(ts,['missing shard'],False),(ts,[],True)]:
            self.assertFalse(architecture_inventory(cfg,items,errors,packed)['complete'])

    def test_hybrid_automatic_partial_plan_and_ledgers(self):
        cfg,ts=self.fixture();i=architecture_inventory(cfg,ts,[],False)
        model=dict(config=cfg,architecture_inventory=i,modality='vision-language',parameter_count=6500)
        r=plan_resources(dict(parameters_b=.001,max_search_gpus=8),model)['training_design']
        self.assertEqual(r['status'],'partial');self.assertFalse(r['blockers'])
        self.assertAlmostEqual(r['assumptions']['expert_parameter_fraction'],4000/6500)
        for c in r['structural_capacity']['candidates']:
            self.assertIsNone(c['training_feasible']);self.assertFalse(c['launch_ready'])
            self.assertEqual(c['gpus'],c['pp']*c['dp']);self.assertEqual(c['dp']%c['ep'],0)
            for rank in c['ranks']:
                self.assertAlmostEqual(rank['accounted_gib'],sum(rank[k+'_gib'] for k in ['weights','gradients','master','adam_m','adam_v','gather','workspace']))
