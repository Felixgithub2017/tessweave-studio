import copy
import unittest
import tempfile
import json
import struct
from pathlib import Path
from workbench.post_training import design_course


class CourseTests(unittest.TestCase):
    def setUp(self):
        self.model=dict(path='/model',name='fixture',model_type='qwen2',modality='text',experts=0,
                        parameter_estimate=7_000_000_000,config=dict(num_hidden_layers=28,hidden_size=3584,
                        num_attention_heads=28,num_key_value_heads=4,max_position_embeddings=32768),
                        tensors=[dict(name='model.layers.0.self_attn.q_proj.weight',shape=[3584,3584])])

    def test_unknown_budgets_never_fit(self):
        r=design_course(self.model,{})
        self.assertTrue(all(b['status']=='incomplete' for b in r['budgets']))
        self.assertEqual(r['schedule']['sft_updates'],125)
        self.assertEqual(r['update']['trainable'],8*(3584+3584))
        self.assertTrue(any('奖励' in b for b in r['blockers']))

    def test_role_accounting(self):
        r=design_course(self.model,dict(reward='math',activation_gib=8,workspace_gib=4))
        self.assertEqual(r['budgets'][0]['status'],'conditional_fit')
        self.assertGreater(r['budgets'][1]['known_gib'],r['budgets'][0]['known_gib'])
        kv=next(x for x in r['rows'] if 'KV' in x['item'])
        self.assertEqual(kv['elements'],2*28*4*128*4*1024)
        self.assertAlmostEqual(kv['gib'],kv['elements']*2/2**30)

    def test_hybrid_not_dense_estimate(self):
        m=copy.deepcopy(self.model);m['experts']=64
        r=design_course(m,dict(activation_gib=8,workspace_gib=4))
        self.assertTrue(all(b['status']=='incomplete' for b in r['budgets']))
        self.assertIsNone(r['rows'][-1]['gib'])

    def test_unknown_parameters(self):
        m=copy.deepcopy(self.model);m['parameter_estimate']=None
        r=design_course(m,dict(mode='full'))
        self.assertIsNone(r['update']['trainable'])
        self.assertIsNone(r['rows'][0]['gib'])

    def test_capacity_and_inputs(self):
        r=design_course(self.model,dict(mode='full',activation_gib=8,workspace_gib=4,capacity_gib=24))
        self.assertEqual(r['budgets'][0]['status'],'over_budget')
        for b in [dict(mode='qlora'),dict(group=1),dict(capacity_gib=float('nan')),dict(samples=True),dict(reward='arbitrary')]:
            with self.assertRaises(ValueError):design_course(self.model,b)
        self.assertTrue(any('上下文' in x for x in design_course(self.model,dict(seq=65536))['blockers']))

    def test_real_api_reads_headers_without_job(self):
        from workbench.server import App
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir()
            (model/'config.json').write_text(json.dumps(dict(model_type='qwen2',num_hidden_layers=1,hidden_size=2,num_attention_heads=1,num_key_value_heads=1)))
            header=json.dumps({'model.layers.0.self_attn.q_proj.weight':dict(dtype='F32',shape=[2,2],data_offsets=[0,16])}).encode()
            (model/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+bytes(16))
            app=App(root/'state',[str(root)])
            r=app.api('/api/post-training/design',dict(model_path=str(model)),{})
            self.assertEqual(r['model']['parameters'],4)
            self.assertEqual(r['update']['targets'],1)
            self.assertEqual(app.jobs.list(),[])

if __name__=='__main__':unittest.main()
