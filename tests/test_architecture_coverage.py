import unittest
import tempfile
from pathlib import Path
from workbench.resources import plan_resources
from workbench.deployment_recipes import assessment
from workbench.engine_deployment import models


def qwen35():
    return dict(repo='Qwen/Qwen3.5-9B',parameter_estimate=9653104368,config=dict(
        model_type='qwen3_5_text',num_hidden_layers=32,hidden_size=4096,
        intermediate_size=12288,num_attention_heads=16,num_key_value_heads=4,head_dim=256,
        max_position_embeddings=262144,layer_types=['linear_attention']*3+['full_attention'],
        linear_conv_kernel_dim=4,linear_key_head_dim=128,linear_num_key_heads=16,
        linear_num_value_heads=32,linear_value_head_dim=128))


class ArchitectureCoverage(unittest.TestCase):
    def test_qwen35_default_context_and_cache(self):
        m=qwen35();m['config']['layer_types']*=8
        r=plan_resources(dict(task='inference',gpu='h100'),m)
        self.assertEqual(r['planning_coverage']['status'],'estimated')
        self.assertEqual(r['deployment_envelope']['target']['context_tokens'],262144)
        self.assertTrue(r['selected']['feasible_under_assumptions'])
        self.assertAlmostEqual(r['detailed_budget']['peak_gib'],r['deployment_envelope']['target']['peak_gib'])

    def test_qwen35_bad_layer_table_rejected(self):
        self.assertFalse(assessment(qwen35())['analytic_supported'])

    def test_pinned_dense_and_standard_moe_are_plannable(self):
        for m in models():
            if m['config'].get('model_type') not in ('qwen2','qwen3','qwen3_moe','phi3','mistral','mixtral'):continue
            with self.subTest(repo=m['repo']):
                r=plan_resources(dict(task='inference',gpu='h100',seq=4096,parameters_b=m['nominal_parameters_b']),m)
                self.assertEqual(r['planning_coverage']['status'],'estimated')
                self.assertTrue(r['selected']['feasible_under_assumptions'])

    def test_unknown_never_silently_gets_standard_kv(self):
        m=qwen35();m['config']['model_type']='future_unknown'
        self.assertFalse(assessment(m)['analytic_supported'])

    def test_offline_selection_and_full_catalog_coverage_contract(self):
        from workbench.server import App
        with tempfile.TemporaryDirectory() as root:
            app=App(Path(root)/'state',[root])
            rows=app.dispatch('/api/modelchoices',{'include_featured':True},{})
            self.assertGreater(len(rows),350)
            for row in rows:
                self.assertIn(row['planning_support']['status'],('metadata_required','analytic_unverified','adapter_required','upstream_recipe'))
                self.assertFalse(row['planning_support']['execution_verified'])
            key='huggingface:Qwen/Qwen3.5-9B'
            selected=app.dispatch('/api/modelchoices/select',{'key':key},{})
            self.assertEqual(selected['parameter_estimate'],9653104368)
            r=app.dispatch('/api/resources',dict(task='inference',gpu='h100',model_key=key),{})
            self.assertEqual(r['planning_coverage']['status'],'estimated')
            self.assertEqual(r['deployment_envelope']['target']['context_tokens'],262144)

    def test_gdn_tp_and_request_state(self):
        from workbench.hybrid_cache import coefficients
        m=qwen35();m['config']['layer_types']*=8
        token,state=coefficients(m['config'],1)
        self.assertEqual(token,8*4*4*256)
        self.assertEqual(state,24*(4*32*128*128+2*(2*16*128+32*128)*3))
        self.assertEqual(coefficients(m['config'],2),(token//2,state//2))
        r=plan_resources(dict(task='inference',gpu='h100',seq=4096,request_mix='4096:2,8192:1'),m)
        e=r['deployment_envelope']
        self.assertAlmostEqual(e['mixed_requests']['kv_gib'],(16384*token+3*state)/2**30)
