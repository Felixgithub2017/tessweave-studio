"""Offline representative-family regression; not a universal support claim."""
import unittest
from workbench.engine_deployment import models
from workbench.model_metadata import with_bundled_metadata
from workbench.resources import plan_resources


class MainstreamPlanning(unittest.TestCase):
    def test_seven_families_quantization_and_concurrency(self):
        pinned={m['repo']:m for m in models()}
        repos=('zai-org/GLM-4.7-Flash','deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
               'moonshotai/Kimi-Linear-48B-A3B-Instruct','Qwen/Qwen3.5-9B',
               'google/gemma-4-26B-A4B-it','meta-llama/Llama-3.1-8B-Instruct',
               'mistralai/Mistral-7B-Instruct-v0.3')
        for repo in repos:
            m=with_bundled_metadata(pinned.get(repo,dict(repo=repo)))
            count=m.get('parameter_estimate') or m.get('nominal_parameters_b',0)*1e9
            self.assertTrue(0<count<50e9,repo)
            if not m.get('config'):
                # Gated Llama config is absent in the offline cache. Keep this
                # visible as missing evidence, never invent the model shape.
                self.assertEqual(repo,'meta-llama/Llama-3.1-8B-Instruct')
                r=plan_resources(dict(task='inference',parameters_b=count/1e9),m)
                self.assertEqual(r['planning_coverage']['status'],'partial')
                continue
            for bits in (4,8,16):
                for batch in (1,4):
                    with self.subTest(repo=repo,bits=bits,batch=batch):
                        r=plan_resources(dict(task='inference',parameters_b=count/1e9,
                            gpu='h100',bits=bits,batch=batch,seq=4097),m)
                        self.assertEqual(r['planning_coverage']['status'],'estimated')
                        self.assertTrue(r['selected']['feasible_under_assumptions'])
                        self.assertAlmostEqual(r['detailed_budget']['peak_gib'],
                                               r['deployment_envelope']['target']['peak_gib'])
