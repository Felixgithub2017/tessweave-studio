"""Offline acceptance matrix; upstream config snapshots, never downloads weights."""
import json
from pathlib import Path
import unittest
from workbench.resources import plan_resources
from workbench.featured_models import MODELS


class CommonModelMatrix(unittest.TestCase):
    def test_catalog_training_and_deployment(self):
        records = json.loads((Path(__file__).parent/'fixtures/common_models_20261014.json').read_text())
        self.assertEqual({x['repo'] for x in records}, {x[0] for x in MODELS})
        for record in records:
            full = record.get('config', {})
            model = dict(repo=record['repo'], config=full.get('text_config', full),
                         parameter_estimate=record.get('parameter_estimate') if not full.get('quantization_config') else None)
            for task, mode, bits in [('train','full',16), ('train','lora',16), ('train','qlora',4),
                                     ('inference','full',16), ('inference','full',8), ('inference','full',4)]:
                for seq in (4096, 32768):
                    with self.subTest(repo=record['repo'], task=task, mode=mode, bits=bits, seq=seq):
                        result = plan_resources(dict(task=task, mode=mode, bits=bits, seq=seq,
                            parameters_b=record['nominal_parameters_b'], gpu='h100'), model)
                        coverage = result['planning_coverage']
                        self.assertGreater(coverage['weight_payload_gib'], 0)
                        if coverage['status'] == 'partial':
                            self.assertTrue(coverage['blockers'])
                        self.assertFalse(coverage['execution_verified'])
                        dense = record['repo'] in ('Qwen/Qwen3-32B', 'Qwen/Qwen3-8B', 'Qwen/Qwen3-0.6B')
                        standard_moe=record['repo'] in ('Qwen/Qwen3-235B-A22B','zai-org/GLM-5','MiniMaxAI/MiniMax-M2')
                        layered=record['repo'] in ('XiaomiMiMo/MiMo-V2.6-Pro-RL','google/gemma-4-31B-it')
                        expected = 'estimated' if (dense and (task == 'inference' or mode == 'full')) or ((standard_moe or layered) and task=='inference') else 'partial'
                        self.assertEqual(coverage['status'], expected)
                        if task == 'inference' and not (dense or standard_moe or layered):
                            self.assertIsNone(result['placement'])
                            self.assertIsNone(result['deployment_envelope']['minimum_gpus_target'])
