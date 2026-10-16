import json
import unittest
from pathlib import Path
from collections import Counter
from tools.build_planning_model_index import records, render


class PlanningModelIndex(unittest.TestCase):
    def test_catalog_has_one_entry_per_repo_and_matches_audit(self):
        rows=records()
        audit=json.loads(Path('docs/planning-coverage-20261016.json').read_text())
        self.assertEqual({x['repo'] for x in rows},{x['repo'] for x in audit['models']})
        self.assertEqual(len(rows),len({x['repo'] for x in rows}))
        self.assertEqual(dict(Counter(x['status'] for x in rows)),audit['summary'])
        for x in rows:
            if x['status']=='analytic_unverified':
                e=x['example']
                self.assertAlmostEqual(e['cache_gib'],((4096+e['extra_tail_tokens'])*e['token_bytes']+e['state_bytes'])/2**30)
            else:
                self.assertIsNone(x['example'])
                self.assertTrue(x['required_calculation'])

    def test_specific_architectures_do_not_share_wrong_equation(self):
        rows={x['repo']:x for x in records()}
        expected={'Qwen/Qwen3.5-9B':'G','moonshotai/Kimi-Linear-48B-A3B-Instruct':'F',
                  'zai-org/GLM-4.7-Flash':'D','deepseek-ai/DeepSeek-V3.2':'E',
                  'google/gemma-4-26B-A4B-it':'C','meta-llama/Llama-3.1-8B-Instruct':'M'}
        for repo,category in expected.items():self.assertEqual(rows[repo]['category'],category)
        self.assertEqual(rows['google/gemma-4-26B-A4B-it']['example']['token_bytes'],25*4*8*256+5*4*2*512)

    def test_committed_index_is_reproducible(self):
        self.assertEqual(Path('docs/MODEL_PLANNING_INDEX.zh-CN.md').read_text(),render(records()))
