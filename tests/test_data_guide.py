import json
import unittest
from workbench.data import dataset_guide


class Guides(unittest.TestCase):
    def test_objectives(self):
        for task in ('cpt','sft','dpo','ppo','grpo','rlhf','rlfh'):
            r=dataset_guide(task)
            self.assertEqual(json.loads(r['jsonl']),r['sample'])
            self.assertTrue(r['fields'])
        self.assertEqual(dataset_guide('rlfh')['task'],'rlhf')
        self.assertEqual(dataset_guide('ppo')['execution_status'],'blocked')

    def test_media_and_independent_examples(self):
        r=dataset_guide('dpo',{'modality':'vision-language','name':'local-vlm'})
        self.assertIn('images',r['sample'])
        self.assertIn('<image>',r['sample']['messages'][0]['content'])
        self.assertNotIn('images',dataset_guide('dpo')['sample'])

    def test_online_rl_no_answer_in_prompt(self):
        for task in ('ppo','grpo'):
            self.assertEqual(dataset_guide(task)['sample']['messages'][-1]['role'],'user')
