import unittest
from workbench.length_curriculum import length_curriculum
from workbench.resources import plan_resources


class CurriculumTests(unittest.TestCase):
    def test_pretrain_quality_does_not_require_longer_context(self):
        c=length_curriculum(dict(training_phase='pretrain',train_tokens_b=1))
        self.assertEqual([x['seq'] for x in c['stages']],[4096,4096,16384,32768])
        self.assertEqual(sum(x['token_share_percent'] for x in c['stages']),100)
        self.assertEqual(sum(x['tokens'] for x in c['stages']),10**9)

    def test_cpt_does_not_replay_pretrain_or_infer_history(self):
        c=length_curriculum({},dict(config={'max_position_embeddings':1048576}))
        self.assertEqual([x['seq'] for x in c['stages']],[8192,16384,32768])
        self.assertIsNone(c['official_training_length'])
        self.assertTrue(all(x['tokens'] is None for x in c['stages']))

    def test_cap_and_rl_roles(self):
        c=length_curriculum({'curriculum_target_seq':131072},{'config':{'max_position_embeddings':8192}})
        self.assertTrue(all(x['seq']<=8192 for x in c['stages']))
        for phase in ('dpo','ppo','grpo'):
            c=length_curriculum({'training_phase':phase})
            self.assertEqual(c['status'],'role_plan_required');self.assertFalse(c['stages'])

    def test_each_stage_has_independent_resource_search(self):
        r=plan_resources(dict(parameters_b=1,layers=16,hidden=1024,heads=16,
                             ffn=4096,vocab=32000,max_search_gpus=8))
        for s in r['length_curriculum']['stages']:
            self.assertEqual(s['seq'],s['resource_design']['assumptions']['seq'])
            self.assertFalse(s['resource_design']['launch_ready'])

    def test_invalid_target(self):
        for value in (0,-1,True,'bad',float('nan')):
            with self.assertRaises(ValueError):length_curriculum({'curriculum_target_seq':value})
