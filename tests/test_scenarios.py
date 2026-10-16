import unittest
from workbench.scenarios import scenario_plan
from workbench.post_training import sharding_brief


class ScenarioTests(unittest.TestCase):
    def test_missing_not_free_or_supported(self):
        r=scenario_plan({})
        self.assertIsNone(r['economics']['affordable_hours'])
        self.assertEqual(r['candidates'][0]['capacity_status'],'unknown')

    def test_budget_and_weights(self):
        r=scenario_plan(dict(memory_gib=4,budget_cny=100,hour_price_cny=10,fixed_cost_cny=20))
        self.assertEqual(r['economics']['affordable_hours'],8)
        self.assertEqual(r['candidates'][0]['capacity_status'],'weights_alone_exceed_budget')

    def test_robot_not_chat_training(self):
        r=scenario_plan(dict(scene='robot',modality='action',target='qualcomm'))
        self.assertTrue(all(c['modality']=='action' for c in r['candidates']))
        self.assertTrue(any('动作策略' in x for x in r['blockers']))
        self.assertIn('急停',r['evaluation'])
        self.assertIn('不能统一',r['training'])

    def test_invalid_input(self):
        for b in [dict(scene='invented'),dict(memory_gib=-1),dict(target='any'),dict(budget_cny=float('inf'))]:
            with self.assertRaises(ValueError):scenario_plan(b)

    def test_zero_equations(self):
        r=sharding_brief(1e9,1e9,'full',dict(dp=8,tp=2,pp=1))
        self.assertEqual(r['gpus'],16)
        self.assertAlmostEqual(r['rows'][0]['state_gib'],18e9/2/2**30)
        self.assertAlmostEqual(r['rows'][3]['state_gib'],18e9/16/2**30)
        self.assertEqual(r['rows'][1]['parts'][0]['divisor'],2)
        self.assertEqual(r['rows'][2]['parts'][1]['divisor'],16)
        self.assertIsNone(sharding_brief(None,None,'full',{})['rows'][0]['state_gib'])

if __name__=='__main__':unittest.main()
