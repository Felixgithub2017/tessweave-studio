import unittest
from workbench.deployment_routes import route,recipes
from workbench.deployment_recipes import assessment
from workbench.model_metadata import with_bundled_metadata
from workbench.resources import plan_resources


class DeploymentRoutes(unittest.TestCase):
    def test_latest_deepseek_has_hardware_recipes(self):
        for repo in ('deepseek-ai/DeepSeek-V4-Flash','deepseek-ai/DeepSeek-V4-Pro',
                     'deepseek-ai/DeepSeek-V4-Flash-0731','deepseek-ai/DeepSeek-V4.1-Flash'):
            m=with_bundled_metadata(dict(repo=repo))
            r=plan_resources(dict(task='inference',gpu='h100',parameters_b=552),m)
            route=r['deployment_route']
            self.assertEqual(route['status'],'upstream_recipe')
            self.assertTrue(route['recipes'])
            self.assertIsNone(r['deployment_envelope']['memory_concurrency'])
            self.assertTrue(all(not x['matches_requested_workload'] for x in route['recipes']))

    def test_exact_identity_hardware_precision_not_inherited(self):
        m=dict(repo='deepseek-ai/DeepSeek-V4.1-Flash')
        r=route(dict(gpu='h200',bits=16),m,assessment(m))
        self.assertEqual(r['recipes'][0]['gpus'],8)
        self.assertEqual(r['recipes'][0]['precision'],'checkpoint_mixed_fp8_fp4')
        self.assertTrue(r['recipes'][0]['matches_selected_hardware'])
        m['repo']+='-invented'
        self.assertEqual(route({},m,assessment(m))['recipes'],[])

    def test_provenance(self):
        self.assertGreater(len(recipes()),100)
        for r in recipes():
            self.assertIn(r['source_commit'],r['source'])
            self.assertFalse(r['locally_verified'])
            self.assertFalse(r['executable'])

    def test_recipe_does_not_need_guessed_parameters(self):
        m=with_bundled_metadata(dict(repo='deepseek-ai/DeepSeek-V4.1-Flash'))
        r=plan_resources(dict(task='inference',gpu='h200',parameters_b=''),m)
        self.assertTrue(r['route_only'])
        self.assertEqual(r['deployment_route']['recipes'][0]['gpus'],8)
        self.assertNotIn('selected',r)
