import unittest
from workbench.serving_plan import plan_serving


class ServingPlanTests(unittest.TestCase):
    def calibrated(self, **kw):
        return plan_serving(dict(replica_rps=4, observed_first_text_p95_s=1,
                                 observed_total_p95_s=20, **kw))

    def test_example_and_gpu_accounting(self):
        r = self.calibrated(gpus_per_replica=4)
        self.assertEqual(r['normal']['replicas'], 4)
        self.assertEqual(r['resilient']['replicas'], 5)
        self.assertEqual(r['resilient']['gpus'], 20)
        self.assertAlmostEqual(r['peak_model_rps'], 100000 / 86400 * 10)
        self.assertIsNone(r['normal']['hourly_gpu_cost'])

    def test_correlated_failure_domains(self):
        r = self.calibrated(replicas_per_domain=2, failed_domains=2)
        self.assertEqual(r['resilient']['replicas'], 8)
        self.assertEqual(r['resilient']['surviving_replicas'], 4)

    def test_unknown_or_slo_violation_never_recommends(self):
        for body in ({}, {'replica_rps': 4}, {'replica_rps':4, 'observed_first_text_p95_s':3, 'observed_total_p95_s':20},
                     {'replica_rps':4, 'observed_first_text_p95_s':1, 'observed_total_p95_s':20, 'observed_error_percent':2}):
            r = plan_serving(body)
            self.assertTrue(r['blockers'])
            self.assertIsNone(r['normal'])

    def test_explicit_peak_amplification_and_price(self):
        r = self.calibrated(peak_rps=8, calls_per_request=2, gpu_hour_price=3)
        self.assertEqual(r['peak_model_rps'],16)
        self.assertEqual(r['normal']['replicas'],5)
        self.assertEqual(r['normal']['hourly_gpu_cost'],15)

    def test_limits(self):
        for key, value in [('replica_rps', float('nan')), ('daily_requests', True), ('failed_domains', 1.2),
                           ('max_output_tokens', 499), ('headroom_percent', 100), ('total_slo_s', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan_serving({key:value})

    def test_monotonic_and_survivors(self):
        for peak in (1, 5, 30, 100):
            r = self.calibrated(peak_rps=peak, replicas_per_domain=3, failed_domains=2)
            self.assertGreaterEqual(r['resilient']['surviving_replicas']*3.2, peak)
            self.assertGreaterEqual(r['resilient']['replicas'], r['normal']['replicas'])
            self.assertEqual(r['evidence'], 'user_entered_scenario_not_verified')
