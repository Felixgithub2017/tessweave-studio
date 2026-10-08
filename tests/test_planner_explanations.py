import tempfile
import unittest
from pathlib import Path
from workbench.model_library import ModelLibrary
from workbench.resources import plan_resources
from workbench.architecture_inventory import architecture_inventory
import test_architecture_inventory as inventory_fixtures


class PlannerExplanationTests(unittest.TestCase):
    def test_featured_sort_and_no_invented_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            lib=ModelLibrary(Path(directory))
            self.assertEqual(lib.list(),[])
            rows=lib.list(True);sizes=[r['nominal_parameters_b'] for r in rows]
            self.assertEqual(sizes,sorted(sizes,reverse=True))
            self.assertGreater(len(rows),7)
            self.assertTrue(all(r['parameter_estimate'] is None for r in rows))
            self.assertEqual(lib.get(rows[0]['key'])['repo'],rows[0]['repo'])
            lib.catalog({'source':'huggingface','models':[{'id':rows[0]['repo']}]})
            self.assertEqual(len(lib.list(True)),len(rows))
            self.assertTrue(lib.list(True)[0]['featured'])

    def test_trace_tracks_actual_solver_results(self):
        body=dict(parameters_b=1,layers=16,hidden=1024,heads=16,ffn=4096,vocab=32000,max_search_gpus=8)
        design=plan_resources(body)['training_design']
        trace=design['decision_trace'];self.assertEqual(len(trace),5)
        self.assertEqual(trace[2]['facts']['feasible_candidates'],design['feasible_count'])
        self.assertFalse(trace[3]['facts']['time_cost_available'])
        self.assertTrue(trace[2]['rules'])
        blocked=plan_resources(dict(body,optimizer='muon'))['training_design']
        self.assertEqual(blocked['decision_trace'][1]['status'],'blocked')
        self.assertEqual(blocked['decision_trace'][2]['status'],'not_run')
        for row in design['recommendations']['minimum_capacity']['rows']:
            self.assertAlmostEqual(row['bytes']/2**30,row['per_gpu_gib'])
            if row['bytes_per_element']:
                self.assertAlmostEqual(row['elements']*row['bytes_per_element'],row['bytes'])

    def test_structural_ledger_reconciles_all_ranks(self):
        cfg,ts=inventory_fixtures.InventoryTests().fixture();inventory=architecture_inventory(cfg,ts,[],False)
        model=dict(config=cfg,architecture_inventory=inventory,modality='vision-language')
        design=plan_resources(dict(parameters_b=.001,max_search_gpus=8),model)['training_design']
        self.assertEqual(design['decision_trace'][1]['status'],'partial')
        for candidate in design['structural_capacity']['candidates']:
            for rank in candidate['ranks']:
                self.assertAlmostEqual(sum(r['gib'] for r in rank['ledger']),rank['accounted_gib'])
                for row in rank['ledger']:
                    self.assertTrue(row['formula']);self.assertTrue(row['precision'])
                    if row['elements'] is not None:
                        self.assertAlmostEqual(row['elements']*row['bytes_per_element'],row['bytes'])
                grad=next(r for r in rank['ledger'] if r['item']=='gradients')
                self.assertEqual(grad['bytes_per_element'],2 if candidate['zero_stage']==3 else 4)
