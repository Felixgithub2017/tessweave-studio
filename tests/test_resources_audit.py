import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench.resources import plan_resources, _plan_resources
from workbench.audit import Audit
from workbench.jobs import Jobs
from workbench import hub


class ResourceTests(unittest.TestCase):
    def test_explanation_precision_and_numeric_substitution(self):
        r=plan_resources({'parameters_b':7,'gpu_count':4})
        rows={x['item']:x for x in r['detailed_budget']['rows']}
        self.assertEqual(rows['weights']['precision'],'BF16 · 2 bytes')
        self.assertIn('7,000,000,000 × 2 bytes ÷ 1',rows['weights']['calculation'])
        self.assertIn('7,000,000,000 × 4 bytes ÷ 4',rows['adam_m']['calculation'])
        self.assertEqual(rows['workspace']['source'],'manual_budget')
        for row in rows.values():
            self.assertIn('calculation',row)
            self.assertIn('precision',row)
        infer=plan_resources({'task':'inference','parameters_b':1,'bits':4,'layers':32,'hidden':4096,'heads':32,'kv_heads':8})
        rows={x['item']:x for x in infer['detailed_budget']['rows']}
        self.assertEqual(rows['weights']['precision'],'4-bit')
        self.assertIn('2(K,V) × 2 bytes',rows['kv']['calculation'])

    def test_zero_stages_and_component_divisors(self):
        for stage in range(4):
            r=_plan_resources({'parameters_b':1,'gpu_count':4,'zero_stage':stage})
            rows={x['item']:x['per_gpu_gib'] for x in r['detailed_budget']['rows']}
            self.assertAlmostEqual(rows['weights'],2e9/2**30/(4 if stage==3 else 1))
            self.assertAlmostEqual(rows['gradients'],2e9/2**30/(4 if stage>=2 else 1))
            self.assertAlmostEqual(rows['adam_m'],4e9/2**30/(4 if stage>=1 else 1))
            self.assertEqual(rows['gather'],2 if stage==3 else 0)
            self.assertAlmostEqual(sum(rows.values()),r['selected']['topology']['peak_gib'])
            self.assertEqual(r['placement']['ranks'][0]['weight_fraction'],.25 if stage==3 else 1)
        for value in (-1,4,True,1.5):
            with self.assertRaises(ValueError):_plan_resources({'zero_stage':value})

    def test_auto_zero_recommendation(self):
        small=plan_resources({'parameters_b':1})
        self.assertEqual(small['zero_recommendation']['stage'],0)
        large=plan_resources({'parameters_b':7})
        self.assertEqual(large['zero_recommendation']['stage'],2)
        fixed=plan_resources({'parameters_b':7,'gpu_count':4})
        self.assertEqual(fixed['zero_recommendation']['stage'],1)
        self.assertEqual(len(fixed['zero_recommendation']['alternatives']),4)
        self.assertAlmostEqual(fixed['detailed_budget']['peak_gib'],fixed['selected']['topology']['peak_gib'])
        self.assertEqual(plan_resources({'parameters_b':1,'gpu_count':9})['selected']['allocated_gpus'],9)
        failed=plan_resources({'parameters_b':700,'gpu_count':1})
        self.assertIsNone(failed['zero_recommendation']['stage'])
        self.assertFalse(failed['selected']['feasible_under_assumptions'])
        self.assertEqual(plan_resources({'parameters_b':1,'zero_stage':3})['zero_recommendation']['stage'],0)

    def test_rank_placement_reconciles_and_nodes(self):
        r=plan_resources({'parameters_b':7,'extra_gpus':16,'layers':32})
        m=r['placement'];s=r['selected']
        self.assertEqual(len(m['ranks']),s['allocated_gpus'])
        self.assertEqual(m['axes']['cp'],1)
        self.assertFalse(m['measured'])
        self.assertEqual(m['global_microbatch'],s['allocated_gpus'])
        self.assertEqual(m['groups'][0]['ranks'],list(range(s['allocated_gpus'])))
        for rank in m['ranks']:
            self.assertEqual(rank['node'],rank['rank']//8)
            self.assertEqual(rank['layer_end_exclusive'],32)
            self.assertAlmostEqual(rank['peak_gib'],r['detailed_budget']['peak_gib'])
            self.assertAlmostEqual(rank['peak_gib']+rank['reserve_gib']+rank['headroom_gib'],rank['capacity_gib'])

    def test_inference_placement_and_infeasible(self):
        r=plan_resources({'parameters_b':72,'task':'inference','gpu_count':16,
                         'layers':80,'hidden':8192,'ffn':29568,'heads':64,'kv_heads':8})
        m=r['placement'];a=m['axes']
        self.assertEqual(a['tp']*a['pp']*a['dp']*a['cp'],len(m['ranks']))
        intervals={(x['layer_start'],x['layer_end_exclusive']) for x in m['ranks']}
        self.assertEqual(sum(end-start for start,end in intervals),80)
        self.assertEqual(len([g for g in m['groups'] if g['axis']=='tp']),a['pp'])
        self.assertEqual(m['ranks'][8]['node'],1)
        bad=plan_resources({'parameters_b':1000,'task':'inference'})
        self.assertEqual(bad['placement']['ranks'],[])

    def test_observed_parameters_override_blank_manual_input(self):
        for blank in ('',None,0):
            r=plan_resources({'parameters_b':blank},{'parameter_estimate':7_000_000_000})
            self.assertEqual(r['parameters'],7_000_000_000)
        with self.assertRaises(ValueError):plan_resources({'parameters_b':''})
    def test_detailed_ledger_reconciles(self):
        for body in ({'parameters_b':7},{'mode':'lora'},{'mode':'qlora'},
                     {'task':'inference','bits':4,'layers':32,'hidden':4096,'heads':32,'kv_heads':8},
                     {'communication_gib':2,'other_models_gib':8,'runtime_gib':1}):
            r=plan_resources(body);budget=r['detailed_budget']
            self.assertAlmostEqual(budget['peak_gib'],r['selected']['topology']['peak_gib'])
            self.assertAlmostEqual(budget['peak_gib'],sum(x['per_gpu_gib'] for x in budget['rows']))

    def test_training_has_no_generation_kv(self):
        r=plan_resources({})
        kv=next(x for x in r['detailed_budget']['rows'] if x['item']=='kv')
        self.assertEqual(kv['per_gpu_gib'],0)

    def test_full_ledger(self):
        r=plan_resources({'parameters_b':7})
        self.assertAlmostEqual(r['static_total_gib'],7e9*16/2**30)
        self.assertGreaterEqual(r['selected']['conditional_min_gpus'],2)

    def test_invalid_inputs(self):
        for v in (-1,float('nan'),True,float('inf')):
            with self.assertRaises(ValueError):
                plan_resources({'parameters_b':v})

    def test_unknown_inference_not_guessed(self):
        r=plan_resources({'parameters_b':1000,'task':'inference'})
        self.assertIsNone(r['selected']['conditional_min_gpus'])

    def test_added_resources_and_nodes(self):
        r=plan_resources({'parameters_b':7,'extra_gpus':16})['selected']
        self.assertEqual(r['allocated_gpus']%8,0)
        self.assertGreater(r['nodes'],1)
        self.assertEqual(r['topology']['dp'],r['allocated_gpus'])

    def test_qlora_not_blind_sharded(self):
        r=plan_resources({'parameters_b':1000,'mode':'qlora'})
        self.assertIsNone(r['selected']['conditional_min_gpus'])

    def test_inference_structure(self):
        r=plan_resources({'parameters_b':72,'task':'inference','layers':80,'hidden':8192,'ffn':29568,'heads':64,'kv_heads':8})
        s=r['selected'];self.assertTrue(s['feasible_under_assumptions'])
        self.assertEqual(s['topology']['tp']*s['topology']['pp'],s['allocated_gpus'])


class AuditAndNative(unittest.TestCase):
    def test_redaction(self):
        with tempfile.TemporaryDirectory() as td:
            a=Audit(Path(td)/'trace.log');a.emit('test',password='private',data={'api_key':'hidden'})
            r=a.tail()['events'][0]
            self.assertEqual(r['password'],'[REDACTED]')
            self.assertNotIn('hidden',a.path.read_text())

    def test_native_runner_real_process_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as td:
            a=Audit(Path(td)/'trace.log');j=Jobs(Path(td)/'state',audit=a)
            jid='a'*32
            j.create({'id':jid,'argv':[sys.executable,'-c','print("native-test-output")'],'env':{},'blockers':[]})
            j.update(jid,'starting')
            cmd=[sys.executable,'-m','workbench.native_runner','--state',str(j.state),'--id',jid,'--audit-log',str(a.path)]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(j.get(jid)['status'],'succeeded')
            self.assertIn('native-test-output',a.path.read_text())
            self.assertNotEqual(subprocess.run(cmd,capture_output=True,timeout=15).returncode,0)


class Modelscope(unittest.TestCase):
    def test_manifest_version_and_url(self):
        payload={'Code':200,'Data':{'Files':[{'Type':'blob','Path':'config.json','Size':3,'Sha256':'b'*64,'Revision':'a'*40}]}}
        with patch('workbench.hub.get_json',return_value=payload):
            m=hub.manifest('modelscope','Qwen/test','main')
        self.assertEqual(m['revision'],'master')
        self.assertIn('Revision='+'a'*40,hub.file_url(m,m['files'][0]))
        with self.assertRaises(ValueError):hub.resolve_source(m,'huggingface')
