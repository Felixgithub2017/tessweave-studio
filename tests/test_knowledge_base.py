import json
import unittest
from unittest.mock import patch
from workbench.knowledge_base import ROOT, planning_evidence
from workbench.length_curriculum import length_curriculum
from workbench.resources import plan_resources


class KnowledgeTests(unittest.TestCase):
    def test_rule_references_resolve(self):
        manifest=json.loads((ROOT/'sources.json').read_text())
        rules=json.loads((ROOT/'rules.json').read_text())
        ids=[s['id'] for s in manifest['sources']]
        self.assertEqual(len(ids),len(set(ids)))
        for r in rules['rules']:
            self.assertTrue(set(r['sources'])<=set(ids))
        for r in rules['release_references']:self.assertIn(r['source'],ids)

    def test_exact_identity_not_folder_or_predecessor(self):
        for m in ({'name':'GLM-5'}, {'repo':'zai-org/GLM-5.3-Flash-BF16'},
                  {'repo':'someone/GLM-5'}, {'repo':'deepseek-ai/DeepSeek-V4.1'}):
            self.assertIsNone(planning_evidence({},m)['published_reference'])
        r=planning_evidence({}, {'repo':'deepseek-ai/DeepSeek-V4-Pro'})
        self.assertEqual(r['published_reference']['optimizer'],'Muon + AdamW')

    def test_offline_and_no_official_default_override(self):
        with patch('urllib.request.urlopen',side_effect=AssertionError('Must be offline')):
            c=length_curriculum({'training_phase':'cpt'},{'repo':'zai-org/GLM-5'})
        self.assertIsNone(c['official_training_length'])
        self.assertEqual(c['stages'][0]['seq'],8192)
        self.assertEqual(c['published_reference']['source'],'glm5')

    def test_scopes_and_moe_stage_gate(self):
        m={'config':{'text_config':{'num_experts':256},'vision_config':{}}}
        m['modality']='image-text-to-text'
        r=planning_evidence({'training_phase':'grpo'},m)
        self.assertTrue({'role-liveness','routing-gates','multimodal-freezing'}<={x['id'] for x in r['rules']})
        c=length_curriculum({},m)
        self.assertTrue(all('all-to-all' in s['advance_gate_en'] for s in c['stages']))

    def test_optimizer_fails_closed(self):
        body=dict(parameters_b=1,layers=16,hidden=1024,heads=16,ffn=4096,vocab=32000,
                  max_search_gpus=8,optimizer='muon')
        r=plan_resources(body)['training_design']
        self.assertEqual(r['status'],'blocked')
        self.assertTrue(any('optimizer' in s for s in r['blockers']))
        self.assertEqual(len(r['knowledge_evidence']['rules_sha256']),64)
        self.assertFalse(r['launch_ready'])
