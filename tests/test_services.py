"""Stream protocol tests: synthetic responses, not model benchmarks."""
import io
import json
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from workbench.services import Services


class ServiceTests(unittest.TestCase):
    def test_image_content_requires_capability_and_inline_data(self):
        import base64
        content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode()}}]
        with self.assertRaises(ValueError):self.s.validate_content(content,'user',self.r)
        self.s.validate_content(content,'user',{'supports_images':True})
        with self.assertRaises(ValueError):self.s.validate_content([{'type':'image_url','image_url':{'url':'http://localhost/private'}}],'user',{'supports_images':True})
        with self.assertRaises(ValueError):self.s.validate_content(content,'assistant',{'supports_images':True})

    def test_ready_requires_generated_content(self):
        with patch.object(self.s,'start_chat') as start:
            self.s.chats['probe']={'status':'succeeded','metrics':{'content_chars':0},'stop':__import__('threading').Event()}
            start.return_value={'id':'probe'}
            self.s.readiness(self.r['id'])
            self.assertEqual(self.s.get(self.r['id'])['status'],'generation-unverified')
            self.s.chats['probe']['metrics']['content_chars']=2
            self.s.readiness(self.r['id'])
            self.assertEqual(self.s.get(self.r['id'])['status'],'generation-verified')
    def test_profiler_confirmation_and_payload(self):
        with self.assertRaises(ValueError):self.s.profile({'service_id':self.r['id'],'backend':'vllm','action':'start'})
        with patch('workbench.services.urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=io.BytesIO(b'accepted')
            result=self.s.profile({'service_id':self.r['id'],'confirm':self.r['id'],'backend':'sglang','action':'start'})
            req=opener.return_value.open.call_args.args[0]
        self.assertTrue(req.full_url.endswith('/start_profile'))
        self.assertEqual(json.loads(req.data)['num_steps'],10)
        self.assertEqual(result['status'],'engine-accepted')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(self.tmp.name, Mock())
        self.addCleanup(self.s.shutdown)
        self.r = self.s.register({'url':'http://127.0.0.1:18081','model':'protocol-fixture'})

    def chat(self, usage=None, done=True, delta=None, finish=None, thinking='auto'):
        records = [{'choices':[{'delta':delta if delta is not None else {'reasoning_content':'check','content':'hello'},'finish_reason':finish}]}]
        if usage is not None: records.append({'usage':usage})
        stream = b''.join(b'data: '+json.dumps(x).encode()+b'\n\n' for x in records)
        if done: stream += b'data: [DONE]\n\n'
        with patch('workbench.services.urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = io.BytesIO(stream)
            job = self.s.start_chat({'service_id':self.r['id'],'messages':[{'role':'user','content':'hi'}],'thinking':thinking})
            deadline = time.monotonic()+3
            while time.monotonic()<deadline:
                result = self.s.events(job['id'])
                if result['status']!='running': return result
                time.sleep(.01)
            self.fail('stream did not terminate')

    def test_stream_and_usage(self):
        r = self.chat({'prompt_tokens':2,'completion_tokens':4})
        self.assertEqual(r['status'],'succeeded')
        self.assertEqual([e['type'] for e in r['events'] if e['type'] in ('content','reasoning')],['reasoning','content'])
        self.assertEqual(r['events'][0]['type'],'request')
        self.assertEqual(r['events'][-1]['type'],'completed')
        self.assertEqual(r['metrics']['output_tokens'],4)
        self.assertGreater(r['metrics']['end_to_end_output_tokens_per_s'],0)

    def test_mlx_reasoning_only_length(self):
        r=self.chat({'completion_tokens':512},delta={'reasoning':'thinking'},finish='length')
        self.assertEqual(r['metrics']['content_chars'],0)
        self.assertEqual(r['metrics']['reasoning_chars'],8)
        self.assertEqual(r['metrics']['finish_reason'],'length')
        self.assertEqual([e['text'] for e in r['events'] if e['type']=='reasoning'],['thinking'])

    def test_thinking_mode_logged(self):
        self.assertEqual(self.chat(thinking='off')['events'][0]['thinking'],'off')
        with self.assertRaises(ValueError):self.chat(thinking='invalid')

    def test_reasoning_alias_not_duplicated(self):
        r=self.chat(delta={'reasoning_content':'same','reasoning':'same','content':'answer'})
        self.assertEqual(r['metrics']['reasoning_chars'],4)
        self.assertEqual(r['metrics']['content_chars'],6)

    def test_missing_usage_not_guessed(self):
        self.assertIsNone(self.chat()['metrics']['output_tokens'])

    def test_recording_survives_restart_and_rejects_traversal(self):
        r=self.chat()
        other=Services(self.tmp.name,Mock())
        self.assertEqual(other.recordings()[0]['id'],r['id'])
        replay=other.replay(r['id'])
        self.assertEqual(replay['events'][0]['messages'][0]['content'],'hi')
        self.assertGreater(replay['events'][-1]['metrics']['response_bytes'],0)
        self.assertTrue(any(e['type']=='telemetry' for e in replay['events']))
        with self.assertRaises(ValueError):other.replay('../services')

    def test_invalid_usage_not_guessed(self):
        self.assertIsNone(self.chat({'completion_tokens':-1})['metrics']['output_tokens'])

    def test_truncated_stream_fails(self):
        self.assertEqual(self.chat(done=False)['status'],'failed')

    def test_registry_survives_restart_and_deduplicates(self):
        other = Services(self.tmp.name,Mock())
        self.assertEqual(other.get(self.r['id'])['model'],'protocol-fixture')
        self.assertEqual(other.register({'url':self.r['url'],'model':'protocol-fixture'})['id'],self.r['id'])

    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            self.s.register({'url':'https://example.com','model':'x'})

    def test_invalid_requests_rejected(self):
        base = {'service_id':self.r['id'],'messages':[{'role':'user','content':'hi'}]}
        for update in [{'temperature':float('nan')},{'max_tokens':True},{'messages':[]},{'messages':[{'role':'tool','content':'x'}]}]:
            with self.assertRaises(ValueError): self.s.start_chat(dict(base,**update))

    def test_cursor_validation(self):
        r = self.chat()
        with self.assertRaises(ValueError): self.s.events(r['id'],100)
        self.assertEqual(self.s.events(r['id'],r['cursor'])['events'],[])
