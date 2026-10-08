import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench.telemetry import parse_metrics, sample_host, sample_remote
from workbench.profiler import read_trace


class ObservabilityTests(unittest.TestCase):
    def test_metrics_preserve_ranks_and_reject_nonfinite(self):
        text='vllm:kv_cache_usage_perc{engine="0"} 0.2\nvllm:kv_cache_usage_perc{engine="1"} 0.4\nsglang:token_usage 0.5\nvllm:num_requests_running NaN\nother 99'
        rows=parse_metrics(text)
        self.assertEqual([r['value'] for r in rows],[.2,.4,.5])
        self.assertNotEqual(rows[0]['labels'],rows[1]['labels'])

    def test_apple_driver_not_vram(self):
        with patch('workbench.telemetry.platform.system',return_value='Darwin'),patch('workbench.telemetry.command',return_value={'ok':True,'output':'"PerformanceStatistics" = {"Device Utilization %"=41,"In use system memory"=1048576}' }):
            g=sample_host()['gpus'][0]
        self.assertEqual(g['utilization'],41)
        self.assertEqual(g['driver_in_use_mib'],1)
        self.assertIsNone(g['used_mib'])

    def test_remote_rejects_shell_and_options(self):
        for alias in ['-oProxyCommand=bad','host;whoami','host name','user@host']:
            with self.assertRaises(ValueError):sample_remote(alias)

    def test_remote_fixed_command(self):
        from subprocess import CompletedProcess
        with patch('workbench.telemetry.subprocess.run',return_value=CompletedProcess([],0,json.dumps({'host':'gpu','gpus':[]}),'')) as run:
            result=sample_remote('gpu-server')
        self.assertEqual(result['scope'],'ssh-host, not per-model')
        self.assertIn('StrictHostKeyChecking=yes',run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][-2:],['gpu-server','python3 -'])

    def test_trace_shapes_and_gzip(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'trace.json.gz'
            with gzip.open(p,'wt') as f:json.dump({'traceEvents':[{'ph':'X','ts':100,'dur':4,'name':'aten::mm','args':{'Input Dims':[[2,3],[3,4]]}},{'ph':'X','ts':101,'dur':2,'name':'kernel','cat':'kernel','tid':2}]},f)
            r=read_trace(p,[root])
            self.assertEqual(r['duration_us'],4)
            self.assertEqual(r['events'][0]['input_shapes'],[[2,3],[3,4]])
            self.assertIsNone(r['events'][1]['input_shapes'])
            with self.assertRaises(ValueError):read_trace(p,[str(Path(root)/'other')])

    def test_trace_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'trace.json';p.write_text('{"traceEvents":[]}')
            with self.assertRaises(ValueError):read_trace(p,[root])
