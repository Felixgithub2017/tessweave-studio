"""Protocol fixtures, not model performance measurements."""
import io
import json
import unittest
from unittest.mock import patch
from workbench.benchmark import one, run_benchmark


def stream(usage=3, done=True):
    values = [{"choices": [{"delta": {"content": "323"}}]}]
    if usage is not None:
        values.append({"usage": {"completion_tokens": usage}, "choices": []})
    data = b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in values)
    return io.BytesIO(data + (b"data: [DONE]\n\n" if done else b""))


class BenchmarkProtocol(unittest.TestCase):
    def probe(self, **kwargs):
        with patch("workbench.benchmark.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = stream(**kwargs)
            return one("http://127.0.0.1:8000", {"model": "fixture"}, "323")

    def test_usage_and_exact_match(self):
        r = self.probe()
        self.assertTrue(r["ok"])
        self.assertTrue(r["exact_match"])
        self.assertEqual(r["completion_tokens"], 3)
        self.assertEqual(r["delta_events"], 1)

    def test_no_usage_is_not_estimated(self):
        r = self.probe(usage=None)
        self.assertIsNone(r["completion_tokens"])
        self.assertIsNone(r["output_tokens_per_s"])

    def test_negative_usage_rejected(self):
        self.assertIsNone(self.probe(usage=-1)["completion_tokens"])

    def test_incomplete_stream_not_success(self):
        self.assertFalse(self.probe(done=False)["ok"])

    def test_failed_request_invalidates_aggregate(self):
        with patch("workbench.benchmark.one", return_value={"ok": False, "error": "fixture"}):
            r = run_benchmark("http://127.0.0.1:8000", "fixture", [{"prompt": "x"}])
        self.assertEqual(r["successes"], 0)
        self.assertIsNone(r["aggregate_output_tokens_per_s"])
        self.assertIsNone(r["p95_first_delta_s"])

    def test_empty_prompts_rejected(self):
        with self.assertRaises(ValueError):
            run_benchmark("http://127.0.0.1:8000", "fixture", [])
