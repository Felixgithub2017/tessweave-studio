import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from workbench import hub
from workbench.downloads import Downloads, verify
from workbench.structure import describe


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.content = b'{"model_type":"llama"}'
        self.f = {"name": "config.json", "size": len(self.content), "algorithm": "sha256", "digest": hashlib.sha256(self.content).hexdigest()}
        self.m = {"source": "huggingface", "repo": "org/model", "commit": "a"*40, "files": [self.f], "total_bytes": len(self.content)}
        self.manager = Downloads(self.root/"state", [str(self.root)])

    def tearDown(self):
        self.manager.shutdown()
        self.tmp.cleanup()

    def wait(self, manager, jid):
        manager.threads[jid].join(3)
        self.assertFalse(manager.threads[jid].is_alive())
        return manager.get(jid)

    def plan(self):
        return self.manager.prepare(self.m, str(self.root/"model"), ["config.json"])

    def transfer(self, r, response):
        with patch.object(hub, "manifest", return_value=self.m), patch.object(hub, "open_url", return_value=response):
            self.manager.start(r["id"])
            return self.wait(self.manager, r["id"])

    def test_complete_verified_download(self):
        r = self.transfer(self.plan(), Response(self.content))
        self.assertEqual(r["status"], "completed")
        self.assertEqual((self.root/"model/config.json").read_bytes(), self.content)
        self.assertTrue((self.root/"model/.workbench-download.json").exists())

    def test_remove_archives_record_not_weights(self):
        r=self.transfer(self.plan(),Response(self.content))
        result=self.manager.remove(r['id'])
        self.assertTrue(Path(result['archive']).exists())
        self.assertEqual(self.manager.list(),[])
        self.assertEqual((self.root/'model/config.json').read_bytes(),self.content)
        self.assertEqual(Downloads(self.root/'state',[str(self.root)]).list(),[])

    def test_remove_refuses_active_download(self):
        r=self.plan()
        self.manager.update(r['id'],status='downloading')
        with self.assertRaises(ValueError):self.manager.remove(r['id'])

    def test_short_stream_can_resume_after_restart(self):
        r = self.transfer(self.plan(), Response(self.content[:8]))
        self.assertEqual(r["status"], "failed")
        self.manager = Downloads(self.root/"state", [str(self.root)])
        response = Response(self.content[8:], 206, {"Content-Range": "bytes 8-%d/%d" % (len(self.content)-1, len(self.content))})
        with patch.object(hub, "manifest", return_value=self.m), patch.object(hub, "open_url", return_value=response) as request:
            self.manager.start(r["id"])
            result = self.wait(self.manager, r["id"])
            self.assertEqual(request.call_args.args[1]["Range"], "bytes=8-")
        self.assertEqual(result["status"], "completed")

    def test_ignored_range_restarts_without_corruption(self):
        r = self.transfer(self.plan(), Response(self.content[:4]))
        r = self.transfer(r, Response(self.content))
        self.assertEqual(r["status"], "completed")
        self.assertEqual((self.root/"model/config.json").read_bytes(), self.content)

    def test_hash_mismatch_preserved_not_published(self):
        r = self.transfer(self.plan(), Response(b'x'*len(self.content)))
        self.assertEqual(r["status"], "failed")
        self.assertFalse((self.root/"model/config.json").exists())
        self.assertTrue(list((self.root/"model/.workbench-parts").glob('*.invalid-*')))

    def test_switch_source_must_match(self):
        changed = dict(self.m, files=[dict(self.f, digest="b"*64)])
        r = self.plan()
        with patch.object(hub, "manifest", return_value=changed), patch.object(hub, "open_url") as request:
            self.manager.start(r["id"], "hf-mirror")
            r = self.wait(self.manager, r["id"])
        request.assert_not_called()
        self.assertEqual(r["status"], "failed")

    def test_existing_destination_refused(self):
        (self.root/"model").mkdir()
        with self.assertRaises(ValueError):
            self.plan()

    def test_outside_root_refused(self):
        with self.assertRaises(ValueError):
            self.manager.prepare(self.m, "/tmp/should-not-create", ["config.json"])

    def test_unknown_file_refused(self):
        with self.assertRaises(ValueError):
            self.manager.prepare(self.m, str(self.root/"model"), ["not-in-manifest"])

    def test_symlink_target_refused(self):
        r = self.transfer(self.plan(), Response(self.content[:4]))
        (self.root/"other").write_bytes(b"user file")
        (self.root/"model/config.json").symlink_to(self.root/"other")
        r = self.transfer(r, Response(self.content))
        self.assertEqual(r["status"], "failed")
        self.assertEqual((self.root/"other").read_bytes(), b"user file")

    def test_git_hash(self):
        p = self.root/"file"; p.write_bytes(self.content)
        digest = hashlib.sha1(("blob %d\0" % len(self.content)).encode()+self.content).hexdigest()
        self.assertTrue(verify(p, dict(self.f, algorithm="git-sha1", digest=digest)))

    def test_pause_keeps_partial_and_continue(self):
        manager = self.manager
        r = self.plan()
        class Pausing(Response):
            def read(self, n):
                data = super().read(4)
                manager.stops[r["id"]].set()
                return data
        r = self.transfer(r, Pausing(self.content))
        self.assertEqual(r["status"], "paused")
        self.assertEqual(self.transfer(r, Response(self.content))["status"], "completed")


class HubValidation(unittest.TestCase):
    def test_https_308_supported_on_python39(self):
        from urllib.request import Request
        with patch.object(hub, "check_url"):
            r = hub.Redirect().redirect_request(Request("https://hf-mirror.com/a"), None, 308, "", {}, "https://hf-mirror.com/b")
        self.assertEqual(r.full_url, "https://hf-mirror.com/b")

    def test_private_dns_refused(self):
        with patch.object(hub.socket, "getaddrinfo", return_value=[(2,1,6,"",("127.0.0.1",443))]):
            with self.assertRaises(ValueError):
                hub.check_url("https://huggingface.co")

    def test_catalog_partial_errors_are_reported(self):
        with patch.object(hub, "get_json", side_effect=ValueError("bad metadata")):
            r = hub.catalog("huggingface", "Qwen")
        self.assertFalse(r["models"])
        self.assertEqual(r["errors"][0]["author"], "Qwen")

    def test_paths(self):
        for name in ["../x", "/tmp/x", "a//b", "a/../b", "a\\b", ".workbench-download.json", "C:/x"]:
            with self.assertRaises(ValueError):
                hub.safe_name(name)

    def test_sources_and_urls(self):
        for url in ["http://huggingface.co", "https://127.0.0.1/x", "https://huggingface.co.attacker.test/x", "https://user:pass@huggingface.co"]:
            with self.assertRaises(ValueError):
                hub.check_url(url)

    def test_wrong_range(self):
        with self.assertRaises(ValueError):
            hub.range_start(Response(b"", 206, {"Content-Range": "bytes 0-9/20"}), 10, 20)

    def test_manifest_missing_digest_refused(self):
        with patch.object(hub, "get_json", return_value={"sha":"a"*40, "siblings":[{"rfilename":"weights","size":3}]}):
            with self.assertRaises(ValueError):
                hub.manifest("huggingface", "org/model")

    def test_manifest_normalization(self):
        with patch.object(hub, "get_json", return_value={"sha":"a"*40, "siblings":[{"rfilename":"config.json","size":3,"blobId":"b"*40}]}):
            m = hub.manifest("huggingface", "org/model")
        self.assertEqual(m["files"][0]["algorithm"], "git-sha1")

    def test_structure_is_inventory_not_claimed_trace(self):
        t={"name":"model.layers.0.self_attn.q_proj.weight","elements":16,"storage_bytes":32,"component":"."}
        s=describe({"model_type":"llama","num_hidden_layers":2}, {}, [t], False)
        self.assertTrue(s["flow"])
        self.assertEqual(s["groups"][0]["storage_bytes"],32)
        self.assertEqual(s["layers"][0]["stored_elements"],16)
        self.assertFalse(describe({"model_type":"unknown"}, {}, [t], False)["flow"])
