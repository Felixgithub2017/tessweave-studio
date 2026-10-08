import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.knowledge_status import status


class ArchiveIntegrityTests(unittest.TestCase):
    def test_failed_refresh_keeps_good_copy_and_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'workbench/knowledge').mkdir(parents=True)
            (root/'knowledge').mkdir()
            (root/'workbench/knowledge/sources.json').write_text(json.dumps({'sources':[
                {'id':'a','title':'A','year':2026},{'id':'b','title':'B','year':2026}]}))
            path=root/'report.pdf';path.write_bytes(b'%PDF-example')
            receipt={'id':'a','status':'downloaded','local_path':'report.pdf',
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (root/'knowledge/receipts.jsonl').write_text(json.dumps(receipt)+'\n'+
                json.dumps({'id':'a','status':'failed'})+'\n')
            self.assertEqual([r['status'] for r in status(root)],['verified','missing'])
            path.write_bytes(b'corrupt')
            self.assertEqual(status(root)[0]['status'],'missing_or_corrupt')
