"""Verify local archive bytes against receipts; never downloads or imports sources."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def status(root=ROOT):
    sources=json.loads((root/'workbench/knowledge/sources.json').read_text())['sources']
    receipt=root/'knowledge/receipts.jsonl'
    events=[json.loads(line) for line in receipt.read_text().splitlines()] if receipt.exists() else []
    result=[]
    for source in sources:
        attempts=[e for e in events if e['id']==source['id']]
        successful=[e for e in attempts if e['status']=='downloaded']
        entry={'id':source['id'],'title':source['title'],'year':source['year'],'status':'missing'}
        if successful:
            record=successful[-1];path=root/record['local_path']
            # A later failed refresh does not erase an earlier valid archive.
            good=path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
            entry.update(status='verified' if good else 'missing_or_corrupt',path=record['local_path'],sha256=record['sha256'])
        if attempts:entry['latest_attempt']=attempts[-1]['status']
        result.append(entry)
    return result


if __name__=='__main__':
    report=status();print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"Verified {sum(x['status']=='verified' for x in report)} / {len(report)} sources")
