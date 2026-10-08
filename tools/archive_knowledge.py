"""Bounded, proxy-free archival of curated public sources; never executes them.

Run from the project root: python3 tools/archive_knowledge.py [--id SOURCE_ID]
Full texts remain in .knowledge-cache pending redistribution review. Receipts
are append-only and content-addressed. Requires no account credentials.
"""
import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
LIMIT=40*1024*1024


def retrieve(source, transport='urllib'):
    record={'id':source['id'],'requested_url':source['url'],
            'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'proxy_policy':'urllib ProxyHandler({}); transparent routing not controlled',
            'redistribution':'local reference only; see source license review'}
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url=source.get('download_url',source['url'])
        if url.startswith('https://raw.githubusercontent.com/'):
            parts=url.split('/');repo='/'.join(parts[3:5]);ref=parts[5];name='/'.join(parts[6:])
            if repo=='stanford-cs336/lectures':ref='de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e'
            url=f'https://api.github.com/repos/{repo}/contents/{name}?ref={ref}'
        if transport=='curl':
            with tempfile.TemporaryDirectory(prefix='workbench-reference-') as tmp:
                destination=Path(tmp)/'download'
                command=['curl','--noproxy','*','--fail','--location','--silent','--show-error',
                         '--proto','=https','--proto-redir','=https','--max-time','60',
                         '--max-filesize',str(LIMIT),'--output',str(destination),
                         '--write-out','%{url_effective}\n%{content_type}',url]
                run=subprocess.run(command,capture_output=True,timeout=65)
                if run.returncode:raise ValueError(run.stderr.decode('utf-8','replace')[:500])
                data=destination.read_bytes();final_url,_,content_type=run.stdout.decode().partition('\n')
                record.update(final_url=final_url,content_type=content_type,transport='curl --noproxy *')
        else:
            request=urllib.request.Request(url,headers={'User-Agent':'ModelWorkbench-KnowledgeArchive/1.0'})
            with opener.open(request,timeout=35) as response:
                data=response.read(LIMIT+1)
                content_type=response.headers.get('Content-Type','')
                record.update(final_url=response.url,content_type=content_type,
                              etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'))
        if len(data)>LIMIT:raise ValueError('Source exceeds 40 MiB limit')
        if url.startswith('https://api.github.com/') and '/contents/' in url:
            metadata=json.loads(data)
            record['git_blob_sha']=metadata['sha']
            if metadata.get('encoding')!='base64':
                with opener.open(metadata['git_url'],timeout=35) as response:
                    metadata=json.loads(response.read(LIMIT+1))
            data=base64.b64decode(metadata['content'],validate=False)
            content_type='application/octet-stream'
            if len(data)>LIMIT:raise ValueError('Decoded source exceeds limit')
        if source['kind']=='pdf' and not data.startswith(b'%PDF-'):
            raise ValueError('Expected PDF magic, received another format')
        digest=hashlib.sha256(data).hexdigest()
        if source.get('expected_sha256') and digest!=source['expected_sha256']:
            raise ValueError('Downloaded bytes do not match the publisher SHA256')
        suffix='.pdf' if data.startswith(b'%PDF-') else '.html' if 'html' in content_type else '.txt'
        path=ROOT/'.knowledge-cache'/source['id']/(digest+suffix)
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(data)
        record.update(status='downloaded',sha256=digest,bytes=len(data),local_path=str(path.relative_to(ROOT)))
        # Optional local searchable extraction; no dependency in the application.
        if suffix=='.pdf':
            try:
                from pypdf import PdfReader
                reader=PdfReader(path)
                text='\n\n'.join(f'=== PDF PAGE {n+1} ===\n'+(p.extract_text() or '') for n,p in enumerate(reader.pages))
                path.with_suffix('.txt').write_text(text,encoding='utf-8')
                record['pages']=len(reader.pages)
            except Exception as error:record['extraction_error']=str(error)[:300]
    except Exception as error:
        record.update(status='failed',error=str(error)[:500])
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--id',action='append')
    parser.add_argument('--transport',choices=('urllib','curl'),default='urllib');args=parser.parse_args()
    sources=json.loads((ROOT/'workbench/knowledge/sources.json').read_text())['sources']
    sources=[s for s in sources if not args.id or s['id'] in args.id]
    if args.id and set(args.id)-{s['id'] for s in sources}:raise ValueError('Unknown source ID')
    if any(not re.fullmatch('[a-z0-9-]+',s['id']) or not s['url'].startswith('https://') for s in sources):
        raise ValueError('Invalid curated source')
    receipts=ROOT/'knowledge/receipts.jsonl';receipts.parent.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(retrieve,s,args.transport) for s in sources]
        for future in concurrent.futures.as_completed(pending):
            record=future.result()
            with receipts.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(record['id'],record['status'],record.get('bytes',record.get('error')),flush=True)


if __name__=='__main__':main()
