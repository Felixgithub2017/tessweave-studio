"""Read-only Chrome/PyTorch trace import. No pickle, execution or inferred shapes."""
import gzip
import hashlib
import json
import math
from .inspection import permitted


def read_trace(path, roots):
    p=permitted(path,roots)
    limit=32*1024*1024
    if not p.is_file() or p.stat().st_size>limit:raise ValueError('Trace must be a file <=32 MiB')
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rb') as f:raw=f.read(limit+1)
    if len(raw)>limit:raise ValueError('Expanded trace exceeds 32 MiB; capture a shorter window')
    obj=json.loads(raw)
    events=obj.get('traceEvents') if isinstance(obj,dict) else obj
    if not isinstance(events,list):raise ValueError('Expected Chrome traceEvents array')
    rows=[];total=0
    for e in events:
        if not isinstance(e,dict) or e.get('ph')!='X':continue
        ts,dur=e.get('ts'),e.get('dur')
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (ts,dur)) or dur<0:continue
        total+=1
        if len(rows)>=3000:continue
        args=e.get('args') if isinstance(e.get('args'),dict) else {}
        shapes=args.get('Input Dims',args.get('Input Shapes'))
        if shapes is not None and len(json.dumps(shapes))>2000:shapes='shape metadata too large'
        rows.append({'name':str(e.get('name','unknown'))[:240],'category':str(e.get('cat','unknown'))[:120],
                     'lane':str(e.get('pid','?'))[:40]+':'+str(e.get('tid','?'))[:40],
                     'start_us':ts,'duration_us':dur,'input_shapes':shapes})
    if not rows:raise ValueError('No complete duration events (ph=X) in trace')
    origin=min(r['start_us'] for r in rows)
    for r in rows:r['start_us']-=origin
    return {'file':p.name,'sha256':hashlib.sha256(raw).hexdigest(),'events':rows,'total_events':total,
            'truncated':total>len(rows),'duration_us':max(r['start_us']+r['duration_us'] for r in rows),
            'scope':'independent profiler capture; not automatically correlated with this conversation',
            'note':'CPU operators and GPU kernels can overlap. Durations must not be summed as wall time. Shapes exist only when recorded.'}
