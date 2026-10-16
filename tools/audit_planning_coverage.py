"""Offline catalog-wide coverage inventory; unknown metadata is NEVER a pass.

Run from repository: python tools/audit_planning_coverage.py --output PATH
No weights, network, installation or execution. Generated artifact is evidence
of capacity coverage only, not model/runtime compatibility.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workbench.featured_models import featured_models
from workbench.engine_deployment import models
from workbench.deployment_recipes import coverage
from workbench.resources import plan_resources
from workbench.model_metadata import with_bundled_metadata


def audit():
    rows={m['repo']:m for m in featured_models()}
    rows.update({m['repo']:m for m in models()})
    rows={repo:with_bundled_metadata(m) for repo,m in rows.items()}
    snapshots=Path(__file__).resolve().parents[1]/'workbench/knowledge/model_snapshots'
    for path in snapshots.glob('*.json'):
        s=json.loads(path.read_text());repo=s['repo']
        if repo in rows and not rows[repo].get('config'):
            rows[repo]=dict(rows[repo],config=s['config'].get('text_config') or s['config'],
                           parameter_estimate=s.get('parameter_estimate'),revision=s.get('revision'))
    result=[]
    for repo,m in sorted(rows.items()):
        support=coverage(m)
        item=dict(repo=repo,**support,parameter_count=m.get('parameter_estimate') or (m.get('nominal_parameters_b') or 0)*1e9 or None,cases=[])
        if support['status']=='analytic_unverified' and item['parameter_count']:
            for seq in sorted({4096,m['config'].get('max_position_embeddings') or m['config'].get('model_max_length') or 4096}):
                for batch in (1,4):
                    for bits in (4,8,16):
                        body=dict(task='inference',gpu='h100',parameters_b=item['parameter_count']/1e9,seq=seq,batch=batch,bits=bits)
                        try:
                            report=plan_resources(body,m);selected=report['selected']
                            item['cases'].append(dict(context=seq,batch=batch,bits=bits,status='candidate' if selected['feasible_under_assumptions'] else 'no_fit_in_search',gpus=selected['allocated_gpus']))
                        except (ValueError,TypeError,KeyError,ZeroDivisionError) as error:
                            item['cases'].append(dict(context=seq,batch=batch,bits=bits,status='error',error=str(error)))
        result.append(item)
    return dict(scope='catalog_inventory_and_H100_capacity_only_not_execution',
        summary=dict(Counter(r['status'] for r in result)),
        case_summary=dict(Counter(c['status'] for r in result for c in r['cases'])),models=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    report=audit();text=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(text+'\n')
    else:print(text)
    print(json.dumps({k:report[k] for k in ('summary','case_summary')},ensure_ascii=False),file=sys.stderr)
    sys.exit(1 if report['case_summary'].get('error') else 0)
