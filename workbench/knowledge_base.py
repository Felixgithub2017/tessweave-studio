"""Offline, curated evidence. Downloaded documents are never executable rules."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).with_name('knowledge')


def planning_evidence(body, model=None):
    raw=(ROOT/'rules.json').read_bytes()
    catalog=json.loads((ROOT/'sources.json').read_text(encoding='utf-8'))
    rules=json.loads(raw)
    m=model or {};cfg=m.get('config') or {};text=cfg.get('text_config') or cfg
    scopes={'all'}
    if any(text.get(k) for k in ('num_experts','n_routed_experts','num_local_experts')) or body.get('expert_count'):
        scopes.add('moe')
    if body.get('training_phase') in ('dpo','ppo','grpo','rlhf'):scopes.add('rl')
    if 'vision_config' in cfg or 'audio_config' in cfg or m.get('modality') not in (None,'text','text-generation'):
        scopes.add('multimodal')
    applicable=[r for r in rules['rules'] if r['scope'] in scopes]
    # Never match a folder basename, broad family substring, or a linked predecessor.
    repo=m.get('repo') or cfg.get('_name_or_path')
    reference=next((r for r in rules['release_references'] if repo in r['repos']),None)
    ids={i for r in applicable for i in r['sources']}
    if reference:ids.add(reference['source'])
    return {'version':rules['version'],'rules_sha256':hashlib.sha256(raw).hexdigest(),
            'source_manifest_sha256':hashlib.sha256((ROOT/'sources.json').read_bytes()).hexdigest(),
            'cutoff':catalog['review_cutoff'],'rules':applicable,'published_reference':reference,
            'sources':[s for s in catalog['sources'] if s['id'] in ids],
            'identity_basis':'declared exact repository; not weight authenticity' if reference else 'no exact release match',
            'policy':'Evidence informs constraints and validation; never implies backend support or measured performance.'}
