"""Small, sourced reference catalog. Never infer training hardware from weights.

Matching metadata identifies a declared model, not checkpoint authenticity.
Unknown releases remain unknown instead of inheriting an older family recipe.
"""
from copy import deepcopy

CARD = 'https://huggingface.co/zai-org/GLM-5.3-Flash-BF16'
REFERENCES = {
    'zai-org/GLM-5.3-Flash-BF16': {
        'name':'GLM-5.3-Flash-BF16', 'family':'GLM-5.3-Flash',
        'model_types':['glm5_next','glm5_next_text'],
        'total_parameters':320_000_000_000,'active_parameters':18_000_000_000,
        'training':{'phase':'multimodal pretraining','tokens':30_000_000_000_000,
                    'gpu':None,'gpu_count':None,'precision':None,'parallelism':None,
                    'zero':None,'batch':None,'sequence_length':None},
        'sources':[{'url':CARD,'title':'Official model card · Introduction','checked':'2026-10-07'}],
        'scope_note':'GLM-5 report 2602.15763 is linked by the card, but is NOT evidence for the exact GLM-5.3-Flash training hardware. Evaluation context lengths are not training sequence lengths.',
    },
}
REFERENCES['zai-org/GLM-5.3-Flash']=deepcopy(REFERENCES['zai-org/GLM-5.3-Flash-BF16'])
REFERENCES['zai-org/GLM-5.3-Flash']['name']='GLM-5.3-Flash'
REFERENCES['zai-org/GLM-5.3-Flash']['sources'][0]['url']='https://huggingface.co/zai-org/GLM-5.3-Flash'


def model_identity(model):
    m=model or {};cfg=m.get('config') or {}
    repo=m.get('repo')
    if repo not in REFERENCES:
        repo=next((key for key,ref in REFERENCES.items()
                   if m.get('model_card_title')==ref['name']
                   and (m.get('model_type') in ref['model_types'] or cfg.get('model_type') in ref['model_types'])),None)
    ref=deepcopy(REFERENCES.get(repo))
    return {'display_name':ref['name'] if ref else m.get('model_card_title') or m.get('repo') or m.get('name') or None,
            'official_name':ref['name'] if ref else None,'repo':repo or m.get('repo'),
            'identity_basis':'matched_release_metadata' if ref else 'unverified_metadata',
            'checkpoint_verified':False,
            'local_folder':m.get('path'), 'stored_parameter_count':m.get('parameter_estimate'),
            'count_basis':m.get('count_basis') or 'Remote metadata; not locally verified',
            'official':ref}
