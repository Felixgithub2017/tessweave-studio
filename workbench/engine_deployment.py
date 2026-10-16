"""Use the Engine's single offline catalog; never execute deployment recipes."""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=1)
def engine():
    try:
        from flow_engine import deployment_catalog
        return deployment_catalog
    except ModuleNotFoundError as error:
        if error.name not in ('flow_engine', 'flow_engine.deployment_catalog'):
            raise
    source = Path(__file__).resolve().parents[2] / 'flow-inference/flow_engine/deployment_catalog.py'
    if not source.is_file():
        # Standalone installs must not depend on a sibling Engine checkout.
        from . import bundled_deployment_catalog
        return bundled_deployment_catalog
    spec = importlib.util.spec_from_file_location('tessweave_deployment_catalog', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(body):
    return engine().deployment_plan(body['model'], body['hardware'], **{
        key: body[key] for key in ('bits', 'context', 'concurrency', 'memory_gib', 'devices', 'artifact')
        if key in body})


def models():
    """Pinned offline entries use the normal planner model-picker contract."""
    return [dict(key='deployment:'+m['id'],kind='hub',featured=True,repo=m['id'],name=m['id'],
        nominal_parameters_b=m['nominal_b'],config=m['config'].get('text_config') or m['config'],
        revision=m['revision'],metadata_status=m['metadata_status'],metadata_origin=m['metadata_status'],source='huggingface',config_url=m.get('config_source',m['source']),architecture_reference=m.get('architecture_reference'))
        for m in engine()._models()]


def model(key):
    return next((m for m in models() if m['key']==key),None)
