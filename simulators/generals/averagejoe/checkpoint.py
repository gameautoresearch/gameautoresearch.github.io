"""Strict source/config/tensor/checksum admission for reference checkpoints."""
from pathlib import Path
import json
from .bootstrap import binding,sha


def verify(directory,label='final-state'):
    root=Path(directory);source=json.loads((root/'source.json').read_text());meta=json.loads((root/(label+'.json')).read_text())
    if source['sha256']!=binding()['sha256'] or meta['binding_sha256']!=source['sha256']:raise ValueError('Reference checkpoint source binding changed')
    if sha(root/(label+'.eqx'))!=meta['sha256']:raise ValueError('Reference checkpoint bytes changed')
    from .framework import recipe
    if json.loads((root/'config.json').read_text())!=recipe().to_dict():raise ValueError('Reference configuration drift')
    return meta
