"""Content-addressed policy packages. Candidate source is never imported here."""
from pathlib import Path
import json
import shutil
from .contract import EDITABLE, BENCHMARK_ID, identity, sha, write

FILES = (*EDITABLE, 'config.json', 'weights.pt', 'binding.json')

def verify(directory, expected=None):
    root = Path(directory)
    if root.is_symlink():
        raise ValueError('Symlinked bundle')
    manifest = json.loads((root/'manifest.json').read_text())
    if set(manifest['files']) != set(FILES) or manifest['benchmark_id'] != BENCHMARK_ID:
        raise ValueError('Wrong bundle contract')
    for name in FILES:
        path = root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024:
            raise ValueError('Invalid bundle file: '+name)
        if sha(path) != manifest['files'][name]:
            raise ValueError('Bundle content changed: '+name)
    from .contract import source_signature, validate_recipe
    config=validate_recipe(json.loads((root/'config.json').read_text()))
    binding=json.loads((root/'binding.json').read_text())
    if binding.get('source_sha256')!=source_signature(root) or binding.get('config_sha256')!=identity(config) or binding.get('benchmark_id')!=BENCHMARK_ID:
        raise ValueError('Bundle source/config binding changed')
    actual = identity({k:v for k,v in manifest.items() if k != 'bundle_sha256'})
    if actual != manifest['bundle_sha256'] or expected is not None and expected != actual:
        raise ValueError('Bundle identity mismatch')
    return manifest

def seal(output, source, config, checkpoint, binding):
    """Called in a clean trusted sealer after every unprivileged trainer is dead."""
    import torch
    from .contract import source_signature, runtime_signature, validate_recipe
    validate_recipe(config)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = {'source_sha256':source_signature(source), 'runtime_sha256':runtime_signature(),
                'config_sha256':identity(config), 'benchmark_id':BENCHMARK_ID}
    if any(cp.get(k) != v for k,v in expected.items()):
        raise ValueError('Source/config/checkpoint binding mismatch')
    weights = cp['model']
    if not isinstance(weights, dict) or not weights or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Nonfinite/non-tensor weights')
    if sum(v.numel() for v in weights.values()) > 10_000_000:
        raise ValueError('Policy exceeds ten million parameter/buffer elements')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    for name in EDITABLE:
        shutil.copyfile(Path(source)/name, root/name)
    write(root/'config.json', config)
    torch.save(weights, root/'weights.pt')
    write(root/'binding.json', dict(binding, **expected, iteration=cp['iteration']))
    body = {'schema_version':1, 'benchmark_id':BENCHMARK_ID, 'files':{n:sha(root/n) for n in FILES}}
    body['bundle_sha256'] = identity(body)
    write(root/'manifest.json', body)
    for path in root.iterdir(): path.chmod(0o444)
    return verify(root)
