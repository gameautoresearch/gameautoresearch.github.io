"""Tensor-only reference admission, rooted in a separately attested export manifest."""
import ast,hashlib,json
from pathlib import Path
import torch
from ...network import Policy
from ..contract import ROOT,HERE,RUNTIME,sha,identity


def engine_signature():
    engine=ROOT/'candidates/pokemon-showdown'
    files={str(p.relative_to(engine)):sha(p) for directory in (engine/'dist',engine/'node_modules')
        for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.js','.cjs','.json','.node')}
    if not files:raise ValueError('Executed engine metadata missing')
    return identity(files)


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def load(directory,expected_manifest_sha256):
    """Expected identity must come from the original strict export receipt, not the candidate."""
    directory=Path(directory);meta=json.loads((directory/'opponent.json').read_text())
    if not expected_manifest_sha256 or meta.get('manifest_sha256')!=expected_manifest_sha256:raise ValueError('Reference export attestation differs')
    if identity({k:v for k,v in meta.items() if k!='manifest_sha256'})!=expected_manifest_sha256:raise ValueError('Reference manifest changed')
    if meta.get('schema')!='pokemon-reference-inference-v1' or not meta.get('inference_only') or meta.get('resume_supported') is not False:raise ValueError('Not an admitted inference-only export')
    if meta.get('interface')!={'observation_values':700,'actions':14,'visibility':'own700value projection','format':'gen9randombattle'}:raise ValueError('Reference public interface differs')
    if set(meta['runtime_file_sha256'])!=set(RUNTIME):raise ValueError('Reference source bindings incomplete')
    for name,digest in meta['runtime_file_sha256'].items():
        if sha(HERE.parent/name)!=digest:raise ValueError('Reference public runtime changed: '+name)
    node=next(n for n in ast.parse((HERE.parent/'network.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Policy')
    if identity(ast.dump(node,include_attributes=False))!=meta['policy_architecture_ast_sha256']:raise ValueError('Reference architecture differs')
    if engine_signature()!=meta['original_engine_runtime_signature']:raise ValueError('Reference executed engine differs')
    if sha(directory/'weights.pt')!=meta['weights_sha256']:raise ValueError('Reference tensor bytes changed')
    weights=torch.load(directory/'weights.pt',weights_only=True,map_location='cpu')
    if state_hash(weights)!=meta['model_sha256']:raise ValueError('Reference tensor identity changed')
    model=Policy(**meta['network']);model.load_state_dict(weights,strict=True);model.eval()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite reference model')
    return model,meta
