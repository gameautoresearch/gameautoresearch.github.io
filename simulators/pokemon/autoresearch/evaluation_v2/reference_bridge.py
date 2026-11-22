"""Strict, versioned inference-only export from a fully admitted reference checkpoint."""
import ast
import json
from pathlib import Path
import torch
from integrations.pokemon.network import Policy
from integrations.pokemon.reference_training import trainer
from ..contract import RUNTIME,HERE,sha,identity,write


def architecture():
    node=next(n for n in ast.parse((HERE.parent/'network.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Policy')
    return identity(ast.dump(node,include_attributes=False))


def export(checkpoint,config,destination):
    # This function must run under the checkpoint's original immutable reference
    # source and executed engine, never under a patched mismatch-tolerant loader.
    checkpoint=Path(checkpoint);cp=trainer.admit(checkpoint,config)
    actual=trainer.state_hash(cp['model'])
    if actual!=cp['model_sha256']:raise ValueError('Checkpoint model identity differs')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    weights={key:value.detach().cpu().contiguous() for key,value in cp['model'].items()}
    torch.save(weights,destination/'weights.pt')
    meta={'schema':'pokemon-reference-inference-v1','inference_only':True,'resume_supported':False,
        'checkpoint_sha256':sha(checkpoint),'weights_sha256':sha(destination/'weights.pt'),'model_sha256':actual,
        'original_executable_signature':cp['executable_signature'],
        'original_reference_source_signature':cp['reference_source_signature'],
        'original_engine_runtime_signature':cp['engine_runtime_signature'],
        'original_config_signature':cp['config_signature'],'original_manifest_sha256':cp['manifest_sha256'],
        'network':cp['network'],'policy_architecture_ast_sha256':architecture(),
        'runtime_file_sha256':{n:sha(HERE.parent/n) for n in RUNTIME},
        'interface':{'observation_values':700,'actions':14,'visibility':'own700value projection','format':'gen9randombattle'},
        'admission':'All original source/config/executed-engine/population bindings passed trainer.admit; only tensors exported. Extended horizon changes evaluator termination policy, not model inputs/actions.'}
    meta['manifest_sha256']=identity(meta);write(destination/'opponent.json',meta,exclusive=True)
    load(destination)
    return meta


def load(directory):
    directory=Path(directory);meta=json.loads((directory/'opponent.json').read_text())
    if meta.get('schema')!='pokemon-reference-inference-v1' or not meta.get('inference_only'):raise ValueError('Unrecognized reference inference bundle')
    if identity({k:v for k,v in meta.items() if k!='manifest_sha256'})!=meta['manifest_sha256']:raise ValueError('Reference manifest changed')
    if sha(directory/'weights.pt')!=meta['weights_sha256']:raise ValueError('Reference tensor bytes changed')
    if architecture()!=meta['policy_architecture_ast_sha256']:raise ValueError('Reference architecture changed')
    for name,digest in meta['runtime_file_sha256'].items():
        if sha(HERE.parent/name)!=digest:raise ValueError('Reference public runtime changed: '+name)
    if trainer.engine_runtime_signature()[0]!=meta['original_engine_runtime_signature']:raise ValueError('Executed engine changed')
    weights=torch.load(directory/'weights.pt',weights_only=True,map_location='cpu')
    if trainer.state_hash(weights)!=meta['model_sha256']:raise ValueError('Reference model identity changed')
    model=Policy(**meta['network']);model.load_state_dict(weights,strict=True);model.eval()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite reference model')
    return model,meta
