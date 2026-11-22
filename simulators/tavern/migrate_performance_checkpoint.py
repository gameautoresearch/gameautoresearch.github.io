"""Explicitly migrate only scheduling/binding fingerprints; verify every mechanic/data file.

Weights are untouched. The migration receipt binds the original checkpoint and verifies
that card data and all mechanics sources still match the pilot's immutable source map.
"""
import hashlib
import json
from pathlib import Path
import torch
import taverngym as tg
from provenance import ROOT, ENGINE, sha


def migrate(source, destination):
    source,destination=Path(source),Path(destination)
    state=torch.load(source,map_location='cpu',weights_only=False)
    old=state['native_source_hash']
    allowed={'crates/taverngym/src/batch.rs','crates/taverngym/src/workers.rs',
             'crates/taverngym-py/src/lib.rs','crates/taverngym/src/rl_obs.rs'}
    checked=[]
    for name,fingerprint in state['manifest_source_sha256'].items():
        prefix='candidates/AutoTavern/'
        if not name.startswith(prefix):continue
        relative=name[len(prefix):]
        if (relative.endswith('.rs') and relative not in allowed or
            relative in {'crates/taverngym/data/cards.json','Cargo.toml','Cargo.lock','rust-toolchain.toml','pyproject.toml'}):
            if sha(ROOT/name)!=fingerprint:raise RuntimeError(f'mechanics/data drift: {name}')
            checked.append(name)
    if len(checked)<50:raise RuntimeError('insufficient immutable mechanic evidence')
    state['native_source_hash']=tg.build_info()['source_hash']
    receipt={'original_checkpoint_sha256':sha(source),'original_native_fingerprint':old,
       'new_native_fingerprint':state['native_source_hash'],'checked_mechanic_files':checked,
       'allowed_performance_files':sorted(allowed),'weights_unchanged':True}
    state['performance_migration']=receipt
    torch.save(state,destination)
    destination.with_suffix('.migration.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return destination


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    if args.destination.exists():raise FileExistsError(args.destination)
    migrate(args.source,args.destination)
