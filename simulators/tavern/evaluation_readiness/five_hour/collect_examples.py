"""Read-only collection of predeclared first-group/seat0 native traces."""
import argparse,json,os,hashlib
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .budget import CAMPAIGN


def collect(attempt,out):
    import modal
    root=CAMPAIGN/'tavern/evaluator'/attempt;launch=json.loads((root/'provider.json').read_text());allocation=json.loads((root/'allocation.json').read_text());panel=json.loads((root/'panel.json').read_text());registry=json.loads((root/'assets/registry.json').read_text())
    if not allocation['provider'].get('finished_at'):raise ValueError('Finished provider allocation required')
    if hashlib.sha256((root/'panel.json').read_bytes()).hexdigest()!=launch['panel_sha256']:raise ValueError('Schedule changed after launch')
    out=Path(out);out.mkdir(parents=True,mode=0o700,exist_ok=False);files={};volume=modal.Volume.from_name(launch['volume_name'],environment_name='staging')
    for profile in registry['profiles']:
        name=f"{profile}-{panel['seeds'][0]}-0";dest=out/name;dest.mkdir(mode=0o700)
        for f in ('initial.json','final.json','actions.npz','result.json'):
            raw=b''.join(volume.read_file('/'+attempt+'/screen/traces/'+name+'/'+f));target=dest/f;target.write_bytes(raw);target.chmod(0o600);files[name+'/'+f]=hashlib.sha256(raw).hexdigest()
        for f in ('actual-terminal-game.json','terminal-state.json'):
            try:raw=b''.join(volume.read_file('/'+attempt+'/screen/traces/'+name+'/'+f))
            except FileNotFoundError:
                if launch.get('runtime_protocol')=='private-paths-terminal-v2-all-lobbies':raise
                continue
            target=dest/f;target.write_bytes(raw);target.chmod(0o600);files[name+'/'+f]=hashlib.sha256(raw).hexdigest()
    (out/'collection.json').write_text(json.dumps(dict(attempt=attempt,choice='All3profiles,first committed group,physical candidate seat0; fixed before inspecting traces',files=files,volume=launch['volume_name'],private_native_states=True),indent=2)+'\n')
    return dict(attempt=attempt,collected_lobbies=3,all_files_hashed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(collect(a.attempt,a.out)))
