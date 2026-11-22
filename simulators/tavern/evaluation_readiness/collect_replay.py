"""Collect preselected seat0/first-group traces, without outcome-based case choice."""
import argparse,json,os,hashlib
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .budget import CAMPAIGN

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=CAMPAIGN/'tavern/evaluator';launch=json.loads((root/a.attempt/'provider.json').read_text());allocation=json.loads((root/a.attempt/'allocation.json').read_text())
    if not allocation['provider'].get('finished_at'):raise ValueError('Worker not authoritatively finished')
    panel=json.loads((root/'private-final-v1/panel.json').read_text());registry=json.loads((root/'registry-v5/registry.json').read_text())
    a.out.mkdir(parents=True,mode=0o700,exist_ok=False)
    import modal
    volume=modal.Volume.from_name(launch['volume_name'],environment_name='staging');files={}
    for profile in registry['profiles']:
        name=f"{profile}-{panel['seeds'][0]}-0";dest=a.out/name;dest.mkdir(mode=0o700)
        for f in ('initial.json','final.json','actions.npz','result.json'):
            data=b''.join(volume.read_file('/'+a.attempt+'/screen/traces/'+name+'/'+f));target=dest/f;target.write_bytes(data);target.chmod(0o600);files[name+'/'+f]=hashlib.sha256(data).hexdigest()
    (a.out/'collection.json').write_text(json.dumps(dict(attempt=a.attempt,choice='All3profiles,first committed group,physical candidate seat0; fixed before inspecting traces',volume=launch['volume_name'],files=files),indent=2)+'\n')
    print(json.dumps(dict(attempt=a.attempt,collected_lobbies=3,all_files_hashed=True)))
