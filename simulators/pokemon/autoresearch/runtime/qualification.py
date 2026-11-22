"""Live exact native learning/resume, fresh initialization, isolation and export proof."""
import copy,json,os,subprocess,sys,time
from pathlib import Path

def run(spec,out,inputs):
    import torch
    from .supervisor import train
    from .sdk import STARTER,STARTER_CONFIG,write,sha,source_signature,runtime_signature
    from .bundle import verify
    from .env_service import snapshots_equal
    from ...native.resident_suite import assert_exact_tree
    from ..arena import drop_privileges
    source=STARTER;cfg=json.loads(STARTER_CONFIG.read_bytes())
    if 'L40S'not in torch.cuda.get_device_name().upper():raise ValueError('Actual L40S required')
    for label,updates,parent in [('first',2,None),('resumed',2,out/'first/checkpoint.pt'),('uninterrupted',4,None)]:
        train(source,cfg,out/label,120,'cuda:0',resume=parent,updates=updates)
    a=torch.load(out/'resumed/checkpoint.pt',map_location='cpu',weights_only=True);b=torch.load(out/'uninterrupted/checkpoint.pt',map_location='cpu',weights_only=True)
    if not snapshots_equal(a.pop('environment'),b.pop('environment'),inputs/'env-state-key.bin'):raise ValueError('Authenticated native resume differs')
    assert_exact_tree(a,b)
    fresh=dict(cfg,seed=cfg['seed']+17,training_seed_start=cfg['training_seed_start']+1000000)
    train(source,fresh,out/'fresh-isolated',120,'cuda:0',updates=2)
    c=torch.load(out/'fresh-isolated/checkpoint.pt',map_location='cpu',weights_only=True)
    if c['iteration']!=2 or c['binding']['config_sha256']==a['binding']['config_sha256']:raise ValueError('Fresh final initialization/config gate')
    if all(torch.equal(v,c['model'][k])for k,v in a['model'].items()):raise ValueError('Fresh seeds did not change model')
    for label in ('first','resumed','uninterrupted','fresh-isolated'):
        verify(out/label/'bundle');r=json.loads((out/label/'result.json').read_bytes())
        if r['trained_samples']<=0 or r['optimizer_updates']<=0 or r['parameter_change_l2']<=0 or r['faults']or r['truncations']:raise ValueError('Actual learning/export failed')
    code="import os,pathlib; assert os.getuid()==65534; paths=['/pilotjob/input','/campaign','/image-pilot-cache','/image-public-cache','/opt/pk-image-cache','/opt/pilot-public-image-cache','/opt/pk','/runtime','/job','/private'];\nfor p in paths:\n try: list(pathlib.Path(p).iterdir())\n except (PermissionError,FileNotFoundError):continue\n raise AssertionError('Private path readable: '+p)\nassert not any(k in os.environ for k in ['OPENROUTER_API_KEY','MODAL_TOKEN_SECRET','POKEMON_PILOT_ENV_KEY'])\nprint('isolation-pass')"
    probe=subprocess.run([sys.executable,'-c',code],env={'PATH':os.environ['PATH']},preexec_fn=drop_privileges,capture_output=True,text=True,timeout=15)
    (out/'isolation.log').write_text(probe.stdout+probe.stderr)
    if probe.returncode:raise ValueError('Live UID/private-path isolation failed')
    assessment=copy.deepcopy(spec['assessment']);assessment.update(candidate=dict(kind='bundle-native',name='weak-proof',path=str(out/'resumed/bundle'),bundle_sha256=verify(out/'resumed/bundle')['bundle_sha256']),seeds=[900418201],panel=[dict(kind='power',name='power')],public_development=True)
    from .arena import run as assess
    directory=out/'public-panel';directory.mkdir();write(directory/'request.json',assessment);assess(assessment,directory,str(directory/'request.json'))
    reference=json.loads((directory/'summary.json').read_bytes())
    assessment['panel']=[dict(assessment['candidate'],name='weak-peer')]
    directory=out/'public-peer';directory.mkdir();write(directory/'request.json',assessment);assess(assessment,directory,str(directory/'request.json'))
    peer=json.loads((directory/'summary.json').read_bytes())
    if reference['status']!='complete'or peer['status']!='complete':raise ValueError('Public reference/pair workflow failed')
    def cgroup(name):
        p=Path('/sys/fs/cgroup')/name
        return p.read_text()if p.exists()else None
    write(out/'qualification.json',dict(passed=True,kind='exact-native-L40S',runtime_sha256=runtime_signature(),starter_source_sha256=source_signature(source),exact_full_resume=True,fresh_initialization_no_parent=True,uid_private_path_isolation=True,source_only_submission=True,public_reference=reference,public_peer=peer,hardware=dict(gpu=torch.cuda.get_device_name(),cpu_max=cgroup('cpu.max'),memory_max=cgroup('memory.max')),private_seeds_created=False))


def run_cpu(spec,out,inputs):
    """Public exported-graph workflow gate; no private bank or model inference."""
    from .arena import run as assess
    from .sdk import write
    request=copy.deepcopy(spec['assessment'])
    seeds=request.get('seeds')
    if not request.get('public_development') or not seeds or any(type(s)is not int or not 900_000_000<=s<1_000_000_000 for s in seeds):
        raise ValueError('Qualification requires explicit public development seeds')
    candidate=request['candidate']
    from .bundle import verify
    manifest=verify(candidate['path'],candidate['bundle_sha256'])
    assess(request,out,str(inputs/'request.json'))
    summary=json.loads((out/'summary.json').read_bytes())
    valid=(summary.get('status')=='complete' and summary.get('natural_games')==summary.get('scheduled_games')
        and summary.get('scheduled_games',0)>0 and not summary.get('faults') and not summary.get('administrative_caps')
        and not summary.get('unstarted_games'))
    write(out/'qualification-cpu.json',dict(passed=valid,public_development=True,
        bundle_sha256=manifest['bundle_sha256'],candidate_identity=summary.get('candidate_identity'),
        natural_games=summary.get('natural_games'),scheduled_games=summary.get('scheduled_games'),
        score_valid=valid,private_seeds_created=False))
    if not valid:raise ValueError('Public CPU workflow did not complete every scheduled game')
