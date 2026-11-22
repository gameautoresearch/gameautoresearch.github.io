"""Offline release construction: frozen worker source, the researcher workspace and private grading data.

No training, worker allocation or network access happens here.
"""
from pathlib import Path
import json
import secrets
import shutil
import subprocess
import time

from .contract import HERE,ROOT,GAME,RUNTIME,DEFAULT,digest,sha,frozen_manifest,write_json
from .bundle import export_existing

CONTROL=('__init__.py','contract.py','bundle.py','wire.py','policy_worker.py','arena.py','grading.py','train_entry.py','evaluate_entry.py','export_entry.py','build_cuda.py')

def prepare(out,include_existing=True,practice_bundle=None,practice_name='trained-starter'):
    """Build a new release in `out`.

    include_existing imports the historical development policies under
    evidence/doudizhu-implementation as panel bots; that conversion needs torch.
    practice_bundle adds one sealed network trained from the unmodified starter (a `harness retrain
    --starter` final bundle, `.../final/jobs/final/bundle`) to the public practice panel, so practice
    has an opponent at trained-network strength instead of only random and heuristic play.
    """
    out=Path(out).resolve()
    trained=None
    if practice_bundle is not None:
        from .bundle import verify as verify_bundle
        trained=Path(practice_bundle).resolve()
        verify_bundle(trained)
        if sha(trained/'policy.py')!=sha(HERE/'starter/policy.py'):
            raise ValueError('practice_bundle must be trained from the unmodified starter (its policy.py differs)')
    if out.exists():raise FileExistsError('Choose a fresh release directory; releases/seed manifests are immutable')
    if include_existing:
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            raise RuntimeError('include_existing converts historical checkpoints and needs torch: construct the release with .venv-doudizhu/bin/python, or pass include_existing=false')
    out.mkdir(parents=True)
    frozen=out/'frozen';public=out/'public';private=out/'private'
    frozen.mkdir();public.mkdir();private.mkdir(mode=0o700)
    (frozen/'integrations').mkdir();(frozen/'integrations/__init__.py').write_text('')
    game=frozen/'integrations/doudizhu'
    for name in RUNTIME:
        target=game/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(GAME/name,target)
    for name in CONTROL:
        target=game/'autoresearch'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/name,target)
    (game/'autoresearch/starter').mkdir();shutil.copy2(HERE/'starter/policy.py',game/'autoresearch/starter/policy.py')
    source=ROOT/'candidates/DouZero';dest=frozen/'candidates/DouZero'
    lock=json.loads((GAME/'source.lock.json').read_text())
    for name in lock['files']:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    # Preserve actual offline upstream objects/HEAD. Drop hooks, logs and local
    # configuration; no credential-bearing Git config is staged into any box.
    shutil.copytree(source/'.git',dest/'.git',ignore=shutil.ignore_patterns('hooks','logs','config','index.lock'))
    (dest/'.git/config').write_text('[core]\n\trepositoryformatversion = 0\n\tbare = false\n')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip()
    if actual!=lock['commit']:raise ValueError('Staged upstream metadata differs from pin')
    for name,value in lock['files'].items():
        if sha(dest/name)!=value:raise ValueError('Staged pinned source mismatch')
    shutil.copytree(HERE/'starter',public/'agent',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(HERE/'recipes',public/'recipes')
    for name in ('benchmark.json','CONTRACT.md'):shutil.copy2(HERE/name,public/name)
    # The researcher sees only the canonical workspace built from this public tree.
    from harness.workspace import stage as stage_workspace
    stage_workspace('doudizhu',public)
    write_json(public/'session.json',DEFAULT)
    write_json(public/'engine-manifest.json',frozen_manifest())
    write_json(private/'grade-seeds.json',{'split':'heldout','deal_seed':secrets.randbits(128),'policy_seed':secrets.randbits(128),'deals':DEFAULT['grade_deals']},exclusive=True)
    write_json(private/'calibration-seeds.json',{'split':'heldout','deal_seed':secrets.randbits(128),'policy_seed':secrets.randbits(128),'deals':DEFAULT['grade_deals']},exclusive=True)
    write_json(private/'anchors.json',{'qualified':False,'status':'not_run','reason':'Full-budget starter/reference calibration is required before an official numeric grade'},exclusive=True)
    bots=[]
    if include_existing:
        base=ROOT/'evidence/doudizhu-implementation'
        candidates=[('learned-small',base/'gpu/followup/learning-seed-17/latest.pt'),
            ('learned-full-a',base/'structural-frontier-learning-gpu-20261108T054525-54f17a/learning-17-frontier/latest.pt'),
            ('learned-full-b',base/'structural-frontier-learning-gpu-20261108T054525-54f17a/learning-29-frontier/latest.pt')]
        for name,checkpoint in candidates:
            if not checkpoint.exists():continue
            identity=export_existing(checkpoint,private/'bots'/name)
            bots.append({'name':name,'kind':'checkpoint','bundle':'bots/'+name,'bundle_sha256':identity,
                'qualification':'historical development policy; no competitive-strength or full-budget anchor claim'})
    simple=[{'name':'random','kind':'random'},{'name':'cooperating-shedder','kind':'heuristic'}]
    practice=simple+bots[:1]
    if trained is not None:
        shutil.copytree(trained,private/'bots'/practice_name)
        identity=json.loads((trained/'manifest.json').read_text())['bundle_sha256']
        practice=practice+[{'name':practice_name,'kind':'checkpoint','bundle':'bots/'+practice_name,'bundle_sha256':identity,
            'qualification':'the unmodified starter trained once at the fresh-final budget; a practice reference, not a grading anchor'}]
    write_json(private/'panel.json',{'practice':practice,'development':simple+bots[1:],
        'official':None,'status':'calibration_required'},exclusive=True)
    write_json(public/'practice-opponents.json',[{'name':p['name'],'kind':p['kind']} for p in practice])
    plan=[]
    for family in ('starter','reference'):
        for seed in (101,211,307):
            plan.append({'name':f'{family}-{seed}','family':family,'seed':seed,
                'allocated_seconds':DEFAULT['gpu_minutes']*60,'gpu':'L40S','cpu_cores':4,'memory_mib':16384,
                'recipe':json.loads((HERE/('recipes/starter.json' if family=='starter' else 'hidden/reference.json')).read_text())})
    shutil.copytree(HERE/'hidden',private/'reference',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(private/'calibration-plan.json',{'status':'not_run','jobs':plan,
        'gate':'equal full allocated budgets; fresh private paired deals; reference gap >=.05 and paired bootstrap lower bound >0; multi-seed validation'},exclusive=True)
    manifest={'version':DEFAULT['version'],'created_unix':time.time(),'engine':frozen_manifest(),
        'public_files':{str(p.relative_to(public)):sha(p) for p in public.rglob('*') if p.is_file()},
        'private_seed_commitments':{n:sha(private/n) for n in ('grade-seeds.json','calibration-seeds.json')},
        'existing_development_bots':len(bots),'gpu_jobs_started':0,'model_calls':0,
        'official_grading_ready':False,'readiness':'prepared; full-budget calibration intentionally unrun'}
    manifest['frozen_files']={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()}
    write_json(out/'release.json',manifest,exclusive=True)
    audit_release(out)
    return manifest

def audit_release(out):
    out=Path(out);m=json.loads((out/'release.json').read_text())
    for tree,key in (('public','public_files'),('frozen','frozen_files')):
        actual={str(p.relative_to(out/tree)) for p in (out/tree).rglob('*') if p.is_file()}
        if actual!=set(m[key]):raise ValueError('Unmanifested or missing release files: '+tree)
    for name,value in m['public_files'].items():
        if sha(out/'public'/name)!=value:raise ValueError('Public context drift: '+name)
    for name,value in m['frozen_files'].items():
        if sha(out/'frozen'/name)!=value:raise ValueError('Frozen context drift: '+name)
    for name,value in m['private_seed_commitments'].items():
        if sha(out/'private'/name)!=value:raise ValueError('Private seed commitment drift')
    forbidden=('private','hidden','grade-seeds','calibration','reference','weights.pt','.env','.key')
    for root in (out/'public',out/'frozen'):
        for p in root.rglob('*'):
            if p.is_symlink():raise ValueError('Symlink in staged release')
            if any(part in ('private','hidden') for part in p.relative_to(root).parts) or p.name in ('grade-seeds.json','calibration-seeds.json') or p.suffix in ('.key','.pt'):
                raise ValueError('Private asset staged: '+str(p))
    return {'public_files':len(m['public_files']),'frozen_files':sum(p.is_file() for p in (out/'frozen').rglob('*')),
        'hidden_files_staged':0,'official_grading_ready':m['official_grading_ready']}
