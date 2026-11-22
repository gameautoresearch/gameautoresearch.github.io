"""Fast-v4 release constructor: the current starter and contract on an existing evaluation panel.

This is the constructor of the full-length (180+120 s, 4 decisions/s) form. The research runtime now plays
the short form (common.MATCH_FORM, short_release.py), so it refuses to build or verify a release here; use
it from git tag full-size-20261113.

`prepare(out, base_release)` reads a protected-public v3/v3.1 development release
and writes a NEW fast-v4 release; the base is never changed. The base's own
evaluator-source fingerprint is recorded but not relied on. What is checked is
what practice still uses: the engine binding, the base's public files, its
evaluation interface, the teacher opponent source, the schedule commitment and
every opponent bundle. No private final bank is created or opened.
"""
import json,shutil
from pathlib import Path
from .common import HERE,EDITABLE,MATCH_FORM,dump,sha,digest,engine_binding
from .fast_entry import runtime_binding
from .bundle import verify as verify_bundle
from .release_v3 import selected_interface,resolve_schedule,fixed_panel

BENCHMARK_ID='clash-autoresearch-fixed8-v4'
# The evaluation panel declaration, carried over byte for byte from the base release.
PANEL_FILES=('interface.json','practice.json','opponent-registry.json','observation-schema.json')

def runtime_source():
 """Every top-level module here: worker trainer and evaluator, ledger, bundle code and this constructor."""
 return {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.glob('*.py'))}

def check_panel(root,release):
 """The protected schedule and its opponent bundles still match the release's commitments."""
 root=Path(root);schedule=json.loads((root/'private/grade.json').read_text())
 if digest(schedule)!=release['private_schedule_commitment']:raise ValueError('frozen schedule drift')
 for opponent in schedule['opponents']:
  if opponent['kind']=='checkpoint' and verify_bundle(opponent['bundle'],opponent['identity'])['identity']!=release['opponent_identities'][opponent['name']]:raise ValueError('opponent identity differs')
 # Public practice names only identities; each must resolve to the same protected bundle.
 resolve_schedule(root,json.loads((root/'public/practice.json').read_text()))
 return schedule

def full_form_only():
 if MATCH_FORM!='full':raise ValueError(f'fast-v4 is the full-length form; this runtime plays {MATCH_FORM} (short_release.py). Use fast-v4 from git tag full-size-20261113')

def base_binding(base_release):
 """Everything a v3/v3.1 development release still proves about its evaluation panel."""
 full_form_only()
 base=Path(base_release);old=json.loads((base/'release.json').read_text())
 if old.get('format')!=3 or old.get('mode')!='provisional-development':raise ValueError('existing protected-public v3/v3.1 development release required; no private final bank is created or opened')
 if old['engine']!=engine_binding():raise ValueError('base engine binding differs')
 if not set(PANEL_FILES)<=set(old['public_files']) or any(sha(base/'public'/n)!=h for n,h in old['public_files'].items()):raise ValueError('base public release drift')
 interface=json.loads((base/'public/interface.json').read_text())
 if digest(interface)!=old['interface_sha256'] or interface!=selected_interface(old.get('grading','sequential')):raise ValueError('base evaluation interface differs from the retained evaluator declaration')
 if old['teacher_sha256']!=sha(HERE/'visible_teacher.py'):raise ValueError('base teacher opponent source differs')
 if not (base/'private/grade.json').is_file():
  # A base restored from an archive that never held its protected schedule: only the opponent bundles
  # remain, and each must still carry the identity the base release recorded for it.
  for name,identity in old['opponent_identities'].items():
   if verify_bundle(base/'private/opponents'/name,identity)['identity']!=identity:raise ValueError('opponent identity differs: '+name)
  return old
 check_panel(base,old)
 return old

def reconstructed_schedule(base,old,seeds=32):
 """A new private registry for a base whose protected schedule is lost: the fixed scripted panel plus the
 base's verified opponent bundles, on fresh secret engine seeds. Never the original protected schedule."""
 import secrets
 opponents=fixed_panel()+[{'name':n,'kind':'checkpoint','identity':i,'bundle':str(base/'private/opponents'/n)}
                          for n,i in sorted(old['opponent_identities'].items())]
 return {'engine_seeds':[secrets.randbelow(2**31-1)+1 for _ in range(seeds)],'policy_seed':secrets.randbelow(2**31-1)+1,
         'opponents':opponents,'schedule_kind':'reconstructed-protected-development'}

def prepare(out,base_release,reconstruct_schedule=False):
 from harness import workspace
 base=Path(base_release).resolve();old=base_binding(base)
 lost=not (base/'private/grade.json').is_file()
 if lost and str(reconstruct_schedule).lower() not in ('1','true','yes'):
  raise ValueError('The base release has no protected schedule (private/grade.json); pass reconstruct_schedule=true to build a new, labelled private registry from its verified opponent bundles')
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);public=out/'public';public.mkdir()
 for n in EDITABLE:shutil.copyfile(HERE/'starter'/n,public/n)
 shutil.copyfile(HERE/'recipes/starter.json',public/'config.json')
 shutil.copyfile(HERE/'recipes/smoke.json',public/'smoke-config.json')
 shutil.copyfile(HERE/'benchmark-v4.json',public/'benchmark.json')
 shutil.copyfile(HERE/'CONTRACT-fast-v4.md',public/'fast-contract.md')
 for n in PANEL_FILES:shutil.copyfile(base/'public'/n,public/n)
 # The protected schedule and panel bundles are copied unchanged. The schedule keeps
 # naming the base release's bundle paths, so the base must stay where it is.
 shutil.copytree(base/'private',out/'private')
 schedule=reconstructed_schedule(base,old) if lost else None
 if schedule:dump(out/'private/grade.json',schedule)
 # The researcher sees only the canonical workspace built from this public tree.
 meta=workspace.stage('clash',public)
 r={**old,'format':4,'benchmark_id':BENCHMARK_ID,'parent_release':str(base),'parent_release_sha256':sha(base/'release.json'),'parent_runtime_source_sha256':old.get('runtime_source_sha256'),'trusted_fast_runtime':runtime_binding(),'runtime_source_sha256':runtime_source(),'public_files':{str(p.relative_to(public)):sha(p) for p in sorted(public.iterdir()) if p.is_file()},'workspace_sha256':meta['sha256'],'training_resources':{'gpu':'NVIDIA L40S','cpu_cores':4,'memory_gib':16},'private_schedule_unchanged':not lost}
 if schedule:r.update(private_schedule_commitment=digest(schedule),protected_schedule='reconstructed: the base release\'s protected schedule was never archived; a new private registry was built from its verified opponent bundles on fresh secret seeds (development only)')
 dump(out/'release.json',r);verify(out);return {'release':str(out),'commitment':digest(r),'benchmark_id':r['benchmark_id'],'private_schedule_unchanged':not lost}

def verify(release):
 from harness import workspace
 full_form_only()
 p=Path(release);r=json.loads((p/'release.json').read_text())
 if r.get('format')!=4 or r['benchmark_id']!=BENCHMARK_ID or r['engine']!=engine_binding():raise ValueError('fast release engine/version binding differs')
 if r['trusted_fast_runtime']!=runtime_binding():raise ValueError('fast runtime source differs; prepare explicit new release')
 if r['runtime_source_sha256']!=runtime_source():raise ValueError('research runtime source differs; prepare explicit new release')
 if any(sha(p/'public'/n)!=h for n,h in r['public_files'].items()):raise ValueError('fast public release drift')
 if r.get('workspace_sha256')!=workspace.verify(p)['sha256']:raise ValueError('researcher workspace differs from the release')
 check_panel(p,r)
 return r
