"""Offline fixtures (numpy only): a sealed stand-in bundle and a development base release shaped like the real ones."""
import json,shutil,tempfile
from pathlib import Path
import numpy as np
from simulators.clash.autoresearch.common import HERE,dump,digest,sha,engine_binding
from simulators.clash.autoresearch.bundle import seal
from simulators.clash.autoresearch.release_v3 import PUBLIC_SEEDS,VALIDATION_SEEDS,fixed_panel,selected_interface


def stand_in_bundle(path):
 """The starter source with placeholder weights. It seals and verifies; it is never imported or played."""
 with tempfile.TemporaryDirectory() as temp:
  source=Path(temp)
  for n in ('policy.py','learner.py','train.py'):shutil.copyfile(HERE/'starter'/n,source/n)
  dump(source/'config.json',{'width':8});np.savez(source/'weights.npz',placeholder=np.zeros(4,np.float32))
  return seal(source,Path(path))


def development_base(path,grading='parallel3',mode='provisional-development'):
 """A format-3 protected-public development release: fixed panel, one learned reference, public and protected schedules.

 Like the releases under evidence/, it also carries files only older harnesses read
 and an evaluator-source fingerprint that no longer matches this tree.
 """
 path=Path(path);public=path/'public';public.mkdir(parents=True);private=path/'private';private.mkdir(mode=0o700)
 identity=stand_in_bundle(private/'opponents/retained-test')['identity']
 panel=fixed_panel()+[{'name':'retained-test','kind':'checkpoint','identity':identity,'bundle':str(private/'opponents/retained-test')}]
 shown=[{k:v for k,v in p.items() if k!='bundle'} for p in panel];interface=selected_interface(grading)
 dump(public/'opponent-registry.json',{'fixed_panel':shown,'learned':[{'name':'retained-test','kind':'checkpoint','identity':identity}],'strong_panel_qualified':False})
 dump(public/'interface.json',interface)
 dump(public/'observation-schema.json',{'schema_version':4,'observation_ints':7024,'actions':2307,'entity_rows':200})
 dump(public/'practice.json',{'engine_seeds':PUBLIC_SEEDS,'policy_seed':20261109,'opponents':shown,'schedule_kind':'public-development'})
 for n in ('instruction.md','brief.md','rules_reader.py'):(public/n).write_text('# read only by an earlier harness\n')
 shutil.copyfile(HERE/'visible_teacher.py',public/'visible_teacher.py')
 dump(private/'grade.json',{'engine_seeds':VALIDATION_SEEDS,'policy_seed':20261109,'opponents':panel,'schedule_kind':'protected-public-workflow-validation'})
 release={'format':3,'benchmark_id':'clash-autoresearch-fixed8-v3.1','mode':mode,'grading':grading,'engine':engine_binding(),'interface_sha256':digest(interface),
  'teacher_sha256':sha(HERE/'visible_teacher.py'),'runtime_source_sha256':{'evaluation_v3.py':'0'*64,'strong_worker.py':'1'*64},'registry_commitment':digest([{'name':'retained-test','identity':identity}]),
  'public_files':{p.name:sha(p) for p in sorted(public.iterdir()) if p.is_file()},'private_schedule_commitment':digest(json.loads((private/'grade.json').read_text())),
  'opponent_identities':{'retained-test':identity},'private_final_feedback_allowed_for_training':False,'normalized_reference_score':None}
 dump(path/'release.json',release);return path
