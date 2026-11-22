import json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from simulators.tavern.autoresearch.evaluation_v2.release import write,sha,verify,RUNTIME,GAME
from simulators.tavern.autoresearch.evaluation_v2.registry import append

def minimal_release(root):
 root.mkdir();(root/'frozen').mkdir();(root/'frozen/marker.py').write_text('protected=True\n');private=root/'private';(private/'registry').mkdir(parents=True)
 pin={'benchmark_id':'task-ABI-preserved','rules':{'num_players':4}};reg={'binding':pin,'opponents':[dict(policy_id='ppo',kind='learned'),*(dict(policy_id=s,kind='script',style=s) for s in ('tempo','economy','synergy'))],'profiles':{'scripts':['tempo','economy','synergy'],'mixed':['ppo','economy','tempo'],'learned':['ppo','ppo','synergy']}}
 write(private/'registry/registry.json',reg);write(private/'registry/manifest.json',{'registry.json':sha(private/'registry/registry.json')});write(root/'binding.json',pin)
 write(root/'release.json',{'evaluator_id':'tavern-four-small-v1','frozen_files':{'marker.py':sha(root/'frozen/marker.py')},'registry_manifest_sha256':sha(private/'registry/manifest.json'),'binding_sha256':sha(root/'binding.json'),'decision_source':{f:sha(GAME/'evaluation_readiness'/f) for f in RUNTIME}})
 (root/'public').mkdir();return root

def test_mixed_release_rejects_mutated_or_missing_registry(tmp_path):
 r=minimal_release(tmp_path/'release');assert verify(r)['evaluator_id']=='tavern-four-small-v1'
 (r/'private/registry/registry.json').write_text('{}')
 with pytest.raises(ValueError,match='registry mutated'):verify(r)

def test_registration_never_imports_entrant_code(tmp_path):
 r=minimal_release(tmp_path/'release');bundle=tmp_path/'bundle';bundle.mkdir();sentinel=tmp_path/'IMPORTED'
 for f in ('policy.py','learner.py','train.py'):(bundle/f).write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').touch()\nraise RuntimeError("Must never import on host")\n')
 np.savez(bundle/'weights.npz',weight=np.ones(3,np.float32));write(bundle/'identity.json',{'binding':json.loads((r/'binding.json').read_text()),'source':{f:sha(bundle/f) for f in ('policy.py','learner.py','train.py')},'config':{}});write(bundle/'summary.json',{'identity_sha256':sha(bundle/'identity.json'),'weights_sha256':sha(bundle/'weights.npz')})
 out=append(r/'private/registry',bundle,tmp_path/'augmented','entrant-safe')
 assert not sentinel.exists()
 data=json.loads((out/'registry.json').read_text());assert data['profiles']==json.loads((r/'private/registry/registry.json').read_text())['profiles'];assert data['opponents'][-1]['role']=='isolated-entrant'

def test_stream_exclusions_include_nested_forks_and_historical_panels(tmp_path):
 from simulators.tavern.autoresearch.evaluation_v2.seed_exclusions import collect
 from simulators.tavern.evaluation_readiness.fresh_bank import lobby_key,training_overlap
 write(tmp_path/'identity.json',{'config':{'seed':71,'n_games':4,'updates':99},'reference_lineage':{'parent':{'config':{'seed':79,'n_games':8,'updates':200}}}})
 write(tmp_path/'public-panel.json',{'seeds':[123,456],'games_per_seat':3})
 keys,configs,receipt=collect(roots=[tmp_path])
 assert {lobby_key(s,g) for s in (123,456) for g in range(3)}<=keys
 assert {(c['seed'],c['updates']) for c in configs}=={(71,99),(79,200)}
 assert any(training_overlap(71,c) for c in configs)
 assert receipt['entire_declared_update_ceilings'] and len(receipt['schedule_declarations'])==2

def test_strong_admission_rejects_unindexed_parent_and_forged_training_summary():
 from simulators.tavern.evaluation_readiness.five_hour.admission import parent_chain,verify_training_summary
 with pytest.raises(ValueError,match='admitted full checkpoint'):
  parent_chain({'kind':'explicit-strong-parent-warmstart','parent_resume_sha256':'unregistered','not_an_exact_old_source_resume':True},{},{})
 ck={'schema':'tavern-strong-population-v1','update':1,'history':[{'learning':{'samples':10,'loss':0.3,'parameter_delta_l2':0.1}}]}
 summary={'updates':1,'samples':10,'finite_losses':True,'changed_weights':True}
 assert verify_training_summary(ck,summary,ck['schema'])==10
 summary['samples']=11
 with pytest.raises(ValueError,match='training history differs'):verify_training_summary(ck,summary,ck['schema'])

def test_paired_seed_comparison_never_filters_faulted_required_cell(tmp_path):
 from simulators.tavern.evaluation_readiness.five_hour.compare import compare
 rows=[dict(profile=p,seed=s,seat=i,lineup=['same-reference']*8,native='fixed',complete_valid_lobby=True,score_eligible=True,failure=None,candidate_placement=4) for p in ('scripts','incumbent-mixed','diverse-history') for s in (123,456) for i in range(8)]
 challenger=[dict(r,candidate_placement=3) for r in rows];a=tmp_path/'a.json';b=tmp_path/'b.json';write(a,rows);write(b,challenger)
 result=compare(a,b);assert result['paired_cases']==48 and result['seed_groups']==2 and result['raw_score_gain']==pytest.approx(100/7)
 challenger[0].update(complete_valid_lobby=False,score_eligible=False,failure={'kind':'simulator_fault'});write(b,challenger)
 assert compare(a,b)['raw_score_gain'] is None
 write(b,challenger[:-1])
 with pytest.raises(ValueError,match='schedules differ'):compare(a,b)
 write(a,rows[:32]);write(b,rows[:32])
 with pytest.raises(ValueError,match='three-profile'):compare(a,b,expected_groups=2)

def test_terminal_recording_survives_batch_reset_and_matches_independent_replay(tmp_path):
 import taverngym as tg
 from simulators.tavern.evaluation_readiness import screen,lobby
 from simulators.tavern.evaluation_readiness.opponents import VisibleScript
 from simulators.tavern.evaluation_readiness.five_hour.recording import install
 from simulators.tavern.evaluation_readiness.five_hour.terminal_replay import audit_dir
 from simulators.tavern.autoresearch.contract import RULES
 original=screen.run_lobby;undo=install();trace=tmp_path/'trace'
 try:result=screen.run_lobby([VisibleScript('tempo') for _ in range(RULES['num_players'])],50101,RULES,trace=trace)
 finally:undo()
 assert screen.run_lobby is original and lobby.tg is tg and result['complete_valid_lobby']
 terminal=trace/'actual-terminal-game.json';assert sha(terminal)==result['actual_terminal_game_sha256']
 assert tg.Game.from_json(terminal.read_text()).is_over()
 assert not tg.Game.from_json(json.dumps(json.loads((trace/'final.json').read_text())['game'])).is_over()
 digest=sha(terminal);proof=audit_dir(trace)
 assert proof['exact_true_terminal_game_state'] and proof['exact_all8_natural_terminals'] and sha(terminal)==digest

def test_direct_game_demo_seed_excludes_equivalent_native_batch_stream(tmp_path):
 import taverngym as tg
 from simulators.tavern.autoresearch.contract import RULES
 from simulators.tavern.autoresearch.evaluation_v2.seed_keys import sm,inverse_sm,lobby_key
 from simulators.tavern.autoresearch.evaluation_v2.seed_exclusions import collect
 base=531101;direct=sm(sm(lobby_key(base)))
 batch=tg.BatchEnv(1,seed=base,controlled=range(RULES['num_players']),config=RULES,threads=1)
 assert batch.game(0).state_json()==tg.Game(direct,json.dumps(RULES)).state_json()
 write(tmp_path/'public-film.json',{'provenance':{'demoSeed':direct}})
 keys,_,receipt=collect(roots=[tmp_path]);assert lobby_key(base) in keys
 for value in (0,1,12345,2**64-1):assert inverse_sm(sm(value))==value
 assert receipt['direct_Game_and_BatchEnv_public_seed_interpretations_excluded']

def test_final_seed_host_import_does_not_require_torch():
 import subprocess,sys
 script='''import importlib.abc,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='torch':raise RuntimeError('Operator environment has no Torch')
sys.meta_path.insert(0,NoTorch())
from harness.games.tavern.final_bank import panel_for
from simulators.tavern.autoresearch.evaluation_v2.seed_exclusions import collect
assert 'torch' not in sys.modules
'''
 subprocess.run([sys.executable,'-c',script],check=True,capture_output=True,text=True,timeout=15)
def test_public_selection_keeps_best_and_latest_without_private_feedback():
 from simulators.tavern.evaluation_readiness.five_hour.selection import choose
 def row(name,update,score):return dict(candidate=name,update=update,public_score=score,checkpoint_sha256=name,policy_sha256='fixed')
 rows=[row('new-plain266',266,70),row('new-quarter1',1,75),row('new-quarter2',2,75),row('new-quarter3',3,None),row('new-anchor1',1,73),row('new-anchor2',2,72)]
 result=choose(rows,['new-plain266'])
 assert result['arms']['quarter']['best']['candidate']=='new-quarter1'
 assert result['arms']['quarter']['latest']['candidate']=='new-quarter3'
 assert result['primary']['candidate']=='new-quarter1' and result['private_results_used'] is False
 assert result['roster']==['new-plain266','new-quarter1','new-quarter3','new-anchor1','new-anchor2']


def test_public_selection_requires_complete_evidence_for_each_arm():
 from simulators.tavern.evaluation_readiness.five_hour.selection import choose
 import pytest
 row=lambda n,u,s:dict(candidate=n,update=u,public_score=s,checkpoint_sha256=n,policy_sha256='fixed')
 with pytest.raises(ValueError,match='complete original PUBLIC192'):
  choose([row('new-plain266',266,70),row('new-quarter1',1,75),row('new-anchor1',1,None)],['new-plain266'])


def test_final_bank_rejects_roster_drift_after_global_close(tmp_path,monkeypatch):
 from simulators.tavern.evaluation_readiness.five_hour import final_bank
 root=tmp_path/'campaign';coord=root/'tavern/coordination';coord.mkdir(parents=True);write(coord/'rl.json',{'selection_closed':True})
 own=root/'tavern/evaluator';own.mkdir();write(own/'final-assessment-policy-v1.json',{'frozen':'policy'})
 write(own/'public-selection-closed-v1.json',{'selection_closed':True,'selection_policy_sha256':sha(own/'final-assessment-policy-v1.json'),'private_results_used':False,'roster':['selected']})
 monkeypatch.setattr(final_bank,'CAMPAIGN',root)
 monkeypatch.setattr(final_bank,'Budget',lambda:SimpleNamespace(auth={'selection_close_utc':'2000-01-01T00:00:00Z','worker_stop_utc':'2099-01-01T00:00:00Z'}))
 output=tmp_path/'bank'
 with pytest.raises(ValueError,match='closed precommitted public-only selection'):final_bank.freeze(tmp_path/'unused-registry',['different'],output)
 assert not output.exists()



def test_mixed_release_rejects_wrong_profile_width_and_scripts_only_opposition(tmp_path):
 r=minimal_release(tmp_path/'release');registry=r/'private/registry'
 def reseal(change):
  reg=json.loads((registry/'registry.json').read_text());change(reg);write(registry/'registry.json',reg);write(registry/'manifest.json',{'registry.json':sha(registry/'registry.json')})
  manifest=json.loads((r/'release.json').read_text());manifest['registry_manifest_sha256']=sha(registry/'manifest.json');write(r/'release.json',manifest)
 reseal(lambda reg:reg['profiles'].update(scripts=['tempo','economy','synergy','tempo']))
 with pytest.raises(ValueError,match='profile contract'):verify(r)
 reseal(lambda reg:reg.update(profiles={k:['tempo','economy','synergy'] for k in ('a','b','c')}))
 with pytest.raises(ValueError,match='No learned reference'):verify(r)


def test_scores_scale_with_the_lobby_size():
 from simulators.tavern.evaluation_readiness.screen import summarize
 rows=[dict(complete_valid_lobby=True,candidate_placement=p,seed=s) for s in (1,2) for p in (1,2,3,4)]
 four=summarize(rows,8,4);assert four['raw_score']==pytest.approx(50) and four['top_half_rate']==.5 and 'top4_rate' not in four
 assert four['placement_histogram']=={'1':2,'2':2,'3':2,'4':2}
 eight=summarize(rows,8,8);assert eight['raw_score']==pytest.approx(100*(8-2.5)/7) and eight['top4_rate']==1.0
