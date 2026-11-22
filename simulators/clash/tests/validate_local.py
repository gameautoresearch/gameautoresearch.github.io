"""Bounded serial local end-to-end validation; no cloud allocation or remote calls."""
from pathlib import Path
import json,subprocess,sys,hashlib
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent
import fcntl
_perf_lock=(ROOT/'evidence/local-performance.lock').open('a')
fcntl.flock(_perf_lock,fcntl.LOCK_EX)
OUT=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'evidence/clash-implementation/final-validation')
OUT.mkdir(exist_ok=False)
def execute(name,cmd):
 with (OUT/(name+'.log')).open('w') as f:subprocess.run([sys.executable,*map(str,cmd)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def run(name,mode,*extra):
 execute(name,[HERE/'runner.py',mode,'--config',HERE/'configs/smoke.json','--out',OUT/name,*extra])
 return json.loads((OUT/name/'result.json').read_text())
def clean(result):
 assert not result['fault_signatures']
 assert result['counts']['illegal_actions']==result['counts']['unsupported_actions']==0
execute('contracts',[HERE/'tests/test_contract.py'])
execute('resume',[HERE/'tests/resume_check.py',str(OUT/'resume-equivalence')])
train=run('training','train')
assert train['updates']==100 and train['parameter_l2_change']>0
assert train['ppo_counts']['natural_terminals']>0
clean(train)
cp=OUT/'training/checkpoint.pt';initial=OUT/'training/initial.pt'
initial_eval=run('initial-practice','evaluate','--checkpoint',initial,'--schedule',HERE/'configs/eval-practice.json')
practice=run('practice','evaluate','--checkpoint',cp,'--schedule',HERE/'configs/eval-practice.json')
heldout=run('heldout','evaluate','--checkpoint',cp,'--schedule',HERE/'configs/eval-heldout.json')
qualification=run('scripted-qualification','evaluate','--policy','scripted','--schedule',HERE/'configs/eval-practice.json')
for r in (initial_eval,practice,heldout,qualification):
 clean(r);assert r['invalid_episodes']==0 and r['completed_episodes']==r['planned_episodes']==48
# Neural-to-neural frozen-opponent evaluation (recipe support; not an independent strong panel).
schedule={'visibility':'public frozen-opponent smoke','policy_seed':8128,'engine_seeds':[2000100,2000101],
          'opponents':['checkpoint:'+str(cp)]}
(OUT/'frozen-schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
frozen=run('frozen-evaluation','evaluate','--checkpoint',cp,'--schedule',OUT/'frozen-schedule.json')
clean(frozen);assert frozen['completed_episodes']==4
# Cooperative budget accounting: all setup remains charged even when no rollout fits.
stop=run('budget-stop','train','--budget-seconds','0.001','--updates','1')
assert stop['status']=='budget_exhausted' and stop['updates']==0
assert stop['allocated_worker_seconds']>stop['limit_seconds'] and stop['budget_overshoot_seconds']>0
assert stop['allocated_gpu_hours'] is None
# Evaluation scheduling should not alter a job's engine/policy seed stream.
c=json.loads((HERE/'configs/smoke.json').read_text());c['envs']=1
(OUT/'single-env.json').write_text(json.dumps(c,indent=2)+'\n')
execute('single-env',[HERE/'runner.py','evaluate','--config',OUT/'single-env.json','--checkpoint',cp,
 '--schedule',HERE/'configs/eval-practice.json','--out',OUT/'single-env'])
def episodes(path):
 return sorted((json.loads(line) for line in (path/'episodes.jsonl').read_text().splitlines()),key=lambda x:(x['opponent'],x['engine_seed'],x['seat']))
assert episodes(OUT/'single-env')==episodes(OUT/'practice'),'per-job seed streams changed with vector width'
warm=json.loads((OUT/'training/warmstart.json').read_text())
assert warm['teacher_nll_after']<warm['teacher_nll_before']*.7
allcards=c['rules']['deck'];coverage={card:train['counts'].get('card_play_requests_'+card,0) for card in allcards}
assert all(coverage.values())
ledger=json.loads((OUT/'training/allocation-ledger.json').read_text())
assert ledger['allocated_worker_seconds']>train['allocated_worker_seconds']
summary={'status':'passed','source':'integrations/clash/tests/validate_local.py','contract_tests':8,
 'ppo_updates':train['updates'],'ppo_counts':train['ppo_counts'],'teacher_learning':warm,
 'initial_practice':initial_eval,'practice':practice,'heldout':heldout,
 'scripted_qualification':qualification,'frozen_evaluation':frozen,
 'budget_stop':stop,'card_request_coverage':coverage,'evaluation_batch_width_invariance':'passed (1 vs 4 lanes, identical episode records)',
 'resume':json.loads((OUT/'resume-equivalence/result.json').read_text()),
 'cumulative_training_and_development_worker_seconds':ledger['allocated_worker_seconds'],
 'gpu_measurements_executed':False,
 'strength_claim':'no strength improvement established; public diagnostic panel and teacher-data fit only',
 'configuration_sha256':hashlib.sha256((HERE/'configs/smoke.json').read_bytes()).hexdigest()}
(OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
