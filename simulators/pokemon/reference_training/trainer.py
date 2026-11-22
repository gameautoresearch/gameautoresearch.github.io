"""Sparse-outcome current/history PPO, using the unchanged supported Showdown stack."""
from __future__ import annotations
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from .. import cli
from ..env import vector_for, evaluation_vector_for
from ..network import Policy
from ..rl import Population, Timings, rollout, inference_for, update
from ..provenance import Ledger, canonical, digest, manifest as original_manifest, ENGINE

HERE=Path(__file__).resolve().parent

def source_signature():
    return hashlib.sha256(canonical({str(p.relative_to(HERE)):digest(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.cjs') and '__pycache__' not in p.parts}).encode()).hexdigest()

def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def engine_runtime_signature():
    files={str(p.relative_to(ENGINE)):digest(p) for directory in (ENGINE/'dist',ENGINE/'node_modules') for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.js','.cjs','.json','.node')}
    if not files:raise ValueError('Missing executed engine build/dependencies')
    return hashlib.sha256(canonical(files).encode()).hexdigest(),files

def manifest(out,cfg,command,inputs):
    pin_path=HERE.parents[2]/'ENGINE-PIN.json'
    if not pin_path.exists():return original_manifest(out,cfg,command,inputs)
    # Explicit manifest-based source validation for the historical image, which
    # intentionally contains tracked files without .git or the git executable.
    expected=json.loads(pin_path.read_text())
    for name,sha in expected['tracked_source_sha256'].items():
        if digest(ENGINE/name)!=sha:raise ValueError('Frozen Showdown source mismatch: '+name)
    if expected['engine_pin']!='567c1cedddeaa2f35576005d3bfd526cf4f8a13e':raise ValueError('Engine pin mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    signature,files=engine_runtime_signature()
    record={'schema':'reference-source-manifest-v1','config':cfg,'command':command,'inputs':inputs,'engine_source':expected,'engine_runtime_sha256':signature,'engine_runtime_files':files,'executable_signature':cli.executable_signature(),'reference_source_signature':source_signature(),'source_snapshot':'immutable release tar contains full executed adapter and training source','host':{'torch':torch.__version__,'numpy':np.__version__},'created_unix':time.time()}
    atomic(out/'manifest.json',record);return digest(out/'manifest.json')

def admit(path,cfg):
    # Campaign-generated files only. External checkpoints are never deserialized here.
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp.get('executable_signature')!=cli.executable_signature():raise ValueError('Runtime source mismatch')
    if cp.get('engine_runtime_signature')!=engine_runtime_signature()[0]:raise ValueError('Executed engine runtime mismatch')
    if cp.get('reference_source_signature')!=source_signature():raise ValueError('Reference source mismatch')
    if cp['config_signature']!=cli.signature(cfg):raise ValueError('Recipe mismatch')
    model=Policy(**cp['network']);model.load_state_dict(cp['model'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite model')
    if cp.get('reference_schema')!=1 or len(cp['history'])!=len(cp['population_lineage']):raise ValueError('Population lineage mismatch')
    if cp.get('strong_schema'):
        from .strong_runtime import validate_bindings
        validate_bindings(cp['config'])
    for state,entry in zip(cp['history'],cp['population_lineage'],strict=True):
        if state_hash(state)!=entry['policy_sha256']:raise ValueError('Population identity mismatch')
    return cp

def save(path,model,optimizer,cfg,iteration,rng,population,lineage,ledger,manifest_hash):
    cli.checkpoint(path,model,optimizer,cfg,iteration,rng,population,ledger,manifest_hash)
    cp=torch.load(path,map_location='cpu',weights_only=False)
    cp.update(reference_schema=1,reference_source_signature=source_signature(),engine_runtime_signature=engine_runtime_signature()[0],population_lineage=lineage,
              model_sha256=state_hash(cp['model']),initialization='random; no teacher or imported weights',
              config=cfg,matchmaking='independent fixed mixture; uniform FIFO history; alternating learner seat; latest/latest trains both seats')
    tmp=path.with_suffix('.tmp');torch.save(cp,tmp);tmp.replace(path)

def train(cfg,out,resume=None,*,persist=lambda:None):
    start=time.perf_counter();out=Path(out)
    mh=manifest(out,cfg,'reference-selfplay-train',{'resume_sha256':digest(resume) if resume else None})
    model=cli.initialize(cfg);device=cfg['hardware']['device'];optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],eps=1e-5)
    rng=np.random.default_rng(cfg['seed']);population=Population(cfg,model,device);iteration=0;prior=0;lineage=[]
    if resume:
        cp=admit(resume,cfg);model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
        population=Population(cfg,model,device,cp['history']);lineage=cp['population_lineage']
        rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng']);random.setstate(cp['python_rng'])
        if cp['cuda_rng'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in cp['cuda_rng']])
        iteration=cp['iteration'];prior=cp['allocated_worker_seconds']
    else:
        population.add(model);lineage=[{'iteration':0,'policy_sha256':state_hash(model.state_dict())}]
    ledger=Ledger(start,prior);timings=Timings();initial=state_hash(model.state_dict());reports=[]
    initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not resume:
        save(out/'initial.pt',model,optimizer,cfg,iteration,rng,population,lineage,ledger,mh)
    deadline=start+cfg['allocation_seconds'];episodes=cfg['hardware']['workers']*cfg['hardware']['environments_per_worker']
    with vector_for(cfg) as vector:
        inference=inference_for(cfg,timings)
        while iteration<cfg['iterations'] and time.perf_counter()<deadline:
            seeds=list(range(cfg['training_seed_start']+iteration*episodes,cfg['training_seed_start']+(iteration+1)*episodes))
            assignments=population.assignments(episodes,iteration*episodes,rng,model)
            opponent_ids=[]
            for pair in assignments:
                opponent_ids.append([p if isinstance(p,str) else 'latest' if p is model else state_hash(p.state_dict()) for p in pair])
            samples,report=rollout(vector,model,assignments,seeds,cfg,rng,inference,timings,deadline)
            stats=update(model,optimizer,samples,cfg,device,timings) if time.perf_counter()<deadline else {'samples':0,'skipped':'allocation_expired'}
            if any(isinstance(v,float) and not np.isfinite(v) for v in stats.values()):raise FloatingPointError('Nonfinite PPO diagnostics')
            iteration+=1
            if iteration%cfg['snapshot_every']==0:
                population.add(model);lineage.append({'iteration':iteration,'policy_sha256':state_hash(model.state_dict())});lineage=lineage[-cfg['history_size']:]
            report.update(iteration=iteration,ppo=stats,opponent_identities=opponent_ids,seeds=seeds,population_lineage=copy.deepcopy(lineage),allocated_worker_seconds=ledger.elapsed)
            reports.append(report)
            with (out/'metrics.jsonl').open('a') as f:f.write(canonical(report)+'\n')
            save(out/'checkpoint.pt',model,optimizer,cfg,iteration,rng,population,lineage,ledger,mh)
            persist()
            print(canonical({'iteration':iteration,'samples':stats['samples'],'natural':report['natural_terminals'],'caps':report['truncations'],'ppo':stats}),flush=True)
    if not (out/'checkpoint.pt').exists():save(out/'checkpoint.pt',model,optimizer,cfg,iteration,rng,population,lineage,ledger,mh)
    cp=admit(out/'checkpoint.pt',cfg)
    result={'iterations_completed':iteration,'iterations_this_process':len(reports),'trained_samples':sum(r['ppo']['samples'] for r in reports),'natural_terminals':sum(r['natural_terminals'] for r in reports),'truncations':sum(r['truncations'] for r in reports),'visible_rejections':sum(r['rejected_visible_choices'] for r in reports),'initial_policy_sha256':initial,'final_policy_sha256':state_hash(model.state_dict()),'parameter_change_l2':sum(float((v.detach().cpu()-initial_state[k]).square().sum()) for k,v in model.state_dict().items())**.5,'checkpoint_sha256':digest(out/'checkpoint.pt'),'source_admitted':True,'population_size':len(cp['history']),'timings':timings.result(),'gpu_executed':device.startswith('cuda'),'allocation_this_process_seconds':time.perf_counter()-start,**ledger.result()}
    atomic(out/'results.json',result);atomic(out/'accounting.json',ledger.result());persist();return result

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b,strict=True))
    return a==b

def resume_proof(cfg,out,*,persist=lambda:None):
    """Exact same next batch, including Adam, both RNGs and frozen population."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    base=copy.deepcopy(cfg);base.update(iterations=2,allocation_seconds=90)
    train(base,out/'first',persist=persist)
    continuation=dict(base,iterations=3)
    train(continuation,out/'resume-a',out/'first/checkpoint.pt',persist=persist)
    train(continuation,out/'resume-b',out/'first/checkpoint.pt',persist=persist)
    a=admit(out/'resume-a/checkpoint.pt',continuation);b=admit(out/'resume-b/checkpoint.pt',continuation)
    keys=['model','optimizer','numpy_rng','torch_rng','python_rng','cuda_rng','history','population_lineage','iteration','next_battle_index']
    checks={k:equal(a[k],b[k]) for k in keys}
    if not all(checks.values()):raise AssertionError(checks)
    rejected=False
    broken=dict(continuation,seed=cfg['seed']+1)
    try:admit(out/'resume-a/checkpoint.pt',broken)
    except ValueError:rejected=True
    result={'exact_resume':checks,'changed_recipe_rejected':rejected,'valid':all(checks.values()) and rejected}
    atomic(out/'proof.json',result);persist();return result

def assess(cfg,checkpoint,schedule,out,opponents=None):
    cp=admit(checkpoint,cfg);model=Policy(**cp['network']).to(cfg['hardware']['device']);model.load_state_dict(cp['model']);model.eval()
    rows=[];out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name,path in (opponents or {'power':None,'random':None}).items():
        frozen=None
        if path:
            peer=torch.load(path,map_location='cpu',weights_only=False)
            # Explicit same-interface peer path supplied by evaluator; record source separately.
            frozen=Policy(**peer['network']).to(cfg['hardware']['device']);frozen.load_state_dict(peer['model']);frozen.eval()
        panel=dict(schedule,games=[dict(s,opponent='checkpoint' if path else name) for s in schedule['games']])
        with evaluation_vector_for(cfg,len(panel['games'])) as vector:
            result=cli.evaluation(vector,model,panel,cfg,Timings(),frozen,deadline=time.perf_counter()+cfg['allocation_seconds'])
        result.update(opponent_name=name,opponent_checkpoint_sha256=digest(path) if path else None)
        atomic(out/(name+'.json'),result);rows.append(result)
    atomic(out/'summary.json',{'checkpoint_sha256':digest(checkpoint),'panels':rows});return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','resume-proof']);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    cfg=cli.config_load(a.config)
    print(json.dumps(train(cfg,a.output,a.resume) if a.command=='train' else resume_proof(cfg,a.output),indent=2))
if __name__=='__main__':main()
