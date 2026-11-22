"""Source-admitted CPU assessment of immutable campaign checkpoints; no training."""
from __future__ import annotations
import argparse,ast,hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def worker(request,index):
    task=request['tasks'][index];root=Path('/tmp/reference-assess')/task['release']
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'source').exists():
        with tarfile.open('/runs/releases/'+task['release']+'/source.tar') as tar:tar.extractall(root,filter='data')
    if not (root/'source/candidates').exists():os.symlink('/opt/pk/candidates',root/'source/candidates')
    sys.path.insert(0,str(root/'source'))
    import copy,numpy as np,torch
    from integrations.pokemon.reference_training import trainer as t
    from integrations.pokemon import cli
    from integrations.pokemon.network import Policy,batch
    from integrations.pokemon.env import evaluation_vector_for
    from integrations.pokemon.rl import Timings
    path=Path(task['checkpoint']);raw=torch.load(path,map_location='cpu',weights_only=False);cfg=raw['config']
    cp=t.admit(path,cfg) # Exact historical trainer/runtime/recipe/engine admission before CPU projection.
    if request.get('bridge'):
        import shutil
        extension=root/'source/integrations/pokemon/autoresearch';(extension/'evaluation_v2').mkdir(parents=True,exist_ok=True)
        for name in ('__init__.py','evaluation_v2/__init__.py'):(extension/name).write_text('')
        for name in ('contract.py','evaluation_v2/reference_bridge.py'):
            src=Path('/runs/assessments')/request['label']/'bridge-source'/name
            if hashlib.sha256(src.read_bytes()).hexdigest()!=request['bridge_source_sha256'][name]:raise ValueError('Evaluator bridge source changed')
            shutil.copyfile(src,extension/name)
        from integrations.pokemon.autoresearch.evaluation_v2.reference_bridge import export
        export(path,cfg,Path('/runs/assessments')/request['label']/task['name']/'bridge')
    cfg=copy.deepcopy(cfg);cfg['hardware'].update(device='cpu',cpu_threads=1);torch.set_num_threads(1)
    model=Policy(**cp['network']);model.load_state_dict(cp['model']);model.eval()
    out=Path('/runs/assessments')/request['label']/task['name'];out.mkdir(parents=True,exist_ok=True)
    rows=[];deadline=time.monotonic()+request['per_candidate_seconds']
    for opp in request['opponents']:
        frozen=None
        if opp['kind']=='development':
            metadata_path=Path('/runs')/opp['metadata'].lstrip('/')
            weights=Path('/runs')/opp['weights'].lstrip('/')
            metadata=json.loads(metadata_path.read_text())
            if t.digest(weights)!=metadata['weights_sha256']:raise ValueError('Historical opponent weight identity mismatch')
            frozen=Policy(**metadata['network']);frozen.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));frozen.eval()
        elif opp['kind']=='campaign':
            # Each campaign peer is source-admitted in its own isolated worker task before assessment.
            peer=torch.load(opp['checkpoint'],map_location='cpu',weights_only=False)
            if t.digest(opp['checkpoint'])!=opp['sha256']:raise ValueError('Frozen campaign peer identity mismatch')
            frozen=Policy(**peer['network']);frozen.load_state_dict(peer['model']);frozen.eval()
        panel={'split':request['split'],'deterministic_policy':True,'max_decision_phases':4096,'games':[{'seed':s,'learner_seat':seat,'opponent':'checkpoint' if frozen else opp['name']} for s in request['seeds'] for seat in (0,1)]}
        with evaluation_vector_for(cfg,len(panel['games'])) as vector:
            result=cli.evaluation(vector,model,panel,cfg,Timings(),frozen,deadline=time.perf_counter()+max(1,deadline-time.monotonic()))
        result.update(opponent_name=opp['name'],source_admission='exact immutable reference source before CPU inference projection',checkpoint_sha256=t.digest(path))
        atomic(out/(opp['name']+'.json'),result);rows.append(result);subprocess.run(['sync','/runs'],check=True,timeout=10)
    atomic(out/'summary.json',{'task':task,'source_admitted':True,'panels':rows})

def supervise(path):
    request=json.loads(Path(path).read_text());start=time.time();out=Path(path).parent
    result={'status':'running','started_at':start,'tasks':[]};atomic(out/'status.json',result)
    for i,task in enumerate(request['tasks']):
        try:
            p=subprocess.run([sys.executable,__file__,'worker','--request',path,'--index',str(i)],timeout=request['per_candidate_seconds']+40,capture_output=True,text=True)
            (out/(task['name']+'-stdout.log')).write_text(p.stdout);(out/(task['name']+'-stderr.log')).write_text(p.stderr)
            result['tasks'].append({'name':task['name'],'returncode':p.returncode})
        except BaseException as e:result['tasks'].append({'name':task['name'],'error':str(e)})
        atomic(out/'status.json',result);subprocess.run(['sync','/runs'],check=True,timeout=10)
    result.update(status='complete' if all(x.get('returncode')==0 for x in result['tasks']) else 'failed',finished_at=time.time());atomic(out/'status.json',result)
    subprocess.run(['sync','/runs'],check=True,timeout=10)

def launch(label,tasks,split,seeds,opponents,seconds=900,bridge=False):
    import modal
    from .remote import BASE,VOLUME,APP,IMAGE,budget,role_usage
    if role_usage()['resource_usd']+seconds*.00024+.03>32:raise ValueError('RL CPU resource subcap')
    json.loads((BASE.parent/'coordination/evaluator.json').read_text()) # Required counterpart read before paid work.
    target=BASE/'assessments'/label;target.mkdir(parents=True,exist_ok=False)
    request={'bridge':bridge,'label':label,'tasks':tasks,'split':split,'seeds':seeds,'opponents':opponents,'per_candidate_seconds':min(180,(seconds-60)//len(tasks)-40)}
    if bridge:
        from .remote import ROOT
        bridge_root=ROOT/'simulators/pokemon/autoresearch'
        request['bridge_source_sha256']={name:hashlib.sha256((bridge_root/name).read_bytes()).hexdigest() for name in ('contract.py','evaluation_v2/reference_bridge.py')}
    atomic(target/'request.json',request)
    volume=modal.Volume.from_name(VOLUME,environment_name='staging')
    with volume.batch_upload() as upload:
        upload.put_file(target/'request.json','/assessments/'+label+'/request.json')
        upload.put_file(__file__,'/assessments/'+label+'/assessment.py')
        if bridge:
            for name in request['bridge_source_sha256']:upload.put_file(bridge_root/name,'/assessments/'+label+'/bridge-source/'+name)
        for opp in opponents:
            if opp['kind']=='development':
                for key in ('metadata','weights'):
                    upload.put_file(opp['local_'+key],opp[key])
    operation=budget().reserve_worker('pokemon/rl/assess/'+label,seconds,gpu=None,cpu_cores=4,memory_gib=8,sandbox=True,extra_usd=.02)
    atomic(target/'reservation.json',{'operation':operation,'seconds':seconds,'assessment_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    app=modal.App.lookup(APP,environment_name='staging',create_if_missing=True)
    sb=modal.Sandbox.create('python','/runs/assessments/'+label+'/assessment.py','supervise','--request','/runs/assessments/'+label+'/request.json',app=app,image=modal.Image.from_id(IMAGE),cpu=(4,4),memory=(8192,8192),timeout=seconds-25,volumes={'/runs':volume},env={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},block_network=True,secrets=[],include_oidc_identity_token=False,tags={'pk_role':'reference-cpu-assessment','pk_job':label})
    budget().attach(operation,sb.object_id);atomic(target/'provider.json',{'app_id':app.app_id,'sandbox_id':sb.object_id,'operation':operation});return {'sandbox_id':sb.object_id,'operation':operation}

def collect(label):
    import modal
    from .remote import BASE,VOLUME,budget,provider_info
    target=BASE/'assessments'/label;p=json.loads((target/'provider.json').read_text());info=provider_info(p['app_id'],p['sandbox_id'])
    if not info or not info.get('finished_at'):return {'status':'running','provider':info}
    op=next(r for r in budget().snapshot()['operations'] if r['id']==p['operation'])
    if op['status']!='settled':budget().settle_worker(p['operation'],info)
    atomic(target/'allocation.json',info);volume=modal.Volume.from_name(VOLUME,environment_name='staging')
    from .remote import download_tree
    download_tree(volume,'/assessments/'+label,target/'collected')
    return {'provider':info,'result':json.loads((target/'collected/status.json').read_text())}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['worker','supervise','collect']);p.add_argument('--request');p.add_argument('--index',type=int);p.add_argument('--label');a=p.parse_args()
    if a.command=='worker':worker(json.loads(Path(a.request).read_text()),a.index)
    elif a.command=='supervise':supervise(a.request)
    else:print(json.dumps(collect(a.label),indent=2))
if __name__=='__main__':main()
