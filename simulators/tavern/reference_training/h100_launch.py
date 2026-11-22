"""Explicit new1,800 allocated-H100-second pipeline verification allowance."""
import argparse,json,os,re,shutil,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .strong.common import ROOT,GAME,write,sha
EVIDENCE=ROOT/'evidence/tavern-h100-pipeline-20261109'
OLD=ROOT/'evidence/tavern-population-throughput-20261109'
VOLUME='tavern-h100-pipeline-20261109'

def budget():
    from tools.overnight_budget import CampaignBudget
    p=EVIDENCE/'authorization.json'
    if not p.exists():
        EVIDENCE.mkdir(parents=True,exist_ok=True);old=json.loads((OLD/'authorization.json').read_text());end=(datetime.now(timezone.utc)+timedelta(hours=4)).isoformat()
        write(p,dict(campaign_id='tavern-h100-pipeline-20261109',authorized_by='Explicit user dispatch: bounded H100 collector/inference/PPO pass,2x first target,3x stretch',scope='Same quarter157 population/model/rules/objective; no strength/private grading/model calls/GPU rules rewrite',deadline_utc=end,worker_stop_utc=end,global_limits=dict(gpu_allocated_seconds=1800,resource_charged_or_reserved_usd=5,model_charged_or_reserved_usd=0,max_concurrent_gpu_workers=1),initial_grants=dict(tavern=dict(gpu_seconds=1800,resource_usd=5,model_usd=0)),pricing=old['pricing'],predecessors={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'CLOSED.json',OLD/'campaign.sqlite3',ROOT/'evidence/tavern-throughput-attribution-20261109/CLOSED.json',ROOT/'evidence/tavern-throughput-attribution-20261109/campaign.sqlite3']}))
    return CampaignBudget('tavern',EVIDENCE/'campaign.sqlite3',p)

def snapshot(dest):
    game=dest/'simulators/tavern';game.mkdir(parents=True)
    for f in GAME.glob('*.py'):shutil.copy2(f,game/f.name)
    shutil.copytree(GAME/'reference_training',game/'reference_training',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    helpers=game/'autoresearch';helpers.mkdir()
    for f in ['__init__.py','contract.py','provider.py','benchmark.json','policy_worker.py']:shutil.copy2(GAME/'autoresearch'/f,helpers/f)
    shutil.copytree(GAME/'autoresearch/recipes',helpers/'recipes')
    (dest/'tools').mkdir()
    for f in ['open_credit_campaign.py','overnight_budget.py']:shutil.copy2(ROOT/'tools'/f,dest/'tools'/f)
    write(dest.parent/'source-files.json',{str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()})

def launch(name,cap,mode,variant,reverse,proof,candidate_cache=16,validated_sampler=False):
    if (EVIDENCE/'CLOSED.json').exists():raise ValueError('H100 task closed')
    if cap not in range(120,701):raise ValueError('Finite120..700 second stage')
    if candidate_cache not in (16,32,64):raise ValueError('Bounded actor-cache comparison')
    import modal
    from simulators.tavern.autoresearch.provider import configure_sandbox_waits
    configure_sandbox_waits();b=budget();out=EVIDENCE/name;out.mkdir(exist_ok=False);snapshot(out/'source')
    if not (EVIDENCE/'baseline-source').exists():shutil.copytree(out/'source',EVIDENCE/'baseline-source');shutil.copy2(out/'source-files.json',EVIDENCE/'baseline-source-files.json')
    config=json.loads((GAME/'reference_training/strong/curriculum-throughput.json').read_text());config.update(machine=dict(gpu='H100',cpu_cores=4,memory_gib=16),updates=2,batched_policy_dispatch=variant=='candidate',ppo_backward_graph=variant=='candidate')
    if variant=='candidate':config.update(actor_graph_max_shapes=candidate_cache,validated_sampler=validated_sampler)
    write(out/'config.json',config);runner=(Path(__file__).parent/'h100_run.py').read_text();(out/'runner.py').write_text(runner)
    baseid=re.findall(r'Built image (im-\w+)',(OLD/'l4-bounded-v4-launch.log').read_text())[-1]
    image=modal.Image.from_id(baseid).add_local_dir(out/'source','/opt/task',copy=True).add_local_dir(EVIDENCE/'baseline-source','/baseline',copy=True).run_commands('ln -s /opt/task/vendor /baseline/vendor')
    app=modal.App('tavern-h100-'+name);volume=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
    build=b.reserve_worker('image:'+name,180,None,4,16,sandbox=False,extra_usd=.1);write(out/'build-reservation.json',dict(operation=build))
    with modal.enable_output(),app.run(detach=True,environment_name='staging'):
        b.attach(build,app.app_id);image.build(app);write(out/'image.json',dict(image_id=image.object_id))
        operation=b.reserve_worker('gpu:'+name,cap,'H100',4,16);write(out/'reservation.json',dict(operation=operation,cap=cap))
        args=['--mode',mode,'--variant',variant,'--candidate-cache',str(candidate_cache)]
        if validated_sampler:args+=['--validated-sampler']
        if reverse:args+=['--reverse']
        if proof:args+=['--proof']
        command=("import sys,runpy\nfrom pathlib import Path\n"+f"root=Path('/retained/{name}');root.mkdir(parents=True,exist_ok=False)\n(root/'runner.py').write_text({runner!r})\n(root/'config.json').write_text({json.dumps(config)!r})\nsys.argv=[str(root/'runner.py'),'--root',str(root),'--config',str(root/'config.json'),'--cap',{str(cap-25)!r}]+{args!r}\nrunpy.run_path(str(root/'runner.py'),run_name='__main__')\n")
        (out/'worker-command.py').write_text(command)
        source_root='/baseline' if mode=='single' and variant=='baseline' else '/opt/task'
        env={**config['host_thread_environment'],'PYTHONPATH':source_root}
        sb=modal.Sandbox.create('python','-u','-c',command,app=app,image=image,gpu='H100',cpu=(4,4),memory=(16384,16384),timeout=cap-20,block_network=True,secrets=[],env=env,volumes={'/retained':volume},include_oidc_identity_token=False,name='tavern-h100-'+name,tags={'campaign':'tavern-h100-pipeline-20261109','stage':name})
        b.attach(operation,sb.object_id);write(out/'provider.json',dict(app_id=app.app_id,sandbox_id=sb.object_id,image_id=image.object_id,volume_name=VOLUME,name=name,gpu_operation=operation,build_operation=build,cap_seconds=cap,source_manifest_sha256=sha(out/'source-files.json'),runner_sha256=sha(out/'runner.py'),mode=mode,variant=variant,reverse=reverse,proof=proof,candidate_cache=candidate_cache,validated_sampler=validated_sampler,automatic_retries=0,launched_unix=time.time()));sb.detach()
    print((out/'provider.json').read_text(),flush=True)

def status(name,collect):
    import modal
    from simulators.tavern.autoresearch.provider import configure_sandbox_waits,allocation_info
    configure_sandbox_waits();b=budget();out=EVIDENCE/name;l=json.loads((out/'provider.json').read_text());r=allocation_info(l['app_id'],l['sandbox_id']);write(out/'provider-status.json',dict(provider=r,checked_unix=time.time()))
    if r and r['started_at'] and r['finished_at']:
        ops={x['id']:x for x in b.snapshot()['operations']}
        if ops[l['gpu_operation']]['status']!='settled':b.settle_worker(l['gpu_operation'],r)
    if collect:
        if not r or not r['finished_at']:raise ValueError('Active stage')
        v=modal.Volume.from_name(VOLUME,environment_name='staging');dest=out/'collected'
        for e in v.iterdir('/'+name,recursive=True):
            if e.type!=1:continue
            f=e.path.lstrip('/');assert f.startswith(name+'/')
            if Path(f).suffix not in ('.json','.jsonl','.log','.csv','.py'):continue
            p=dest/f[len(name)+1:]
            if p.exists():continue
            p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.download')
            with t.open('wb') as h:
                for data in v.read_file('/'+f):h.write(data)
            t.replace(p)
    write(EVIDENCE/'budget-snapshot.json',b.snapshot());print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','status']);p.add_argument('--name',required=True);p.add_argument('--cap',type=int,default=200);p.add_argument('--mode',choices=['single','pair','crossover'],default='single');p.add_argument('--variant',choices=['baseline','candidate'],default='baseline');p.add_argument('--reverse',action='store_true');p.add_argument('--proof',action='store_true');p.add_argument('--collect',action='store_true');p.add_argument('--candidate-cache',type=int,default=16);p.add_argument('--validated-sampler',action='store_true');a=p.parse_args()
    if a.command=='launch':launch(a.name,a.cap,a.mode,a.variant,a.reverse,a.proof,a.candidate_cache,a.validated_sampler)
    else:status(a.name,a.collect)
