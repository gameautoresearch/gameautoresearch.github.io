"""Reconcile and stop only this RL role's recorded new-campaign apps."""
import argparse,json,os,sys,time
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
ROOT=Path(__file__).resolve().parents[3]
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z'

def cleanup(out,stop_completed=False,stop_all=False):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    from tools.open_credit_campaign import OpenCreditBudget
    sys.path.insert(0,str(ROOT/'simulators/tavern/autoresearch'))
    from provider import configure_sandbox_waits,allocation_info
    import modal
    configure_sandbox_waits();budget=OpenCreditBudget('tavern',CAMPAIGN/'campaign.sqlite3')
    launches=[json.loads(p.read_text()) for p in (CAMPAIGN/'tavern/rl').glob('*/provider.json')]
    metadata=[]
    for p in launches:
        info=allocation_info(p['app_id'],p['sandbox_id'])
        if stop_all and (not info or not info.get('finished_at')):
            sandbox=modal.Sandbox.from_id(p['sandbox_id']);sandbox.terminate(wait=True);info=allocation_info(p['app_id'],p['sandbox_id'])
        rows={r['id']:r for r in budget.snapshot()['operations']};operation=p['reservations']['gpu']
        if info and info.get('finished_at') and info.get('started_at') and rows[operation]['status']!='settled':budget.settle_worker(operation,info)
        metadata.append(dict(app_id=p['app_id'],name=p['name'],provider=info))
    ops=[o for o in budget.snapshot()['operations'] if o['game']=='tavern' and o['kind'].startswith('rl:')]
    ids={p['app_id'] for p in launches}|{o['provider_id'] for o in ops if o['kind'].startswith('rl:image:') and o.get('provider_id')}
    finished_apps={x['app_id'] for x in metadata if x['provider'] and x['provider'].get('finished_at')}
    from modal._utils.async_utils import synchronizer
    async def inspect():
        from modal.client import _Client
        from modal_proto import api_pb2
        client=await _Client.from_env();listed=await client.stub.AppList(api_pb2.AppListRequest(environment_name='staging'));apps={a.app_id:a for a in listed.apps};before=[];after=[]
        for app_id in sorted(ids):
            tasks=await client.stub.TaskList(api_pb2.TaskListRequest(environment_name='staging',app_id=app_id));app=apps.get(app_id);before.append(dict(app_id=app_id,state=app.state if app else None,tasks=[t.task_id for t in tasks.tasks]))
            if stop_all or (stop_completed and app_id in finished_apps and not tasks.tasks):
                if not app or app.state!=api_pb2.APP_STATE_STOPPED:await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app_id,source=api_pb2.APP_STOP_SOURCE_CLI))
            current=await client.stub.TaskList(api_pb2.TaskListRequest(environment_name='staging',app_id=app_id));after.append(dict(app_id=app_id,tasks=[t.task_id for t in current.tasks]))
        return before,after
    before,after=synchronizer.create_blocking(inspect)();remaining=[o for o in budget.snapshot()['operations'] if o['game']=='tavern' and o['kind'].startswith('rl:') and o['gpu_hold'] and o['status']!='settled']
    result=dict(checked_unix=time.time(),owned_apps=len(ids),stop_completed=stop_completed,stop_all=stop_all,allocations=metadata,before=before,after=after,active_owned_tasks=sum(len(x['tasks']) for x in after),unsettled_GPU_holds=len(remaining),unknown_image_storage_or_observer_holds_retained=True,other_roles_and_historical_campaigns_untouched=True)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['owned_apps','active_owned_tasks','unsettled_GPU_holds','stop_completed','stop_all']}),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-completed',action='store_true');p.add_argument('--stop-all',action='store_true');a=p.parse_args();cleanup(a.out,a.stop_completed,a.stop_all)
