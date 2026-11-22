"""Stop only recorded evaluator apps, after all own CPU allocations reconcile."""
import argparse,json,os,time
from pathlib import Path
from .budget import CAMPAIGN,EvaluatorBudget
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')

def cleanup(out,stop=False):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    root=CAMPAIGN/'tavern/evaluator';launches=[json.loads(p.read_text()) for p in root.glob('*/provider.json')]
    ids={r['app_id'] for r in launches};usage=EvaluatorBudget().usage()
    unresolved=[r for r in usage['operations'] if r['kind'].startswith('tavern-evaluator:cpu-') and r['status']!='settled']
    if stop and unresolved:raise ValueError('Reconcile every evaluator CPU worker before teardown')
    from modal._utils.async_utils import synchronizer
    async def work():
        from modal.client import _Client
        from modal_proto import api_pb2
        client=await _Client.from_env();apps=await client.stub.AppList(api_pb2.AppListRequest(environment_name='staging'));before=[];after=[]
        for app in apps.apps:
            if app.app_id not in ids:continue
            tasks=await client.stub.TaskList(api_pb2.TaskListRequest(environment_name='staging',app_id=app.app_id))
            before.append(dict(app_id=app.app_id,state=app.state,tasks=[t.task_id for t in tasks.tasks]))
            if stop and app.state!=api_pb2.APP_STATE_STOPPED:await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app.app_id,source=api_pb2.APP_STOP_SOURCE_CLI))
            tasks=await client.stub.TaskList(api_pb2.TaskListRequest(environment_name='staging',app_id=app.app_id));after.append(dict(app_id=app.app_id,tasks=[t.task_id for t in tasks.tasks]))
        active=sum(len(r['tasks']) for r in after)
        return dict(complete=stop and active==0 and not unresolved,checked_unix=time.time(),owned_recorded_apps=len(ids),stop_requested=stop,active_owned_tasks=active,unresolved_CPU_workers=len(unresolved),GPU_workers=0,unknown_build_observer_storage_holds_retained=True,before=before,after=after)
    result=synchronizer.create_blocking(work)();out.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stop',action='store_true');a=p.parse_args();print(json.dumps(cleanup(a.out,a.stop),indent=2))
