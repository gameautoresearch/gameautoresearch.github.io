"""Inventory/stop only new-ledger evaluator apps, including build-only attempts."""
import argparse,asyncio,json,os,time
from pathlib import Path
from datetime import datetime
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .budget import Budget,CAMPAIGN
from simulators.tavern.autoresearch.evaluation_v2.release import write


def cleanup(out,stop=False):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    b=Budget();ops=b.usage()['operations'];root=CAMPAIGN/'tavern/evaluator';apps={r['provider_id'] for r in ops if str(r.get('provider_id','')).startswith('ap-')}
    for pattern in ('*/provider.json','*/image-build.json'):
        for p in root.glob(pattern):apps.add(json.loads(p.read_text())['app_id'])
    workers=[r for r in ops if json.loads(r['spec']).get('sandbox')]
    unresolved=[r['id'] for r in workers if r['status']!='settled'];deadline=datetime.fromisoformat(b.auth['worker_stop_utc'].replace('Z','+00:00')).timestamp()
    if stop and unresolved and time.time()<deadline:raise ValueError('Reconcile own worker allocations before early teardown')
    from modal._utils.async_utils import synchronizer
    async def work():
        from modal.client import _Client
        from modal_proto import api_pb2
        client=await _Client.from_env();errors=[]
        async def inspect(app_id):
            try:
                state=await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id));tasks,sandboxes=await asyncio.gather(client.stub.TaskList(api_pb2.TaskListRequest(app_id=app_id,environment_name='staging')),client.stub.SandboxList(api_pb2.SandboxListRequest(app_id=app_id,environment_name='staging',include_finished=False)))
                return dict(app_id=app_id,state=state.lifecycle.app_state,active_task_ids=[t.task_id for t in tasks.tasks],active_sandbox_ids=[s.id for s in sandboxes.sandboxes])
            except Exception as exc:return dict(app_id=app_id,unresolved_inventory=type(exc).__name__+': '+str(exc)[:250])
        before=await asyncio.gather(*(inspect(app_id) for app_id in sorted(apps)))
        if stop:
            for row in before:
                if row.get('state')!=api_pb2.APP_STATE_STOPPED:
                    try:await client.stub.AppStop(api_pb2.AppStopRequest(app_id=row['app_id'],source=api_pb2.APP_STOP_SOURCE_CLI))
                    except Exception as exc:errors.append(dict(app_id=row['app_id'],error=type(exc).__name__+': '+str(exc)[:250]))
        after=await asyncio.gather(*(inspect(app_id) for app_id in sorted(apps))) if stop else before
        for _ in range(3):
            if not stop or not any(r.get('active_task_ids') or r.get('active_sandbox_ids') for r in after):break
            await asyncio.sleep(2);after=await asyncio.gather(*(inspect(app_id) for app_id in sorted(apps)))
        active_tasks=sum(len(r.get('active_task_ids',[])) for r in after);active_sandboxes=sum(len(r.get('active_sandbox_ids',[])) for r in after);unknown=[r for r in after if r.get('unresolved_inventory')]
        return dict(complete=bool(stop and not active_tasks and not active_sandboxes and not unresolved and not unknown and not errors),checked_unix=time.time(),owned_recorded_apps=len(apps),includes_build_only_apps=True,stop_requested=stop,active_owned_tasks=active_tasks,active_owned_sandboxes=active_sandboxes,unresolved_worker_operations=unresolved,unresolved_app_inventory=unknown,stop_errors=errors,unknown_build_observer_storage_holds_retained=True,before=before,after=after)
    result=synchronizer.create_blocking(work)();write(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stop',action='store_true');a=p.parse_args();print(json.dumps(cleanup(a.out,a.stop),indent=2))
