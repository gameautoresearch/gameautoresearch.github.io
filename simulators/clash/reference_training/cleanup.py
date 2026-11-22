"""Stop only recorded apps owned by this RL campaign, after compute settles."""
import json,time
from .launch import CAMPAIGN,LEDGER
from simulators.clash.autoresearch.common import dump

def close():
 from tools.overnight_budget import CampaignBudget
 root=CAMPAIGN/'clash/rl';snapshot=CampaignBudget('clash',path=LEDGER).snapshot()
 rows=[r for r in snapshot['operations'] if r['game']=='clash' and r['kind'].startswith('clash/rl/')]
 if any(r['gpu_hold'] and r['status']!='settled' for r in rows):raise ValueError('Reconcile owned GPU allocations before cleanup')
 if any(r['kind']=='clash/rl/final-cpu-worker' and r['status']!='settled' for r in rows):raise ValueError('Reconcile final CPU allocation before cleanup')
 ids=set();sources=[]
 for folder in root.iterdir():
  if not folder.is_dir() or not (folder/'reservations.json').exists():continue
  for name in ('app.json','provider.json','failure.json','prelaunch-recovery.json','no-gpu-created.json'):
   path=folder/name
   if not path.exists():continue
   value=json.loads(path.read_text());found=[]
   for key in ('app_id','old_app_id','previous_app_id','recovered_app_id'):
    if value.get(key):ids.add(value[key]);found.append(value[key])
   if found:sources.append({'receipt':str(path.relative_to(root)),'app_ids':found})
 from modal._utils.async_utils import synchronizer
 async def stop():
  import asyncio
  from modal.client import _Client
  from modal_proto import api_pb2
  client=await _Client.from_env();results=[]
  for app_id in sorted(ids):
   life=(await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))).lifecycle
   if life.app_state!=api_pb2.APP_STATE_STOPPED:
    await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app_id,source=api_pb2.APP_STOP_SOURCE_CLI))
    life=(await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))).lifecycle
   tasks=await client.stub.TaskList(api_pb2.TaskListRequest(app_id=app_id))
   for _ in range(30):
    if life.app_state==api_pb2.APP_STATE_STOPPED and not tasks.tasks:break
    await asyncio.sleep(.5)
    life=(await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))).lifecycle
    tasks=await client.stub.TaskList(api_pb2.TaskListRequest(app_id=app_id))
   results.append({'app_id':app_id,'state':api_pb2.AppState.Name(life.app_state),'running_tasks':len(tasks.tasks)})
  return results
 results=synchronizer.create_blocking(stop)()
 receipt={'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'only_recorded_rl_owned_apps':True,'sources':sources,'apps':results,'all_stopped_zero_tasks':bool(results) and all(x['state']=='APP_STATE_STOPPED' and x['running_tasks']==0 for x in results),'volumes_retained':True,'unpriced_build_observer_storage_holds_retained':True}
 target=root/('cleanup-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json');dump(target,receipt)
 return {'receipt':str(target),**receipt}

if __name__=='__main__':print(json.dumps(close(),indent=2))
