"""Stop only this role's recorded new-campaign apps after compute reconciles."""
import argparse,json,time
from .provider import OWN,LEDGER
from tools.open_credit_campaign import OpenCreditBudget
from simulators.clash.autoresearch.common import dump

def inventory():
 budget=OpenCreditBudget('clash',LEDGER)
 rows=[r for r in budget.snapshot()['operations'] if r['game']=='clash' and r['kind'].startswith('clash/rl/')]
 unresolved=[r['id'] for r in rows if r['status']!='settled' and (r['gpu_hold'] or r['kind'].startswith('clash/rl/dev-cpu/'))]
 sources=[];ids=set()
 for path in sorted([*OWN.glob('*/app.json'),*OWN.glob('cpu-development/*/app.json')]):
  value=json.loads(path.read_text());identity=value['app_id'];ids.add(identity)
  sources.append({'receipt':str(path.relative_to(OWN)),'app_id':identity})
 return {'app_ids':sorted(ids),'sources':sources,'unresolved_compute_operations':unresolved}

def close():
 value=inventory()
 if value['unresolved_compute_operations']:raise ValueError('Reconcile every owned GPU/CPU allocation before cleanup')
 from modal._utils.async_utils import synchronizer
 async def stop():
  import asyncio
  from modal.client import _Client
  from modal_proto import api_pb2
  client=await _Client.from_env();results=[]
  for app_id in value['app_ids']:
   life=(await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))).lifecycle
   if life.app_state!=api_pb2.APP_STATE_STOPPED:
    await client.stub.AppStop(api_pb2.AppStopRequest(app_id=app_id,source=api_pb2.APP_STOP_SOURCE_CLI))
   for _ in range(30):
    life=(await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))).lifecycle
    tasks=await client.stub.TaskList(api_pb2.TaskListRequest(app_id=app_id))
    if life.app_state==api_pb2.APP_STATE_STOPPED and not tasks.tasks:break
    await asyncio.sleep(.5)
   results.append({'app_id':app_id,'state':api_pb2.AppState.Name(life.app_state),'running_tasks':len(tasks.tasks)})
  return results
 results=synchronizer.create_blocking(stop)()
 receipt={'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'explicit_ledger':str(LEDGER),'only_recorded_new_rl_owned_apps':True,'sources':value['sources'],'apps':results,'all_stopped_zero_tasks':bool(results) and all(x['state']=='APP_STATE_STOPPED' and x['running_tasks']==0 for x in results),'volumes_retained':True,'unpriced_build_observer_storage_holds_retained':True}
 target=OWN/('cleanup-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json');dump(target,receipt)
 return {'receipt':str(target),**receipt}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stop',action='store_true');args=parser.parse_args()
 print(json.dumps(close() if args.stop else inventory(),indent=2))
