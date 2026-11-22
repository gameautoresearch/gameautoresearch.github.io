"""Owned-app inventory/teardown; no operation on another campaign's workers."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
ROOT=Path(__file__).resolve().parents[4]

def main():
 p=argparse.ArgumentParser();p.add_argument('--stop',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 root=ROOT/'evidence/reference-readiness-20261109-0251Z';rl=root/'tavern/rl'
 launches=[json.loads(f.read_text()) for f in rl.glob('*/provider.json')]
 ids={x['app_id'] for x in launches};names={'tavern-reference-'+d.name+'-20261109' for d in rl.glob('attempt-*') if d.is_dir()}
 from tools.overnight_budget import CampaignBudget
 budget=CampaignBudget('tavern',path=root/'campaign.sqlite3').snapshot()
 active=[o for o in budget['operations'] if o['game']=='tavern' and o['kind'].startswith('rl-') and o['gpu_hold'] and o['status']!='settled']
 if a.stop and active:raise ValueError('Reconcile unresolved owned GPU allocations before cleanup')
 from modal._utils.async_utils import synchronizer
 async def work():
  import asyncio
  from modal.client import _Client
  from modal_proto import api_pb2
  client=await _Client.from_env();response=await client.stub.AppList(api_pb2.AppListRequest(environment_name='staging'))
  owned=[x for x in response.apps if x.app_id in ids or x.description in names or x.name in names]
  before=[]
  for x in owned:
   tasks=await client.stub.TaskList(api_pb2.TaskListRequest(environment_name='staging',app_id=x.app_id))
   before.append(dict(app_id=x.app_id,name=x.name,description=x.description,state=x.state,n_running_tasks=x.n_running_tasks,task_ids=[t.task_id for t in tasks.tasks]))
   if a.stop and x.state!=api_pb2.APP_STATE_STOPPED:await client.stub.AppStop(api_pb2.AppStopRequest(app_id=x.app_id,source=api_pb2.APP_STOP_SOURCE_CLI))
  after=[]
  for x in owned:
   tasks=await client.stub.TaskList(api_pb2.TaskListRequest(environment_name='staging',app_id=x.app_id));after.append(dict(app_id=x.app_id,task_ids=[t.task_id for t in tasks.tasks]))
  return dict(checked_unix=time.time(),stop_requested=a.stop,before=before,after=after,active_owned_tasks=sum(len(x['task_ids']) for x in after),owned_gpu_unresolved_holds=len(active),unpriced_image_and_CPU_holds_retained=True)
 result=synchronizer.create_blocking(work)();a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
