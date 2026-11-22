"""Capture/settle the single-use observer's full authoritative container interval."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
ROOT=Path(__file__).resolve().parents[3]
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z'

def inspect(name,settle=False):
    root=CAMPAIGN/'tavern/rl'/name;launch=json.loads((root/'provider.json').read_text())
    from modal._utils.async_utils import synchronizer
    from modal_proto import api_pb2
    from google.protobuf.json_format import MessageToDict
    previous=root/'observer-container.json'
    saved=json.loads(previous.read_text()) if previous.exists() else None
    async def read():
        from modal.client import _Client
        client=await _Client.from_env();graph=await client.stub.FunctionGetCallGraph(api_pb2.FunctionGetCallGraphRequest(function_call_id=launch['observer_call_id']))
        calls=[r for r in graph.function_calls if r.function_call_id==launch['observer_call_id']]
        if len(calls)!=1 or calls[0].function_name!='observe_allocation':raise ValueError('Observer call mapping differs')
        task_id=saved['task_id'] if saved else None
        if task_id is None:
            tasks=await client.stub.TaskList(api_pb2.TaskListRequest(app_id=launch['app_id'],environment_name='staging'));cpu=[]
            for t in tasks.tasks:
                response=await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=t.task_id));i=response.info
                if response.app_id!=launch['app_id']:raise ValueError('Task belongs to another app')
                if i.gpu_type=='CPU' and not i.sandbox_id:cpu.append(i.id)
            if len(cpu)>1:raise ValueError('Ambiguous CPU container; retain observer hold')
            if not cpu:return None
            task_id=cpu[0]
        response=await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=task_id));i=response.info
        if response.app_id!=launch['app_id'] or i.gpu_type!='CPU' or i.sandbox_id:raise ValueError('Observer container identity differs')
        return dict(provider_id=launch['observer_call_id'],task_id=i.id,app_id=response.app_id,started_at=i.started_at,finished_at=i.finished_at,gpu_type=i.gpu_type,checked_unix=time.time(),mapping='Only CPU task in owned app; source declares exactly one single-use observer function and one call, zero retries/warm pool',call_graph=MessageToDict(graph,preserving_proto_field_name=True))
    receipt=synchronizer.create_blocking(read)()
    if receipt is None:
        print('No captured observer container; existing hold retained',flush=True);return
    temp=previous.with_suffix('.tmp');temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(previous)
    if settle and receipt['started_at'] and receipt['finished_at']:
        from tools.open_credit_campaign import OpenCreditBudget
        budget=OpenCreditBudget('tavern',CAMPAIGN/'campaign.sqlite3');operation=launch['reservations']['observer'];ops={r['id']:r for r in budget.snapshot()['operations']}
        if ops[operation]['status']!='settled':budget.settle_worker(operation,receipt)
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--settle',action='store_true');a=p.parse_args();inspect(a.name,a.settle)
