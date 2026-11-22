"""Reattach a recorded bounded Modal invocation; never submit another experiment."""
import json
from pathlib import Path
import time
import modal


def reattach(log:Path,receipt:Path,run_id:str,timeout=3900):
    calls=[]
    for line in log.read_text().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if isinstance(row,dict) and row.get('run_id')==run_id and row.get('function_call_id'):
            calls.append(row['function_call_id'])
    if not calls:return None
    if len(set(calls))!=1:raise RuntimeError('ambiguous experiment invocation; refuse to resubmit')
    start=time.perf_counter();record={'run_id':run_id,'function_call_id':calls[0],'new_experiment_submitted':False}
    try:
        result=modal.FunctionCall.from_id(calls[0]).get(timeout=timeout)
        record['result']=result
        return result
    except Exception as error:
        record['error']=repr(error)
        return None
    finally:
        record['reattachment_wall_seconds']=time.perf_counter()-start
        with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
