"""Fixed-work query parity and remote-only timing under the pinned4CPU allocation."""
import concurrent.futures,contextlib,json,time,statistics,re
from pathlib import Path
import torch
from ..autoresearch.evaluation_v2.env import EvaluationVector
from ..autoresearch.evaluation_v2.foul_process import FoulPlayProcess
from ..autoresearch.evaluation_v2.qualification import prepare_search
from .strong_runtime import ReferenceVector
from .strong_campaign import write

def normalize(value):
 if isinstance(value,str):return re.sub(r'^\|t:\|[0-9]+$', '|t:|0',value,flags=re.MULTILINE)
 if isinstance(value,list):return [normalize(x) for x in value]
 if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
 return value

def run(spec,job):
 torch.set_num_threads(1);(job/'search-setup').mkdir();source=prepare_search(spec,job/'search-setup',str(job/'request.json'));cases=[]
 for repetition in range(3):
  with contextlib.ExitStack() as stack:
   old=stack.enter_context(EvaluationVector(24));new=stack.enter_context(ReferenceVector(1,24));seeds=list(range(930300100+repetition*24,930300124+repetition*24));a=old.reset(seeds,4096);b=new.reset(seeds,4096)
   if a!=b:raise AssertionError('Sparse wrapper changes initial700-values or rule state')
   seats=[None]*24;offsets=[0]*24
   for i,seat in ((2,0),(7,1),(19,0)):seats[i]=seat
   original=old.visible([x if x is not None else 0 for x in seats],offsets);sparse=new.visible(seats,offsets)
   selected=[i for i,s in enumerate(seats) if s is not None]
   if any(normalize(original[i])!=normalize(sparse[i]) for i in selected):raise AssertionError('Requested public teacher channel differs')
   if any(sparse[i] is not None for i in range(24) if i not in selected):raise AssertionError('Unrequested views should not be serialized')
   kwargs={'mode':'fixed-work','iterations':1000,'time_ms':10,'samples':2,'timeout':1.,'deny_file':str(job/'request.json')}
   serial_bots=[stack.enter_context(FoulPlayProcess(source,spec['public_sets'],**kwargs)) for i in selected];parallel_bots=[stack.enter_context(FoulPlayProcess(source,spec['public_sets'],**kwargs)) for i in selected]
   before=time.perf_counter();serial=[bot.choose(original[i]) for bot,i in zip(serial_bots,selected,strict=True)];serial_seconds=time.perf_counter()-before
   before=time.perf_counter()
   with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:parallel=list(pool.map(lambda x:x[0].choose(sparse[x[1]]),zip(parallel_bots,selected,strict=True)))
   parallel_seconds=time.perf_counter()-before
   checks=[s[0]==p[0] and s[1]['visits']==p[1]['visits'] and s[1]['belief_state_sha256']==p[1]['belief_state_sha256'] and s[1]['input_sha256']==p[1]['input_sha256'] for s,p in zip(serial,parallel,strict=True)]
   if not all(checks):raise AssertionError({'serial':serial,'parallel':parallel})
   cases.append({'seeds':seeds,'queried_game_indices':selected,'exact_actions_visits_beliefs_inputs':checks,'original_visible_json_bytes':len(json.dumps(original)),'sparse_visible_json_bytes':len(json.dumps(sparse)),'serial_search_seconds':serial_seconds,'parallel_search_seconds':parallel_seconds})
 result={'valid':True,'presentation_timestamp_normalization_only':True,'cases':cases,'median_query_wall_speedup':statistics.median(c['serial_search_seconds']/c['parallel_search_seconds'] for c in cases),'qualification':'Exact testedpublicqueries only; not a fullPPO throughput or strength certificate'};write(job/'transport-proof.json',result);return result
