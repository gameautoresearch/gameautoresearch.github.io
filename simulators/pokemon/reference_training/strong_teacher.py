"""Public-channel FoulPlay labels; no hidden battle object or silent action fallback."""
from __future__ import annotations
import concurrent.futures,json,time,traceback,subprocess,os
from pathlib import Path
import numpy as np
import torch
from ..network import Policy,batch,scripted
from ..autoresearch.evaluation_v2.env import EvaluationVector
from ..autoresearch.evaluation_v2.foul_process import FoulPlayProcess
from ..autoresearch.evaluation_v2.qualification import prepare_search
from ..autoresearch.evaluation_v2.reference_inference import load
from .strong_campaign import write,sha

KEYS=('tokens','mons','global','global_tokens','action_tokens','actions','mask')
def collect_game(spec,index):
 torch.set_num_threads(1);out=Path(spec['out']);seed=spec['seed_start']+index
 record={'index':index,'seed':seed,'teacher_seats':[],'status':'not-started','natural':False,'truncated':False,'fault':None,'teacher_config':spec['teacher_config']}
 if time.time()>spec['deadline_unix']-20:return record
 observations=[];labels=[];seats=[];teacher_metadata=[];before=time.monotonic();last_visible=None
 try:
  import contextlib
  teacher_seats=(0,1) if index%3==0 else (index%2,);record['teacher_seats']=list(teacher_seats)
  opponent='self-search' if len(teacher_seats)==2 else 'power' if index%3==1 else 'incumbent';record['opponent']=opponent
  peer=None
  if opponent=='incumbent':
   if spec.get('incumbent_weights'):
    if sha(spec['incumbent_weights'])!=spec['incumbent_weights_sha256']:raise ValueError('Teacher opponent tensor identity changed')
    peer=Policy(**spec['incumbent_network']);peer.load_state_dict(torch.load(spec['incumbent_weights'],map_location='cpu',weights_only=True),strict=True);peer.eval()
   else:peer=load(spec['incumbent_path'],spec['incumbent_manifest'])[0]
  with contextlib.ExitStack() as stack:
   vector=stack.enter_context(EvaluationVector(1));bots={seat:stack.enter_context(FoulPlayProcess(spec['source'],spec['public_sets'],deny_file=spec['request_path'],**spec['teacher_config'])) for seat in teacher_seats}
   state=vector.reset([seed],4096)[0];offsets=[0,0];rng=np.random.default_rng(np.random.SeedSequence([seed,1008675309]))
   while not(state['terminated'] or state['truncated']):
    if time.monotonic()-before>120 or time.time()>spec['deadline_unix']-10:
     record['truncated']=True;record['reason']='collection_wall_limit';break
    actions=[None,None]
    for seat in (0,1):
     if not state['ready'][seat]:continue
     obs=state['observations'][seat]
     if seat in bots:
      visible=vector.visible([seat],[offsets[seat]])[0];offsets[seat]=visible['protocol_offset'];last_visible=visible;action,metadata=bots[seat].choose(visible)
      if not obs['mask'][action]:raise ValueError('Teacher action is masked in the700-value observation')
      observations.append(obs);labels.append(action);seats.append(seat);teacher_metadata.append(metadata)
     elif peer is not None:
      with torch.no_grad():logits,_=peer(batch([obs],'cpu'));action=int(logits.argmax(-1))
     else:action=scripted(obs,rng,'power')
     actions[seat]=action
    state=vector.step([actions])[0]
   record.update(status='complete',natural=state['terminated'],truncated=record['truncated'] or state['truncated'],reason=record.get('reason',state.get('terminal_reason',state['reason'])),returns=state['returns'] if state['terminated'] else None,visible_rejections=sum(state['rejections']),turn=state['turn'],phases=state['steps'])
 except BaseException as e:record.update(status='fault',fault=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),fault_visible=last_visible)
 if observations:
  tensors={k:np.asarray([o[k] for o in observations],dtype=np.int16 if k in ('tokens','global_tokens','action_tokens') else np.bool_ if k=='mask' else np.float32) for k in KEYS}
  tensors.update(action=np.asarray(labels,np.int8),seat=np.asarray(seats,np.int8),value=np.asarray([record['returns'][seat] if record.get('returns') else np.nan for seat in seats],np.float32))
  path=out/f'game-{index:05d}.npz';temporary=path.with_suffix('.tmp')
  with temporary.open('wb') as f:np.savez_compressed(f,**tensors)
  temporary.replace(path);record.update(samples=len(labels),shard=path.name,sha256=sha(path))
  write(out/f'teacher-metadata-{index:05d}.json',teacher_metadata)
 record['wall_seconds']=time.monotonic()-before;write(out/f'game-{index:05d}.json',record);subprocess.run(['sync','/runs'],timeout=10,check=True);return record

def run(spec,job):
 out=job/'teacher-data';out.mkdir();spec=dict(spec,out=str(out),request_path=str(job/'request.json'))
 (job/'search-setup').mkdir()
 source=prepare_search(spec,job/'search-setup',str(job/'request.json'))
 spec['source']=str(source)
 records=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(collect_game,spec,index) for index in range(spec['games'])]
  for future in concurrent.futures.as_completed(futures):
   record=future.result();records.append(record);records.sort(key=lambda r:r['index']);write(out/'index.json',records)
   write(out/'summary.json',{'scheduled_games':spec['games'],'attempted_games':sum(r['status']!='not-started' for r in records),'natural_games':sum(r['natural'] for r in records),'caps':sum(r['truncated'] for r in records),'faults':sum(r['fault'] is not None for r in records),'samples':sum(r.get('samples',0) for r in records),'natural_samples':sum(r.get('samples',0) for r in records if r['natural'] and not r['fault']),'teacher_config':spec['teacher_config'],'information_boundary':'Each teacher receives only its own request and player-channel history; stored student input is only700values; labels legal14actions. Public dev seeds only.'})
 result=json.loads((out/'summary.json').read_text());result['status']='complete';return result
