"""Small, host-private public-information Foul Play qualification; no model training."""
import argparse,hashlib,json,os,pathlib,subprocess,tarfile,time,traceback,sysconfig
import numpy as np
import torch
from ..contract import write,sha
from ..arena import PolicyProcess,drop_privileges
from ...env import WorkerError
from ...network import Policy,batch,scripted
from .env import EvaluationVector
from .foul_process import FoulPlayProcess
from .scoring import summarize

class AdministrativeWallLimit(RuntimeError):
    pass

def prepare_search(spec,out,request_path):
    torch.set_num_threads(1)
    wheels=pathlib.Path(spec['wheels'])
    for name,digest in spec['wheel_sha256'].items():
        if sha(wheels/name)!=digest:raise ValueError('Pinned search wheel changed')
    subprocess.run(['python','-m','pip','install','--no-index','--no-deps',*[str(wheels/name) for name in spec['wheel_sha256']]],check=True,stdout=subprocess.DEVNULL,timeout=40)
    site=pathlib.Path(sysconfig.get_paths()['purelib'])
    for name in ['poke_engine','requests','websockets','dateutil','six.py','charset_normalizer','idna','urllib3','certifi']:
        root=site/name
        if root.exists():
            root.chmod(0o755 if root.is_dir() else 0o444)
            if root.is_dir():
                for path in root.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o444)
    source=out/'public-source';source.mkdir()
    if sha(spec['source_archive'])!=spec['source_archive_sha256']:raise ValueError('Foul Play archive changed')
    if sha(spec['public_sets'])!=spec['public_sets_sha256']:raise ValueError('Public random-set data changed')
    with tarfile.open(spec['source_archive']) as tar:tar.extractall(source,filter='data')
    source=next(source.iterdir())
    # Reject access to private schedules/history from both inference identities.
    denied=subprocess.run(['python','-I','-c',"import sys;open(sys.argv[1],'rb').read()",request_path],capture_output=True,preexec_fn=drop_privileges,cwd='/tmp')
    if denied.returncode==0:raise RuntimeError('Candidate UID can read private grading inputs')
    write(out/'candidate-isolation.json',{'uid':65534,'private_request_denied':True})
    return source


def run(spec,out,request_path):
    source=prepare_search(spec,out,request_path)
    schedule=[(o['name'],i,s) for o in spec['opponents'] for i in range(len(spec['seeds'])) for s in (0,1)]
    rows=[];decisions=[];started=time.monotonic();scheduled=len(schedule);faulted=False
    for opponent in spec['opponents']:
        model=None;candidate=None
        if opponent['kind']=='checkpoint':
            p=pathlib.Path(opponent['path']);meta=json.loads((p/'opponent.json').read_text())
            if sha(p/'weights.pt')!=meta['weights_sha256']:raise ValueError('Opponent tensors changed')
            model=Policy(**meta['network']);model.load_state_dict(torch.load(p/'weights.pt',map_location='cpu',weights_only=True));model.eval()
        elif opponent['kind']=='bundle':candidate=PolicyProcess(opponent['path'])
        try:
            for index,seed in enumerate(spec['seeds']):
                for seat in (0,1):
                    before=time.monotonic();row={'opponent':opponent['name'],'seed_index':index,'seat':seat,'natural':False,'truncated':False,'points':None,'fault':None,'fault_category':None}
                    actor='reference'
                    try:
                        with EvaluationVector(1) as vector,FoulPlayProcess(source,spec['public_sets'],mode=spec['mode'],
                                iterations=spec['iterations'],time_ms=spec['time_ms'],samples=spec['samples'],timeout=spec['inference_timeout_seconds'],deny_file=request_path) as foul:
                            write(out/f'isolation-{opponent["name"]}-{index}-{seat}.json',foul.startup)
                            states=vector.reset([seed],spec['max_adapter_phases']);offset=0;rng=np.random.default_rng(np.random.SeedSequence([seed,1-seat,8675309]))
                            while not(states[0]['terminated'] or states[0]['truncated']):
                                if time.monotonic()-before>spec['game_wall_seconds']:raise AdministrativeWallLimit('Whole-game administrative wall safety cap')
                                state=states[0];actions=[None,None]
                                if state['ready'][seat]:
                                    actor='reference'
                                    visible=vector.visible([seat],[offset])[0];offset=visible['protocol_offset']
                                    inference_started=time.monotonic()
                                    action,metadata=foul.choose(visible);actions[seat]=action
                                    metadata['end_to_end_seconds']=time.monotonic()-inference_started
                                    decisions.append({'opponent':opponent['name'],'seed_index':index,'seat':seat,'phase':state['steps'],**metadata})
                                if state['ready'][1-seat]:
                                    actor='candidate'
                                    obs=state['observations'][1-seat]
                                    if candidate:actions[1-seat]=candidate.choose([obs])[0]
                                    elif model:
                                        with torch.no_grad():logits,_=model(batch([obs],'cpu'));actions[1-seat]=int(logits.argmax(-1).item())
                                    else:actions[1-seat]=scripted(obs,rng,opponent['kind'])
                                actor='engine';states=vector.step([actions])
                            final=states[0]
                            row.update(natural=final['terminated'],truncated=final['truncated'],points=(final['returns'][seat]+1)/2 if final['terminated'] else None,
                                terminal_reason=final['terminal_reason'],engine_turns=final['engine_turns'],adapter_phases=final['adapter_phases'],accepted_choices=final['accepted_choices'],unavailable_choice_retries=final['unavailable_choice_retries'],visible_choice_rejections=sum(final['rejections']))
                            audit=vector.audit()[0];write(out/f'trace-{opponent["name"]}-{index}-{seat}.json',audit)
                    except Exception as exc:
                        if isinstance(exc,AdministrativeWallLimit):
                            row.update(truncated=True,terminal_reason='process_wall_safety_cap')
                        else:
                            if isinstance(exc,WorkerError):category=exc.category
                            elif isinstance(exc,TimeoutError):category=actor+'_inference_timeout'
                            elif isinstance(exc,ValueError):category=actor+('_illegal_action' if 'illegal action' in str(exc).lower() or 'masked action' in str(exc).lower() else '_adapter_fault')
                            else:category='infrastructure_or_adapter_fault'
                            row.update(fault=type(exc).__name__+': '+str(exc),fault_category=category)
                        (out/f'fault-{opponent["name"]}-{index}-{seat}.log').write_text(traceback.format_exc());faulted=True
                    row['wall_seconds']=time.monotonic()-before;rows.append(row)
                    write(out/'rows.json',rows);write(out/'decision-metadata.json',decisions)
                    write(out/'summary.json',dict(summarize(rows,scheduled,schedule),seconds=time.monotonic()-started,inference_budgets={k:spec[k] for k in ['mode','iterations','time_ms','samples','inference_timeout_seconds','game_wall_seconds','max_adapter_phases']}))
                    if faulted:break
                if faulted:break
        finally:
            if candidate:candidate.close()
        if faulted:break
    durations=[r['end_to_end_seconds'] for r in decisions]
    result=summarize(rows,scheduled,schedule);result.update(seconds=time.monotonic()-started,search_decisions=len(decisions),
        search_latency_seconds={'mean':float(np.mean(durations)),'p95':float(np.quantile(durations,.95)),'max':max(durations)} if durations else None,
        inference_budgets={k:spec[k] for k in ['mode','iterations','time_ms','samples','inference_timeout_seconds','game_wall_seconds','max_adapter_phases']})
    write(out/'summary.json',result)


def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 run(json.loads(pathlib.Path(a.request).read_text()),pathlib.Path(a.output),a.request)
if __name__=='__main__':main()
