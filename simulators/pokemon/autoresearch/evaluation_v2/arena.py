"""Reusable entrant-facing evaluator-v2 against frozen public-information opponents."""
import argparse,ast,contextlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from ..arena import PolicyProcess
from ..bundle import verify
from ..contract import HERE,sha,runtime_signature,write
from ...network import Policy,batch,scripted
from ...env import WorkerError
from .env import EvaluationVector
from .foul_process import FoulPlayProcess
from .qualification import prepare_search,AdministrativeWallLimit
from .reference_inference import load as load_reference
from .scoring import summarize

PLAIN_STARTER_POLICY_SHA256='638d949c10a5f57d235a3080f12ce3a304c88849495ecfbeaed485568cb42003'


def trusted_opponent(spec):
    path=Path(spec['path'])
    if spec['kind']=='reference-inference':return load_reference(path,spec['manifest_sha256'])[0]
    if spec['kind']=='plain-starter-bundle':
        meta=verify(path,spec['bundle_sha256']);binding=json.loads((path/'binding.json').read_text())
        if binding['runtime_sha256']!=runtime_signature():raise ValueError('Original opponent runtime binding differs')
        if meta['files']['policy.py']!=PLAIN_STARTER_POLICY_SHA256:raise ValueError('Only exact audited plain starter source may be projected into trusted inference')
        cfg=json.loads((path/'config.json').read_text());network=cfg['network']
    elif spec['kind']=='checkpoint':
        if sha(path/'opponent.json')!=spec['metadata_sha256']:raise ValueError('Frozen opponent metadata changed')
        meta=json.loads((path/'opponent.json').read_text());network=meta['network']
        if sha(path/'weights.pt')!=meta['weights_sha256']:raise ValueError('Frozen opponent weights changed')
    else:raise ValueError('Unsupported trusted neural opponent')
    model=Policy(**network);model.load_state_dict(torch.load(path/'weights.pt',weights_only=True,map_location='cpu'),strict=True);model.eval()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite opponent tensor')
    return model


def run(spec,out,request_path):
    torch.set_num_threads(1)
    for name,digest in spec['qualified_core_file_sha256'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('Qualified evaluator core changed: '+name)
    if set(spec['qualified_core_file_sha256'])!={'worker.cjs','env.py','foul_play.py','foul_process.py','scoring.py'}:raise ValueError('Incomplete qualified evaluator core binding')
    source=prepare_search(spec,out,request_path) if any(o['kind']=='search' for o in spec['panel']) else None
    if spec['limit']!=4096:raise ValueError('Use the frozen evaluator-v2 phase safety limit')
    schedule=[(o['name'],i,s) for o in spec['panel'] for i in range(len(spec['seeds'])) for s in (0,1)]
    rows=[];started=time.monotonic();candidate=spec['candidate'];stop=False
    with contextlib.ExitStack() as stack:
        if candidate['kind']=='bundle':
            verify(candidate['path'],candidate['bundle_sha256']);binding=json.loads((Path(candidate['path'])/'binding.json').read_text())
            if binding['runtime_sha256']!=runtime_signature():raise ValueError('Original candidate runtime binding differs')
            process=stack.enter_context(PolicyProcess(candidate['path']));model=None
        elif candidate['kind']=='reference-inference':model=load_reference(candidate['path'],candidate['manifest_sha256'])[0];process=None
        else:raise ValueError('Candidate must have an original sealed binding or attested reference inference export')
        for opponent in spec['panel']:
            peer=trusted_opponent(opponent) if opponent['kind'] in ['reference-inference','plain-starter-bundle','checkpoint'] else None
            for index,seed in enumerate(spec['seeds']):
                for seat in (0,1):
                    row={'opponent':opponent['name'],'seed_index':index,'seat':seat,'natural':False,'truncated':False,'points':None,'fault':None,'fault_category':None};actor='candidate';before=time.monotonic()
                    try:
                        with contextlib.ExitStack() as game:
                            vector=game.enter_context(EvaluationVector(1));foul=None;offset=0
                            if opponent['kind']=='search':
                                actor='reference';foul=game.enter_context(FoulPlayProcess(source,spec['public_sets'],mode='production',iterations=1000,time_ms=10,samples=2,timeout=1.,deny_file=request_path))
                            states=vector.reset([seed],4096);rng=np.random.default_rng(np.random.SeedSequence([seed,1-seat,8675309]))
                            while not(states[0]['terminated'] or states[0]['truncated']):
                                if time.monotonic()-before>120:raise AdministrativeWallLimit('Whole-game safety cap')
                                state=states[0];actions=[None,None]
                                if state['ready'][seat]:
                                    actor='candidate'
                                    if process:actions[seat]=process.choose([state['observations'][seat]])[0]
                                    else:
                                        with torch.no_grad():logits,_=model(batch([state['observations'][seat]],'cpu'));actions[seat]=int(logits.argmax(-1).item())
                                if state['ready'][1-seat]:
                                    actor='reference'
                                    if foul:
                                        visible=vector.visible([1-seat],[offset])[0];offset=visible['protocol_offset'];actions[1-seat],_=foul.choose(visible)
                                    elif peer:
                                        with torch.no_grad():logits,_=peer(batch([state['observations'][1-seat]],'cpu'));actions[1-seat]=int(logits.argmax(-1).item())
                                    elif opponent['kind'] in ['power','random']:actions[1-seat]=scripted(state['observations'][1-seat],rng,opponent['kind'])
                                    else:raise ValueError('Unknown opponent kind')
                                actor='engine';states=vector.step([actions])
                            final=states[0];row.update(natural=final['terminated'],truncated=final['truncated'],points=(final['returns'][seat]+1)/2 if final['terminated'] else None,
                                terminal_reason=final['terminal_reason'],engine_turns=final['engine_turns'],adapter_phases=final['adapter_phases'],accepted_choices=final['accepted_choices'],
                                unavailable_choice_retries=final['unavailable_choice_retries'],visible_choice_rejections=sum(final['rejections']))
                            write(out/f'trace-{opponent["name"]}-{index}-{seat}.json',vector.audit()[0])
                    except Exception as exc:
                        if isinstance(exc,AdministrativeWallLimit):row.update(truncated=True,terminal_reason='process_wall_safety_cap')
                        else:
                            kind=exc.category if isinstance(exc,WorkerError) else actor+'_inference_timeout' if isinstance(exc,TimeoutError) else actor+'_illegal_action' if 'illegal action' in str(exc).lower() or 'masked action' in str(exc).lower() else actor+'_adapter_fault'
                            row.update(fault=type(exc).__name__+': '+str(exc),fault_category=kind)
                        (out/f'fault-{opponent["name"]}-{index}-{seat}.log').write_text(traceback.format_exc());stop=True
                    row['wall_seconds']=time.monotonic()-before;rows.append(row);write(out/'rows.json',rows)
                    if stop:break
                if stop:break
            if stop:break
    result=summarize(rows,len(schedule),schedule);result['by_opponent_and_seat']={k.replace('/foul-seat','/candidate-seat'):v for k,v in result['by_opponent_and_seat'].items()}
    result.update(perspective='candidate',seconds=time.monotonic()-started,candidate_identity={k:v for k,v in candidate.items() if k!='path'},qualified_core_release_sha256=spec['qualified_core_release_sha256'])
    write(out/'summary.json',result)


def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 spec=json.loads(Path(a.request).read_text());out=Path(a.output)
 try:run(spec,out,a.request)
 except Exception as exc:
  write(out/'failure.json',{'status':'failed','raw_score_0_100':None,'normalized_reference_score':None,'failure_category':'source_interface_or_setup_admission','fault':type(exc).__name__+': '+str(exc)});raise
if __name__=='__main__':main()
