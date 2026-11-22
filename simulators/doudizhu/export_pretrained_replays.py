"""Local official pretrained self-play, upstream score parity and public movies."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from simulators.doudizhu.export_demo_replays import record_game
from simulators.doudizhu.pretrained import PretrainedDouZero
from simulators.doudizhu.environment import SOURCE, ROLES, get_obs
from simulators.doudizhu.artifacts import write_json

PUBLIC_SEED=202611090934


class UpstreamParity:
    def __init__(self,bot):
        spec=importlib.util.spec_from_file_location('pinned_douzero_reference_models',SOURCE/'douzero/dmc/models.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.models={r:module.model_dict[r]().cpu().eval() for r in ROLES}
        for r,model in self.models.items():model.load_state_dict(bot.policy.roles[r].state_dict(),strict=True)
        self.bot=bot;self.states={r:0 for r in ROLES};self.actions=0;self.max_error=0.

    @torch.inference_mode()
    def __call__(self,game,obs,choice):
        # Only this differential checker sees the oracle infoset. The actual bot
        # has already chosen using its public Observation and no engine handle.
        raw=get_obs(game._engine.game_infoset)
        keys=[tuple(sorted(a)) for a in raw['legal_actions']]
        if set(keys)!=set(obs.legal):raise AssertionError('Legal-action set mismatch')
        x,z,actions=obs.encode();actual=np.concatenate((np.repeat(x[None],len(actions),axis=0),actions),axis=1)
        positions={key:keys.index(key) for key in obs.legal};order=[positions[key] for key in obs.legal]
        np.testing.assert_array_equal(raw['x_batch'][order],actual)
        np.testing.assert_array_equal(raw['z_batch'][order],np.repeat(z[None],len(actions),axis=0))
        values=self.models[obs.role](torch.tensor(raw['z_batch'],dtype=torch.float32),torch.tensor(raw['x_batch'],dtype=torch.float32),return_value=True)['values'].flatten().numpy()[order]
        adapted=self.bot.values([obs])[0]
        np.testing.assert_allclose(adapted,values,atol=1e-6,rtol=1e-5)
        if int(np.argmax(values))!=choice:raise AssertionError('Upstream greedy action mismatch')
        self.states[obs.role]+=1;self.actions+=len(actions);self.max_error=max(self.max_error,float(np.abs(adapted-values).max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--games',type=int,default=3)
    parser.add_argument('--public-seed',type=int,default=PUBLIC_SEED)
    parser.add_argument('--id-prefix',default='douzero-pretrained-selfplay')
    args=parser.parse_args()
    if not 1<=args.games<=12:raise ValueError('Bounded viewing batch requires1..12games')
    if not args.id_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.id_prefix):raise ValueError('Invalid replay ID prefix')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    bot=PretrainedDouZero(args.model);audit=UpstreamParity(bot);entries=[];receipts=[]
    slug=args.output.name
    for index in range(args.games):
        film=record_game(bot,bot.identity,index,observer=audit,demo_seed=args.public_seed)
        name=f'{args.id_prefix}-{index+1:02d}'
        film.update(id=name,title=f'DouZero pretrained self-play · {index+1:02d}',subtitle='Official pretrained bot · all three seats')
        film['players']={r:'DouZero pretrained · '+r.replace('_',' ') for r in ROLES}
        film['policy'].update(name='Official DouZero pretrained',description='Official demo archive role checkpoints; no local training. The same pretrained three-role policy controls every seat. Fresh public local self-play, not a scored strength estimate.')
        film['provenance'].update(sourceType='official-pretrained-demo',sourceUrl=bot.manifest['source']['source_url'],localTraining=False,scoredEvaluation=False,inference='CPU FP32 greedy; upstream input/Q/argmax parity checked at every state',roleCheckpoints={r:bot.manifest['roles'][r]['sha256'] for r in ROLES},publicDealProtocol='practice deterministic deal; public demonstration seed and index')
        write_json(args.output/(name+'.json'),film)
        entries.append({k:film[k] for k in ('id','gameId','title','subtitle')}|{'url':f'replays/doudizhu/{slug}/{name}.json','moves':len(film['frames'])-1,'matchType':'self-play'})
        receipts.append({'id':name,'outcome':film['outcome'],'trajectory_sha256':film['provenance']['trajectorySha256'],'policy_identity':bot.identity,'public_seed':args.public_seed,'deal_index':index,'deal_sha256':film['provenance']['dealSha256'],'role_checkpoints':film['provenance']['roleCheckpoints']})
        print(f"{name}: {film['outcome']['decisions']} moves, natural finish, full replay verified",flush=True)
    write_json(args.output/'index.json',entries)
    write_json(args.output/'verification.json',{'games':receipts,'upstream_states_by_role':audit.states,'legal_action_scores_checked':audit.actions,'max_Q_absolute_error':audit.max_error,'all_public_inputs_and_greedy_actions_match_upstream':True,'all_movies_replayed_on_view_engine':True,'no_training_or_paid_workers':True})


if __name__=='__main__':main()
