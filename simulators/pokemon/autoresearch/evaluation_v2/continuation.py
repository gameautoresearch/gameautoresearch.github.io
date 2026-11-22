"""Retained 12-game topology and immutable policy replay, with exact 512-phase prefixes."""
import argparse
import json
from pathlib import Path
import numpy as np
from integrations.pokemon.autoresearch import arena
from integrations.pokemon.autoresearch.contract import write
from integrations.pokemon import env
from .env import EvaluationVector


def summary(state):
    hp=[];active=[]
    for obs in state['observations']:
        mons=np.asarray(obs['mons'])[:6]
        hp.append(float(mons[:,2].sum()));ids=np.flatnonzero(mons[:,1]>.5)
        active.append(int(ids[0]) if len(ids) else None)
    return {'hp_sums':hp,'active_slots':active,'ready':state['ready'],
        'phases':state['steps'],'natural':state['terminated'],'truncated':state['truncated']}


def run(request,output):
    context={};original=env.Vector
    class Traced(EvaluationVector):
        def __init__(self,workers,per_worker,**kwargs):
            assert workers==1
            super().__init__(per_worker);context['vector']=self;self.records=[]
        def step(self,actions):
            i=context['target_env'];before=self.states[i];record=None
            if not before['terminated'] and not before['truncated']:
                record={'before':summary(before),'actions':list(actions[i])}
            states=super().step(actions)
            if record is not None:
                record['after']=summary(states[i]);self.records.append(record)
            return states
        def close(self):
            if self.workers and self.workers[0].process.poll() is None:
                context['audit']=self.audit()
            super().close()
    env.Vector=Traced
    cases=[]
    try:
        for case in request['cases']:
            target=case['target'];index=target['local_seed_index']*6+target['opponent_index']*2+target['seat']
            context.clear();context['target_env']=index
            result=arena.evaluate(case['bundle'],request['panel'],case['seeds'],limit=request['max_adapter_phases'])
            records=context['vector'].records;recorded=json.loads(Path(case['retained_trace']).read_text())
            prefix=records[:512]
            # Only terminal safety metadata differs on the original v1 capped phase.
            normalized=json.loads(json.dumps(prefix))
            if len(normalized)==512:
                normalized[-1]['after']['truncated']=True
                normalized[-1]['after']['ready']=[False,False]
            exact=normalized==recorded
            target_row=next(r for r in result['rows'] if r['seed_index']==target['local_seed_index'] and r['seat']==target['seat'] and r['opponent']==target['opponent'])
            audited=context['audit'][index]
            final=audited['result']
            item={'label':case['label'],'original_batch_topology':12,'original_bundle_sha256':case['bundle_sha256'],
                'original_512_phase_prefix_exact':exact,'prefix_comparison':'Actions and before/after own-view HP, active slots, readiness and phase counters; original512truncation metadata normalized only',
                'original_primary_score_unchanged':None,'original_result_remains_null':True,
                'adapter_phases':final['adapter_phases'],'engine_turns':final['engine_turns'],
                'accepted_choices':final['accepted_choices'],'choice_attempts':final['choice_attempts'],
                'unavailable_choice_retries':final['unavailable_choice_retries'],
                'termination_policy':final['termination_policy'],'terminal_reason':final['terminal_reason'],
                'natural':final['terminated'],'truncated':final['truncated'],'winner':final['winner'],
                'new_evaluation_points':target_row['points'],'fault':target_row['fault'],
                'native_tie_evidence':'Pinned engine protocol emitted tie' if final['natural_tie'] else None,
                'public_trace_sha256':audited['hash']}
            folder=output/case['label'];folder.mkdir(parents=True,exist_ok=False)
            write(folder/'summary.json',item);write(folder/'trace.json',records)
            write(folder/'batch-result.json',result);write(folder/'audit.json',context['audit'])
            cases.append(item)
    finally:env.Vector=original
    write(output/'summary.json',{'cases':cases,'all_original_prefixes_exact':all(r['original_512_phase_prefix_exact'] for r in cases),
        'all_reached_engine_outcome':all(r['natural'] for r in cases),'original_grades_unchanged':True,
        'max_adapter_phases':request['max_adapter_phases'],'retry_safety_cap_per_seat':16})


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(json.loads(Path(a.request).read_text()),Path(a.output))
if __name__=='__main__':main()
