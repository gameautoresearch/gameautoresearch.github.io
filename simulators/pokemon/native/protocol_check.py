"""Hidden-choice rejection and pending-choice checkpoint qualification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .bridge import Native, build
from .rules import compile_moves
from .teams_check import load_team_fixture, team_comparison, observation_comparison
from .visible import PublicTables


def run(output, fixtures):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=build(output/'build')
    rows=[];checks=waiting=dual=truncations=0;digest=hashlib.sha256()
    with Native(library)as native:
        blank=native.snapshot()
        with Path(fixtures).open('rb')as stream:
            for case,line in enumerate(stream):
                digest.update(line);data=json.loads(line);ids={m['id']:i for i,m in enumerate(data['moves'])}
                moves=compile_moves(data['moves']);chart=np.asarray(data['chart'],dtype=np.int8)
                tables=PublicTables(data['metadata'],data['chart'],ids)
                # The oracle's decisions are frozen; a retained rejected choice
                # remains a decision, rather than being silently replaced.
                native.restore(blank)
                load_team_fixture(native,data,ids);count=0
                for phase,frame in enumerate(data['frames']):
                    before=native.snapshot();old=type(native.records[0]).from_buffer_copy(before)
                    native.step([frame['actions']],moves,chart);state=native.records[0]
                    mismatch=team_comparison(native,frame['state']) or observation_comparison(native,frame['state'],tables)
                    if mismatch:raise AssertionError((case,phase,mismatch))
                    changed=sum(state.unavailable_choices)-sum(old.unavailable_choices)
                    if not changed:continue
                    if state.rng!=old.rng or state.turn!=old.turn or state.mid_turn!=old.mid_turn:
                        raise AssertionError('Rejected switch advanced the simulator')
                    if state.fault:raise AssertionError('Unavailable choice became a fault')
                    checks+=1;count+=1;waiting+=bool(state.pending_mask);dual+=changed==2
                    snapshot=native.snapshot()
                    if phase+1<len(data['frames']):
                        actions=data['frames'][phase+1]['actions']
                        native.step([actions],moves,chart);expected=native.snapshot()
                        native.restore(snapshot);native.step([actions],moves,chart)
                        if native.snapshot()!=expected:raise AssertionError('Pending-choice resume diverged')
                        native.restore(snapshot)
                    # The same rejected submission must honor the decision cap.
                    native.restore(before);native.records[0].limit=old.phases+1
                    native.step([frame['actions']],moves,chart)
                    if not native.records[0].truncated or native.records[0].fault:
                        raise AssertionError('Rejected-choice boundary lost truncation')
                    truncations+=1;native.restore(snapshot)
                rows.append(dict(case=case,rejected_boundaries=count))
    if not checks or not waiting or not dual:raise AssertionError('Missing hidden-choice protocol coverage')
    result=dict(full_gen9_admission=False,cases=len(rows),rejected_boundaries=checks,
                pending_opponent_choices=waiting,dual_rejections=dual,
                exact_pending_choice_resume=True,decision_cap_checks=truncations,
                input_sha256=digest.hexdigest(),rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixtures',required=True)
    args=parser.parse_args();run(args.output,args.fixtures)
