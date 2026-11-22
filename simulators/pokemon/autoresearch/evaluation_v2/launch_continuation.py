"""Explicit new-ledger CPU continuation of historical caps; no historical receipt edits."""
import json
import os
import argparse
from pathlib import Path
from .campaign import BASE
from .remote import launch
from ..bundle import FILES,verify
from ..contract import ROOT,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
    old=ROOT/'evidence/overnight-20261108-1533Z/pokemon'
    release=Path(json.loads((old/'active-release.json').read_text())['path'])
    schedule=json.loads((release/'private/schedules.json').read_text())['calibration']
    retained=json.loads((old/'retained-capped-case-evidence.json').read_text())['rows']
    label=args.label;mount=Path('/data')/label/'input'
    files=[];cases=[]
    for index,row in enumerate(retained):
        start=row['opaque_seed_index']//2*2
        submission=json.loads((old/row['session']/'gpu-ledger.json').read_text())['submissions'][-1]
        verify(submission['bundle'],submission['bundle_sha256'])
        bundle=Path(submission['bundle']);name='bundle-'+row['session']
        for filename in (*FILES,'manifest.json'):files.append((bundle/filename,name+'/'+filename))
        trace=old/'capped-trace-diagnosis'/f'capped-case-{index+1}'/'trace.json';trace_name=f'trace-{index+1}.json'
        files.append((trace,trace_name))
        cases.append({'label':f'capped-case-{index+1}','bundle':str(mount/name),'bundle_sha256':submission['bundle_sha256'],
            'retained_trace':str(mount/trace_name),'seeds':schedule[start:start+2],
            'target':{'local_seed_index':row['opaque_seed_index']-start,'opponent_index':['power','development-a','development-b'].index(row['opponent']),
                'opponent':row['opponent'],'seat':row['seat']}})
    panel=[{'name':'power','kind':'power'}]
    for name in ['development-a','development-b']:
        for filename in ['weights.pt','opponent.json']:files.append((release/'private/opponents'/name/filename,name+'/'+filename))
        panel.append({'name':name,'kind':'checkpoint','path':str(mount/name)})
    # Repeated GLM bundle inputs are identical; upload each immutable file only once.
    files=list({name:(src,name) for src,name in files}.values())
    print(json.dumps(launch(label,'continuation',{'cases':cases,'panel':panel,'max_adapter_phases':4096},files,seconds=300)))
if __name__=='__main__':main()
