"""Append a corrected analysis without altering any frozen baseline receipt."""
import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    source=json.loads((a.run/'original-summary.json').read_text());native=[];training=[]
    for n in [32,128,1024]:
        rows=[r for r in source['simulation'] if r['n_games']==n]
        rates={mode:statistics.median(r['decisions_per_complete_protocol_second'] for r in rows if r['mode']==mode) for mode in ['upstream','candidate']}
        validations=[json.loads((a.run/f'n{n}-{mode}-validation.json').read_text()) for mode in ['upstream','candidate']]
        keys=['decisions','natural_episodes','views_masks_results_sha256','final_state_rng_sha256']
        checks={key:validations[0][key]==validations[1][key] for key in keys}
        if not all(checks.values()):raise AssertionError(checks)
        native.append({'n_games':n,'median_decisions_per_complete_protocol_second':rates,'speedup':rates['candidate']/rates['upstream'],
            'oracle_parity':checks,'decisions':validations[0]['decisions'],'natural_episodes':validations[0]['natural_episodes'],
            'protocol':'Three interleaved repetitions, all eight-player lobbies active in both arms, identical precomputed actions with seven internal scripted bots; initialization, observations, stepping and final checkpoint included.'})
    for n in [32,128,512]:
        for trial in range(1 if n==32 else 2):
            rows={r['mode']:r for r in source['training'] if (r['n_games'],r['trial'])==(n,trial)}
            metrics={m:[json.loads(line) for line in (a.run/f'learn-n{n}-t{trial}-{m}/metrics.jsonl').read_text().splitlines()] for m in rows}
            totals={m:sum(x['counts']['accepted_learning_samples'] for x in ms) for m,ms in metrics.items()}
            count_keys=['batch_steps','ready_learning_decisions','ready_opponent_decisions','rejected','invalid_lobbies','accepted_learning_samples','completed_learning_episodes']
            same_length=len(metrics['earliest'])==len(metrics['candidate'])
            parity={'episodes':same_length and all(x['episodes']==y['episodes'] for x,y in zip(metrics['earliest'],metrics['candidate'])),
                'semantic_counts':same_length and all(all(x['counts'][k]==y['counts'][k] for k in count_keys) for x,y in zip(metrics['earliest'],metrics['candidate'])),
                'accepted_samples':same_length and all(x['counts']['accepted_learning_samples']==y['counts']['accepted_learning_samples'] for x,y in zip(metrics['earliest'],metrics['candidate']))}
            first_difference=next((i+1 for i,(x,y) in enumerate(zip(metrics['earliest'],metrics['candidate'])) if x['episodes']!=y['episodes'] or any(x['counts'][k]!=y['counts'][k] for k in count_keys)),None)
            old_check=next(x['checks'] for x in source['training_parity'] if (x['n_games'],x['trial'])==(n,trial))
            seconds={m:r['process_seconds'] for m,r in rows.items()};rates={m:totals[m]/seconds[m] for m in rows}
            training.append({'n_games':n,'trial':trial,'completed_updates':{m:len(ms) for m,ms in metrics.items()},'accepted_samples':totals,
                'process_seconds':seconds,'samples_per_process_second':rates,'throughput_ratio':rates['candidate']/rates['earliest'],
                'corrected_parity':parity,'semantic_count_keys':count_keys,'first_episode_or_count_difference_update':first_difference,
                'policy_bitwise':old_check['policy'],'optimizer_bitwise':old_check['optimizer'],'cuda_rng_bitwise':old_check['cuda_rng']})
    result={'source_run':str(a.run.resolve()),'source_receipt_unchanged':True,'native_simulator':native,'complete_neural_trainer':training,
        'qualification':{'gpu_simulation':False,'tenfold_speedup':False,'original_neural_learning_bitwise':False},
        'initialization_diagnostic':'Separate CPU-only diagnostic found earliest hardcoded four-thread versus candidate one-thread orthogonal actor QR differs by <=1.862645149230957e-09 in 27961 elements before any action/update. Each thread-count repeat is exact. Training rows are therefore throughput comparisons of the declared recipe, not bitwise-equivalent learning claims.',
        'correction':'Frozen v1 original-summary training_parity episodes erroneously compared missing checkpoint stats fields (None==None). This appended analysis uses every actual metrics.jsonl episode and shared semantic count record (candidate-only diagnostic counters are listed separately in raw receipts); no original receipts or runs changed.',
        'scope':'Pinned untouched upstream simulator and separately pinned earliest trustworthy complete trainer; direct current comparisons only, no multiplication of ratios from older campaigns.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
