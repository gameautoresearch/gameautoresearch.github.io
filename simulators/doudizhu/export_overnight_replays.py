"""Fresh CPU viewing games: immutable overnight AI finals vs public panel bots."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulators.doudizhu import export_demo_replays as demo
from integrations.doudizhu.autoresearch.arena import TrustedPanelPolicy
from integrations.doudizhu.autoresearch.bundle import verify
from integrations.doudizhu.environment import heuristic, ROLES

CAMPAIGN = ROOT / 'evidence/overnight-20261108-1533Z/doudizhu'

class Match:
    def __init__(self, candidate, role):
        self.candidate, self.role = candidate, role
    def choose(self, observations):
        return [self.candidate.choose([o])[0] if o.role == self.role else heuristic(o) for o in observations]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    demo.PUBLIC_DEMO_SEED = 202611090901
    entries, receipts = [], []
    rows = json.loads((CAMPAIGN / 'researcher-outcomes.json').read_text())['rows']
    for row, label, slug in zip(rows, ['DeepSeek', 'GLM Flash', 'Qwen'], ['deepseek', 'glm', 'qwen']):
        bundle = Path(row['final_source_or_bundle_path'])
        identity = verify(bundle)['bundle_sha256']
        candidate = TrustedPanelPolicy(bundle)
        for seat, role in enumerate(ROLES):
            film = demo.record_game(Match(candidate, role), identity, seat)
            film.update(id=f'doudizhu-{slug}-shedder-{seat + 1:02d}',
                        title=f'{label} vs Cooperating Shedder · {seat + 1:02d}',
                        subtitle=f'AI {role.replace("_", " ")} · ' + ('AI wins' if film['outcome']['landlord_won'] == (role == 'landlord') else 'AI loses'))
            film['players'] = {r: label + ' · final' if r == role else 'Cooperating Shedder' for r in ROLES}
            film['policy'].update(name=f'{label} overnight final', description=f'Exact November 8–9 explicit {label} submission controls {role}; the other two seats use the evaluation panel’s cooperating-shedder heuristic. Fresh public CPU viewing game, separate from scored evaluation.')
            film['provenance'].update(campaign='overnight-20261108-1533Z', researcherModel=row['model'], candidateRole=role, opponent='cooperating-shedder', inference='CPU greedy, same policy ABI', scoredEvaluation=False)
            path = args.output / (film['id'] + '.json')
            path.write_text(json.dumps(film, separators=(',', ':')) + '\n')
            entries.append({k: film[k] for k in ('id','gameId','title','subtitle')} | {'url': f'replays/doudizhu/overnight-20261109/{path.name}', 'moves':len(film['frames'])-1})
            receipts.append({'id':film['id'], 'bundle_sha256':identity,'decisions':film['outcome']['decisions'],'natural':True,'verification':film['provenance']['verification']})
            print(f"{film['id']}: {film['outcome']['decisions']} moves verified", flush=True)
    (args.output / 'index.json').write_text(json.dumps(entries, indent=2)+'\n')
    (args.output / 'verification.json').write_text(json.dumps(receipts,indent=2)+'\n')

if __name__ == '__main__': main()
