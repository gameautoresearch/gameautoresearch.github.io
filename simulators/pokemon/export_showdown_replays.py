"""Local CPU public demonstrations from sealed final graphs, with full Showdown logs.

Does not import candidate Python, train, spend on providers, or read private seed banks.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .autoresearch.runtime.bundle import verify
from .autoresearch.runtime.policy_process import PolicyProcess
from .autoresearch.runtime.sdk import runtime_signature

ROOT = Path(__file__).resolve().parents[2]
MATCHES = [
    ('sol61-a1', 'astra6-a1', 'GPT-6.1 Sol · a1', 'GPT-6 Astra · a1'),
    ('inkling2s-b2', 'deepseek-v4pro-a2', 'Inkling 2 Small · b2', 'DeepSeek V4 Pro · a2'),
    ('luna6-a1', 'kimi-k3-a2', 'GPT-6 Luna · a1', 'Kimi K3 · a2'),
    ('sol61-a2', 'glm53-a3', 'GPT-6.1 Sol · a2', 'GLM-5.3 · a3'),
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entrants', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    entrants = json.loads(a.entrants.read_text())['entrants']
    library = []
    for i, (left, right, left_name, right_name) in enumerate(MATCHES):
        seed = 900_111_500 + i
        identities = []
        with contextlib.ExitStack() as stack:
            policies = []
            for key in (left, right):
                bundle = Path(entrants[key]['bundle'])
                meta = verify(bundle)
                binding = json.loads((bundle/'binding.json').read_text())
                if binding['runtime_sha256'] != runtime_signature():
                    raise ValueError('Sealed runtime differs: ' + key)
                identities.append(dict(name=key, bundleSha256=meta['bundle_sha256'], weightsSha256=meta['files']['weights.pt'], graphSha256=meta['files']['graph.json']))
                policies.append(stack.enter_context(PolicyProcess(bundle, development=True)))
            proc = subprocess.Popen(['node', str(Path(__file__).with_name('showdown_capture.cjs')), '--team-size=3'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            def call(x):
                proc.stdin.write(json.dumps(x)+'\n'); proc.stdin.flush()
                value = json.loads(proc.stdout.readline())
                if not value['ok']: raise RuntimeError(value['error'])
                return value
            try:
                state = call(dict(op='reset', seed=seed))['state']
                actions = []; started = time.monotonic()
                while not (state['terminated'] or state['truncated']):
                    if time.monotonic()-started > 180: raise TimeoutError('Demo exceeded 180 seconds')
                    joint = [policies[s].choose([state['observations'][s]])[0] if state['ready'][s] else None for s in (0, 1)]
                    actions.append(joint)
                    state = call(dict(op='step', actions=joint))['state']
                checked = call(dict(op='verify', seed=seed, actions=actions))
                if not checked['matched']: raise AssertionError('Independent action/protocol replay differs')
                if not state['terminated'] or state['truncated']: raise ValueError('Demo did not end naturally')
                # Labels affect presentation only; original trace is retained separately.
                log = checked['log'].replace('|player|p1|p1|', '|player|p1|'+left_name+'|').replace('|player|p2|p2|', '|player|p2|'+right_name+'|')
                log = log.replace('|win|p1', '|win|'+left_name).replace('|win|p2', '|win|'+right_name)
                rid = left+'-vs-'+right
                movie = dict(schemaVersion=1, gameId='pokemon-showdown', id=rid, title=left_name+' vs '+right_name, players=[left_name,right_name], log=log,
                    turns=state['turn'], phases=state['steps'], outcome=dict(natural=True, truncated=False, winner=state['winner'], fault=None),
                    provenance=dict(kind='fresh-public-demo', publicSeed=seed, format='gen9randombattle@@@Max Team Size = 3', policies=identities,
                        traceSha256=checked['trace'], logSha256=hashlib.sha256(log.encode()).hexdigest(), verifiedIndependentReplay=True,
                        note='Fresh local CPU viewing match using the November 14–15 sealed finals. Not a historical private tournament game or a new strength estimate. Only player-visible observations enter each frozen graph.'))
                (a.output/(rid+'.json')).write_text(json.dumps(movie, indent=2)+'\n')
                (a.output/(rid+'.log')).write_text(log+'\n')
                library.append(dict(id=rid,title=movie['title'],players=movie['players'],turns=movie['turns'],url='replays/pokemon/showdown-20261115/'+rid+'.json',winner=state['winner']))
                print(rid, state['turn'], 'turns', state['winner'], flush=True)
            finally:
                proc.stdin.close(); proc.wait(timeout=10)
        (a.output/'index.json').write_text(json.dumps(library,indent=2)+'\n')

if __name__ == '__main__': main()
