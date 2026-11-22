"""Private final banks sealed after selection in a NEW session only.

The four-player small form draws 24 seed groups: 24 groups x 4 seats x 3 profiles = 288 lobbies, the eight-player
form's lobby count (12 x 8 x 3).
"""
import fcntl
import json
from pathlib import Path
import secrets
from harness.workspace import sha
from harness.policy import verify_release
from simulators.tavern.autoresearch.evaluation_v2.seed_keys import lobby_key,training_overlap
from simulators.tavern.autoresearch.evaluation_v2.seed_exclusions import collect
from simulators.tavern.autoresearch.rules_enabled.release import FINAL_GROUPS


def panel_for(release,session):
    release,session=Path(release).resolve(),Path(session).resolve()
    binding=verify_release('tavern',release)
    state=json.loads((session/'state.json').read_text())
    if not state.get('finalized')or state.get('final_submission')!=state.get('submission'):raise ValueError('Freeze selection before generating a private final bank')
    selected=session/'submissions'/state['final_submission'];manifest=json.loads((selected/'submission.json').read_text())
    if any(sha(selected/name)!=value for name,value in manifest['files'].items()):raise ValueError('Immutable final submission changed')
    expected=dict(release_sha256=binding['sha256'],submission_sha256=sha(selected/'submission.json'))
    private=session/'grading-private';private.mkdir(mode=0o700,exist_ok=True)
    panel=private/'panel.json';receipt=private/'bank-commitment.json'
    with (private/'bank.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if panel.exists()or receipt.exists():
            if not panel.is_file()or not receipt.is_file():raise ValueError('Partial private bank; retain it for explicit recovery')
            prior=json.loads(receipt.read_text())
            if prior.get('binding')!=expected or prior.get('panel_sha256')!=sha(panel):raise ValueError('Private bank/selection binding changed')
            return panel
        roots=[session,release/'public',release/'private/registry']
        excluded,configs,evidence=collect(roots=roots)
        seeds=[]
        for _ in range(100000):
            seed=secrets.randbits(64)
            if seed not in seeds and lobby_key(seed)not in excluded and not any(training_overlap(seed,c)for c in configs):seeds.append(seed)
            if len(seeds)==FINAL_GROUPS:break
        if len(seeds)!=FINAL_GROUPS:raise ValueError('Unable to construct disjoint final bank')
        seats=int(json.loads((release/'private/registry/registry.json').read_text())['binding']['rules']['num_players'])
        panel.write_text(json.dumps(dict(label='successor-private-final-after-selection',seeds=seeds))+'\n');panel.chmod(0o600)
        receipt.write_text(json.dumps(dict(binding=expected,panel_sha256=sha(panel),groups=FINAL_GROUPS,seats=seats,profiles=3,required_lobbies=FINAL_GROUPS*seats*3,generated_after_selection=True,exclusions=evidence),indent=2)+'\n');receipt.chmod(0o400)
        return panel
