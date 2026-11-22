"""Tavern tournament worker: uploaded into the release's CPU evaluation sandbox and run there as root.

Both entrants run exactly as the mixed-v2 evaluator runs its candidate (`IsolatedVisible`: the frozen
`autoresearch/policy_worker.py` in a separate process, empty environment, visible arrays over a pipe, the
15-second forward deadline), each under its own unprivileged uid (A 65534, B 65533) with a private copy of
only its own bundle; neither is imported in-process. The other N-2 seats of an N-player lobby are the release
registry's trusted visible scripts. Lobbies are played by the frozen `evaluation_readiness.lobby.run_lobby`
(every seat, natural terminals only, explicit fault accounting).

A lobby is (seed, k, swap): A and B sit opposite each other in seats k and k+N/2; the scripts fill the other
seats clockwise from seat k in a fixed order; `swap` exchanges A and B and nothing else. So each lobby has an
exact mirror in which the two networks trade places, and over k = 0..N/2-1 each entrant plays every seat
(four players: seats k and k+2, k = 0..1; eight players: k and k+4, k = 0..3).

Only the standard library is imported at module level so the host can import `lobbies`.
"""
import json
import os
import shutil
import sys
import time

UIDS = {'a': 65534, 'b': 65533}
BUNDLE_FILES = ('policy.py', 'learner.py', 'train.py', 'weights.npz', 'identity.json', 'summary.json')


def lobbies(unit_seeds, styles):
    """[(unit, seed, k, swap, seats)] with seats[i] in {'a', 'b', ('script', style)}; N per seed.

    The lobby has N = len(styles) + 2 players (even, 4..8)."""
    players = len(styles) + 2
    if players not in (4, 6, 8):
        raise ValueError('An even lobby of 4-8 players (two entrants plus scripts) required')
    half = players // 2
    plan = []
    for unit, seed in unit_seeds:
        for k in range(half):
            for swap in (0, 1):
                seats = [None] * players
                seats[k], seats[k + half] = ('a', 'b') if swap == 0 else ('b', 'a')
                others = [(k + d) % players for d in range(1, players) if d != half]
                for seat, style in zip(others, styles):
                    seats[seat] = ('script', style)
                plan.append(dict(unit=unit, seed=seed, k=k, swap=swap, seats=seats))
    return plan


def stage(request):
    """Root-private request/registry, one owner-only bundle copy per entrant uid."""
    os.chmod('/private', 0o700)
    for name in os.listdir('/private'):
        path = os.path.join('/private', name)
        if os.path.isfile(path):
            os.chmod(path, 0o600)
    staged = {}
    for label, uid in UIDS.items():
        source = request['bundles'][label]
        dest = '/tmp/tournament-entrants/' + label
        os.makedirs(dest)
        for name in BUNDLE_FILES:
            shutil.copy2(os.path.join(source, name), os.path.join(dest, name))
            os.chown(os.path.join(dest, name), uid, uid)
            os.chmod(os.path.join(dest, name), 0o400)
        os.chown(dest, uid, uid)
        os.chmod(dest, 0o500)
        staged[label] = dest
    os.chmod('/tmp/tournament-entrants', 0o711)
    return staged


def privacy(uid, other, other_bundle):
    """The guest's privacy probe, per entrant uid: the schedule, the uploads and the other bundle are unreadable."""
    import subprocess
    paths = ['/private/tournament.json', '/private/entrants/%s/weights.npz' % other, os.path.join(other_bundle, 'weights.npz')]
    probe = ("import os,json,sys; assert os.geteuid()==%d; denied=[]\n"
             "for p in %r:\n try: open(p,'rb').read(1)\n except PermissionError: denied.append(True)\n"
             " else: denied.append(False)\n"
             "assert all(denied), denied; assert not any(k in os.environ for k in "
             "['OPENROUTER_API_KEY','MODAL_TOKEN_SECRET','MODAL_TOKEN_ID'])\n"
             "print(json.dumps({'uid':os.geteuid(),'private_paths_denied':len(denied),'credential_env_absent':True}))"
             ) % (uid, paths)

    def demote():
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
    done = subprocess.run([sys.executable, '-I', '-c', probe], env={'PATH': '/usr/local/bin:/usr/bin:/bin',
                          'OMP_NUM_THREADS': '1'}, cwd='/tmp', preexec_fn=demote, timeout=30, capture_output=True,
                          text=True, check=True)
    return json.loads(done.stdout)


class Entrant:
    """IsolatedVisible under a chosen uid, restarted after any fault it causes."""

    def __init__(self, label, bundle, log):
        self.label, self.bundle, self.log = label, bundle, log
        self.inner = None
        self.faulted = False
        self.starts = 0

    def start(self):
        import subprocess
        from simulators.tavern.evaluation_readiness import policies
        policies.admit_bundle(self.bundle)
        uid = UIDS[self.label]

        def demote():
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
        inner = policies.IsolatedVisible.__new__(policies.IsolatedVisible)
        inner.bundle, inner.policy_id = self.bundle, 'entrant-' + self.label
        inner.p = subprocess.Popen([sys.executable, '-I', str(policies.HERE.parent / 'autoresearch/policy_worker.py'),
                                    str(self.bundle)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                   env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'OMP_NUM_THREADS': '1'}, cwd='/tmp',
                                   bufsize=0, preexec_fn=demote)
        self.inner = inner
        self.starts += 1

    @property
    def policy_id(self):
        return 'entrant-' + self.label

    def remote_act(self, obs, indices):
        import numpy as np
        try:
            if self.inner is None:
                self.start()
            out = self.inner.remote_act(obs, indices)
        except BaseException:
            self.faulted = True
            raise
        a = np.asarray(out[0])
        if a.shape != (len(indices),) or a.dtype.kind not in 'iu' or ((a < 0) | (a >= 245)).any() or \
                not obs['action_mask'][np.asarray(indices), a].all():
            self.faulted = True  # run_lobby's own check then rejects it as an illegal action
        return out

    def reset(self):
        if self.inner is not None:
            try:
                self.inner.close()
            except Exception:
                pass
        self.inner = None
        self.faulted = False

    def close(self):
        self.reset()


def play(request, staged, deadline, log):
    from simulators.tavern.evaluation_readiness.lobby import run_lobby
    from simulators.tavern.evaluation_readiness.opponents import VisibleScript
    entrants = {label: Entrant(label, staged[label], log) for label in UIDS}
    scripts = {}
    rows, pending = [], []
    try:
        for lobby in request['lobbies']:
            if time.time() > deadline - request['lobby_seconds'] - 15:
                pending.append({k: lobby[k] for k in ('unit', 'k', 'swap')})
                continue
            policies = []
            for seat in lobby['seats']:
                if isinstance(seat, str):
                    policies.append(entrants[seat])
                else:
                    style = seat[1]
                    if style not in scripts:
                        scripts[style] = VisibleScript(style)
                        scripts[style].policy_id = 'script-' + style
                    policies.append(scripts[style])
            for e in entrants.values():
                if e.inner is None:
                    e.start()
            row = run_lobby(policies, lobby['seed'], request['rules'], seconds=request['lobby_seconds'])
            faulted = [label for label, e in entrants.items() if e.faulted]
            a_seat, b_seat = lobby['seats'].index('a'), lobby['seats'].index('b')
            terminals = row['terminals']
            rows.append(dict(unit=lobby['unit'], k=lobby['k'], swap=lobby['swap'], a_seat=a_seat, b_seat=b_seat,
                             valid=row['complete_valid_lobby'], failure=row['failure'],
                             fault_by=faulted[0] if len(faulted) == 1 else ('both' if faulted else None),
                             a_placement=terminals.get(a_seat, {}).get('placement'),
                             b_placement=terminals.get(b_seat, {}).get('placement'),
                             placements={str(s): t.get('placement') for s, t in terminals.items()},
                             process_seconds=row['process_seconds'], frames=row['counts'].get('frames'),
                             latency={name: dict(calls=len(v), seconds=sum(x['seconds'] for x in v))
                                      for name, v in row['latency'].items()},
                             initial_state_sha256=row['initial_state_sha256'],
                             final_state_sha256=row['final_state_sha256'], native=row['native']))
            for label in faulted:
                entrants[label].reset()  # a fresh process for the next lobby
            with open('/private/progress.json', 'w') as handle:
                json.dump(dict(completed=len(rows), scheduled=len(request['lobbies'])), handle)
    finally:
        for e in entrants.values():
            e.close()
    return rows, pending, {label: e.starts for label, e in entrants.items()}


def main(request_path):
    import torch
    started = time.time()
    request = json.loads(open(request_path).read())
    deadline = float(request['absolute_stop'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    staged = stage(request)
    checks = {label: privacy(uid, other, staged[other]) for (label, uid), other in zip(UIDS.items(), ('b', 'a'))}
    with open('/private/entrant-stderr.log', 'w') as log:
        rows, pending, starts = play(request, staged, deadline, log)
    import taverngym as tg
    result = dict(rows=rows, pending=pending, privacy=checks, process_starts=starts, seconds=time.time() - started,
                  native=tg.build_info()['source_hash'], uids=UIDS)
    with open('/private/tournament-result.json', 'w') as handle:
        json.dump(result, handle)


if __name__ == '__main__':
    main(sys.argv[1])
