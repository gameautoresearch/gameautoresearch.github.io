# Pokémon autoresearch

Game-side substrate for Pokémon research sessions: Gen 9 random battle 6v6, trained
on the admitted native C++ engine and evaluated on pinned Showdown. Sessions are
run by the one research harness; read [harness/README.md](../../../harness/README.md)
for how. The host-side backend (controller, provider, diagnostics) is
`harness/games/pokemon/`. Earlier harness generations are in git history at `ba5c0d5`.

Read [CONTRACT.md](CONTRACT.md) and [benchmark.json](benchmark.json) for the task,
the worker profile (L40S, 4 physical CPU cores, 16 GiB), accounting and evaluation.

## What is here

| Path | What it is |
|---|---|
| `runtime/` | The native runtime a worker runs: `bootstrap.py` (trusted sandbox entry, source pack), `worker.py`, `supervisor.py` (isolated trainer, deadlines, failed-stage retention), `env_service.py` / `env_client.py` / `fastenv.py` (the engine behind a root-owned service), `graph_export.py` / `graph_worker.py` / `bundle.py` / `policy_process.py` (the scored tensor-graph policy), `arena.py` (evaluator), `practice.py`, `qualification.py`, `recovery.py`, `sdk.py` (runtime identity) and [GRAPH_CONTRACT.md](runtime/GRAPH_CONTRACT.md) |
| `runtime/starter/` | The starter the researcher edits: `policy.py`, `learner.py`, `train.py` |
| `recipes/` | Its configs: `starter.json` and the small `smoke.json` |
| `prepare.py` | Release constructor (`prepare`) and `audit` |
| `ledger.py` | One session's allocation ledger |
| `task/brief.md`, `task/game_rules/` | The game brief and the offline rulebook staged into the workspace |
| `evaluation_v2/`, `strong_bots/` | The qualified Showdown evaluator and its public-information search opponent, plus the reference-opponent campaign code they came from. Files named in the evaluator registry are hash-bound; do not edit them |
| `contract.py`, `bundle.py`, `arena.py`, `policy_worker.py` | Identities, package format and isolated inference of the original Showdown trainer. The evaluator still uses them for its two retained opponents and for privilege dropping |
| `starter/` | The original Showdown starter. `starter/learner.py` and `contract.py` are pinned byte for byte by the native engine's admission certificate, so they must not change or move; nothing trains with them |
| `wiki_maintenance/` | Sources and generator of the rulebook |
| `requirements-host.txt` | Host dependencies of `.venv-pokemon` |

The editable starter is in `runtime/starter/` and not in `starter/` only because of
that certificate: a worker rebuilds the engine with the certificate and refuses to
if `autoresearch/starter/learner.py` differs from the certified file.

## Releases

`python -m harness prepare pokemon --out NEW_RELEASE` calls `prepare.py`. It is
offline and takes no options. A release holds the workspace, one frozen snapshot of
the worker source with its manifest and pack, the native admission certificate and
engine data, the evaluator's inputs and two retained opponents, and a fresh private
environment-state key. It contains no private evaluation seeds: those are created in
the session ledger only after a final policy is locked.

The constructor reads retained local inputs and verifies each against the digest its
own record declares: `evidence/pokemon-implementation/native-cpu-completion-20261109/`
(admission and engine data) and `evidence/strong-bots-20261109-0516Z/pokemon/evaluator/`
(evaluator registry, search source and wheels, opponents). Without them it fails.

Workers start from the retained runtime image named in `runtime/bootstrap.py`. A new
release has a new source identity; it needs its own exact-source live qualification
on real L40S workers (`runtime/qualification.py` is the worker side) before a scored
session. No host entry for that gate exists in the harness yet.

## Tests

Run from the repository root with `.venv-pokemon/bin/python`:

```sh
python -m unittest harness.tests.test_pokemon
python -m unittest discover -s simulators/pokemon/autoresearch/runtime/tests -t .
python -m unittest discover -s simulators/pokemon/autoresearch/evaluation_v2/tests -t .
python -m unittest discover -s simulators/pokemon/autoresearch/strong_bots -t .
```

They are offline. `runtime/tests` trains the starter on the admitted engine on CPU
and checks exact resume, export and sealing; it does not cover the Linux privilege
separation, the runtime image, GPU training or the search opponent.
