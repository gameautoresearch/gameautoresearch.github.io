# Tavern research substrate

The game side of Tavern (Hearthstone Battlegrounds on AutoTavern 37.0.2) research sessions, in the
**four-player small form** `tavern-four-small-v1`: four players, regular tavern tiers 1–4, three lobby minion
types, two hero choices (the eight-player form is at git tag `full-size-20261113`). Sessions
are run by the single research harness: see [harness/README.md](../../../harness/README.md) for the
workspace, the two task variants, commands and what a session leaves behind. The Tavern backend that
drives the code here is [harness/games/tavern/](../../../harness/games/tavern/). Earlier harnesses,
model drivers and campaign tooling are in git history at `ba5c0d5`.

## What is here

| Path | Purpose |
|---|---|
| `starter/` | The editable starter: `policy.py` (masked MLP), `learner.py` (plain PPO), `train.py` (training loop) |
| `recipes/` | `starter.json` and `smoke.json`, staged as `configs/train.json` and `configs/smoke.json` |
| `task/brief.md` | The game brief the harness adds to the researcher's instructions |
| `task/game_rules/` | The offline Markdown rulebook staged as `game_rules/` |
| `wiki_maintenance/` | Sources and tooling records for regenerating that rulebook |
| `rules_enabled/release.py` | The release constructor (`prepare`) |
| `evaluation_v2/` | The current benchmark contract (`benchmark.json`, `CONTRACT.md`), release verification, the selected reference-pool receipt and its builder (`build_four_pool.py`), entrant registration and private-bank seed exclusions |
| `game_length.py` | Typical game length of a rules form (turns, learner decisions, whole-lobby decisions) and the zero-padding check for absent opponents |
| `contract.py` | Frozen rules, config validation and the native source binding |
| `ledger.py` | The session's allocated-worker-time ledger |
| `train_entry.py` | Trusted training entry run on the GPU worker; loads the editable files, exports weights and identity |
| `policy_worker.py` | Unprivileged inference process that loads a submitted policy during evaluation |
| `diagnostic_trace.py` | Opt-in bounded action traces for failing lobbies |
| `provider.py` | Modal sandbox helpers still used by `../reference_training/` and `../evaluation_readiness/` |
| `grade.py`, `benchmark.json`, `CONTRACT.md` | The v1 evaluator and declaration, restated for four seats. The artifact binding is `tavern-four-small-v1`; `../evaluation_readiness/` and replay export use them |
| `setup.sh`, `requirements-host.txt` | Create the research interpreter `.venv-tavern-ar` |
| `tests/` | Offline tests of the ledger, contract, training entry, policy worker and traces |

The evaluator itself (lobby runner, reference policies, isolation and terminal-replay proof) is
[`../evaluation_readiness/`](../evaluation_readiness/); practice and private grading run its
`five_hour.guest` entry inside the release's frozen image.

## Releases and sessions

```sh
.venv-tavern-ar/bin/python -m harness prepare tavern --out NEW_RELEASE
.venv-tavern-ar/bin/python -m harness run tavern recipe --release NEW_RELEASE --out NEW_SESSION --model MODEL
```

`prepare` is offline. It needs the private reference pool named by
`evaluation_v2/selected_registry-four-small-v1.json` (local `evidence/`, never published; rebuild it with
`python -m simulators.tavern.autoresearch.evaluation_v2.build_four_pool --out NEW_DIR` under `.venv-tavern`; `--option registry=DIR`
selects another compatible pool) and the game interpreter `.venv-tavern`, which holds the current
native AutoTavern build used to compute the source binding. A release contains the public starter and
configs, the researcher workspace, the frozen worker source for the image and a private copy of the pool.

Training runs on one L40S with four CPU cores and 16 GiB (the repository GPU with the mixed-v2 CPU/RAM), one run at a
time. Each run reserves 150 seconds beyond its training time for setup, export and teardown, and is
charged the worker's actual allocated time from the provider's receipt; an allocation whose end is
unknown keeps its reservation and blocks further runs. Public practice (48 lobbies: 4 seed groups × 4 seats × 3 profiles) and
private grading (288 fresh lobbies: 24 × 4 × 3, drawn only after the session is finalized) run on a CPU-only worker with
no network and are not charged to the research allocation.

## Tests

```sh
.venv-tavern-ar/bin/python -m unittest harness.tests.test_tavern
.venv-tavern/bin/python -m pytest simulators/tavern/autoresearch/tests simulators/tavern/tests -q
```
