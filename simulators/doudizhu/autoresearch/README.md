# Dou Dizhu research substrate

The game-side half of Dou Dizhu research sessions: the starter agent, the
contract, the release constructor, the worker entrypoints and the evaluator for
`doudizhu-fixed-role-research-v2` (fixed roles, no bidding, full 54-card deck).
Sessions themselves are run by the single research harness; read
[harness/README.md](../../../harness/README.md) for the workspace, the `recipe`
and `network` variants and the commands. The Dou Dizhu backend is
`harness/games/doudizhu/` (session controller and Modal provider). Earlier
harnesses that lived here are in git history at `ba5c0d5`.

## Setup

```sh
python3 tools/restore_upstream_metadata.py        # after a fresh clone
sh simulators/doudizhu/setup.sh                   # .venv-doudizhu: torch and the simulator
uv venv --python 3.12.9 .venv-doudizhu-ar         # research interpreter: Modal, no torch
uv pip install --python .venv-doudizhu-ar/bin/python -r simulators/doudizhu/autoresearch/requirements-host.txt
.venv-doudizhu/bin/python -m pytest simulators/doudizhu/autoresearch/tests -q
.venv-doudizhu-ar/bin/python -m unittest harness.tests.test_doudizhu
```

Run everything from the repository root. `integrations/doudizhu` is a
compatibility link to `simulators/doudizhu`; staged worker trees and the
starter's imports keep that package name.

## What is here

| Path | Purpose |
|---|---|
| `benchmark.json`, `CONTRACT.md` | The benchmark contract (machine and prose forms) |
| `contract.py` | Session defaults (`DEFAULT`), editable file list, frozen runtime file list and its digest |
| `starter/`, `recipes/` | The starter policy, learner and trainer, and its `starter.json` / `smoke.json` configs |
| `task/brief.md`, `task/game_rules/` | The game brief and Markdown rulebook staged into every workspace |
| `prepare.py` | Release constructor (`prepare`) and `audit_release` |
| `ledger.py` | SQLite session ledger: allocations, submissions, milestone points, finalization |
| `bundle.py` | Sealed policy packages: tensor-only weights, tensor cap, source- and engine-bound verification |
| `build_cuda.py` | Builds the frozen native and L40S (sm_89) CUDA engines inside the worker image, before any GPU is attached |
| `train_entry.py`, `export_entry.py` | Worker side of a training run: unprivileged supervised trainer, then root-side export |
| `evaluate_entry.py`, `arena.py`, `policy_worker.py`, `wire.py`, `grading.py` | Worker side of practice and grading: isolated policy process, role-balanced panel games, scoring |
| `manage.py` | Host-private `grade_session`, anchor `calibrate`, `panel_paths` |
| `transport.py` | Guard for the pinned Modal 1.5.2 gRPC transport used by the provider |
| `hidden/` | Reference trainer and recipe for anchor calibration; staged only into a release's private tree |
| `wiki_maintenance/` | Maintainer sources and provenance for `task/game_rules/` |
| `requirements-host.txt`, `tests/` | Research interpreter requirements and this directory's tests |

## Releases

```sh
.venv-doudizhu/bin/python -m harness prepare doudizhu --out NEW_RELEASE
.venv-doudizhu-ar/bin/python -m harness prepare doudizhu --out NEW_RELEASE --option include_existing=false
.venv-doudizhu-ar/bin/python -m harness verify doudizhu --release NEW_RELEASE
```

`prepare(out, include_existing=True)` is offline and starts no worker. With
`include_existing` it converts the three historical development policies under
`evidence/doudizhu-implementation/` into panel bots through the source-bound
checkpoint loader; that needs torch, so use `.venv-doudizhu`. Without it the
panels hold only the random and heuristic bots and either interpreter works.
It writes:

- `frozen/`: the simulator runtime files (`contract.RUNTIME`), the pinned
  DouZero sources with their offline Git metadata and the `CONTROL` modules
  listed in `prepare.py`; this tree becomes the worker image at `/opt/dd`;
- `public/`: starter (`agent/`), configs (`recipes/`), contract, practice
  opponent names and the researcher workspace (`agent_workspace/`,
  `workspace_manifest.json`) built by `harness.workspace.stage`;
- `private/`: fresh grade and calibration seeds, the practice / development /
  official panels (`panel.json`; official stays `null` until calibration),
  development bot packages, the reference trainer and the calibration plan;
- `release.json`: file hashes of `public/` and `frozen/` and the seed
  commitments, checked by `audit_release` every time a provider is created.

Only `public/agent_workspace/` reaches the researcher. A training worker gets
the frozen image plus exactly the three training files and the chosen config of
that run; it has no network, secrets, volumes or private files.

## Sessions

The backend maps the harness operations onto the controller and ledger here:
one L40S worker at a time with four CPU cores and 16 GiB, 7,200 allocated
seconds by default (`benchmark.json`), every allocated second charged from
provider timestamps including failed runs, at least 120 seconds per training
run and 10 per practice run. `--smoke` selects the contract's bounded smoke:
600 seconds, runs of 30 to 360 seconds, 8 practice deals and a 20-second
teardown reserve that is never granted. A lease the provider cannot confirm as
ended stays held as `uncertain` and blocks new work until the session is
reopened and the provider reports its end.

Practice (`ar eval`) is a budgeted worker run on the public practice deals
against the practice panel. Private grading runs on a separate CPU-only worker
that is not charged to the research allowance and is reachable only through
`python -m harness grade`.

## Grading and calibration

`grade_session` grades the submission held at each ledger point (one third, two
thirds and the end of the allowance); the harness reports the final point as the
session's answer. The raw score is 100 × role- and opponent-balanced win
probability on the private deals. Reference-normalized reward needs calibrated
anchors: `manage.calibrate(release, out, provider)` trains three starter and
three reference seeds at the full allowance, checks the paired gap on the
calibration schedule and then on the grading schedule, and writes
`calibrated-panel.json`. It has no command of its own and has never been run at
full budget. Until a catalog exists every grade is development-only (raw score,
normalized `null`); pass one with `--backend-option catalog=PATH`, and force a
development grade of a full-allowance session with `--backend-option development=true`.
A session whose allowance differs from 7,200 seconds, which includes smoke
sessions and the fresh final training of a recipe session, is always graded as
development and never receives an official normalized score. Smoke grades use a
16-deal prefix of the private schedule.
