# Hex research substrate

The game-side half of Hex research sessions: the starter agent, the contract, the release
constructor, the worker entrypoints and the evaluator for `hex13-no-swap-research-v1`. Sessions are
run by the single research harness; read [harness/README.md](../../../harness/README.md) for the
workspace, the `recipe` and `network` variants and the commands. The Hex backend is
`harness/games/hex/` (session controller and Modal provider).

## Setup

```sh
.venv-hex/bin/python -m pytest -q simulators/hex/tests simulators/hex/autoresearch/tests
.venv-hex/bin/python -m unittest harness.tests.test_hex
```

See [../README.md](../README.md) for the interpreter and the optional local MoHex build. Run everything
from the repository root.

## What is here

| Path | Purpose |
|---|---|
| `benchmark.json`, `CONTRACT.md` | The benchmark contract (machine and prose forms) |
| `contract.py` | Session defaults (`DEFAULT`), MoHex levels, the panel, editable files, frozen runtime list |
| `starter/`, `recipes/` | The starter policy, learner and trainer, and its `starter.json` / `smoke.json` configs |
| `task/brief.md`, `task/game_rules/`, `task/dryrun.py` | Brief, Markdown rulebook and the in-box CPU dry run behind `ar check` |
| `prepare.py` | Release constructor (`prepare`) and `audit_release` |
| `ledger.py` | SQLite session ledger (shared design with Dou Dizhu): allocations, submissions, milestone points |
| `bundle.py` | Sealed policy packages: tensor-only weights, tensor cap, source- and engine-bound verification |
| `train_entry.py`, `export_entry.py` | Worker side of a training run: unprivileged supervised trainer, then root-side export |
| `evaluate_entry.py`, `arena.py`, `policy_worker.py`, `wire.py`, `grading.py` | Practice and grading: isolated policy process, MoHex pool, seat-paired openings, scoring |
| `manage.py` | Host-private `grade_session` |
| `ladder.py` | Host-only panel calibration: opponent-vs-opponent matches and the Elo fit behind `PANEL`'s ratings |
| `transport.py` | Guard for the pinned Modal 1.5.2 gRPC transport used by the provider |
| `wiki_maintenance/` | Maintainer notes and provenance for `task/game_rules/` |
| `tests/` | Substrate tests (MoHex games skip without a binary) |

## Releases

```sh
.venv-hex/bin/python -m harness prepare hex --out NEW_RELEASE
.venv-hex/bin/python -m harness verify hex --release NEW_RELEASE
.venv-hex/bin/python -m harness task hex recipe --release NEW_RELEASE --out NEW_TASK_DIR
```

`prepare(out)` is offline and starts no worker. It writes:

- `frozen/`: the engine, the worker control modules and the complete pinned MoHex source, laid out as
  in the repository, for the worker image (which compiles MoHex at build time, before any GPU).
- `public/`: the starter, recipes, contract and the researcher workspace.
- `private/`: `grade-seeds.json` (a fresh 128-bit opening seed and a policy seed) and `panel.json`.
- `release.json`: every public and frozen file hash and the private seed commitment.

There is no calibration step: the raw panel score anchored on MoHex is the grade. Releases prepared before
2026-11-14 are 9x9 releases; they do not verify against this source and are not to be re-prepared in place.

## Workers

Training: one L40S, 4 CPU, 16 GiB, network blocked; the trainer runs as uid 65534 and the root
exporter seals `model.json`, `weights.pt` and the exact source. Leases: a shape the researcher chooses,
their own commands as uid 65534, the session's scratch Volume at `/scratch` (named in the session's
`research/scratch-volume.json`; it is cloud-only until collected for the run archive), every command and
its last output recorded under `research/jobs/<lease>/exec/`. Training and leases use the research image,
which has no MoHex. Practice: an 8-CPU, 16 GiB CPU sandbox on the evaluation image, charged to the
session's research budget at its shape's rate per second. Private grading: the same shape, host-only and not charged. In both, the
candidate runs as an unprivileged child process seeing only public arrays, under the per-move inference
clock (`position_seconds`, `call_seconds`), and MoHex runs as up to seven GTP processes. The 9x9 contract
ran on Modal (suite of 2026-11-13); no 13x13 image has been built and no 13x13 session has run yet.

## Tournament

Every campaign ends with `tools/hex_tournament.py`: a round robin of the sessions' final networks with the
five strongest panel opponents as references, under the same clock. `tools/nanny.py` starts it by itself.
