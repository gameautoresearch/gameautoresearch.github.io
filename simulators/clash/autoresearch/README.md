# Clash research substrate

The game-side half of Clash research sessions: the starter agent, the contract,
the release constructor, the worker entrypoints and the evaluator for
`clash-fixed8-v1` (fixed eight-card mirror deck, level 11) under the
**short-v1 contract** ([CONTRACT-short-v1.md](CONTRACT-short-v1.md)): 60 s regulation + 30 s
overtime at 2 decisions per simulated second (`common.MATCH_FORM`, `../native/match_forms.py`),
on a scripted four-opponent panel. It replaced fast-v4 (180+120 s, 4 decisions per second) on
2026-11-13; fast-v4's constructor is at git tag `full-size-20261113` and `fast_release.py` here
refuses to build or verify. Sessions themselves are run by the single research harness; read
[harness/README.md](../../../harness/README.md) for the workspace, the `recipe`
and `network` variants and the commands. The Clash backend is
`harness/games/clash/` (session controller and Modal provider). Earlier
harnesses that lived here are in git history at `ba5c0d5`.

## Setup

```sh
python3 tools/restore_upstream_metadata.py        # after a fresh clone
sh simulators/clash/autoresearch/setup.sh          # creates .venv-clash-ar (Modal host tooling, no torch)
.venv-clash-ar/bin/python -m unittest harness.tests.test_clash
.venv-clash/bin/python -m pytest simulators/clash/autoresearch/tests -q    # learning checks need torch
```

Run everything from the repository root. `.venv-clash-ar` runs the harness and
the constructor; `.venv-clash` (torch, pytest) runs this directory's tests.

## What is here

| Path | Purpose |
|---|---|
| `benchmark-current.json`, `benchmark-short-v1.json`, `CONTRACT-short-v1.md` | The current contract `clash-autoresearch-short-v1`: the short match form, L40S / 4 CPU / 16 GiB, 7,200 s sessions, native fast starter, scripted panel |
| `benchmark-v4.json`, `CONTRACT-fast-v4.md` | The previous contract `clash-autoresearch-fixed8-v4` (full-length matches), kept as a record |
| `CONTRACT.md`, `CONTRACT-evaluation-v2.md`, `-v3.md`, `-v3.1.md`, `benchmark.json`, `benchmark-v3.json`, `benchmark-v3.1.json` | The earlier contract layers fast-v4 is written as a supplement to: task, admission rules and the v3.1 evaluation interface practice still uses. Commands named inside them belong to removed harnesses |
| `starter/`, `recipes/` | The starter policy, learner and trainer, and `starter.json` (1,024 worlds) / `smoke.json` (64 worlds); both name `match_form` `short-v1` with `rules.decision_ticks` 10 |
| `task/brief.md`, `task/game_rules/` | The game brief and Markdown rulebook staged into every workspace |
| `short_release.py` | Release constructor `prepare(out)` and `verify(release)`: new interface, panel, public practice and 32 fresh protected seeds; no base release |
| `fast_release.py` | The fast-v4 constructor; refuses while the runtime plays the short form (use the tag) |
| `release_v3.py` | Public seed sets, evaluation interfaces, fixed panel and schedule resolution of the v3/v3.1 evaluation contract |
| `ledger.py` | JSON session ledger: reservations, provider-receipt settlement, submissions, finalization |
| `bundle.py` | Sealed policy packages: safe weight files, one-million-parameter cap, source and engine binding |
| `train_entry.py`, `fast_entry.py`, `fast_adapter.py`, `fast_policy.py`, `checkpoint_writer.py` | Trusted worker side of a training run; together with `../model.py` they are the runtime hash bound into every fast bundle, so changing one invalidates existing bundles |
| `fast_worker.py`, `fast_qualification.py` | Finite worker wrapper (`train`, `historical`, `resume-check` modes) and the GPU resume/export check it can run |
| `evaluation_v3.py`, `evaluation_v3_parallel.py`, `evaluation_v2.py`, `arena.py`, `policy_worker.py`, `wire.py` | Worker side of practice: isolated unprivileged policy processes, paired-seat panel games, scoring |
| `readiness_contract.py`, `refusal_diagnostics.py`, `visible_teacher.py` | Policy admission review, the execution observer, and the `teacher` / `intermediate` panel opponents |
| `decision_trace.py`, `public_choice_probe.py`, `public_score_probe.py`, `viewing_replays_v3.py` | Standalone evaluator diagnostics on sealed bundles; no session uses them |
| `wiki_maintenance/` | Maintainer sources and provenance for `task/game_rules/` |
| `setup.sh`, `requirements-host.txt`, `tests/` | Interpreter setup and this directory's tests (`tests/fixtures.py` builds an offline base release) |

## Releases

```sh
.venv-clash-ar/bin/python -m harness prepare clash --out NEW_RELEASE \
  --option base_release=evidence/strong-bots-20261109-0516Z/clash/evaluator/releases/development-v31-2
```

`prepare(out, base_release)` is offline and starts no worker. `base_release` is
an existing protected-public v3/v3.1 development release (format 3, mode
`provisional-development`). It is only read, and it must stay where it is: the
copied schedule keeps naming its opponent bundle paths. The constructor checks
what practice still depends on and nothing else:

- the base's engine binding equals this tree's (`common.engine_binding()`);
- every base public file matches the base's recorded hash;
- the base's evaluation interface is the one `release_v3.selected_interface` declares;
- the base's teacher source hash is the current `visible_teacher.py`;
- the protected schedule matches its commitment, each learned opponent bundle
  verifies against its identity, and the public practice schedule resolves to
  those same identities.

The base's own evaluator-source fingerprint is not checked, because it names
files of removed harnesses; it is kept in the new release as
`parent_runtime_source_sha256` for the record. No code here builds a new base
panel release or a private final bank.

The new release holds `public/` (starter, `config.json`, the contract, the
panel declaration `interface.json` / `practice.json` / `opponent-registry.json`
/ `observation-schema.json` copied byte for byte, and the researcher workspace),
`private/` (the protected schedule and panel bundles, unchanged) and
`release.json`. `verify(release)` refuses a release when the engine binding, the
trusted fast runtime, any top-level `*.py` in this directory, a public file, the
workspace, the schedule or an opponent bundle differs.

## Sessions

One training launch takes 90 to 3,300 worker seconds (`--seconds`, the worker's
hard stop). The ledger reserves that plus 30 seconds for settlement and teardown
and is then charged the provider's own start-to-end interval; `ar budget` shows
the largest launch that still fits as `allocatable_seconds`. In the contract's
seconds mode one job runs at a time; a session in the harness's dollar mode
(`--budget-mode usd`) runs up to four jobs, two of them training runs, at once,
and its ledger holds each run's whole grant in dollars until it settles. Practice is a separate CPU-only worker (8 CPU / 32 GiB, 600 s),
limited to six per session and not charged to the GPU ledger. A run is sealed
only if its returned source matches the snapshot and it shows a finite, non-zero
learning update.

Private grading is closed for this contract. `Backend.grade()` refuses, so the
public practice result of the submitted network is the only evaluation a Clash
session produces.

Private schedules, opponent weights and generated native table assets stay
local; nothing here is published.
