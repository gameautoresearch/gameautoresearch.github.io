# Tavern benchmark contract and DeepSeek smoke — 2026-11-08

The new task is `tavern-eight-implemented-v1`: full eight-player AutoTavern 37.0.2
Rust CPU games with CUDA learning, five random minion types, two hero choices,
seven board/ten hand slots and 30 actions per recruit turn. Guards remain faults.
The independent GPU-rules rewrite remains paused. No historical ledger was reset.

`simulators/tavern/autoresearch/benchmark.json` and `CONTRACT.md` define the task.
`task/instruction.md` and `task/brief.md` provide the initial agent context.
The editable random-init starter is a 64-hidden/4-embedding masked MLP and plain
PPO with gamma 1. Training, resume, public evaluation, immutable submission,
finalization and independent private grading are implemented. The full 7200-second
entrypoint lets DeepSeek choose experiment lengths and submit repeatedly, with
2400/4800-second milestone snapshots. Paid commands require `--execute`.

## Actual autonomous smoke

DeepSeek V4 Flash inspected the starter and budget, edited policy defaults and the
recipe to 128 hidden units/8-dimensional embeddings, ran actual CUDA learning,
evaluated public practice, submitted `smoke1` and finalized. One model request
failed with HTTP 2 transport error before any GPU allocation; its reservation was
retained, and the same campaign resumed using HTTP 1.1. A rejected run name caused
no allocation. No operator supplied the agent's source change or submission.

The 90-second learning window completed 59 PPO updates and 242,572 accepted samples
with finite losses and changed weights. One training no-progress guard discarded
its entire lobby;1,887 natural learner episodes remained. No masked action was
rejected. Public greedy practice scored 0/100 across 16 natural games: the policy
remains weak. Independent private grading scored**31.25/100**, mean placement
**5.8125**, top-four rate**25%**, against seven copies of a separately trained,
frozen four-update PPO opponent. All 16 private games were natural, with zero
faults, truncations or illegal actions. The two-seed-group bootstrap interval
[25,37.5] is diagnostic; reference normalization remains null.

Provider metadata settles the sole L4/four-core/16 GiB allocation at
**126.350456953 seconds**, including startup/transfer/IO/idle/teardown, below 600.
No unresolved GPU hold remains. Separate CPU-only practice, grading and isolation
workers total 110.25125 seconds. Model-reported charges are$0.001251610866;
conservative model reservations total$0.02068627561, including the failed call,
well below$2. Estimated known worker resource cost is about$0.04885 at retained
historical rates, excluding CPU image builds/storage/egress; invoice cost remains
unsettled. All owned apps/workers are confirmed stopped. A live boundary probe
confirmed UID 65534, inaccessible private files, blocked outbound networking and
no credentials in the candidate process environment.

## Full packaging and checks

The finalized offline release-v4 has a 128-lobby starter,64-game public practice,
and 16 private seed groups×8 seats×4 lobbies against each of three committed frozen
CPU-trained development opponents:1,536 private games. Smoke has its own 32-lobby
recipe and 16-game panel against one opponent. Exact inventory/weight/schedule
hashes are committed; missing/mutated declared opponents fail preflight.
These weak opponents do not unlock strong-reference normalization.

34 local tests pass; seven CUDA-only tests skip on this Mac. Checks include real
CPU PPO, exact CPU resume across model/optimizer/RNG/environment/history, mismatch
rejection, budget holds/caps/finalization, immutable submission tamper rejection,
and offline full/smoke plans. New-harness exact CUDA resume remains unrun.
The observed smoke uses immutable release-v1; final full-panel packaging was
verified offline and was not substituted into that score. No full 7200-second
research allocation,1536-game grade or strength calibration was executed.

Reproduction from repository root:

```sh
sh simulators/tavern/autoresearch/setup.sh
.venv-tavern/bin/python -m pytest simulators/tavern/autoresearch/tests simulators/tavern/tests -q
.venv-tavern/bin/python simulators/tavern/autoresearch/manage.py prepare --out NEW_RELEASE
.venv-tavern-ar/bin/python simulators/tavern/autoresearch/manage.py smoke --release NEW_RELEASE --out NEW_SESSION
.venv-tavern-ar/bin/python simulators/tavern/autoresearch/manage.py research --release NEW_RELEASE --out NEW_FULL_SESSION
```

The last two commands are offline plans. Add `--execute` only for an authorized
allocation. The executed smoke used the pre-existing host Modal interpreter;
portable commands now use the pinned game-specific `.venv-tavern-ar` environment.
Resume an interrupted smoke with the same output path and `--resume`; never clear
or replace its budgets. Private panels, weights, seeds and transcripts stay in
ignored local evidence, not published artifacts. Exact receipts and structured
limits are indexed by the companion JSON report.
