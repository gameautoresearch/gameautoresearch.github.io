# AverageJoe reference trainer

This is an independent reference-training track, outside the frozen 600k-parameter,
stateless entrant benchmark. It uses the actual upstream **15,351,761-parameter
Equinox HistoryTransformer** from `strakam/AverageJoe` at
`5041e635dcf6c87b8b844cb2e244ad3e0abe40d7` and its unmodified
`configs/custom/L_7d_gae90.yaml`.

The [November 9 smoke readout](../../../reports/generals/averagejoe-smoke-2026-11-09.md)
records 20 updates, 10.49M player transitions and 563.319 total allocated H100
seconds across both attempts. The engineering check passed; random-opponent
draw-adjusted points fell, so this is not yet a qualified strong reference.

The optional reference track expects a separate upstream checkout at
`vendor/engines/AverageJoe`. See [the upstream repository](https://github.com/strakam/AverageJoe)
and match the pinned commit and hashes in `upstream.json` before running this track.
The release does not distribute the upstream source or the reference weights.

## Fidelity

- 7 transformer blocks, width448, 8 heads,3×3 patches, BF16 forward/FP32 master
  weights, full38-plane history/visibility observation and512-step opponent
  scoreboard history; original categorical value head/HL-Gauss loss.
- Original direction-major9×24×24 action logits, including per-cell pass logits.
  This is deliberately different from the benchmark entrant's4609-ID codec.
- Spawn-distance curriculum:2–6,4–8,6–13,11–17,17–28. Only a completed512-game
  paired-seat current-policy evaluation against upstream random may advance a
  stage, at60% wins. EMA is retained separately and does not drive the gate.
-512 environments×512 rollout ticks; both seats train; one PPO epoch; the top
  quarter of absolute normalized advantages; original clipping, power-law
  learning/entropy schedules, EMA0.999, sparse terminal win/loss reward.
-200,000 requested pool entries and refresh every20iterations. The pinned native
  generator truncates this to199,969 to balance all49rectangular shapes equally.
- Upstream's exact GAE/truncation mask and optimizer-schedule arithmetic are
  preserved rather than silently corrected during a reproduction.

The orchestration removes multi-device parameter replication for a one-GPU run,
fuses GAE/filtering/diagnostics, keeps EMA/optimizer/rollouts on device, and freezes
finished evaluation games with early stopping. The model and PPO equations are
imported unchanged. This has not yet established a throughput advantage over
upstream's full transformer pipeline.

Initial map banks are prepared with the exact pinned generator on CPU and
source-bound/checksummed. CPU setup and image build are reported separately;
loading/transferring these inputs is included in GPU allocation. Later pool
refreshes happen inside the allocated worker. Initial23×23 states follow the
upstream `init_state`; subsequent auto-resets use the mixed17–23pool. No old seed
bank, checkpoint binding, submission, campaign ledger or benchmark is changed.

## Local checks and new inputs

```sh
uv pip install --python .venv-generals-ar/bin/python equinox==0.13.8 ruamel.yaml==0.19.1
.venv-generals-ar/bin/python -m pytest simulators/generals/averagejoe/tests -q
JAX_PLATFORMS=cpu .venv-generals-ar/bin/python -m simulators.generals.averagejoe.prepare \
  --output evidence/generals-averagejoe-NEW/inputs
```

Every output must be new. Local checks cover memory/action-codec semantics,
published model/curriculum settings, finite learning, exact next-update restore
(model, optimizer, RNG, engine/reset cursor and memories), and paired evaluation
outcomes/legality.

## Bounded Modal pilot

The November9 user instruction explicitly authorized a roughly10-minute Modal
qualification. Its **new cumulative600GPU-worker-second budget** is recorded in
`evidence/generals-averagejoe-20261109/authorization.json`; all attempts/failures,
setup, compilation, training, evaluation, saving, transfer and idle time count.
No researcher model calls, old allowance reuse, warm workers or automatic GPU
retries. The driver pins one H100,4physical CPU cores and16GiB RAM in staging,
blocks network access inside the worker, reserves before allocation, and settles
from provider task start/end timestamps. Estimated billing is separate from a
provider invoice. Image build happens before GPU reservation/allocation.

The completed attempts are `smoke-01` (startup failure, retained and charged) and
`smoke-02` (successful, 570-second grant after accounting for the first attempt).
The pilot's cumulative ledger is closed. A future paid run requires a separately
authorized budget; do not reuse these tags or reset its ledger.

The process reserves time for final evaluation, source-bound checkpoints and
export. It logs captures, natural losses, caps, illegal actions, city ownership,
curriculum stage and actual complete-iteration throughput. A short run that does
not reach the full-distance stage is not a qualified full-task reference and is
never labeled superhuman.

The saved full trainer state includes network, optimizer, EMA, PRNG, games,
reset cursors and both observation memories. Stage0 continuation also requires
the original cached map file/hash. Curriculum continuation after a later stage
requires retaining that stage's refreshed pool; no claim of portable later-stage
resume is made by this first smoke implementation. The original smoke trainer
has no resume CLI; the separate continuation runner below adds parent restore.
Exact next-update restoration is covered by the small CPU correctness test.

## Detached one-hour continuation, November 9

**Completed:** see the [final continuation readout](../../../reports/generals/averagejoe-continuation-2026-11-09.md).
The worker and observer exited cleanly after 53m 07s of GPU allocation. Updates
20 → 240 added 115.34M player transitions; fixed stage-0 random wins improved
4.69% → 53.13%, below the 60% promotion gate. Final artifacts were verified and
four examples loaded into the replay viewer. No cities were captured in those
four examples. The separate one-hour ledger is closed; no full-task/expert
qualification or further paid allowance is implied.

The user subsequently authorized continuing end to end with the computer off.
The new `continuation.py` resumes all seven components of the verified smoke
state using `lineage.py`: it independently checks the archived old source and
checkpoint, requires every existing model/engine/framework file to remain
byte-identical, and records the newly added orchestration modules. The original
smoke code, inputs, state and closed ledger remain unchanged.

`detached.py` starts exactly one H100 Sandbox with the trainer as its autonomous
entrypoint. The provider timeout is 3,500 seconds against a fresh cumulative
3,600-second allowance. An in-container watchdog stops the child earlier to
reserve final assessment, saving and shutdown time. No GPU retry is automatic.
The detached CPU observer can only inspect/terminate this specific worker; it
cannot create another one. It settles provider allocation and verifies final
checkpoint hashes after worker exit, then writes a remote readout.

All inputs, archived parent source, current source, logs, regenerated map pools,
checkpoint state, current/EMA weights and assessments are retained privately on
the staging v2 Volume `generals-averagejoe-continuation-20261109`. Closed files
are explicitly committed with `sync` from the Sandbox. Checkpoints occur after
the first resumed update, at multiples of 25, around curriculum promotions and
at completion. Each includes the active pool hash and evaluation starts.
Scheduled assessment retains the upstream 50-update frequency and 60% win gate;
it uses a separate public assessment RNG and explicitly matches self-play's
disabled general-trade rule. The historical smoke evaluator enabled that rule;
its counts remain unchanged and are retained separately, not silently treated as
the new baseline. A simultaneous-general-capture regression test checks the rule
distinction. The model/PPO/reward implementation is unchanged. Only one seed is
being continued.

At completion the worker exports four public viewing games, two against random
and two against Hunter, if the reserved export time remains. All sampled actions
are checked and an independent engine-only resimulation must match final states.
These demonstrations are distinct from the 512-game scored curriculum checks.
They use the current stage's spawn distances and do not calibrate expert skill.

Read progress or retrieve durable results without starting a GPU:

```sh
.venv-generals-ar/bin/python -m simulators.generals.averagejoe.status
# Once the worker has finished, collect the entire private Volume into a new folder:
.venv-generals-ar/bin/python -m simulators.generals.averagejoe.status --collect
```

Local launch/authorization receipts are under
`evidence/generals-averagejoe-continuation-20261109/`. The remote observer writes
`control/allocation.json`, `control/gpu-budget.json` and `control/readout.md`.
The Sandbox writes `output/result.json`, `output/latest.json`,
`output/checkpoints/`, `output/pools/` and `output/replays/`. Neither training nor
its finalization requires a local client heartbeat. Infrastructure failure can
still interrupt a run; verified partial checkpoints remain on the Volume, and
failures never silently trigger another paid GPU attempt.
