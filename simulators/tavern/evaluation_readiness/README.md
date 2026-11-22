# Tavern mixed-lobby evaluator

The active five-hour campaign uses the [new-ledger wrappers](five_hour/README.md)
and [mixed-v2 benchmark](../autoresearch/evaluation_v2/README.md). The earlier
bounded workflow below remains historical; its unused allowance is not renewed.

This is the November 9 evaluator release for native AutoTavern 37.0.2,
`217577b83fc4525e` / admission-v2. It composes visible-only scripts and exact
source-compatible neural exports. Read the current
[readout](../../../reports/tavern/evaluation-readiness-2026-11-09.md) before use.
Strong-panel and superhuman flags remain false. The new self-play reference is
owned separately by `../reference_training/`; its admission does not establish
human or expert strength.

Run from the repository root. `.venv-tavern` has the native simulator and Torch;
`.venv-tavern-ar` is the pinned Modal host environment. All launch commands
require a current explicit authorization, the new shared campaign ledger and
available evaluator subcap. Historical unused budgets authorize no work.

## Contract

- Numeric features `[3227]`, card IDs `[124]`, legal mask `[245]`; exact frozen
  rules and binding checked before every inference artifact or worker launch.
- Tempo/economy/synergy scripts use only these arrays and the static current card
  catalog. No environment handle, hidden state, opponent hand or future RNG.
  The immutable development release and full-lobby driver are under the ignored
  owning evidence root, published to the RL owner through its coordination file.
- Choose-One option descriptions, some target location/entity distinctions and
  action-use counters are absent from the encoder. References do not receive
  an extra observation channel to recover them.
- Candidate: reviewed inference source, at most 2M parameters / 32MB NPZ,
  deterministic masked argmax, separate Linux UID/process, 15 seconds for the
  complete visible-input/response exchange. Parent deadline also covers blocked
  writes and incomplete response lines. Registry/panels/traces are root-private.
- References: reviewed finite feedforward code or pure scripts, one CPU Torch
  thread, identical visible inputs. No search or external model calls.
- Seven explicit clockwise opponent seats; all eight candidate seats, every
  declared seed group and all three frozen profiles must be present.
- Full eight-seat natural ending and permutation of placements 1–8 required.
  Any required unresolved lobby makes the primary score null. Early candidate
  death alone does not qualify a later-faulted lobby. No fallback action.
- Natural placement score is `100*(8-placement)/7`; top four is reported
  separately. Pairwise placement order is not a duel win rate. Normalize only
  after an independent reference qualification; current normalized scores null.
- Faults: illegal response, inference deadline, visible native rejection,
  simulator fault/guard, infrastructure fault and administrative cap remain
  distinct. Conditional valid statistics never replace the required aggregate.
- Public screen: 2 seed groups x8 seats x3 profiles =48 lobbies/policy.
  Fresh final: 6 seed groups x8 seats x3 profiles =144 lobbies/policy. Generate
  after exact selection closes; seed-group uncertainty preserves seats/profiles.

## Local verification and private records

```sh
.venv-tavern/bin/python -m pytest -q simulators/tavern/tests/test_evaluation_readiness.py
.venv-tavern/bin/python -m simulators.tavern.evaluation_readiness.replay \
  --traces /absolute/private/traces --out /absolute/new/replay-proof.json
```

The synthetic long no-damage fixture deliberately reaches the unchanged fault
guard; it is separate from natural workload completion. Mask coverage does not
claim every natural effect path is tested. Retained full-lobby records contain
native initial/final JSON, compressed actions, all terminals, counts and latency.
Replay requires the exact native fingerprint and validates preframe masks,
final state, all eight terminals, rejections and deferrals.

`freeze_registry` admits an exact training-source export and appends it to a new
immutable registry while keeping profiles unchanged. `fresh_bank` requires a
closed-selection record and matching registry hash; it excludes historical
schedule keys and entire declared training-cohort ceilings using the native
SplitMix seed derivation. Do not disclose final seeds to training or public files.

## Bounded provider workflow

`launch` prints a plan unless `--execute` is supplied. It snapshots source and
manifest-listed assets, reserves image/storage, CPU worker and observer liability
before creation in the explicitly named new ledger, then starts a finite detached
network-blocked worker plus remote receipt observer on the private v2 Volume.
No automatic retries, warm pool, researcher calls or GPU allocation. Launch reads
counterpart interface pins. The observer uses provider allocation start time and
an absolute campaign deadline; receipt settlement uses provider start/end only.

`status --attempt NAME` is read/reconcile only. Add `--collect-summary` once to
collect outcomes and receipts while leaving bulk native traces privately remote;
`--collect` downloads all records to a new folder. `cleanup --stop --out NEW_PATH`
refuses unresolved own CPU allocations, stops only recorded evaluator apps and
checks active tasks. Unpriced build/observer/storage liability stays held.
`readout` regenerates canonical MD/JSON from retained evidence and never copies
private seeds or native state into those reports. No commit/push/publication is
part of this workflow.
