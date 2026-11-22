# Five-hour Tavern evaluator, November 9

This is the active campaign integration. Read the rolling
[evaluator report](../../../../reports/tavern/strong-bots-five-hour-2026-11-09-evaluator.md)
and `evidence/strong-bots-20261109-0516Z/tavern/coordination/{evaluator,rl}.json`.
The RL owner controls `reference_training/`; this evaluator owns admission,
opposition, scoring, benchmark integration and its own cleanup. Coordination is
through each owner's atomic status file, with no new chats or messages.

Only the explicit new `campaign.sqlite3` and `OpenCreditBudget` apply here.
Historical capped campaigns, seeds, checkpoints and failures stay unchanged.
Modal aggregate credits/GPU seconds are uncapped; paid model APIs are zero.
Workers are finite, GPU concurrency remains two per game/eight campaign-wide,
and each L4 has four fixed CPUs/16GiB. Training/selection closes09:36:36UTC;
all workers stop10:11:36; reporting ends10:16:36, November9. These helpers do not
renew an allowance or authorize a future campaign.

The supported native fingerprint remains `217577b83fc4525e`, admission-v2,
with the exact 3,227/124/245 visible interface and frozen eight-player rules.
The GPU simulator rewrite remains paused. `opponents.py`, `lobby.py`,
`policies.py`, `screen.py` and the earlier qualification source stay frozen;
new wrappers snapshot their complete source per operation.

`admission` verifies exported source/identity/finite NPZ against a complete
checkpoint and its exact archived source. It checks optimizer/RNG/population,
native states, logical cohorts and training history; strong-parent imports
recursively return to the original r3 checkpoint. No source identity is
relabelled, and inference always uses the export's original policy class.
`collect_checkpoint` only reads an already-published immutable snapshot and
verifies its digest; it cannot create or exec a worker.

`launch` starts one new bounded CPU evaluator plus a remote receipt observer,
after source/interface checks and new-ledger reservations. It reuses the
verified public-only runtime foundation, overlays exact source and private
grading assets, blocks network, and protects private paths before starting a
UID65534 candidate with an empty credential environment. No private opposition
weights or banks enter a training image. All eight seats must finish naturally;
faults and incomplete schedules make the primary score null.

```sh
.venv-tavern-ar/bin/python -m simulators.tavern.evaluation_readiness.five_hour.launch \
  --attempt NEW_NAME --candidate POLICY_ID --registry FROZEN_REGISTRY \
  --panel PUBLIC_PANEL --cap 1800
.venv-tavern-ar/bin/python -m simulators.tavern.evaluation_readiness.five_hour.status --attempt NAME
```

The launch is a plan until `--execute`; actual execution requires the existing
authorization. Add `--collect-summary` only after authoritative completion, once
per summary directory. Bulk native traces remain privately retained on the
Volume; `collect_examples` reads the predeclared first-group/seat0 cases from
all three profiles. A copy/host failure does not justify another paid worker.
Preserve a partial collection and recover it read-only.

The new guest probes denied private-path access and independently replays three
predeclared terminal cases. `BatchEnv` saves the next episode after the last
terminal; `terminal_replay` also reconstructs the actual terminal Game state
under the pinned pending-context/seat-order admission rule. Both state identities
and all eight terminals are checked. `viewing` uses fixed PUBLIC seeds and
verifies native actions, true terminal state and combat projections. Only its
projected film belongs in a viewer; full states/RNG/checkpoints remain private.

The default benchmark is [mixed-v2](../../autoresearch/evaluation_v2/README.md).
`freeze_pool` creates a new immutable, publicly selected opposition version;
promoted learned references must appear in its actual seven-opponent profiles.
`final_bank` refuses to draw the common 12-group/288-lobby bank until clock and
RL selection closure, then freezes exact roster/source/weights/profiles before
fresh private grading. No private feedback returns to training or selection.
Comparisons preserve identical seed/profile/seat cells, reject missing profiles,
retain faults, and bootstrap whole environment groups with input/helper hashes.

`cleanup` inventories all new-ledger evaluator apps, including build-only
failures. `--stop` refuses early teardown of unresolved own workers; the hard
deadline still requires stopping owned apps. It checks both active tasks and
Sandboxes, and never stops RL-owned apps. Unknown build/observer/storage costs
remain held. Use new receipt paths for each inspection/stop attempt; only a
verified complete receipt becomes `cleanup-final.json`.

No commit, push, publication, commercial ladder play, Polytopia or Dou Dizhu
work is authorized by this evaluator workflow. Strong/expert/superhuman and
reference-normalization flags require independent evidence and remain separate
from workflow correctness or gains against this simulator's retained pool.
