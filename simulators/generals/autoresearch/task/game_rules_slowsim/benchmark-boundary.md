# Benchmark boundary: results, learning and validity

[Wiki home](README.md)

The gameplay rules above do not depend on a particular public or private map seed, opponent checkpoint, model route or allocation. This chapter describes the discoverable public submission/scoring boundary without exposing a hidden bank.

## Result and score

Each completed valid game contributes capture-win 1, capture-loss 0 or cap-draw .5. Panel points are `100*mean(game_points)` on [0,100], with equal physical-seat and opponent weights and the declared map weights. Every map is normally played in both physical seats against every panel opponent. Confidence intervals bootstrap whole map clusters to preserve paired seats/opponents. Raw points do not prove human/expert strength. Reference-normalized improvement remains unqualified until an independent equal-full-budget multi-seed calibration passes.

Public training uses a finite 256-map bank and ordinary practice uses a separate public bank. Those counts describe data partitions, not permission to inspect private maps or reverse-engineer a hidden evaluation bank. The private grader, its schedules and opposing weights remain host-owned and are not included in the wiki. Development feedback cannot be represented as independently unseen final performance.

## Outcome and boundary semantics

The only reward the environment gives is the outcome: a natural ending gives the winner +1 and the loser −1, and a cap gives both players zero. Anything a training method adds to that is its own and changes neither natural game outcomes nor the grading score. The source-bound simulator, observation channels, legal mask and task settings are not editable research knobs.

A step exposes the true final state without autoreset. Separate reset logic chooses the next map; reading a restarted board as the end of the previous game would misread the final board.

## Policy boundary

The discoverable v1 submission contract admits stateless greedy visible-only inference, no memory interface, no inference search or external calls, safe finite float32 NPZ parameters, and a compute-per-move cap of 1,000,000,000 FLOPs per board (no parameter cap; weights stored up to 32,000,000 elements). Exact sources, architecture, effective config, weights and rules identity are sealed together. A new versioned release may alter an inference resource limit only by explicit declaration; it cannot silently relabel old results. Do not read old research/reference recipe details into the baseline task.

## Failures remain distinct

- A mask-false, malformed or nonfinite policy output violates the policy/submission boundary.
- A pre-frame legal move that becomes inapplicable due to earlier opponent resolution is ordinary gameplay.
- Simulator faults, mismatched source/data/checksum identities, exceptions and panel failures fail validation rather than becoming capture losses.
- An incomplete schedule has no ordinary partial headline score.
- A budget stop is not an in-game resignation. Unfinished work is recorded separately.

The broker's actual current budget and hardware contract are authoritative. This wiki starts no worker and renews no allowance. Setup, compile, simulation, transfer, learning, evaluation attached to a worker, IO, idle, failures and teardown retain their accounting.
