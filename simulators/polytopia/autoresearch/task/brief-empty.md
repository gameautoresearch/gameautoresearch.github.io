# Polytopia research brief

The game is 1v1 Polytopia: two players, both Bardur, on an 11×11 board with fog of war, human-equivalent income
and the frozen 2.17.3 rules. Every map is of one type, drylands: almost all land with light mountains and forest
(in a host check of 3,032 layouts the capitals were always 6 tiles apart and joined by land). Training draws from
a fixed pool of 100,000 layouts; public practice and private grading use separate maps. Diplomacy is the engine
default (enabled), resignation is disabled, and a game that reaches 30 rounds goes to the higher in-game score
(equal scores draw). A game ends earlier only when a player loses its last city. A policy sees only observation v3
(player-visible features with leak-free visible history) and the legal masks of 17 hierarchical action families.

**Starter: boilerplate only.** There is no working recipe to start from. `training/train.py` is the GPU rollout
plumbing: the CUDA engine steps thousands of self-play games on the GPU, one policy samples every decision of both
seats, finished games restart on fresh maps, and the trainer writes checkpoints, exports, `training.jsonl` and
`protocol.json` and honours `ar stop`. It has no learning: every `batch` decisions it calls `_update()`, which does
nothing, and `training/learner.py` is empty. `training/policy.py` is a linear placeholder network. Trained as it is,
a run keeps its random initialization. What to store from the rollout, the learning signal, the loss, the optimizer
and the architecture are yours to write; the docstrings of `training/train.py` (what each step gives you and where
an update belongs) and `training/policy.py` (the network interface the evaluator uses) describe the interfaces.
`simulator/` shows the worker's Python package (engines, observation encoder, action masks, checkpoint writer) and
the host's worker entry points; the project's own training code and its reference networks are not part of this
task and are withheld.

Training and evaluation use the same outcome rule, and it is the only result the environment gives: +1 or −1 when a
player is eliminated and, for a game cut at the 30-round cap, +1 for the higher in-game score, −1 for the lower and
0 for equal scores. Anything denser has to be computed by your own code from what the rollout already gives you.
Every run begins at fresh random weights.

Limits: every network descends from random initialization inside this session; no external weights, search,
privileged state or engine changes. The policy has at most 500,000 parameters and the training-only critic (the
second network the trainer writes as `final-critic`) at most 500,000: `ar check` counts both, and a network over the
policy limit cannot be packaged. A configuration keeps `size` 11, `players` 2, `maps` drylands, `max_turns` 30 and
map seeds 200000-300000, a `learning_rate` in (0, 0.01] and an `entropy_coef` in [0, 0.1] (the host checks both
fields even where your trainer does not use them) and `trainer.games`, `sets` and `ring` (read by
`training/launch.py`). A network you package needs the `final.npz` and `final.json` the trainer writes, in a run
folder or in one of its `exports/update-N` folders.

**The host's final training of your recipe.** It runs `/opt/v1/train.py` (shown under `simulator/`) on a fresh
worker with no `/scratch`: that script imports `GpuResidentTrainer` from `training/train.py`, wraps its `run` and
`_update`, and reads `root`, `params`, `run_config`, `version`, `_host_state()` and `_write()`; keep those names.
The run must leave the trainer's outputs (initial policy, final policy and critic, `protocol.json`,
`training.jsonl`, in which any `policy_loss` or `value_loss` must be finite, `training-complete.json`) and a final policy that differs from
its initial weights: the boilerplate as it is fails this check. The graded artifact is the final checkpoint with
your `training/policy.py` and architecture: it is rebuilt from those alone and plays `sample(...,
deterministic=True)` in an unprivileged process on a CPU worker, given only features and masks, with 120 seconds
per decision.

Write a usable policy early and keep it current. The boilerplate writes the policy and critic as a whole export
(`exports/update-N/`, renamed into place only once complete) after its first `_update()` and then every
`export_seconds` (60), and its final checkpoint when its `--seconds` are up. A training command you end early
(`ar exec NAMEw -- 'pkill -f gpu_resident_train'`) leaves its exports: `training/launch.py` then packages the newest
whole one, and you can package any of them yourself. A worker that ends (`ar stop`, or its time running out) takes
its processes with it and packages nothing, but what they wrote under `/scratch` stays and a later worker can
package it. The fresh final training of a recipe is one training command the host runs alone: cut off at its time
limit it counts with its last complete export, with none it is a failure, and a trainer that fails by itself is a
failed run. If you rewrite the training loop, keep the exports working.

Hardware: a GPU worker has one L40S (48 GB). The engine's buffers (torch) grow with `trainer.games`, and the
training command gives JAX 45% of the card (`--jax-mem-fraction`). A setting that does not fit fails with an
out-of-memory error. The engine steps about 100,000 decisions a second on this board with a small network. A
training command spends a minute or two compiling and allocating before its first step, and
`training/launch.py` keeps 180 seconds of the worker's time for its start, packaging and end, so a 360-second
worker has only about a minute of updates. `ar poll NAMEw 0001` shows the trainer's output while it runs.

## Workers

All training happens on workers you lease: a worker of a shape you choose (0 or 1 L40S, 1-32 CPU cores, 2-64 GiB),
held for as long as you say, on which you run any commands you like with `ar lease`, `ar wait NAME --started`,
`ar exec NAME -- COMMAND`, `ar poll`, `ar push`, `ar sync`, `ar package` and `ar stop`.

- `python training/launch.py NAME --seconds S` does the plain case in one command: a worker `NAMEw` with one L40S,
  4 CPU cores and 16 GiB (the shape a recipe's final training gets; `--gpu`, `--cpu` and `--memory` choose
  another), the training command on the files as they are now, the result sealed as run `NAME`, the
  worker stopped. The training command it runs is
  `python experiments/gpu_resident_train.py --config configs/train.json --arch configs/architecture.json --run-root /scratch/runs --run-id NAME --seconds T --games G --sets S --ring R`,
  with `G`, `S` and `R` from the config's `trainer`. Run it yourself with `ar exec` when you want something else.
- Commands run in `/work` as an unprivileged user with no network. `/work/src` and `/work/experiments` are a
  writable copy of the worker's Python package and launch script in which the four trainer paths are links to
  your pushed files: `/work/training/policy.py` is `polytopia_bot/rl/network.py`, `learner.py` is `rl/ppo.py`,
  `train.py` is `rl/gpu_resident.py` and `collector.py` is `experiments/gpu_resident_train.py`. Your configs are
  in `/work/configs`. `python` has JAX, torch, both engines and the training map table, and JAX uses the GPU on a
  worker that has one. Quote a command that uses shell syntax: `ar exec gpu1 -- 'cd /scratch && ls'`.
- `/work` is the worker's own disk and is gone when it ends. `/scratch` is one volume shared by all your workers
  in this session and persists between them. A worker sees `/scratch` as it was when it started; what it writes
  becomes visible to others when it ends or after `ar sync`. Two workers should not write the same file. You may
  continue from checkpoints your own earlier work in this session left in `/scratch`.
- `ar budget` shows `max_lease_seconds`: when it is set, no lease may be longer. A recipe's fresh final training
  is the session section's length.
- How many workers may be active at once is set by this session's budget mode (the session section below and
  `ar budget` say which). One practice job runs at a time either way.
- A worker is charged from the moment it starts until it ends, busy or idle, at the charge of its shape, so
  stop it when you are done. It serves commands until about 35 seconds before its time is up.
- Your allocation and its unit are in the session section and `ar budget`. Every worker is charged its seconds
  times the charge of its shape, which follows the shape's list price by GPU, CPU cores and memory: the training
  worker (one L40S, 4 cores, 16 GiB, about $2.90 an hour) counts in full, one with 16 cores and 64 GiB about
  twice as much and one without a GPU far less; the boilerplate runs its games on the GPU.
  `ar budget` lists each shape's charge, what you have left and how many seconds of each shape that buys. A
  queued or running worker holds its whole grant until it ends and is then charged only for the time it lived.
  Public practice (`ar eval`) is not charged.
- A worker grades nothing by itself. Seal a checkpoint with `ar package LEASE NAME --path DIR`, where `DIR` holds
  `final.npz` and `final.json` as the trainer writes them (a run folder, or one of its `exports/update-N`
  folders), then `ar eval` and `ar submit` that name. It is sealed with the `training/policy.py` and
  `configs/architecture.json` you send with the command, which must be the ones that network was built with
  (`training/launch.py` keeps a copy from the launch for this). In a recipe session what is graded is your recipe
  trained alone from scratch by `/opt/v1/train.py` with your trainer, on a fresh worker with no `/scratch`: use workers
  to find what trains best that way, and train the exact recipe you submit at least once with `launch.py`.

## Evaluation

Public practice plays a completed run on 64 fixed public maps, both seats, against four retained opponents, in
rising strength: `random` (an untrained network), and `starter-20m`, `starter-65m` and `starter-200m` (the
project's reference recipe, not part of this task, after 20, 65 and 200 million samples). It reports game points from 0
to 100 (win 1, draw 0.5, loss 0) by seat and opponent, natural and capped games separately, with a map-cluster
interval. Practice plays several workers at once and takes roughly ten minutes. A smoke session plays one public
map against a random network. An illegal action or simulator fault invalidates the evaluation. The opponents are
uncalibrated: practice shows progress, not expert strength.
