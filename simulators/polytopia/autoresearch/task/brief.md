# Polytopia research brief

The game is 1v1 Polytopia: two players, both Bardur, on an 11×11 board with fog of war, human-equivalent income
and the frozen 2.17.3 rules. Every map is of one type, drylands: almost all land with light mountains and forest
(in a host check of 3,032 layouts the capitals were always 6 tiles apart and joined by land). Training draws from
a fixed pool of 100,000 layouts; public practice and private grading use separate maps. Diplomacy is the engine
default (enabled), resignation is disabled, and a game that reaches 30 rounds goes to the higher in-game score
(equal scores draw). A game ends earlier only when a player loses its last city. A policy sees only observation v3
(player-visible features with leak-free visible history) and the legal masks of 17 hierarchical action families.

The starter is plain PPO self-play from random weights: a 2-block, 32-channel network (`training/policy.py`),
the loss (`training/learner.py`), the GPU-resident trainer and its configuration fields (`training/train.py`)
and the launch script (`training/collector.py`). Known weaknesses: a small network, one policy playing both
seats with no opponent pool, a separate critic that reads the same view as the policy, per-decision GAE and a
constant learning rate. Training and evaluation use the same outcome rule: the starter's only reward is the
game's result, +1 or −1 when a player is eliminated and, for a game cut at the 30-round cap, +1 for the higher
in-game score, −1 for the lower and 0 for equal scores. The starter's training command begins at fresh random
weights; exact training resume is not something it does for you.

Limits: every network descends from random initialization inside this session; no external weights, search,
privileged state or engine changes.
The policy has at most 500,000 parameters and the training-only critic at most 500,000: `ar check` counts both, and
a network over the policy limit cannot be packaged. A configuration keeps `size` 11, `players` 2, `maps` drylands, `max_turns` 30 and map seeds
200000-300000, a learning rate in (0, 0.01], an entropy coefficient in [0, 0.1] and `trainer.games`, `sets` and
`ring`, with `trainer.ring` at least 5 × `segment` (the starter's 160 for 32: a seat's rows must stay in the device
ring until their segment is trained on, and a smaller ring fails a few minutes into the run); a smoke run keeps the
2×32 architecture. A network you package needs the `final.npz` and `final.json` the trainer writes, in a run folder
or in one of its `exports/update-N` folders. The host's fresh final training of a recipe must also leave the
trainer's other outputs (initial policy, final critic, `protocol.json`, `training.jsonl` with finite losses,
`training-complete.json`) and a final policy that differs from the initial one. The graded artifact is the final
checkpoint with your `training/policy.py` and architecture: it is rebuilt from those alone and plays greedily in an
unprivileged process on a 4-core, 16 GiB CPU worker, given only features and masks, with 120 seconds per decision.

Write a usable policy early and keep it current. The starter writes the policy and critic as a whole export
(`exports/update-N/`, renamed into place only once complete) after its first update and then every
`export_seconds` (60), and its final checkpoint when its `--seconds` are up. A training command you end early
(`ar exec NAMEw -- 'pkill -f gpu_resident_train'`) leaves its exports: `training/launch.py` then packages the newest
whole one, and you can package any of them yourself. A worker that ends (`ar stop`, or its time running out) takes
its processes with it and packages nothing, but what they wrote under `/scratch` stays and a later worker can
package it. The fresh final training of a recipe is one training command the host runs alone: cut off at its time
limit it counts with its last complete export, with none it is a failure, and a trainer that fails by itself is a
failed run. If you rewrite the training loop, keep the exports working.

Hardware: a GPU worker has one L40S (48 GB). The engine's rollout buffers (torch) grow with `trainer.games`, and JAX
gets 45% of the card for the update, which grows with `batch`. The starter (2048 games, batch 65536) fits; how
far larger settings go has not been measured on this board, and one that does not fit fails with an out-of-memory
error. A training command spends about 130 seconds compiling and allocating before its first update, and
`training/launch.py` keeps 180 seconds of the worker's time for its start, packaging and end, so a 360-second
worker has only about a minute of updates. `ar poll NAMEw 0001` shows the trainer's output while it runs.

## Workers

All training happens on workers you lease: a worker of a shape you choose (0 or 1 L40S, 1-32 CPU cores, 2-64 GiB),
held for as long as you say, on which you run any commands you like with `ar lease`, `ar wait NAME --started`,
`ar exec NAME -- COMMAND`, `ar poll`, `ar push`, `ar sync`, `ar package` and `ar stop`.

- `python training/launch.py NAME --seconds S` does the plain case in one command: a worker `NAMEw` with one L40S,
  4 CPU cores and 16 GiB (the shape a recipe's final training gets; `--gpu`, `--cpu` and `--memory` choose
  another), the starter's training command on the files as they are now, the result sealed as run `NAME`, the
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
- How many workers may be active at once is set by this session's budget mode (the session section below and
  `ar budget` say which). One practice job runs at a time either way.
- A worker is charged from the moment it starts until it ends, busy or idle, at the charge of its shape, so
  stop it when you are done. It serves commands until about 35 seconds before its time is up.
- Your allocation and its unit are in the session section and `ar budget`. Every worker is charged its seconds
  times the charge of its shape, which follows the shape's list price by GPU, CPU cores and memory: the training
  worker (one L40S, 4 cores, 16 GiB, about $2.90 an hour) counts in full, one with 16 cores and 64 GiB about
  twice as much and one without a GPU far less; the starter runs its games and its updates on the GPU.
  `ar budget` lists each shape's charge, what you have left and how many seconds of each shape that buys. A
  queued or running worker holds its whole grant until it ends and is then charged only for the time it lived.
  Public practice (`ar eval`) is not charged.
- A worker grades nothing by itself. Seal a checkpoint with `ar package LEASE NAME --path DIR`, where `DIR` holds
  `final.npz` and `final.json` as the trainer writes them (a run folder, or one of its `exports/update-N`
  folders), then `ar eval` and `ar submit` that name. It is sealed with the `training/policy.py` and
  `configs/architecture.json` you send with the command, which must be the ones that network was built with
  (`training/launch.py` keeps a copy from the launch for this). In a recipe session what is graded is your recipe
  trained alone from scratch by the starter's training command, on a fresh worker with no `/scratch`: use workers
  to find what trains best that way, and train the exact recipe you submit at least once with `launch.py`.

## Evaluation

Public practice plays a completed run on 64 fixed public maps, both seats, against four retained opponents, in
rising strength: `starter-200m` (the starter after 200 million samples), `starter-1h` (the unmodified starter
recipe trained fresh for one hour), and `research-mid` and `research-top` (two networks trained in earlier
sessions of this game mode, stronger than `starter-1h`). It reports game points from 0 to 100 (win 1, draw 0.5,
loss 0) by seat and opponent, natural and capped games separately, with a map-cluster interval. Practice plays
several workers at once and takes roughly ten minutes. A smoke session plays one public map against a random
network. An illegal action or simulator fault invalidates the evaluation. The opponents are uncalibrated: practice
shows progress, not expert strength.
