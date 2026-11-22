# Generals: mid-size fog-of-war 1v1

Mode `generals-research-1v1-medium-v2`: two players under fog of war, map sides drawn independently
from 12–15 and padded to 16×16, 4–6 neutral castles of 10–15 army, generals at least 10 apart, general trade on,
1,000 joint ticks.
Capturing the enemy general scores 1, being captured 0, and reaching the cap is a draw (½) that
land and army totals never break.

The starter is deliberately plain: a width-16 CNN of three 3×3 convolutions (a 7×7 receptive
field on a 16×16 board), a per-cell action head and a mean-pooled value head, trained by PPO/GAE
on current-policy self-play with outcome-only reward; a cap bootstraps the final visible value.
It runs end to end but has no qualified playing strength. Known weaknesses: no memory under fog,
sparse reward, a single self-play opponent, and a fixed bank of 256 training maps
(`/data/train-maps.npz` on a worker) that invites overfitting. It imports two fixed helper
modules that are not under `simulator/`: `integrations.generals_autoresearch.rollout`
(`load_pool`, `initial_carry`, `make_collect`, `logprob`: batched self-play collection) and
`integrations.generals_autoresearch.bundle.save_weights`. Keep using them or write your own
collection from the simulator source.

Limits every policy must respect:

- `training/policy.py` keeps `create(config)` returning a Flax model with `init(key, x, mask)` and
  `apply({'params': p}, x, mask) -> (logits[batch, 2049], value[batch])` on the visible
  `[batch, 16, 16, 14]` input and the `[batch, 2049]` legal mask (action 0 is pass). The sizes are
  `BOARD_SHAPE` and `ACTION_COUNT` in `integrations.generals.environment`.
- `training/train.py` is run unprivileged (256 processes/threads, 512 open files) as
  `train.py --config FILE --out DIR --seed N --seconds S` and must leave `weights.npz` (written
  with `save_weights`) and `model.json` (the config `create` needs) in `DIR`; `progress.jsonl`
  and `result.json` there are the training's metrics, which you read on the worker. In a recipe's
  final training `S` is the worker's time less setup and an export reserve (5 % of it, at least 40
  and at most 60 seconds) and the trainer is ended when that reserve begins. A practice is at least
  120 seconds. A training that short is mostly overhead: worker setup, imports and compiling up to
  the first update and the trainer's own 10-second closing margin leave little time for updates.
  The starter writes a `progress.jsonl` row about every 2 seconds (`progress_seconds` in the
  config), each covering the updates since the last (`updates_in_row`; losses are their means,
  `update`, `learner_samples` and the capture counts are totals so far).
- A network is packaged from the `weights.npz` and `model.json` the trainer last left, whether it
  reached its own end, you ended it early, or (in a recipe's final training) its time limit cut it
  off: the starter writes `model.json` before its first update and replaces `weights.npz` after the
  first update and then every 30 seconds (`save_seconds` in the config). With neither there is
  nothing to package. Keep saving early and whole (write a temporary file, then rename it) if you
  rewrite the loop.
- No parameter cap. What is capped is compute per move: one forward pass of your policy for one board
  may take at most 1,000,000,000 FLOPs. The count comes from the traced computation (the jaxpr) of
  `apply` at batch 1 on the padded 16×16 board: 2 per multiply-add in convolutions (dilated or not)
  and matrix products, 1 per output element of other arithmetic, 1 per input element of a reduction.
  It needs no device or compiler, so it is the same on every worker and grader and does not depend on
  timing (for scale: the starter is about 3.5 million, a width-48 residual CNN of 12 3×3 convolutions
  about 130 million, a width-128 one about 920 million). A scan counts its body once per step; a while
  loop has no countable trip count and is refused.
  `python -m integrations.generals_autoresearch.flops --dir DIR` prints the count for a folder with
  `policy.py`, `model.json` and `weights.npz`; the starter's `train.py` checks it before training, and
  practice and grading check it before the first game. Weights are finite float32 (bfloat16
  computation is fine) and stored up to 32,000,000 parameters. The three training files stay under
  512 KiB together, a config under 64 KiB.
- Every network descends from random initialization inside this session: no external weights or data
  and no engine changes. The starter's training command starts from random weights. You may continue
  from checkpoints your own earlier work in this session left in `/scratch`.
- **The policy must be learned.** This is a reinforcement-learning task: what decides a move is weights your
  training produced from playing games. `training/policy.py` is a neural network (learned layers and generic
  operations on the visible board) and may not contain a hand-written player or any part of one: no fixed
  scores, bonuses, penalties or rules for particular moves or situations (taking a castle, attacking a general,
  expanding, guarding a stack), added to or multiplied into the action scores, and no learned switch or scale in
  front of such a rule. Do not write a player by hand to imitate either. What you know about the game goes into
  training (the learning signal, the opponents a network plays, what it sees as input), where it has to change
  the weights. The host enforces this on every submitted policy while it plays: it replays the policy's
  decisions with its weights re-initialized, and a policy that still chooses the same move on more than 25 % of
  boards is rejected as not learned and its evaluation is invalid (trained networks keep under 6 %; a
  hand-written player keeps 80–100 %). The code of every submission is also read after the session.
- Inference is one stateless greedy forward per decision: no memory between decisions, no search,
  no hidden-state input. The evaluator runs your policy unprivileged (64 processes, 64 open files),
  sends up to 32 boards at a time and needs a legal action for each within 15 seconds, the first,
  compiling call included. Loading the policy and the host's own checks before the first batch are not on that
  clock (they have 180 seconds). An illegal or late answer invalidates the policy instead of scoring losses, and
  the evaluation says which rule was broken (`fault`: deadline, startup, illegal_action, bad_scores, bad_response,
  not_learned, compute_cap or crashed).

## Workers

All training happens on workers you lease: a worker of a shape you choose (0 or 1 L40S, 1-32 CPU cores,
2-64 GiB), held for as long as you say, on which you run any commands you like with `ar lease`,
`ar wait NAME --started`, `ar exec NAME -- COMMAND`, `ar poll`, `ar push`, `ar sync`, `ar package` and
`ar stop`.

- `python training/launch.py NAME --seconds S [--seed N]` does the plain case in one command: a worker
  `NAMEw` with one L40S, 4 CPU cores and 16 GiB (the shape a recipe's final training gets; `--gpu`,
  `--cpu` and `--memory` choose another), the starter's training command on the files as they are now,
  the result sealed as run `NAME`, the worker stopped. The training command it runs is
  `python training/train.py --config configs/train.json --out /scratch/runs/NAME --seed N --seconds T`,
  where `T` is the worker's time less 90 seconds for its start, packaging and end. Run it yourself with
  `ar exec` when you want something else.
- Your current `training/` and `configs/` are copied to `/work`, where commands run as an unprivileged
  user with no network. The engine and the two helper modules import as on any training worker,
  `/work/training` is on the Python path and the training maps are at `/data/train-maps.npz`. JAX uses
  the GPU on a worker that has one. Quote a command
  that uses shell syntax: `ar exec gpu1 -- 'cd /scratch && ls'`. `ar poll NAMEw 0001` shows a command's
  output while it runs.
- `/work` is the worker's own disk and is gone when it ends. `/scratch` is one volume shared by all your
  workers in this session and persists between them. A worker sees `/scratch` as it was when it started;
  what it writes becomes visible to others when it ends or after `ar sync`. Two workers should not write
  the same file. You may continue from checkpoints your own earlier work in this session left there.
- A worker that ends (`ar stop`, or its time running out) takes its processes with it and packages
  nothing, but what they wrote under `/scratch` stays and a later worker can package it. To end a
  training early and keep it, end the command (`ar exec NAMEw -- 'pkill -f train.py'`): `launch.py` then
  packages what the trainer last wrote.
- Workers and practice count alike as jobs: how many may be active at once is in the session section
  below and in `ar budget` (`concurrency`, `concurrent_gpus`).
- A worker is charged from the moment it starts until it ends, busy or idle, at the cost of its shape,
  so stop it when you are done.
- One lease lasts at most 1,500 seconds (`max_run_seconds` in `ar budget`). A training that should run
  longer continues on a new lease from the checkpoint it left in `/scratch`.
- Every worker (a lease, `ar eval`) is charged to one research allocation by its seconds and its shape.
  The session section says whether the allocation is in seconds of the training worker (one L40S, 4
  cores, 16 GiB; another shape is charged by its price relative to that worker's) or in US dollars;
  `ar budget` shows the unit, what each shape costs, what you have left and how many seconds of each
  shape that buys. Workers running side by side spend it proportionally faster.
- A worker grades nothing by itself. Seal a network with `ar package LEASE NAME --path DIR`, where `DIR`
  holds `weights.npz` (written with `save_weights`) and `model.json` as `train.py` writes them, then `ar eval` and `ar submit` that name. It is sealed with the training files you
  send with the command, which must be the ones that network was built with (`training/launch.py` keeps
  a copy from the launch for this), and the limits above apply to it. In a recipe session what is
  graded is `train.py` trained alone from scratch by that command, on a fresh worker with no `/scratch`:
  use workers to find what trains best that way, and train the exact recipe you submit at least once
  with `launch.py`.

## Evaluation

Public practice plays a completed or packaged run on 16 public maps, both seats, against the release's
practice opponents (`practice-opponents.json`): uniform random, an expansion heuristic and, when the
release has them, trained networks. It reports mean game points (×100 = panel points) with a
map-clustered interval, capture wins, capture losses, cap draws, and per-opponent and per-seat means;
where the opponents are rated it also reports `panel_elo`, your network's Elo on the opponents' ladder
(the scripted Hunter, which private grading uses, is 0). Private grading uses fresh maps and adds the
scripted Hunter. Every campaign also ends with a head-to-head tournament between the networks its
sessions submitted.
