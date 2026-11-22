# Hex: 13x13, no swap rule

Mode `hex13-no-swap-v1`: 13x13 Hex with no swap rule, cells numbered `13 * row + col`. `game_rules/` has the
rules, the policy and engine interfaces and the evaluation in full. The starter is a small MLP trained
by REINFORCE self-play that plays without search.

Limits every policy must respect:

- `training/policy.py` keeps `create` and `act`, and `training/train.py` keeps its command line and its
  outputs `weights.pt` and `model.json`, as `game_rules/interface.md` defines them. A recipe's final
  training is cut off at its time limit and whatever is there is packaged, and you package a network
  from whatever those two files hold, so write both early and atomically.
- At most 10 million tensor elements, parameters and buffers together, all finite; `weights.pt` at most
  128 MiB. The three training files stay under 512 KiB together, `model.json` and a config under 64 KiB.
- Every network descends from random initialization inside this session: no external weights, no
  opening books or tables computed outside your allocation, and no engine changes. The starter's
  training command starts from random weights. You may continue from checkpoints your own earlier work
  in this session left in `/scratch`.
- Inference is CPU-only with one Torch thread, and it is capped per move. Loading the model has 60
  seconds. Then the evaluator sends up to 64 positions at a time, and a call with n positions has
  0.1 + 0.01 x n seconds: 10 milliseconds per position plus a tenth of a second per call, so 0.74 seconds
  for 64 positions and 0.11 for one. That is about one forward pass of a residual network of 1 to 3
  million parameters, or a few passes of a smaller one, not a deep search. Do any warm-up in `create`,
  which is not on this clock. A late call is replaced by a random legal move in every game of that batch, in practice, in
  grading and in the tournament alike. Search at evaluation is allowed (the engine is importable there)
  but must fit inside that clock. An illegal or malformed answer, or a model that does not load in
  time, invalidates the policy instead of scoring losses. `ar check` times your `act` against this
  clock on the box's CPU.

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
- Your current `training/` and `configs/` are copied to `/work`, where commands run. Quote a command
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
- A worker is charged from the moment it starts until it ends, busy or idle, at the rate of its shape,
  and it counts as one of your active jobs while it is held, so stop it when you are done.
- Every worker (a lease, `ar eval`) is charged its seconds times the rate of its shape: an `ar eval`
  worker costs about half as much per second as the training worker (one L40S, 4 cores, 16 GiB), a
  bigger lease more. `ar budget` is authoritative for this session's budget: its unit, the rate of each
  shape, what you have left, how many seconds of each shape that buys and how many jobs may run at once
  (see also the session section of these instructions).
- A worker grades nothing by itself. Seal a network with `ar package LEASE NAME --path DIR`, where `DIR`
  holds `weights.pt` and `model.json` as `train.py` writes them, then `ar eval` and `ar submit` that name. It is sealed with the training files you
  send with the command, which must be the ones that network was built with (`training/launch.py` keeps
  a copy from the launch for this), and the limits above apply to it. In a recipe session what is
  graded is `train.py` trained alone from scratch by that command, on a fresh worker with no `/scratch`:
  use workers to find what trains best that way, and train the exact recipe you submit at least once
  with `launch.py`.

## Evaluation

The score is 100 x your win rate over nine equally weighted opponents, from random play up to MoHex
with its solver, every opening played from both seats. Public practice uses fixed public openings;
private grading uses fresh ones with the same opponents. After the sessions end, your final network also
plays the other researchers' final networks in a round robin, reported beside the score.
