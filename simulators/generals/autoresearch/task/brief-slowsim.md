# Generals: mid-size fog-of-war 1v1

Mode `generals-research-1v1-medium-v2`: two players under fog of war, map sides drawn independently
from 12–15 and padded to 16×16, 4–6 neutral castles of 10–15 army, generals at least 10 apart, general trade on,
1,000 joint ticks.
Capturing the enemy general scores 1, being captured 0, and reaching the cap is a draw (½) that
land and army totals never break.

**Castles decide games in this mode.** Every map has 4–6 neutral castles and each costs only 10–15 army to
capture. A castle you hold produces exactly like a second general, one army every second tick, so it has paid
back its price about 25 ticks after you take it and keeps producing for the rest of a game that usually runs
200–400 ticks between trained networks. Host measurements on this ruleset, with trained networks:

- a player given its nearest castle at tick 80, paying the full price in armies, won 82–85 % of games against an
  otherwise equal opponent (67–72 % when it bought at tick 120);
- when the host forced a trained network, from outside its policy, to capture a neutral castle whenever a
  stack next to one could afford it, it beat the same network left alone in 61–64 % of games;
- networks trained from the capture outcome alone did **not** learn this in the host's own trial: after about 780
  million samples no player held a castle at tick 100, and forbidding such a network to attack castles cost it
  nothing. The gain from a castle arrives 50–100 ticks after the capture, and an attack with too small a stack
  only loses armies, so a learner that sees nothing but the final result leaves castles alone.

So a network that leaves castles neutral is giving away a large, measured advantage, and getting your networks
to capture castles early and then use what they produce is likely the single largest improvement open to you.
It has to be learned (see the limits below: a rule written into the policy is rejected), so it is your training
that must make castles worth taking. Do not assume it happens by itself. Design for it, and check it: which cells are castles and who owns them is
in the visible board (the `castles`, `owned_cells`, `neutral_cells` and `opponent_cells` channels and the army
counts, [observations](game_rules/observations.md)), so your own code can count how many castles each player of
a game holds at tick 100 and whether the side holding more wins. Treat a network that holds none as unfinished.

**Starter: boilerplate only.** There is no working recipe to start from. `training/train.py` is self-play
plumbing: it loads the training maps (`/data/train-maps.npz` on a worker, a fixed bank of 256), plays `num_envs`
games with the simulator in `simulator/` and one policy sampling the moves of both seats, restarts finished
games on another training map, and writes `model.json`, `weights.npz`, `progress.jsonl` and `result.json`. It has no learning:
after every `block_steps` joint ticks it calls `learner.update(params, key)`, which returns the weights
unchanged, and `training/learner.py` is otherwise empty. `training/policy.py` is a linear placeholder network.
Trained as it is, a run keeps its random initialization. What to keep from each step, the learning signal, the
loss, the optimizer and the architecture are yours to write; the docstrings of `training/train.py` (what each
step gives you and where an update belongs) and `training/policy.py` (the interface the evaluator uses) describe
the interfaces. `simulator/generals_sim` is the game simulator: the rules of one game, what each player sees
and the action codec (`simulator/README.md`). It is one of your editable files, like `training/`. Besides it, your code on a worker can import the weight writer
(`integrations.generals_autoresearch.bundle.save_weights`) and the compute check
(`integrations.generals_autoresearch.flops`). The project's own training code, reward helpers, reference networks
and scripted players are not part of this task and are withheld.

The capture outcome is the only result the environment gives: +1 or −1 when a general is captured, 0 for both
players at the cap. Anything denser has to be computed by your own code from what a step already gives you.

Limits every policy must respect:

- `training/policy.py` keeps `create(config)` returning a Flax model with `init(key, x, mask)` and
  `apply({'params': p}, x, mask)` returning a tuple whose first element is `logits[batch, 2049]` (the evaluator
  ignores anything after it), on the visible
  `[batch, 16, 16, 14]` input and the `[batch, 2049]` legal mask (action 0 is pass). The sizes are
  `BOARD_SHAPE` and `ACTION_COUNT` in `integrations.generals.environment`.
- `training/train.py` is run unprivileged (256 processes/threads, 512 open files) as
  `train.py --config FILE --out DIR --seed N --seconds S` and must leave `weights.npz` (written
  with `save_weights`) and `model.json` (the config `create` needs) in `DIR`; `progress.jsonl`
  and `result.json` there are the training's metrics, which you read on the worker. In a recipe's
  final training `S` is the worker's time less setup and an export reserve (5 % of it, at least 40
  and at most 60 seconds) and the trainer is ended when that reserve begins. A practice is at least
  120 seconds. A training that short is mostly overhead: worker setup, imports and compiling up to
  the first block and the trainer's own 10-second closing margin leave little time for training.
  The boilerplate writes a `progress.jsonl` row about every 2 seconds (`progress_seconds` in the
  config): blocks and joint ticks so far and how many games ended in a capture or at the cap.
- A network is packaged from the `weights.npz` and `model.json` the trainer last left, whether it
  reached its own end, you ended it early, or (in a recipe's final training) its time limit cut it
  off: the boilerplate writes `model.json` before its first block and replaces `weights.npz` after the
  first block and then every 30 seconds (`save_seconds` in the config). With neither there is
  nothing to package. Keep saving early and whole (write a temporary file, then rename it) if you
  rewrite the loop.
- No parameter cap. What is capped is compute per move: one forward pass of your policy for one board
  may take at most 1,000,000,000 FLOPs. The count comes from the traced computation (the jaxpr) of
  `apply` at batch 1 on the padded 16×16 board: 2 per multiply-add in convolutions (dilated or not)
  and matrix products, 1 per output element of other arithmetic, 1 per input element of a reduction.
  It needs no device or compiler, so it is the same on every worker and grader and does not depend on
  timing (for scale: the linear placeholder is about 63 thousand, and one 3×3 convolution from 64 to 64
  channels over the board about 19 million). A scan counts its body once per step; a while
  loop has no countable trip count and is refused.
  `python -m integrations.generals_autoresearch.flops --dir DIR` prints the count for a folder with
  `policy.py`, `model.json` and `weights.npz`; the boilerplate's `train.py` checks it before training, and
  practice and grading check it before the first game. Weights are finite float32 (bfloat16
  computation is fine) and stored up to 32,000,000 parameters. The three training files stay under
  512 KiB together, a config under 64 KiB.
- Every network descends from random initialization inside this session: no external weights or data
  The boilerplate starts every run from random weights. Practice, grading and the tournament play the rules
of `game_rules/` on the host's own simulator; your `simulator/` is the copy your training plays on, and what
you change in it changes only your own training games. You may continue
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
  `--cpu` and `--memory` choose another), the training command on the files as they are now,
  the result sealed as run `NAME`, the worker stopped. The training command it runs is
  `python training/train.py --config configs/train.json --out /scratch/runs/NAME --seed N --seconds T`,
  where `T` is the worker's time less 90 seconds for its start, packaging and end. Run it yourself with
  `ar exec` when you want something else.
- Your current `training/`, `simulator/` and `configs/` are copied to `/work`, where commands run as an
  unprivileged user with no network. `/work/training` and `/work/simulator` are on the Python path (the
  simulator imports as `generals_sim`) and the training maps are at `/data/train-maps.npz`. JAX uses
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
