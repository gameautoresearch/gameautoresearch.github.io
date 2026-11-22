# Pokémon research brief

**Mode.** Singles `gen9randombattle@@@Max Team Size = 3` at Pokémon Showdown pin `567c1ced`: unmodified Gen 9
Random Battle with three generated Pokémon each instead of six, one active per side; no team building. A decision sees the fixed 700-value player-visible observation and answers
with one of 14 actions: 0-3 moves, 4-7 the same moves with Tera, 8-13 switches in request order (with three
Pokémon, 11-13 and the observation's own rows 3-5 and foe rows 9-11 are always empty). The last 14
observation values are the legal mask. The only reward is +1/-1/0 at a natural end.

**The reward is frozen.** The environment returns nothing but that terminal reward. The starter passes
`faint_reward=0` to `env.step`; it must stay 0, and any other value raises `Frozen terminal rewards only` and fails
the run. A denser learning signal has to be computed by your own learner from the observations it already receives.

**Training** runs on a native C++ engine admitted as exact against the pinned Showdown rules. The trainer never
touches the engine: a trusted environment service hands it `[battles, 2 seats, 700]` observations, takes one action
per seat (-1 for a seat that is not to move) and returns terminal rewards and per-step terminal, cut-off and fault
flags. Evaluation plays the exported policy on Showdown itself.

**Starter.** A 64-wide feed-forward actor-critic over hashed name embeddings (`policy.py`), a clipped-PPO collector
that plays both seats of 256 battles with the current policy (`learner.py`), and a loop that exports the policy
graph once the model is built and replaces the checkpoint and `result.json` after every update (`train.py`). Known
weaknesses: it only ever plays its current self, learns from the sparse terminal reward alone and has no memory
beyond the single observation.

**Keep a complete policy on disk.** Write `graph.json` early and keep `checkpoint.pt` and `result.json` current, as
the starter does: a run that you stop (`ar stop NAME`) or that is still training at its time limit is packaged from
the last complete set of `checkpoint.pt`, `graph.json`, `result.json` and `progress.jsonl` and finishes as complete,
marked `stopped_early` or `cut_short`; a run with no complete set fails. The graph is the architecture and is checked
against the checkpoint's weights, so export it again only if the architecture changes, and replace files whole
(write a temporary file, then rename it).

**Checking and watching a run.** `ar check` also runs your `train.py` here for one small CPU update against random
observations of the contract's shape and then the real graph export, so import, shape, reward-flag, checkpoint and
export-limit mistakes show up before a worker is spent; it cannot tell you whether anything learns. While a run is
going, `ar log NAME` shows what `train.py` has printed so far, a minute or so behind: the starter prints each
progress row as a JSON line, and whatever else you print with `flush=True` appears too. Use `ar wait NAME --timeout
120` and `ar log NAME` to watch a long run and `ar stop NAME` to end one that is going nowhere.

**Checking and watching a run.** `ar check` also runs your `train.py` here for one small CPU update against random
observations of the contract's shape and then the real graph export, so import, shape, reward-flag, checkpoint and
export-limit mistakes show up before a worker is spent; it cannot tell you whether anything learns. While a run is
going, `ar log NAME` shows what `train.py` has printed so far, a minute or so behind: the starter prints each
progress row as a JSON line, and whatever else you print with `flush=True` appears too. Use `ar wait NAME --timeout
120` and `ar log NAME` to watch a long run and `ar stop NAME` to end one that is going nowhere.

**Limits a policy must respect.**
- The scored policy is the tensor graph `train.py` exports from `policy.create(config["network"])`, not your Python:
  `forward(x)` returns `(logits, value)` from the seven-key observation dictionary. Only the pure tensor operations
  listed in the graph contract export; no randomness, mutation, custom operators, control flow or search. At most
  1,000,000 parameter and buffer elements, 2,048 graph nodes and a 2 MiB graph.
- The trainer runs unprivileged with no network and may import only torch, numpy and the public runtime modules the
  starter uses. Its time argument is a maximum for the learning loop: stop by the deadline it is given, then leave
  the checkpoint (with the bindings the starter writes), `result.json`, `progress.jsonl` and `graph.json`.
- The config must keep positive integer `seed`, `training_seed_start`, `environments` (at most 4,096),
  `rollout_steps` (at most 256), `minibatch` and `ppo_epochs`, with training seeds below 800,000,000; other keys
  are yours. Every run starts from a new random initialization. Source is code only: no string literal over 4,096
  characters, no literal table over 256 elements, no `base64`, `pickle`, `marshal`, `requests`, `urllib` or `socket`
  imports, and no state file paths in the config.
- In evaluation the graph runs alone in an unprivileged single-thread CPU process; the host applies the mask and
  plays the highest legal logit. A reply slower than 15 seconds or an invalid one is a policy failure.

**Public practice** plays a completed run on 16 fixed public seeds, both seats, against a max-power heuristic, two
retained learned policies and a pinned public-information search opponent (128 games), and reports 100 x mean points
(win 1, natural tie 0.5, loss 0) per opponent and seat. It runs on a worker of the same profile as a run and is
charged against the research budget like one (see `ar budget` and the session section for the budget and how many
jobs may run at once): about three minutes of worker time a panel, held as a 360-second lease while it runs
(`ar eval NAME --seconds S` sets the lease, 240 at least). Private grading uses fresh seeds against the same panel. A panel with any unfinished or faulted game has no score; faults and
illegal actions never count as wins or draws.
