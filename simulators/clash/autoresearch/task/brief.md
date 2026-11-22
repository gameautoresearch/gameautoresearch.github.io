# Clash: fixed eight-card mirror deck, level 11

Mode `clash-fixed8-v1`: two players with the same deck (Knight, Archer, Musketeer, Giant, MiniPekka,
Fireball, Arrows, Zap), everything at level 11 with Princess towers; no champions or evolutions. Both
seats decide simultaneously twice a second (ten 50 ms ticks per decision; the simulation waits
for inference). A match is the benchmark's short form: 60 s of regulation plus 30 s of overtime
(double elixir from 40 s, triple from 75 s) with the natural tiebreak, about 130–200 decisions; it is
never cut off by an artificial horizon. Every training config names `"match_form": "short-v1"` with
`rules.decision_ticks` 10, and the trainer refuses any other form. Each seat sees 7,024 int32 values (224 header values and 200 entity rows of 34)
and a 2,307-entry legal mask in which action 0, wait, is always legal. The enemy hand, queue and RNG
are never visible. A win scores 1, a draw ½, a loss 0.

The starter is deliberately weak. `training/policy.py` is a thin wrapper around a fixed module that
is not under `simulator/`, `simulators.clash.autoresearch.fast_policy`: the entity/card network of
`simulator/simulators/clash/model.py` at width 64 and embedding 12, fed through
`native/gpu_features.DeviceBatchEncoder` so that `model(observation, mask)` takes the raw arrays.
`training/train.py` subclasses `fast_adapter.Trainer`, which is `native/train.py:Trainer` with two
hooks: the network comes from `policy.create(config)` and `learn(data)` calls
`learner.update(trainer, data)`. `training/learner.py:update` currently hands back to the native
graph PPO; the eager GAE/PPO functions above it are a reference and are not called. Training is
current-policy self-play from random weights on 1,024 worlds with only the terminal +1/0/−1 reward:
no teacher, shaping, opponent pool or curriculum. On the earlier full-length form (180+120 s, four
decisions a second) a 32-update run of this starter scored 5.2/100 on that form's public panel; it has
not been measured on the short form.

Limits every policy must respect:

- `policy.py` keeps `create(config)` returning a torch module whose `model(observation, mask)` returns
  `(logits[batch, 2307], value)`, and `act` must be exactly
  `return model(observation, mask)[0].argmax(-1)`. At most 500,000 parameters over the whole
  `state_dict`, finite float32, exported as `weights.npz` of at most 16 MiB.
- Only the unmodified starter `policy.py` may import `fast_policy`. Any edited `policy.py` may import
  only `torch`, `numpy` and `math`, so a changed network has to be written out in that file. It may
  not use `global`/`nonlocal`/`async`, nor call anything named `open`, `exec`, `eval`, `compile`,
  `__import__`, `load`, `save`, `seed`, `manual_seed`, `random`, `rand`, `randn`, `randint` or
  `Generator`. The native trainer also calls `model.sample_batch`, `model.distribution` and
  `model.values` on encoded batches; keep them or replace collection and learning in `train.py`.
- `train.py` defines `Trainer(config, data_path, resume)` with `collect()`, `learn(data)` returning
  `(metrics, optimizer_steps)` where metrics is a tensor of at least five finite values (the fifth
  is the gradient norm), plus `update`, `samples`, `counters`, `bucket_usage`, `model`,
  `checkpoint(...)` and `close()` as in the native trainer. The starter's `checkpoint` writes model,
  optimizer and RNG state only, not the full engine world, so a long run finishes inside its deadline.
  A run counts only with at least one update, finite losses and changed weights. Every run starts from random weights; nothing is
  carried between runs.
- On the native fast route (`"training_route": "native-fast-v1"`) the config keeps `max_ticks` 0,
  `envs` 1–2,048, `rollout_steps` 1–256, `warmstart_steps` 0, `reward_shaping` 0 and `opponent_mode`
  `"selfplay"`. `hardware_allocation` stays at 4 CPU threads and 16 GiB. The training seed is the
  config's `seed`. The three training files total at most 256,000 bytes, a config 65,536.
- Inference is one stateless greedy forward per decision on a single CPU thread: no memory between
  decisions, no search, no sampling. The evaluator sends up to 256 seat rows at a time, needs a legal
  action for each within 15 seconds, and probes that repeated, reordered and single-row inputs give
  the same answers. An illegal, late or batch-dependent answer invalidates the whole evaluation
  instead of scoring losses.

Public practice plays a completed run on the release's public seeds (eight in the standard panel),
both seats, against every opponent of its panel: scripted `standard` and `pressure` and the
visible-information `intermediate` and `teacher` bots (no learned networks: those of the full-length
form were trained on 300-second matches and were dropped). It reports the raw score (mean points × 100, opponents and seats weighted equally) with a
paired seed-cluster interval and the score per opponent and per seat; any fault, illegal action,
timeout or unfinished game gives no score. Practice runs in a separate CPU worker and is limited by
count rather than time, and one training launch has a minimum and a maximum length; the live budget
shows both. Public practice is the only evaluation you see; after the session the host plays the
final network once against the same four opponents on protected seeds you never see.

`ar stop NAME` ends a training run at its next update and keeps the network trained so far: the run
finishes as `complete`, marked `stopped_early`, and can be practised and submitted like any other. If
no update had finished it ends as `stopped` with nothing kept. The time until it ends is charged.
