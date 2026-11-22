# Dou Dizhu: fixed roles, no bidding

Mode `douzero-fixed-role-v1`: the full 54-card deck, a landlord with 20 cards against two cooperating
peasants with 17 each; the three bottom cards are public and already in the landlord's hand. No bidding,
spring or bid multiplier. A game ends only when a hand is emptied; the 162-decision guard is a simulator
fault, never a result. A policy sees its own hand, public information and the complete canonical list of
legal moves for the current decision, and answers with an index into that list.

The starter is deliberately plain: three role networks (a 3-layer, 128-unit MLP over a 64-unit LSTM history
encoder) trained from random weights by terminal win/loss Deep Monte Carlo regression with RMSprop and a
fixed exploration rate, on pure self-play in the resident GPU simulator. Known weaknesses: no replay, no past
or diverse opponents, no optimizer or exploration schedule, a small network, and it only ever plays beside
and against copies of itself. `policy.py` and `learner.py` are thin wrappers over the fixed worker module
`integrations.doudizhu.model` (`Policy`, `update`); read it at `simulator/simulators/doudizhu/model.py`, and
keep using it through the calls the starter makes or write your own network. On a worker the package shown
under `simulator/simulators/doudizhu/` is imported as `integrations.doudizhu`.

Limits every policy must respect:

- `training/policy.py` keeps `create(model_config) -> torch.nn.Module` and
  `act(policy, role, static, history, legal_actions, counts) -> list[int]`, one legal local index per
  decision; `game_rules/rules/action-interface.md` defines the arrays.
- `training/train.py` is run unprivileged (64 processes/threads, 64 open files) as
  `train.py --config FILE --out DIR --seed N --seconds S` and must leave `weights.pt` (a tensor-only
  `state_dict`) and `model.json` (the argument `create` needs) in `DIR`; write them atomically and early,
  because a run is cut off at its time limit and what is there is packaged. `progress.jsonl` there is
  returned as the run's metrics.
- At most 500,000 tensor elements, parameters and buffers together, all finite; `weights.pt` at most
  128 MiB. The three training files stay under 512 KiB together, `model.json` and a config under 64 KiB.
- Every run starts from random initialization: no external or earlier weights, and no engine changes.
- Inference is CPU-only with one Torch thread on public arrays only. The evaluator sends up to 32 decisions
  at a time and needs an answer within 10 seconds, the first, model-loading call included; any search must
  fit inside that. An illegal, malformed or late answer invalidates the policy instead of scoring losses.

Public practice replays a fixed set of public deals with your policy controlling each of the three roles in
turn and the same practice opponent in both other seats, your teammate included when you play a peasant:
uniform random, a cooperating shedding heuristic and, when the release has one, an earlier learned policy.
It reports 100 × win probability with every role and opponent weighted equally, a deal-clustered interval,
per-role and per-opponent win rates and secondary economic scores. This measures single-seat strength beside
fixed partners, not joint peasant play. Private grading uses fresh deals and a separately held panel.
