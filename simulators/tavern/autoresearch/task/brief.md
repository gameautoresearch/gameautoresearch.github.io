# Tavern (Hearthstone Battlegrounds) research brief

The game is a small form of AutoTavern 37.0.2 implemented-content Battlegrounds, an emulator whose parity
with the commercial game is unverified. Four players share a lobby that draws three minion types and
offers two heroes; seven board slots, ten hand slots, regular tavern tiers 1-4 (tier 5 and 6 minions are
not in the pool), spells, trinkets and Dark Gifts are on, Timewarp and experimental content are off.
With at most four players alive, hero damage is never capped. The observation keeps the eight-player
layout: the slots of the four absent opponents are zero. A policy sees only its own player's visible state (3,227 numeric
features, 124 card indices) and chooses one of 245 legal-masked actions, at most 30 per recruit turn;
the 100-turn guard is a fault, not a loss. Choose-One option text, some target distinctions and
action-use counters are absent from the encoding and nothing replaces them.

The starter is a randomly initialized one-hidden-layer tanh MLP (64 units, 4-dimensional card
embeddings) trained by plain episodic PPO with gamma 1 from one learning seat against three scripted
opponents. It is deliberately weak: a flat view of the board, no memory or opponent modelling, no
curriculum or self-play, and training opponents that differ from the evaluation field.

Interface and limits: `training/policy.py` defines `Policy(**config["network"])`, mapping (features,
cards) to (245 logits, value), and the `act` function the simulator's `rollout.py` imports. The host
entry builds the policy and an Adam optimizer from `learning_rate`, then calls `train(model, optimizer,
config, device, timer, deadline, start_update, on_update)` in `training/train.py`; call `on_update(n,
{"learning": {"loss", "samples", "parameter_delta_l2"}, "episodes", "counts"}, states)` after each
update, which is when weights are exported, and return by `deadline`. A run registers only with at
least one update, finite losses and changed weights. The export is a finite state-dict NPZ of at most
500,000 values and 32 MB, reloaded strictly from that file. Play is greedy: one feedforward pass on
the current observation in an isolated single-thread CPU process, the best legal action, within 15
seconds per exchange including start-up; no search, no external weights. A config keeps the frozen
`rules` block, turn-1 starts, `opponent` "scripted" or "selfplay", 1-256 lobbies per batch, at most 4
native and 4 torch threads, `network.hidden` 8-256 and `network.embedding` 1-32; `seed` seeds training.

Public practice plays a network in 48 public lobbies: four seed groups, each of the four seats, against
three fixed mixes of three opponents drawn from visible tempo/economy/synergy scripts and development
PPO policies trained in this form. It reports mean placement points, 100*(4-placement)/3, with
top-half (first or second) rate and per-seat and per-mix breakdowns. Every lobby must end naturally for
all four players: a fault, illegal
action, missed deadline or unfinished lobby makes the score null rather than a loss. The schedule is
fixed and play is deterministic, so repeating practice for the same network adds nothing.

`ar stop NAME` ends the trainer at once and keeps the network it last exported (the weights are exported
after every update): the run finishes as `complete`, marked `stopped_early`, and can be practised and
submitted like any other; before its first update it ends as `stopped` with nothing kept. A trainer
still running when its training command reaches its limit is cut off the same way and marked
`cut_short`. A trainer that fails by itself is a failed run. All worker time is charged either way.
