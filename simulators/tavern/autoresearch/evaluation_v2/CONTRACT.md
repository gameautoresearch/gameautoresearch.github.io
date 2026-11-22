# Tavern four-player small benchmark v1

The evaluation and artifact/source/rules identity is `tavern-four-small-v1`. It replaces the eight-player
`tavern-eight-mixed-v2` evaluation (artifact ABI `tavern-eight-implemented-v1`), which is preserved at git tag
`full-size-20261113`; networks, releases and receipts of that form do not bind to this one.

The pinned native AutoTavern 37.0.2 fingerprint is `217577b83fc4525e`, admission-v2; no engine code changed.
The task has **four players**, regular tavern tiers **1–4** (`max_regular_tier` 4), **three** lobby minion
types, two hero choices, seven board slots, ten hand slots, 30 recruit actions and a 100-turn safety guard.
Timewarp and experimental mode remain disabled. With at most four players alive the engine never caps hero
damage. The visible interface is unchanged: exactly 3,227 numeric features, 124 card indices and a 245-action
legal mask; the four absent opponents' slots are zero. No privileged replacement observation is supplied for
encoder gaps.

An entrant receives a random starter and may edit its policy, learner, trainer and recipe. Its budget is the
session's research allowance; each training run gets one L40S worker with four physical CPUs and 16 GiB attached
RAM. The session states its budget mode (`ar budget`): allocated worker seconds with one job at a time (the
default), or US dollars of worker time at frozen list prices with several jobs at once, each holding its whole
grant until it ends.
Startup, setup, compilation, simulation, idle GPU, transfer, training, checkpointing and teardown count. Failed
and unknown allocations remain charged or reserved. Migration does not reset a ledger; finalized workflows
cannot launch additional training.

Inference is a reviewed finite feedforward policy within the harness parameter cap and a 32MB finite NPZ. A
separate unprivileged Linux process receives only its own bundle and visible arrays. The 15-second deadline
covers complete input/response exchange, including startup and blocked/partial output. Network is blocked,
credentials are absent, and private references/schedules/traces are root-protected. References are reviewed
feedforward models or static visible scripts using the same inputs; no hidden state, test-time search or LLM
rescue.

Three frozen profiles each provide three explicit actors clockwise around the candidate. Each candidate occupies
every physical seat. Every required lobby continues through all four native natural terminals with placements
forming 1–4, even after candidate elimination. Score is the equal-weight mean of `100*(4-placement)/3`;
top-half rate (first or second) and per-profile/seat diagnostics are secondary. Any required fault or
incomplete cell makes the primary aggregate null. Illegal actions, visible engine rejections, inference
deadlines, simulator faults, infrastructure failures and administrative caps remain distinct; there is no
fallback, imputation or valid-only substitute. Pairwise placement order within a mixed lobby is not a duel
win rate.

The reference pool (`selected_registry-four-small-v1.json`, built by `build_four_pool.py`) holds the three
public visible scripts (tempo, economy, synergy) and three development PPO policies trained from the public
starter on CPU in this form (1, 5 and 15 minutes, distinct seeds). Profiles: `scripts` = tempo, economy,
synergy; `mixed` = ppo-5m, economy, ppo-1m; `learned` = ppo-15m, ppo-5m, synergy. These are development
opponents with no strength qualification; the profile names describe composition, not difficulty.

Practice and engineering qualification each use 48 PUBLIC lobbies (four seed groups × four seats × three
profiles). The scientific final uses 288 fresh private lobbies (24 groups × four seats × three profiles),
drawn only after entrant selection closes. Historical/public native streams and complete declared
training-cohort ceilings are excluded. Roster, exact source/weights, profiles and bank commitment are frozen
before grading; private results never return to training or selection. Uncertainty resamples whole seed
groups while preserving profiles and seats. Reference normalization and human, expert or superhuman claims
remain unqualified without independent calibration.

A new pool requires a new immutable receipt version, exact checkpoint/source admission and public
qualification. Private artifacts, seeds and game assets are not published by this workflow.

`BatchEnv` resets after the final natural terminal. Its saved post-step hash is the next episode state;
explicit stop codes and placements determine the score. The qualification runtime additionally replays
predeclared recorded cases against the pinned native Game admission steps, recovering the true terminal Game
state and checking all four terminals, applied/deferred actions and post-reset state.
