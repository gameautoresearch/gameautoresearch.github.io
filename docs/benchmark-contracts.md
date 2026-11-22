# Research contracts

The accepted first release keeps each game's strategic scope fixed. The editable
research problem is to improve a deliberately simple, randomly initialized RL
starter under a finite compute allocation. Simulator rules, visibility, legal
actions, private schedules and budget enforcement belong to the trusted harness.

| Game | Frozen scope | Agent materials |
|---|---|---|
| Polytopia | Small form: two Bardur players, 11×11, fog, drylands maps, 30-round horizon and score-at-cap adjudication (the 14×14 ten-map-type form is at tag `full-size-20261113`) | [`autoresearch/`](../simulators/polytopia/autoresearch/README.md) |
| Generals.io | Full-size form: two players, sides 17–23 padded to 24×24, fog, 2,048 joint ticks; capture wins, caps draw; graded against scripted and learned opponents on an Elo ladder (the 8×8 small form was last at commit `48f226ee`) | [`autoresearch/`](../simulators/generals/autoresearch/README.md) |
| Pokémon | Small form: pinned Gen 9 Random Battle with three Pokémon per team (`Max Team Size = 3`), one active, switching and Tera; no team construction (the six-Pokémon form is at tag `full-size-20261113`) | [`autoresearch/`](../simulators/pokemon/autoresearch/README.md) |
| Clash Royale | Short form: fixed eight-card level-11 mirror, full arena, 60 s regulation + 30 s overtime at two decisions per simulated second, natural match endings (the 180+120 s, four-decision form is at tag `full-size-20261113`) | [`autoresearch/`](../simulators/clash/autoresearch/README.md) |
| Battlegrounds | Small form: four-player implemented-content lobbies, tavern tiers 1–4, three random minion types, two hero choices, 30 recruit actions (the eight-player form is at tag `full-size-20261113`) | [`autoresearch/`](../simulators/tavern/autoresearch/README.md) |
| Hex | 13x13, no swap rule; 2-ply openings from both seats against Elo-calibrated opponents from random play to a MoHex 2.0 anchor (the method of Jones 2021); inference capped per move (the saturated 9x9 form is at commit `48f226ee`) | [`autoresearch/`](../simulators/hex/autoresearch/README.md) |
| Dou Dizhu | Three players, full 54-card deck, fixed landlord/peasant roles, no bidding | [`autoresearch/`](../simulators/doudizhu/autoresearch/README.md) |

## Public entry points

Researchers always work in the [uniform workspace](research-task-layout.md) built
by the [harness](../harness/README.md). The common instructions are one template
(`harness/templates/instructions.md`); each game adds `task/brief.md` and
`task/game_rules/`, an offline wiki of rules, entities and exact mode boundaries.
The rulebook contains only readable Markdown; generators and audit metadata live
separately under `autoresearch/wiki_maintenance/`.

Every owning `autoresearch/` directory exposes `benchmark.json`, `CONTRACT.md`,
`task/brief.md`, `task/game_rules/`, a minimal editable `starter/`, and starter and
smoke configurations in `recipes/`. The JSON declaration records exact task
settings, hardware, budget, editable files and score definition. The game contract
defines tie handling, completion criteria, policy limits and schedule sizes.
Preparation is offline; paid execution is explicit.

Check declarations without importing game frameworks, contacting providers or
executing their declared commands:

```sh
python3 tools/check_benchmark_contracts.py
python3 tools/check_benchmark_contracts.py --game generals --json
python3 tools/check_benchmark_contracts.py --smoke-receipts --json
```

This checks public files and common declarations. It does not certify simulator
fidelity, isolation, learning strength, cloud operation or paid accounting. Those
claims require the owning game's tests and retained execution receipts.
The optional smoke-receipt check also requires explicit workflow completion,
non-null raw grades, resolved allocations, stopped workers and spending within
the initial campaign ceilings. It checks declarations; it does not independently
query provider billing or prove the underlying receipts.

## Compute and evidence

The main research allowance is **7,200 allocated single-GPU worker seconds**, with
the exact attached CPU/RAM fixed per game. Setup, compile, simulation, transfer,
learning, development evaluation, checkpointing, idle GPU time and teardown count.
Failures retain their charges; unknown allocations retain reservations. Historical
Polytopia process-time records remain historical and are not relabeled.

The initial preparation campaign separately authorizes at most **600 cumulative
allocated GPU-worker seconds per game**, including any GPU-based independent
grading, and **$2 charged-or-reserved OpenRouter usage per game**. CPU-only agent
shells and independent CPU grading are reported separately. These are campaign
limits, not renewable allowances for each retry. No full-budget reference training
is authorized by the preparation campaign.

A workflow smoke must exercise the actual agent prompt, training with a changed
policy, public evaluation, immutable submission, independent private grading,
finalization and worker cleanup. Completing a conversation or testing the trainer
alone is insufficient. Passing a reduced panel is not full difficulty calibration.

## Scores

The headline raw score is on a **0–100** scale. Two-player games use equally
weighted win/half-draw points across seats and opponents. Dou Dizhu uses balanced
win probability across three individually controlled roles and opponents;
peasant tests retain a panel-controlled teammate. Battlegrounds rotates one
candidate through all four seats and uses `100 × (4 − placement) / 3`, with
mean placement and top-half rate also reported.

Natural outcomes, rule-based draws, horizon adjudications, illegal actions,
policy failures and simulator faults remain distinct. Complete schedules and
game-specific validity checks are required for an ordinary score. Confidence
intervals preserve paired seats/roles/opponents by resampling whole maps, deals
or lobbies as appropriate. Elo and economic points are secondary diagnostics.

Raw points do not assert human or competitive strength. Starter/reference
normalization stays unqualified until matched full-budget multi-seed calibration
passes. No private seeds, policies, reference recipes, credentials, game assets or
private trajectories belong in agent inputs or public reports.
