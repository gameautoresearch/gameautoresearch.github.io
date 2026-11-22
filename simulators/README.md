# Simulators

Research sessions for every game run through the single harness in
[`harness/`](../harness/README.md): `python -m harness` from the repository root
with the owning game's research interpreter. Terminus-2 is always the researcher
and the workspace is the same for every game. The harness is checked offline;
live qualification and calibration are still required before scored work. No
spending is authorized by this repository's contents.

Every active game lives here. Each directory owns its environment, native simulator
work, learning code, recipes and tests. Each game's research materials (contract,
brief, rulebook, starter, constructor and trainer/evaluator entrypoints) live under
its `autoresearch/` directory; its harness backend is `harness/games/<game>/`. Pinned third-party engines and rules oracles are
vendored separately in [`vendor/engines/`](../vendor/README.md).

| Directory | Game | Primary implementation | Environment | Readout |
|---|---|---|---|---|
| [clash](clash/README.md) | Clash Royale | C++ CPU/CUDA; Java rules oracle | `.venv-clash` | [Native rewrite](../reports/clash/native-rewrite-2026-11-08.md) |
| [doudizhu](doudizhu/README.md) | Dou Dizhu | Native CPU/CUDA; DouZero oracle | `.venv-doudizhu` | [Research trial](../reports/doudizhu/autoresearch-research30-2026-11-08.md) |
| [generals](generals/README.md) | Generals.io | JAX environment and training | `.venv-generals` | [Full-sized qualification](../reports/generals/fullsize-performance-2026-11-08.md) |
| [pokemon](pokemon/README.md) | Pokémon | Showdown CPU; incomplete C++ rewrite | `.venv-pokemon` | [CPU rewrite](../reports/pokemon/native-rewrite-2026-11-08.md) |
| [polytopia](polytopia/README.md) | Polytopia | Rust/CUDA/WASM engines and Python stack | `.venv-polytopia` | [Overnight research](polytopia/docs/autoresearch-s120-readout-2026-11-08.md) |
| [tavern](tavern/README.md) | Hearthstone Battlegrounds | AutoTavern Rust CPU; paused device rewrite | `.venv-tavern` | [Population throughput](../reports/tavern/population-throughput-2026-11-09.md) |
| [hex](hex/README.md) | Hex 13x13, no swap rule | Batched PyTorch (GPU/CPU); MoHex 2.0 opponent | `.venv-hex` | [13x13 form](../reports/hex/hex13-2026-11-14.md), [integration](../reports/hex/hex9-integration-2026-11-12.md) |

The machine-readable [catalog](catalog.json) is the shared directory index.
Each game supplies the same researcher inputs: `autoresearch/task/brief.md`, a
detailed `autoresearch/task/game_rules/` rulebook, the three common starter roles
`autoresearch/starter/{policy,learner,train}.py` and starter/smoke configurations in
`autoresearch/recipes/`. The harness assembles them into the uniform workspace; see
the [workspace guide](../docs/research-task-layout.md). Each game also owns its exact
benchmark contract under `autoresearch/`. See the
[shared contract guide](../docs/benchmark-contracts.md) for scope, scoring and
allocation conventions, and run `python3 tools/check_benchmark_contracts.py`
from the repository root for the offline declaration check.
The harness runs under each game's research interpreter (`.venv-<game>-ar` where
host dependencies need separation); the catalog records the exact choice. Simulation and learning checks retain the game's main environment. Existing `integrations.*` Python imports remain
supported through compatibility links; the files themselves live here.

Run ordinary commands from the repository root. For Polytopia, change to
`simulators/polytopia` before using its relative commands. Its full snapshot
retains the internal engine/package structure and licenses; the inactive PolyAR
snapshot is retained privately.

Reports remain in `reports/<game>/` and local evidence in `evidence/`. Keep
historical manifests and receipts unchanged, and write new runs to new directories.
