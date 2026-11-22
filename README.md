# Game Autoresearch Bench

Autonomous reinforcement-learning research across seven games. All active game
implementations live in **[`simulators/`](simulators/README.md)**, including Polytopia.
Read the [research article](https://gameautoresearch.github.io/)
for the results, interactive figures, and match replays.

New GPU-backed work defaults to **NVIDIA L40S**, unless explicitly overridden.
See [compute defaults](docs/compute-defaults.md) for exceptions and the matched
H100/L40S smoke-test results. Existing frozen hardware contracts stay unchanged.

The accepted research scopes, compute accounting and scoring conventions are in
[the contract guide](docs/benchmark-contracts.md). Each game owns an
exact machine-readable contract, game brief, rulebook and deliberately minimal RL
starter under its `autoresearch/` directory.

Every research session runs through one harness, **[`harness/`](harness/README.md)**:
Harbor's Terminus-2 is the researcher for every game, in one uniform workspace,
with two task variants (`recipe` and `network`) on the same setup.

| Game | Simulator and setup | Research materials | Latest readout |
|---|---|---|---|
| Clash Royale | [clash](simulators/clash/README.md) | [autoresearch](simulators/clash/autoresearch/README.md) | [Contract and smoke](reports/clash/benchmark-contract-smoke-2026-11-08.md) |
| Dou Dizhu | [doudizhu](simulators/doudizhu/README.md) | [autoresearch](simulators/doudizhu/autoresearch/README.md) | [Contract and smoke](reports/doudizhu/benchmark-contract-smoke-2026-11-08.md) |
| Generals.io | [generals](simulators/generals/README.md) | [autoresearch](simulators/generals/autoresearch/README.md) | [Contract and smoke](reports/generals/benchmark-contract-smoke-2026-11-08.md) |
| Pokémon | [pokemon](simulators/pokemon/README.md) | [autoresearch](simulators/pokemon/autoresearch/README.md) | [Contract and smoke](reports/pokemon/benchmark-contract-smoke-2026-11-08.md) |
| Polytopia | [polytopia](simulators/polytopia/README.md) | [autoresearch](simulators/polytopia/autoresearch/README.md) | [Contract and smoke](reports/polytopia/benchmark-contract-smoke-2026-11-08.md) |
| Hearthstone Battlegrounds | [tavern](simulators/tavern/README.md) | [autoresearch](simulators/tavern/autoresearch/README.md) | [Contract and smoke](reports/tavern/benchmark-contract-smoke-2026-11-08.md) |
| Hex 13x13 | [hex](simulators/hex/README.md) | [autoresearch](simulators/hex/autoresearch/README.md) | [Integration](reports/hex/hex9-integration-2026-11-12.md) |

## Layout

```text
harness/              # The one research harness: Terminus-2 sessions for every game
simulators/           # Seven active games: native engines, environments, training and tests
  clash/
  doudizhu/           # Includes autoresearch/
  generals/           # Includes autoresearch/
  hex/                # Includes autoresearch/
  pokemon/
  polytopia/          # Includes engine-*, src/, training/ and autoresearch/
  tavern/
vendor/
  engines/            # Pinned upstream engines/oracles, with local improvements retained
  upstream-git/       # Upstream commit pins and provenance manifests
reports/              # Per-game readouts and historical research results
docs/                 # Repository layout, research notes and migration records
tools/                # Shared local validation and metadata restoration
external-website/    # Research article and interactive, public demo replays
evidence/             # Ignored local receipts, checkpoints, ledgers and private evaluation
```

Each game has a README and tests, and keeps its own virtual environment. Engines
and research interfaces retain their game-specific APIs; [the simulator catalog](simulators/catalog.json)
records the common paths. The [layout guide](docs/repository-layout.md) explains
ownership and compatibility paths.

## Local setup and checks

After cloning, fetch the vendored engines’ pinned Git metadata before running
setup or pin checks. The script also retrieves a few dependency assets from their
pinned upstream commits and preserves the checked-in source and local changes:

```sh
python3 tools/restore_upstream_metadata.py
python3 tools/check_layout.py
python3 tools/check_benchmark_contracts.py
python3 tools/check_local.py --game generals
python3 -m unittest harness.tests.test_core
```

Follow each game's README for dependencies and commands. Run the five newer games
from this repository root; Polytopia's relative commands run from `simulators/polytopia`.
`check_local.py` uses the individual `.venv-*` environments and runs checks serially;
it provisions no cloud compute and writes to a new evidence directory each time.

The old `integrations/`, `candidates/` and `environments/` paths are compatibility
links for existing commands, editable installs and historical records. New source
belongs in the canonical directories above. Frozen manifests, seed schedules,
receipts and budget ledgers keep their original contents. Source-bound checkpoints
still require their exact source; a relocation does not grant compatibility.

The release includes the benchmark harness, task rulebooks, starters, simulator
source, and the research article. Official held-out inputs, run archives, hosted
authentication state, and proprietary game files are kept separately. Historical
grades require their original private inputs. Use fresh evaluation inputs for
new runs. See [the article build guide](external-website/README.md) for Pages.

The harness and its game backends are checked offline; a live Terminus-2
session is still required before scored work, and short smokes do not establish
competitive strength or calibrated difficulty. See the
[preparation readout](reports/benchmark-preparation-2026-11-08.md) for executed
evidence and remaining qualification. Preserve upstream licenses and game-data
provenance.

## License

The [MIT license](LICENSE) covers original benchmark code where no other license
is specified. Vendored and derived components retain the licenses and notices
identified in their own directories. See [vendor/README.md](vendor/README.md)
and the [Showdown manifest](external-website/src/showdown/MANIFEST.json).
