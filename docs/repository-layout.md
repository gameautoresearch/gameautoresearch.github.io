# Repository layout

`simulators/<game>/` is the canonical home for each of the seven active games.
Polytopia has the same top-level status as the other games. Its established
internal Rust/CUDA/Python structure is retained so relative build commands work.

## Ownership

| Location | Owns |
|---|---|
| `simulators/<game>/` | Game integration, native simulator work, training, evaluation, recipes and tests |
| `harness/` | The one research harness: Terminus-2 sessions, the uniform workspace, the `recipe` and `network` variants, and each game's backend in `harness/games/<game>/` |
| `simulators/<game>/autoresearch/` | Each game's benchmark contract, brief, rulebook, minimal starter, release constructor and trainer/evaluator entrypoints |
| `vendor/engines/` | Pinned third-party source trees (MoHex/benzene for Hex among them), including retained AutoTavern modifications |
| `vendor/upstream-git/` | Upstream commit pins and tracked-file manifests |
| `reports/<game>/` | Historical readouts and measured results; Polytopia also retains its snapshot's `docs/` |
| `docs/research/` | Cross-game benchmark research notes |
| `docs/migration/` | Original migration records and source checksums |
| `tools/` | Shared repository checks, rulebook audits, archiving and offline metadata restoration |
| `external-website/` | Research article and demo replays |
| `evidence/`, `.local/`, `.venv-*` | Ignored local artifacts, private state and per-game dependencies |

[`simulators/catalog.json`](../simulators/catalog.json) records each game's
directory, tests, README, readout, upstream dependencies and virtual environment.
`python3 tools/check_layout.py` checks the catalog and compatibility links without
importing ML frameworks or provisioning compute.

## Compatibility

The following relative symlinks retain old package imports, launcher paths,
editable installs and paths embedded in frozen manifests:

| Historical path | Canonical location |
|---|---|
| `integrations/{clash,doudizhu,generals,hex,pokemon,tavern}` | `simulators/<game>` |
| `integrations/generals_autoresearch` | `simulators/generals/autoresearch` |
| `candidates` | `vendor/engines` |
| `environments/polytopia` | `simulators/polytopia` |
| `environments/polytopia-legacy-polyar` | `archive/polytopia-legacy-polyar` |

These are aliases, with no duplicate source trees. Keep existing `integrations.*`
Python namespaces and prepared container paths stable. A prepared research box
can still use its original standalone `integrations/` layout. New documentation
and file navigation use the canonical paths.

Source/data/seed locks, archived recipes, receipts and existing ledgers are not
rewritten. Old checkpoints must pass their original exact source checks; use the
matching archived source when required. Neither migration nor validation renews
paid experiment budgets. Local checks write timestamped directories beneath
`evidence/local-validation/` and do not launch paid workers.

## Adding a game

Create `simulators/<game>/README.md` with its rules, setup, commands and limitations,
then add its tests and entry to the catalog. Keep game-specific APIs where they
belong; this directory layout does not impose a common observation or action API.
Put separately vendored dependencies in `vendor/engines/`, preserve licenses, and
record immutable source/data pins. Put new readouts in `reports/<game>/` and local
run outputs in fresh ignored evidence directories.
