# Autoresearch game candidates

Local inspection workspace created on 2026-11-08. It is separate from `polytopia-bot`.

| Checkout | Game / role |
|---|---|
| `candidates/pokemon-showdown` | Pokémon battle engine |
| `candidates/poke-env` | Pokémon Python RL interface and starter examples |
| `candidates/crforge` | Clash Royale simulator and Python RL interface |
| `candidates/AutoTavern` | Hearthstone **Battlegrounds** simulator (not constructed Hearthstone) |

All clones are shallow, with exact commits recorded in [evidence/provenance.json](evidence/provenance.json).
Upstream tracked source is unchanged. Build outputs, ignored local configuration, virtual
environments and a Clash game-data cache are local to the inspection.

Read the [environment assessment](reports/environment-assessment-2026-11-08.md).
Raw build/test logs, repeatable probes and probe results are in `evidence/`.

Python environments are `.venv-pokemon`, `.venv-clash` and `.venv-tavern`. Pokémon
uses Node 24; Clash uses the existing Homebrew JDK 21; TavernGym builds a Rust extension.
The Clash data cache is `.cache-crforge`; it is not part of a source release.

## Repeating the probes

From this directory, after the dependencies have been built:

```sh
node evidence/probe-showdown.cjs 30
.venv-tavern/bin/python evidence/probe-tavern.py
```

For the Pokémon Python probe, start the local server in another terminal. Its ignored
`config/config.js` selects port 18000 and binds to 127.0.0.1:

```sh
cd candidates/pokemon-showdown
node pokemon-showdown start --no-security
```

Then, from this workspace root:

```sh
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  .venv-pokemon/bin/python evidence/probe-poke-env.py
```

The local proxy bypass is required on the inspection host. These are random-action
environment probes, not policy-training benchmarks or game-strength results.
