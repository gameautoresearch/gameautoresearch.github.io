# The researcher workspace and game rulebooks

## One workspace for every game

Every research session, for every game and both task variants, puts the
researcher (Harbor's Terminus-2) in the same box layout:

```text
/workspace/
  instructions.md   task instructions: common text, game brief, variant, budgets
  game_rules/       read-only Markdown rules and lookup indexes
  simulator/        read-only public simulator source and engine dependencies
  training/         editable starter network, learner and training entrypoints
  configs/          editable experiment configurations (train.json, smoke.json)
  NOTES.md          the researcher's notes
```

Most games expose `training/policy.py`, `learner.py` and `train.py`. Polytopia
also has `training/collector.py` and `configs/architecture.json` to preserve its
trainer contract. Chapter names and nesting within `game_rules/` and `simulator/`
are game-specific.

`harness/workspace.py` builds this tree inside each new release and keeps the
inventory, checksums and native path mapping outside the researcher's root. It
fails if an engine source group is missing or the release already exists. The
source export covers the complete public engine trees for the pinned mode and
excludes private banks, opponent weights, run state, credentials, artwork and
binaries. Full static source access does not grant access to hidden live
simulator state or private evaluation data.

In the box, rules, simulator source and instructions are root-owned and
read-only; the researcher runs as an unprivileged user that owns `training/`,
`configs/` and `NOTES.md`. It reads everything with ordinary shell tools and
edits the starter files in place. Only the declared editable files leave the
box, inside `ar` requests; the host validates them and copies them through an
explicit mapping into the game's native trainer when a run is requested. That
preserves each game's input schema, legal masks, scoring, source-bound
checkpoint checks and budget ledger. Full runs use `configs/train.json` and
smoke runs `configs/smoke.json`.

See [the harness guide](../harness/README.md) for sessions, the `ar` commands and
the `recipe` and `network` variants. To inspect exactly what a researcher is
given, build the task offline:

```sh
.venv-generals-ar/bin/python -m harness task generals recipe --release RELEASE --out NEW_TASK_DIR
```

## What each game supplies

```text
simulators/<game>/autoresearch/
  task/brief.md         the game mode, the starter and the limits a policy must respect
  task/game_rules/      the offline rulebook: README.md, scope-and-limitations.md, game-specific chapters
  starter/              the minimal editable starter (policy, learner, train)
  recipes/              starter.json and smoke.json configurations
```

The common instructions are a single template, `harness/templates/instructions.md`,
plus one section per variant; a game contributes only its brief and rulebook.
The rulebook describes the exact pinned mode: readable mechanics, entity
statistics, conditions, examples, actions, visible observations and useful
ID/name lookups. Chapter names and nesting are flexible. Every file inside
`game_rules/` is Markdown. Implementation excerpts, source appendices, raw
structured objects and executable examples are excluded; numerical rules remain
as written formulas and tables.

## Maintainer material

Every game's `autoresearch/wiki_maintenance/` owns:

- `tools/`: documentation generators and validators.
- `metadata/`: coverage declarations, source inventories, catalog metadata and hashes.
- `source_markdown/`: preserved original prose and implementation appendices.
- `raw_build/`: maintainer staging output; generators cannot write the researcher's rulebook.

The researcher receives none of those directories. Human-readable scope and known
limits remain in `game_rules/scope-and-limitations.md`. Do not copy raw generated
output directly into `game_rules/`; the Markdown-only checker rejects scripts,
metadata, source blocks and links escaping the rulebook.

```sh
python3 tools/research_tasks.py --check
python3 tools/research_tasks.py --game pokemon --out NEW_DIRECTORY     # export one rulebook
python3 tools/audit_research_wikis.py
```

These checks run offline: they do not launch games, import ML frameworks or
prove every possible simulator interaction.
