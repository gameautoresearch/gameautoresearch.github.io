# Native frozen Gen 9 — C++ CPU

This independent C++ backend implements the pinned `gen9randombattle` reset,
battle and 700-feature / 14-action contract. Native generation, stepping,
observations and ordinary training do not invoke Node or Showdown. The pinned
Showdown engine is the independent differential oracle.

The November 9 CPU completion and its qualification receipts are recorded in
[the completion readout](../../../reports/pokemon/native-cpu-completion-2026-11-09.md)
and [machine-readable status](../../../reports/pokemon/native-cpu-completion-2026-11-09.json).
The supported scope is the frozen random-battle data at
`567c1cedddeaa2f35576005d3bfd526cf4f8a13e`. It is not a general all-format Showdown
implementation or an expert-strength policy claim.

Production admission requires an evidence-bound certificate. A normal developer
build still returns `pkn_gen9_complete() == 0`; `Native.require_full_task()` rejects
it. `build(..., admission=...)` verifies the qualified source, external runtime,
public data and proof receipts before compiling the admitted binary. Production
`NativeVector` verifies these bindings again. Changing any native source/helper,
external runtime or public table requires qualification again. Never bypass a
source mismatch or transfer an old environment/optimizer checkpoint across it.

## Team size (added 2026-11-13)

Team size is a format parameter, 1 to 6, default 6 (unmodified `gen9randombattle`).
`ResetSeeds.team_size` carries it in what was padding, and 0 means six, so
six-Pokémon schedules keep their bytes. Generation, construction, battle start
and reset honour it. Slots beyond the size stay all-zero, and the simulator,
observation and masks already skip empty slots. The observation (700 values,
12 rows), the 14 actions and every `/6` normalizer are unchanged: with three
Pokémon, own rows 3–5, foe rows 9–11 and switch actions 11–13 are always empty.

Showdown plays the same format as `gen9randombattle@@@Max Team Size = N`. The
pinned source is unchanged; its generator honours the rule. The oracles take
`team_size` in a spec, or `POKEMON_TEAM_SIZE` for `teamgen_oracle.cjs`;
`worker.cjs` takes `--team-size=N`.

A certificate qualifies one team size (`team_size`, `format` and its own
`qualification_plan`). `prepare_runtime --team-size N` records the size in
`runtime.json`, and production vectors refuse any other size. The three-Pokémon
qualification is `reports/pokemon/native-cpu-qualification-3v3.json`, read out in
`reports/pokemon/pokemon-3v3-form-2026-11-13.md`. Its banks also found
pre-existing six-Pokémon parity defects; they are fixed here, so the November 9
six-Pokémon certificate no longer verifies against this source. Using six-Pokémon
battles in production from this source needs a new six-Pokémon certificate.

## Components

- `teamgen.hpp` and `reset.hpp` generate both teams and construct/start battles
  directly from frozen public tables. Simulation inputs contain no retained
  seed teams or oracle states, and generation does not filter, resample or fall
  back to Showdown. The separate admission certificate embeds audit receipts;
  their teams/seeds/results are hash-checked evidence, never simulator inputs.
- `sim.hpp` implements fixed-state transitions, action queues, suspended switches,
  Tera/Stellar, forms, Transform/Imposter, Illusion, delayed attacks, move effects,
  abilities and items reachable through the frozen data. Unknown effects fault.
- `observe.hpp` preserves the existing encoder's event-driven public knowledge,
  private request caches, legal masks and Illusion aliases, including terminal
  observations. Natural endings, official draws, caps, illegal actions and
  simulator faults remain distinct.
- `vector.py` implements the ordinary persistent `Vector` interface, slot resets,
  public metadata and complete source-bound state restore.
- `train.py` runs the original editable recipe, network, rollout and starter PPO
  learner with original terminal rewards. Full checkpoints include model, Adam,
  Python/NumPy/Torch RNGs, native state, seed/seat cursors and source/rules binding.
- `cpu_env.py` and `resident_ppo.py` provide batched CPU collection and development
  learning checks. The bounded L4 validation uses CPU simulation with GPU neural
  inference/PPO. No new CUDA simulator qualification is implied.
- `admission.py` checks the frozen proof bundle; `prepare_runtime.py` packages
  normalized public simulation rules, a compiled library, and (for admitted
  builds) the separate evidence certificate. The older preparation-helper
  docstring describes the simulation-data portion, not the added certificate.

## Build and run the admitted runtime

Commands run from the workspace root with `.venv-pokemon`. Use new output paths.
The retained local certificate is
`evidence/pokemon-implementation/native-cpu-completion-20261109/admission-1.json`.
It is valid only while its source and public-data bindings still match.

```sh
.venv-pokemon/bin/python -m integrations.pokemon.native.prepare_runtime \
  --generators evidence/pokemon-implementation/native-rewrite-20261108/teamgen-full-1/oracle.json \
  --constructors evidence/pokemon-implementation/native-rewrite-20261108/materialize-2/oracle.json \
  --chart evidence/pokemon-implementation/native-rewrite-20261108/startup-2/oracle.json \
  --admission evidence/pokemon-implementation/native-cpu-completion-20261109/admission-1.json \
  --output evidence/pokemon-implementation/NEW_ADMITTED_RUNTIME

.venv-pokemon/bin/python -m integrations.pokemon.native.train \
  --config simulators/pokemon/autoresearch/recipes/smoke.json \
  --runtime evidence/pokemon-implementation/NEW_ADMITTED_RUNTIME/runtime.json \
  --out evidence/pokemon-implementation/NEW_NATIVE_TRAIN \
  --seconds 60 --updates 1
```

Resume with `--resume PATH/checkpoint.pt` and the exact same source, rules,
recipe, runtime and device. `--qualification` explicitly selects development
scope and is never a substitute for production admission. Neural weights may be
exported independently; weights-only inference is checked against the policy
that produced them. Qualification is not policy-strength evaluation.

## Correctness and performance

Retained qualification plans, all failed banks/counterexamples, source snapshots
and final receipts live under the completion evidence directory. Differential
checks accept `--library` to bind every proof to one candidate. Keep new outputs
in new directories. The ordinary RL checks can run with:

```sh
.venv-pokemon/bin/python tools/check_local.py --game pokemon
.venv-pokemon/bin/python -m integrations.pokemon.native.ability_flags
```

Local timing comparisons must hold `evidence/local-performance.lock`. Report
simulation/encoding components separately from complete learner throughput and
fully allocated worker time. Both-seat learner rows exclude PPO epoch reuse,
opponent rows, teacher labels, inherited rows and duplicate resume verification.
Provider app lifecycle bounds are not verified financial billing.

## Preserved history

The November 8 readout and `native-rewrite-20261108/` receipts preserve the earlier
7,055-case regression, source-specific CPU/CUDA diagnostics, failed/partial
random-reset qualification and implemented-mechanics timing comparisons. Those
historical rates and CUDA checks do not qualify later CPU additions. GPU
simulator work remains stopped. Historical manifests, receipts and budgets are
unchanged, and the continuation does not authorize restarting old campaigns.
