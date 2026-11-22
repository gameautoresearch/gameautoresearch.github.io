# Pokémon learned-reference track

## November 9 five-hour strong-bot continuation

The active continuation uses `strong_campaign.py` and the explicit new ledger
`evidence/strong-bots-20261109-0516Z/campaign.sqlite3`. The older `remote.py`
campaign below is closed; its unused allowance cannot fund new work. Read
`reports/pokemon/strong-bots-five-hour-2026-11-09-rl.{md,json}` and both new
Pokémon coordination files for current checkpoints, public results and worker state.

`strong_runtime.py` adds two declared initialization paths: strictly audited
retained canonical-policy tensors with a fresh optimizer/RNG, or fresh seeded
canonical weights for a separately budgeted trusted reference architecture.
An existing reference checkpoint must first pass admission under its original
immutable source. An explicit source migration carries model/Adam/RNG/history,
teacher sampler and frozen-anchor state; a same-source resume rechecks all recipe,
engine, evaluator/search and guard bindings. It never rewrites old identities.

Public-player FoulPlay demonstrations support whole-game train/validation splits,
initial behavior cloning and optional auxiliary imitation during actual sparse
PPO. Only natural fault-free shards with verified hashes enter learning. Teacher
request/history is richer legal public information; student input remains 700
values and 14 masked actions. Training search uses reproducible fixed work;
production evaluation search uses the separately qualified two-belief 10 ms mode.
The exported neural policy has no test-time search. Teacher contribution, raw
initial weights, baselines and all failed attempts remain explicit.

`strong_vector_worker.cjs` delegates rules, observations, masks, retries and
termination to the qualified evaluator. It elides unrequested public views;
independent search actors can query concurrently within the same fixed 4 CPU
allocation. Query parity and full-state L4 learning/resume proofs are separate
from strength qualification. New architecture proofs are bounded at 600 seconds;
subsequent stages remain finite and respect the absolute selection/worker cutoffs.

Read-only accounting and collection commands from the repository root:

```sh
.venv-pokemon/bin/python -m simulators.pokemon.reference_training.strong_campaign status
.venv-pokemon/bin/python -m simulators.pokemon.reference_training.strong_campaign collect --label EXACT_EXISTING_JOB
```

`strong_campaign.launch_job` reserves the explicit ledger before every paid
allocation. Supply the checkpoint's original frozen `release` for strict export
or resume; new source is frozen separately after evaluator qualification.
`strong_export.py` uses the strict reference bridge. The current campaign ends
November 9 at 10:16:36 UTC; it does not authorize later paid work, excluded games,
benchmark entrant changes, publication or a simulator rewrite.

## Earlier sparse self-play baseline

This package is separate from the deliberately weak entrant starter. It reuses
our unchanged supported Showdown CPU transition engine, 700-value player-visible
encoder, 14 masked legal actions, Policy network and PPO update implementation.
No native engine work, test-time search, teacher labels or imported initialization
is used in the first baseline. Sparse +1/-1/0 natural outcomes are the sole
training reward. Administrative cutoffs bootstrap the final visible value.

`recipes/selfplay.json` uses 25% public-power opposition, 25% latest/latest games
(training both seats), and 50% uniformly chosen frozen historical opponents.
The opponent seat alternates with the episode index. History starts with the
random initialization, adds an immutable snapshot every four updates, and keeps
the latest eight snapshots. Each rollout records its exact opponent hashes,
team/battle seeds, policy samples, natural outcomes, caps and visible rejections.
The equal-allocation `control.json` changes only opposition to 100% public power.
An untrained initialization is exported separately for every from-zero run.

New recipes use a **4096 adapter-decision safety cap**, not 4096 game turns.
Historical 512-phase records remain unchanged. The evaluator owns the independent
versioned completion/retry contract and final qualification. Until its explicit
source/interface migration is admitted, ordinary development assessments using
this training engine do not count as evaluator-v2 qualification.

Every checkpoint binds exact adapter/trainer source, executed engine/dependency
hashes, recipe, model/Adam, NumPy/Torch/CUDA/Python RNGs, next episode index, frozen
population tensors and their identities. Resume only permits iteration/time
limits to change. The short GPU proof repeats one next batch twice and checks all
of these states for exact equality. Files are campaign-generated trusted local
checkpoints; external pickle is never loaded by this package.

The historical cached image intentionally lacks `.git` and the Git executable.
Our reference manifest verifies every tracked engine file against the clean local
Showdown commit manifest instead, then records and binds every executed compiled
JS/data/dependency hash. This is an explicit versioned provenance path; it does
not disable source-bound checkpoint admission. Local runs retain the original
Git-based manifest verifier.

From the repository root:

```sh
.venv-pokemon/bin/python -m unittest integrations.pokemon.reference_training.tests.test_reference -v
.venv-pokemon/bin/python -m integrations.pokemon.reference_training.trainer resume-proof \
  --config simulators/pokemon/reference_training/recipes/local.json --output NEW_LOCAL_DIRECTORY
.venv-pokemon/bin/python -m integrations.pokemon.reference_training.remote freeze
.venv-pokemon/bin/python -m integrations.pokemon.reference_training.remote launch \
  --label UNIQUE --mode proof --seconds 580
.venv-pokemon/bin/python -m integrations.pokemon.reference_training.remote collect --label UNIQUE
```

The launcher uses only the explicit new shared ledger in
`evidence/reference-readiness-20261109-0251Z/campaign.sqlite3`, checks the evaluator
source acknowledgment and RL role subcaps, reserves before allocation, attaches
provider IDs and settles completed authoritative worker intervals. Failed attempts
count. The detached Sandbox primary process has a finite provider timeout and
persists artifacts on private Modal Volume `pokemon-reference-20261109-private`
(v2, explicit syncs). No warm pool or retry loop exists. Frozen source releases and
all outputs are ignored private campaign evidence. The cached image is reused;
no new image build is launched. The fixed worker is L4/4 physical CPU/8 GiB.

All assessed seeds are development data until selection closes and a fresh bank
is committed. A trained checkpoint, self-play success or one-script win rate does
not establish reference strength. `assessment.py` performs inference only under each
checkpoint's original frozen source, and can invoke the evaluator's exact
`reference_bridge.export/load` before projecting inference to CPU. The bridge
extension is outside the reference training package and never changes checkpoint
bindings or an active training release. `remote launch --release-path PATH` keeps
matched arms on the exact same archived training release after assessment tools
are repaired. `export.py` additionally retains an independently verified raw
tensor export and its complete recipe/lineage. `readout.py` aggregates every GPU
and CPU attempt, receipts and paired seed-cluster intervals; final seed values
remain only in ignored evidence. Read the dated owning campaign report for the
actual executed runs, controls, costs and readiness flags.
