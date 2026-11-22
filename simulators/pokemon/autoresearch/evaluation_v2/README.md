# Pokémon evaluator v2

Read [CONTRACT.md](CONTRACT.md). This package adds explicit completion semantics,
public-information search opposition, strict inference-only reference admission
and a reusable candidate-facing evaluator without changing the canonical runtime.

The admitted release, frozen opponent registry, development/fresh assessments,
all failed attempts, source snapshots and allocation receipts are private under
`evidence/reference-readiness-20261109-0251Z/pokemon/evaluator/`. The dated
[readiness report](../../../../reports/pokemon/evaluation-readiness-2026-11-09.md)
records what actually passed and what remains unqualified.

Run local correctness checks and offline reporting from the repository root:

```sh
node simulators/pokemon/autoresearch/evaluation_v2/test_worker.cjs
.venv-pokemon/bin/python -m unittest integrations.pokemon.autoresearch.evaluation_v2.tests.test_contract -v
.venv-pokemon/bin/python -m integrations.pokemon.autoresearch.evaluation_v2.report
```

`arena.run(spec, output_directory, private_request_path)` evaluates a source-bound
candidate against a panel. Candidate kinds are `bundle` with its exact bundle hash,
or `reference-inference` with its attested export-manifest hash. Panel kinds are
`search`, `power`, `random`, pinned `checkpoint`, `reference-inference`, or an exact
audited `plain-starter-bundle`. Search requires the frozen Foul Play archive,
public-set JSON and Linux wheel hashes. Every request includes the qualified core
release and file hashes. The phase limit is 4,096. Results report candidate points,
per-opponent/seat completion, score bounds and separate outcome/fault categories.

Official execution requires root Linux isolation, not a local development bypass:

```sh
python -m integrations.pokemon.autoresearch.evaluation_v2.arena \
  --request PRIVATE_REQUEST_JSON --output NEW_PRIVATE_OUTPUT
```

For campaign cloud execution, call `evaluation_v2.remote.launch` with a unique
label, module `arena`, complete immutable inputs, finite CPU lifetime and the
frozen source release. It reserves against the explicit new ledger, uploads all
inputs before starting an autonomous Sandbox, mounts only that job's subdirectory,
uses the unchanged cached image and persists outputs through host disconnect.
Recover a known ended job without relaunching or renewing its allowance:

```sh
.venv-pokemon/bin/python -m integrations.pokemon.autoresearch.evaluation_v2.remote \
  --collect EXISTING_UNIQUE_LABEL
```

Cloud collection applies while that campaign Volume is retained. After verified
final cleanup, use the local hashed artifact inventory instead; deletion never
renews a job label or its spent allowance. The admitted general controller source
is selected by `active-entrant-api-release.json`; its qualified opponent core is
the unchanged parent release in `active-evaluator-release.json`.

`reference_bridge.export` must run inside the checkpoint's original frozen
reference source/configuration/engine after its normal admission checks.
`reference_inference.load(directory, trusted_manifest_sha256)` is the standalone
strict tensor-only inference path. It does not require importing the trainer or
unpickling its optimizer/RNG checkpoint, and does not support resume.

Development qualification uses eight public seed clusters and both seats.
`final_bank prepare` closes the already frozen opponent selection and commits
one fresh private bank; `final_bank execute` uses that exact release and schedule.
These commands launch paid work and must only be used within the matching current
authorization. They refuse existing job labels and do not reset historical costs.
Do not rerun the already opened bank to tune policies or repair a failed score.
