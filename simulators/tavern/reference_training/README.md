# Population PPO reference track

The November 9 bounded baseline is complete: 194 updates, 2,814,589 accepted decisions,
fresh mean placement 7.2813 → 4.6354. See
`reports/tavern/selfplay-reference-2026-11-09.{md,json}` for exact costs, failures,
artifacts and qualification limits. Selection is closed; the launcher refuses
further training in this baseline. Strong/superhuman qualification remains false.

Separate from the weak entrant starter. Uses qualified AutoTavern admission-v2
native fingerprint `217577b83fc4525e`, fixed eight-player supported-content rules,
player-visible numeric/card observations and legal masks. No search or test-time LLM.

Each lobby has four current-policy learner seats, one current-policy nonlearner
and three uniform historical-policy seats. Learner parity rotates with update and
lobby index, covering all eight seats. Only executed learner decisions create PPO
rows. Historical bootstrap policies are explicitly random; learned snapshots are
added every four updates to an eight-entry deterministic ring. The initial anchor
is retained. Matchmaking depends only on recipe seed/update/lobby, and checkpoints
save the full population, optimizer, RNG, completed native states and next seed.
Resume requires exact source/interface/recipe equality and verifies native restore.
Faults invalidate every learner trajectory from that lobby; action traces and guard
state remain private. No invented placements or safety-bound bootstraps.

Local correctness (no cloud work):

```sh
.venv-tavern/bin/python -m pytest -q simulators/tavern/reference_training/tests
.venv-tavern/bin/python -m simulators.tavern.reference_training.validate --config CPU_CONFIG --out NEW_PROOF
```

Authorized cloud stages use the explicit new shared ledger. First full worker cap
is 600 seconds, including startup/teardown, with in-container and provider bounds,
remote version-2 Volume persistence and a separate CPU provider observer:

```sh
.venv-tavern-ar/bin/python -m simulators.tavern.reference_training.launch --attempt attempt-01 --cap 600 --proof
.venv-tavern-ar/bin/python -m simulators.tavern.reference_training.status --attempt attempt-01 --collect
```

Every later allocation is separately invoked after first CUDA/export proof,
maximum 1,800 seconds. No automatic retries, warm pools or unbounded controller.
Image and observer liability remain held until authoritative reconciliation.

Export produces finite NPZ weights, exact policy source, strict original admission
binding and full source-bound reference checkpoint lineage. Its policy is the
existing two-layer `model.Policy`; loaders must use exported `policy.py`, rather
than assume the smaller starter architecture. Export is an inference artifact;
resume uses the source-bound full `latest.pt` and immutable runtime release.

Development/fresh assessment, all attempts and receipts are recorded in the dated
reference report. Same-policy self-play placement averages do not measure strength.

The selected checkpoint requires the exact archived `attempt-02/source` runtime.
Canonical host lifecycle helpers include later tested queue/deadline safeguards;
they do not change the already executed immutable training source.
