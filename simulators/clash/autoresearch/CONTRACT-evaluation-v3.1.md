# Explicit v3.1 research and evaluation release

This successor selects the same fixed task, six-actor mixed panel, public-only
information boundary, at-most1M float32 stateless neural entrants, raw scoring,
refusal diagnostics and precommit-before-private-bank workflow documented in
[the v3 contract](CONTRACT-evaluation-v3.md). Its declaration is
`benchmark-v3.1.json`. Prepare it explicitly with `v3 prepare --grading parallel3`.
Public instructions, brief, declaration, contract, schema and opponent identities
are copied into the immutable release; weights and final schedules stay private.

Independent opponent pairs run in at most three CPU processes inside the same
8-core/32GiB allocation. Each pair uses one native simulation thread and distinct
unprivileged policy UIDs. Candidate inputs, NN batch composition within a pair,
legal masks, acting-seat scripted RNG streams,4Hz decisions, natural clock,
inference deadlines and score weights are unchanged. Per-pair manifests/probes,
partial natural rows and failures remain durable. Any failed or incomplete pair
makes the official score null. Concurrency does not grant extra author GPU time.

Adoption requires exact complete-public-trajectory and diagnostic parity against
sequential grading on the pinned public schedule, plus provider-allocation
receipts. A small resumed real-PPO action workflow separately qualifies the
selected successor hooks. Qualification receipts, rather than this declaration,
determine whether these gates passed. Strong-panel/superhuman flags and reference
normalization remain false/false/null without independent calibration.

New author workers remain finite90..3300-second exact-H100 allocations with32
physical CPU cores/128GiB and a7200-second full author-session ceiling. The
current five-hour campaign's earlier selection and absolute worker stops still
win. All resource/model accounting, old failure preservation and source-bound
checks from v3 remain enforced. Current paid external model allowance is zero.

Public practice uses eight paired seed units. Protected workflow validation uses
four declared public units; it is never a private final. Fresh private assessment
uses32 paired units only after the exact candidate/control/registry precommit
has frozen. No checkpoint, panel, recipe or parameter is selected from private
scores. All precommitted identities and failures must be reported.
