# Pokémon evaluation contract v2

The full task remains pinned Gen 9 Random Battle, six Pokémon per player, one
active, 14 actions, 700 candidate observation values and CPU Showdown commit
`567c1cedddeaa2f35576005d3bfd526cf4f8a13e`. Canonical training/runtime files and
all historical manifests, source-bound checkpoints and v1 grades are unchanged.

The first admitted policy retains its original immutable implementation ID
`showdown-completion-v2-draft1`; its frozen release status is qualified development.
The word `draft1` is historical identity, not permission to mutate that release.

## Completion and scoring

- Count last engine-published turn, adapter decision phases, submitted choices,
  accepted choices and unavailable-choice retries separately, per seat.
- Allow the authoritative engine to produce wins and ties. Its native turn-limit
  and Endless Battle Clause ties are legitimate outcomes. Do not infer draws
  from switching cycles, unchanged HP or a visible-state repetition.
- Use 4,096 adapter phases and 16 consecutive unavailable-choice retries per seat
  as administrative safety limits. A waiting seat does not submit another choice.
  A successful own choice resets that seat's consecutive retry count.
- Preserve the engine's own updated request after an unavailable choice. Unknown
  trapping must not appear in the earlier mask. Never substitute an automatic
  rescue action for a rejected, invalid, unmappable or timed-out choice.
- Whole-game wall safety is 120 seconds. Candidate inference retains its 15-second
  batch deadline. Search has a separate one-second end-to-end choice deadline.
  Wall/phase/retry caps are unresolved; inference/process faults are failures.
- Score natural wins as 1, engine ties as 0.5 and losses as 0. An official aggregate
  requires the entire committed schedule to end naturally without faults.
  Otherwise its primary is null. Report completion and total-point bounds across
  capped, faulty and unstarted games. A valid zero remains zero.
- Keep natural outcomes, administrative caps, illegal policy actions, visible
  engine rejections, simulator faults and infrastructure faults distinct.

Historical v1 caps remain null even when a separately recorded v2 continuation
later reaches an outcome. Admission evidence must retain the original containing
12-game batches, action/state prefixes and all 700-value/public-protocol parity
checks; no old checkpoint binding is rewritten.

## Information and opposition

The candidate receives only its 700-value projection and legal mask. The Foul
Play-based opponent receives only its own request and player-channel protocol
history. The authoritative stream excludes omniscient output and ignores the
private end payload. Neither opponent requests nor actual hidden teams, RNG or
true-state search clones are provided.

Search reconstructs a public-information Foul Play state and samples unrevealed
opponents from pinned public random-set data. It uses the pinned poke-engine
search model, never that model for authoritative transitions. Semantic names and
full legal history contain more public information than the neural encoding;
this interface difference is explicit.

The admitted adapter samples two beliefs and aggregates normalized MCTS visit
fractions with deterministic weighted argmax. This differs from upstream's
adaptive sampling and randomized near-best selection. Test mode uses exactly
1,000 iterations per belief, one thread and seeded Python/Rust sampling.
Production uses a soft 10 ms per belief, one thread and a hard one-second parent
process deadline. The upstream engine checks time in 1,000-iteration chunks;
record actual visits and end-to-end latency, including overshoot. This reduced
adapter does not inherit published full-budget Foul Play strength claims.

All source, public data, dependency wheels, the seeded-search/packaging patch,
selection rule, inference limits and baseline weights must be identified in the
host-owned registry and identity audit. Search is an opponent, not a learned
reference. Weak scripts and short-trained policies remain weak baselines.

## Source admission and process boundaries

Original sealed candidates execute under their original exact runtime binding
in the unchanged Linux image, with unprivileged UID 65534. The search process
uses distinct UID 65533 and no-new-privileges. Private job inputs and outputs are
root-only; only the current job's Volume subdirectory is mounted. Network access,
secrets and OIDC are absent from grading workers.

The separate reference trainer may export only after its original source,
configuration, engine, tensor and population checks pass. Its tensor-only export
records those original identities. The standalone inference loader requires an
attested manifest, every canonical runtime-file hash, the Policy architecture,
executed engine and exact tensor identity. It never reinterprets a checkpoint as
a new resume checkpoint or disables source checks.

`arena.py` is the reusable candidate-facing API. It supports original sealed
candidates and attested reference inference exports against search, public scripts
and pinned trusted neural opponents. Exact audited plain-starter bundles may be
projected into trusted tensor inference; arbitrary submitted Python remains in
the isolated candidate process. All summaries use candidate perspective.

## Campaign execution

Use only the explicit new ledger in
`evidence/reference-readiness-20261109-0251Z/campaign.sqlite3`, through this
package's evaluator role guard. The role cap is 1,800 allocated GPU seconds,
$8 resources and $0.50 model calls. No evaluator GPU or model call is required by
this implementation. Old ledgers are not defaults or renewed allowances.

Every paid worker reserves first, has a finite provider lifetime, receives an
immutable input/source snapshot and persists artifacts before terminating.
Charge full authoritative allocation intervals, including failed attempts and
setup/compilation/inference/export/teardown. Unknown resource liabilities remain
held. Prove pre-allocation failures against provider inventory before clearing
reservations. Close selection before committing a fresh private schedule, and
never tune on that bank.

Harness readiness, learned-reference readiness, strong-panel readiness and
superhuman evidence are distinct. This version admits a useful development
search opponent; it does not establish expert or superhuman strength.
