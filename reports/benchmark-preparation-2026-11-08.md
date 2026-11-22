# Six-game benchmark preparation — 2026-11-08

All six accepted game scopes now have exact public contracts, initial agent
instructions and briefs, deliberately minimal RL starters, budgeted research
entrypoints, immutable source-bound submissions, public practice and private
grading. Each completed a real DeepSeek-driven learning/evaluation/submission
workflow followed by independent host grading. This is functional preparation,
not a full-budget strength or difficulty qualification.

The [contract guide](../docs/benchmark-contracts.md) describes shared conventions.
The [machine-readable integration summary](benchmark-preparation-2026-11-08.json)
records exact per-game totals and source report pointers. Private execution
evidence stays ignored and local; the public summaries contain no private seeds,
policy weights, credentials or game assets.

## Smoke results

GPU time below is the complete allocated worker lifetime, including setup, idle
time, teardown and GPU-based evaluation. Each game's cumulative ceiling was 600
seconds; model calls separately had a $2 charged-plus-held ceiling. CPU-only
agent and grading allocations are reported in the owning readout.

| Game | GPU | Allocated seconds | Private games | Raw points /100 | Readout |
|---|---|---:|---:|---:|---|
| Polytopia | H100 | 207.97 | 4 natural | 100.00 | [Report](polytopia/benchmark-contract-smoke-2026-11-08.md) |
| Generals.io | H100 | 312.68 | 20: five capture losses, 15 caps | 37.50 | [Report](generals/benchmark-contract-smoke-2026-11-08.md) |
| Pokémon | L4 | 375.21 | 12 natural | 66.67 | [Report](pokemon/benchmark-contract-smoke-2026-11-08.md) |
| Clash Royale | H100 | 401.19 | 8 natural | 50.00 | [Report](clash/benchmark-contract-smoke-2026-11-08.md) |
| Battlegrounds | L4 | 126.35 | 16 natural | 31.25 | [Report](tavern/benchmark-contract-smoke-2026-11-08.md) |
| Dou Dizhu | H100 | 315.95 | 192 natural | 67.19 | [Report](doudizhu/benchmark-contract-smoke-2026-11-08.md) |

The six workflows used approximately **29 allocated GPU minutes**, below the
60-minute combined maximum. Every score is conditional on its own small weak
development panel; the numbers cannot rank game difficulty or cross-game skill.
Polytopia's four wins against a randomly initialized opponent are especially
limited evidence. Reference-normalized scores remain null for all six games.

Every workflow produced finite learning losses and changed weights, completed
public practice, explicitly selected an immutable submission, finalized research
and received a complete private grade. No private grading game reported an
illegal action, simulator fault or unapproved truncation. Generals' cap draws are
part of its declared rules. Tavern discarded one no-progress training lobby as
a simulator fault; it was not scored or converted into a natural loss.

## What is supplied to the researcher

Each `simulators/<game>/autoresearch/` owns:

- `benchmark.json` and `CONTRACT.md`: task settings, hardware, budgets, editable
  surface, inference limits, panel sizes, scoring and outcome validity.
- `task/instruction.md` and `task/brief.md`: the initial objective and working
  brief, with session-specific smoke/full limits injected by the host.
- Minimal editable policy, learner and trainer code, plus starter/smoke recipes.
- Preparation, budget/status, run, practice evaluation, submit/finalize and
  independent grading interfaces, with full 7,200-second session entrypoints.

The supplied starters initialize randomly and use plain learning methods:
Polytopia's 2×32 PPO, Generals' width-16 PPO, Clash's width-32 PPO, Pokémon's small
64-hidden/8-embedding PPO, Tavern's 64-hidden/4-embedding PPO, and Dou Dizhu's
128×3/LSTM-64 terminal-win DMC. They include no pretrained policy, historical
league, replay curriculum or advanced reference recipe as starter options.
DeepSeek's improvements and trained weights are separate candidate artifacts.

Research still benefits from proven simulator, mask, encoding and rollout
plumbing. Weakness comes from simple models and learning methods, rather than
deliberate bugs, illegal actions or false rewards. A modest policy can beat a
weak diagnostic opponent; these starters are not claimed to be competitively
qualified.

## Fixes and verification

The campaign repaired transport/proxy failures, incomplete model-catalog reads,
editable-workspace permissions, native build-lock ownership, private GPU-grading
accounting, policy isolation, terminal cap-score handling and full-mode schedule
selection. Attempts and unknown model-call reservations remain in their original
campaign ledgers; no historical allowance was reset.

The owning reports retain exact local test commands, checkpoint/source checks,
learning evidence, private score breakdowns, provider start/end receipts and
worker cleanup. Parent review checked the per-game receipts against the summary
and ran offline CLI previews in the declared per-game host environments. Shared
validation is reproducible without importing game frameworks or launching jobs:

```sh
python3 tools/check_layout.py
python3 tools/check_benchmark_contracts.py --smoke-receipts
git diff --check
```

The receipt checker validates completion declarations and bounds; it does not
query billing or independently prove the underlying provider records. The
retained parent CLI-preview logs are in
`evidence/benchmark-preparation-20261108/parent-checks/plan-previews.json`.

All owned smoke workers were stopped, and no GPU allocation remains unresolved.
Reported resource costs are estimates where invoices are unsettled; model
liabilities include any failed or unpriced calls. Source-only CPU image builds,
storage and egress are not silently relabeled GPU training time.

## Remaining qualification

The full 120-minute research campaigns, full private panels and equal-budget
multi-seed reference calibration were not executed. Full-mode preparation,
schedule commitments, available development opponents and entrypoints were
checked offline. Their presence does not establish stronger reference policies.

Smoke attempts included operator repairs. The owning JSON reports distinguish
the executed immutable release from subsequent locally verified host/prompt/full-
mode refinements; this is not a claim that the final packaging received a second
clean unattended cloud run. Historical source snapshots and receipts are retained.
Clash's later reduced package reproduced the original private rows locally and
passed a remote public privilege check; an ancillary remote private repeat
stalled in provider control-plane work and was stopped. Its earlier complete
private smoke grade remains the reported result.
Parent review also found that Clash's executed ledger omitted the two-thirds
milestone when teardown crossed that threshold. The historical record is kept;
its interpretation is missing/null, with no retrospective submission. New
settlement atomically records provider time and crossed milestones, covered by
a local teardown-boundary regression test.

Polytopia's minimal starter does not implement exact training resume. Pokémon
and Tavern verified exact CPU resume but did not run a separate exact CUDA-resume
comparison for their new harnesses. Every registered experiment can start fresh,
as the research contract requires. Simulator fidelity remains bounded by the
owning game evidence: Pokémon uses Showdown CPU, Tavern uses Rust CPU, and their
unfinished/paused GPU simulator rewrites were not resumed.

Changes remain in the working tree. No commits, pushes, private-artifact
publication or full-strength reference-training campaign were performed.
