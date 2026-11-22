# Final from-scratch benchmark design — November 9, 2026

**Superseded:** the current user-selected $2,000 hero-run design is
[benchmark-hero-2k-v2.md](benchmark-hero-2k-v2.md), with an authoritative
[current-design pointer](benchmark-current.json). The 24-hour/five-repeat matrix
below is historical and must not be used for a new launch.

Saved from the user's decisions in this chat. The authoritative machine-readable
design is [benchmark-final-v1.json](benchmark-final-v1.json). This is a design
record, not a paid launch authorization or a claim that all release gates passed.
The latest choice is **24 hours**, superseding the earlier discussed 2-, 8- and
20-hour suggestions. Do not silently inherit the old 120-minute starter contract.

| Dimension | Selected design |
|---|---|
| Task | Build and train a learned game player from the simulator and documentation |
| Supplied code | Simulator, stepping example, public practice and submission interfaces |
| Withheld code | Policy network, optimizer/learning algorithm, collector and trainer |
| Games | Generals, Pokemon, Clash, Tavern/Battlegrounds, Dou Dizhu and Polytopia |
| Models | Ten, listed below |
| Independent research sessions | Five per model/game; fresh conversation, workspace and randomness |
| Research time | 24 hours per session |
| Compute | One H100 SXM5 80GB, 16 physical CPU cores (32 vCPU equivalent), 64 GiB RAM |
| Accounting | Full allocated worker time, including idle GPU and model/tool waits |
| Reasoning | Highest supported native effort, frozen for the served model/route |
| Progress artifacts | At 2, 6, 12 and 24 hours, latest explicit selection available then |
| Primary result | Mean final 24-hour score over five attempts, uncertainty and completion rate |
| Final grading | Isolated, fresh private cases after selection closes; additional budget |

## Model and routing matrix

| Model | Route | Effort |
|---|---|---|
| Claude Fable 5.1 | CLI proxy / existing coding subscription | max |
| Claude Opus 5.5 | CLI proxy / existing coding subscription | max |
| Claude Sonnet 5.5 | CLI proxy / existing coding subscription | max |
| GPT-6 Astra | CLI proxy / existing coding subscription | max |
| GPT-6.1 Sol | CLI proxy / existing coding subscription | max |
| Gemini 4 Carbon preferred | OpenRouter | Highest supported native level; access unresolved |
| Kimi K3.1 Pro | OpenRouter | max |
| GLM-5.3 | OpenRouter | max |
| DeepSeek V4.1 Flash 1017 | OpenRouter | max |
| Tencent Hunyuan Hy4, November 4 release | OpenRouter | Reasoning on, highest native level; mapping unresolved |

Carbon is absent from the OpenRouter catalog checked November 9. The agreed
alternative is **Gemini 4 Argon at high**, selected explicitly before freeze.
The cost analysis uses Argon and does not pretend to quote Carbon. No automatic
model substitution or fallback from the CLI proxy to a paid API is allowed.

The CLI proxy lists the five requested OpenAI/Anthropic IDs in a read-only check.
Only prior Sol/Opus smoke access is documented locally. Listing does not establish
all-model maximum-effort tool use, quota capacity, or benchmark-grade serving.
Local loopback is not reachable from a cloud controller: unattended hosting and
authenticated connectivity must be deliberately configured before launch.

## Matrix size

Ten models × six games × five research sessions = **300 sessions**.
At the full allowance this is **7,200 H100-worker hours**, including **720 per
model** and **1,200 per game**. A research session may conduct multiple training
experiments within its allowance; those experiments are not additional independent
researcher sessions. Unused time and failed attempts remain visible.

Existing competent trainers remain private equal-budget baselines. Do not weaken
them to manufacture gains. Baseline training, calibration, final grading and any
48-hour development extensions are separate work and costs.

## How the 24-hour horizon is justified

It measures what an autonomous researcher achieves in a day, not convergence or
expert game strength. Before launch, use a separate development calibration to
check that observations, environment throughput, submission and grading work;
that the opponent panel leaves headroom; and that useful iteration is possible.
Inspect progress curves and a preselected 48-hour diagnostic subset without using
the final private bank to choose the horizon. The 24-hour final endpoint is not
replaced by an earlier privately better checkpoint.

[MLE-bench's canonical setup](https://github.com/openai/mle-bench#benchmarking)
uses 24 hours with an A10 and at least three seeds.
[FrontierSWE v2](https://www.frontierswe.com/blog/v2) uses 20-hour sessions and
mean@5 at maximum reasoning effort. Different GPUs and task definitions mean
these session lengths are not equivalent compute.

## Release boundary

This is a **new from-scratch release**, not a rewrite of historical task contracts,
source-bound checkpoint checks, accounting or failed attempts. Qualify every
game's new release and common hardware, preserve player-visible information,
legal masks and separately reported endings/faults, and freeze model/provider IDs,
effort mappings, inference limits and external-asset/search rules before launch.
Calibration and normalized cross-game scoring remain unqualified until evidenced.
The old cancelled/closed ledgers and active independently authorized campaigns
are not changed or funded by this file.

## Cost and reproducibility

Read [the dated cost analysis](../reports/benchmark-final-cost-analysis-2026-11-09.md).
Its public rate snapshot and computed results are saved alongside it. Recompute
without any network, model call or cloud allocation:

```sh
python3 tools/estimate_benchmark_final_cost.py --config docs/benchmark-final-v1.json --prices reports/benchmark-final-prices-2026-11-09.json
```

Use `--output NEW_PATH.json` to write a new calculation; existing outputs are not
overwritten. Credentials, proxy gateway state and private evaluation assets do
not belong in these files.
