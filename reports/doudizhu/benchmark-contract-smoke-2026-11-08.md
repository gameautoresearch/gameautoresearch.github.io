# Doudizhu benchmark contract and autonomous smoke — 2026-11-08

**Passed:** DeepSeek V4 Flash inspected and edited the starter, ran real H100 learning,
evaluated public practice, explicitly submitted an immutable policy and finalized.
An independent isolated CPU grader then scored **67.1875/100** across
**192 natural games**, with complete-deal bootstrap 95% interval
**[59.90, 73.96]**. No faults,
illegal actions or truncations occurred. Reference-normalized score remains **null**.

## Frozen benchmark and starter

`simulators/doudizhu/autoresearch/benchmark.json` and `CONTRACT.md` declare
`doudizhu-fixed-role-research-v2`: full 54-card deck, three fixed roles, 20/17/17
hands, no bidding, player-visible observations and complete ragged legal actions.
The 162-decision guard is a fault, not a scored cap. Full research allowance remains
7200 allocated seconds on one strict H100, four physical CPU cores and 16 GiB.
No full-budget run or anchor calibration was launched.

The supplied starter remains random 128x3/LSTM64 three-role DMC, RMSprop and pure
self-play, terminal win targets, no replay/Adam/population/scheduling improvements.
New diagnostics expose finite losses and changed checkpoint hashes. The reusable
initial instruction and brief both enter the driver context. Smoke-only guidance
is injected from actual session settings. Raw score is 100 times equally weighted
role/opponent win probability; economic score remains separate. Peasant tests
control one candidate seat with a panel teammate, not joint peasant-team control.

The new campaign mode requires `--campaign-smoke --execute`, retains an immutable
600-second ledger and $2 model liability ceiling, admits one GPU worker at a time,
and excludes 20 seconds of cleanup reserve from new grants. Uncertain provider
leases retain holds. Existing old smoke/research30 records and rules/source/deal
locks were preserved. Sealed policy provenance now also binds captured recipe/seed.

## Executed smoke

DeepSeek independently chose 256x4/LSTM128 networks (1,483,779 tensor elements),
AdamW and cosine learning-rate scheduling. Those changes exist only in the captured
candidate, not the supplied starter. It trained **1,136,640 natural games /
45,205,065 decisions / 555 waves**, with final MSE **0.650707**. All recorded losses
were finite, initial/final weight hashes differ, and the sealed checkpoint matches
the final progress hash. The host verified tensor-only finite weights and exact
source/configuration binding without importing candidate code.

Public practice: **75/100**, eight deals x three opponents x three roles = 72 games.
Fresh host-private grading: 16 precommitted deals x four opponents x three roles =
192 games. Candidate inference ran as an unprivileged Linux child in a network-
blocked CPU sandbox; private state, seeds and panel weights stayed in the parent.
The researcher received no private feedback. Final role win rates were landlord
68.75%, downstream peasant 60.94%, upstream peasant 71.88%. Against the two hidden
full-size development bots, each win rate was 52.08%. This is a functional smoke,
not evidence of improvement, competitive strength or calibrated difficulty.

The first milestone crossed before any explicit submission, so it remains empty;
the two-thirds clock was not reached. Final grading used explicit submission 1,
bundle `cc570b66d372adf67af6122f386f45987c5e076c32d2499bb487ab558d7e8b4f`.

## Accounting and cleanup

| Allocation | Authoritative H100 seconds |
|---|---:|
| Training | 298.887083 |
| Public practice | 17.065943 |
| Total | **315.953026 / 600** |

The agent chose to stop early after completing the experiment; **284.046974**
seconds were unused. No failed or unresolved GPU attempts occurred in this new
campaign. Model charged-plus-reserved liability was **$0.03108000 / $2**.
The $0.49060675 quote-history field is cumulative reservation history,
not outstanding liability. CPU-only editing and independent grading consumed
735.909 provider worker seconds separately, with no attached GPU.

At the [current Modal list rates](https://modal.com/pricing), estimated GPU-worker
cost including attached CPU/RAM is $0.4301; separately measured CPU
agent/grader cost is $0.1946. Sandbox-specific CPU/RAM prices are used.
These are unsettled estimates excluding unmeasured image-build CPU, egress and
non-compute costs. Both owned apps are stopped with zero tasks; all four owned
worker intervals have provider finish timestamps. No other app was modified.

## Verification and reproduction

77 combined simulator/harness tests passed; the final 32-test harness suite also
passed after adding private-prefix and reusable-context checks. Coverage includes
checkpoint continuation, ledger resume/immutability, budget reserve, unresolved
holds, calibration gate, public visibility, safe packaging and latest final
submission. The final reusable-brief release was prepared/audited offline without
another training session. The live release retains its original prompt snapshot;
both release hashes and a final source snapshot are retained in the JSON summary.

Exact executed commands are in `evidence/doudizhu-autoresearch/campaign-smoke-20261108/commands.json`. The essential invocation is
`session --campaign-smoke --model-dollar-cap 2 --execute`; independent grading uses
`grade --development --private-deals 16 --execute` against the same release/session.
Do not create a new ledger to retry this campaign. No commit, push or publication
was performed. Machine-readable report: `benchmark-contract-smoke-2026-11-08.json`.
Private evidence is under `evidence/doudizhu-autoresearch/campaign-smoke-20261108/`; seeds, weights and full trajectories remain ignored.

Full-budget multi-seed reference calibration and larger independent opponent/deal
panels remain future work. Representative resume checks passed locally; this
campaign did not deliberately interrupt a live GPU worker.
