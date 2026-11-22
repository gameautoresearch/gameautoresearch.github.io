# Generals benchmark contract and bounded DeepSeek smoke — 2026-11-08

**Passed:** a real DeepSeek V4 Flash session inspected and edited the starter,
trained on H100, submitted an immutable policy, evaluated public practice and
finalized. Independent host-private grading completed all 20 games at **37.5/100**.
The three GPU allocations total **312.681146 seconds**, including setup,
compilation, transfers, learning, practice, hidden grading, idle and teardown.
No GPU allocation remains unresolved. The separately authorized ceiling was
600 seconds; the previous $5 campaign and all historical ledgers are unchanged.

Machine-readable result: [benchmark-contract-smoke-2026-11-08.json](benchmark-contract-smoke-2026-11-08.json).
Private receipts, source snapshots, transcripts and weights remain under ignored
`evidence/generals-autoresearch/benchmark-contract-20261108/`.

## Delivered contract

The discoverable files are under `simulators/generals/autoresearch/`:

- `benchmark.json` and `CONTRACT.md`: exact versioned task, resources, budgets,
  observation/action surface, admission and scoring.
- `task/instruction.md` and `task/brief.md`: initial agent prompt and research brief.
- `starter/{policy,learner,train}.py`, `recipes/{starter,smoke}.json`: the functional
  minimal width-16 CNN, actual PPO/GAE and outcome-only self-play. The agent edits
  these files; the supplied starter was not replaced by the agent's optimizer change.
- `agent_cli.py`, `controller.py`, `ledger.py`, `bundle.py`: run/evaluate/submit/
  budget/status/wait/finalize, host-owned timestamps, immutable source/config/weight
  packages, milestone snapshots and final submission.
- `bounded_smoke.py` and `campaign.py`: persistent cumulative GPU accounting
  spanning training, practice, hidden grading and every failed allocation. A
  timeout plus 30-second teardown margin is reserved before allocation; unknown
  ends retain their hold and block another GPU. Completed provider receipts settle
  actual time without clipping overruns. Model calls share a separate $2 ledger.

The game remains `generals-research-1v1-full-v2`: two players, independent uniform
17–23 rows/columns padded to 24×24, fog, 2,048 joint ticks, terrain-density input
0.21, 9–11 castles with armies [40,51), BFS spawn separation at least 17,
general trade enabled and no castle construction/deathtouch. “Two” is player
count, not map size. Actor and critic receive the same visible 24×24×14 features
and 4,609-action mask. The 600,000 combined actor/value parameter limit and
stateless greedy inference remain fixed. The pinned engine and simulator source
were not changed.

The full benchmark locks one strict H100, four physical CPU cores, 16 GiB RAM
and 7,200 allocated research seconds. No full-budget training was started.
An offline full release was prepared and audited with 256 training, 16 practice,
128 private grading and 128 independent calibration maps: 528 valid, distinct
boards with no private seed overlap. The smoke used a separate fresh two-map
grading/two-map calibration release, audited across 276 boards. Neither full
private policy assessment nor reference calibration was run.

## Actual autonomous session and independent result

The corrected fourth attempt ran unattended after the host transport fixes.
DeepSeek wrote a hypothesis and changed only `learner.py`: Adam became AdamW
with `weight_decay=1e-4`. Policy architecture, trainer and starter hyperparameters
remained unchanged. The registered run started from random seed 101.

| Evidence | Result |
|---|---:|
| PPO updates / learner transitions | 760 / 24,903,680 |
| Final loss / KL | −0.044039 / 0.001017 |
| Parameter L2 change / total parameters | 10.269679 / 7,114 |
| Training capture wins / losses / caps | 2 / 1 / 11,262 |
| Illegal training actions | 0 |
| Public practice | 12 games; 41.6667/100; 10 caps, 2 capture losses |
| Private grading | 20 games; 37.5/100; 15 caps, 5 capture losses |
| Private illegal actions / simulator faults | 0 / 0 |

The explicit submitted package is
`61875a8b9c4a3ffb99f94dbbd993e385143379ba4a601cf64c536ab4335e0342`.
Its source/config/weights and current frozen-engine binding verify normally.
The same submission was recorded at the first allocation-third milestone and
at finalization; the session ended before the second-third threshold. No later
policy was substituted. Research remained closed during grading.

Every private map was played in both physical seats against random, expansion,
Hunter, early-starter and trained-fullsize opponents, with equal weights.
Scores were win=1, loss=0, genuine cap draw=0.5. Opponent means were 0.5, 0.375,
0, 0.5 and 0.5 respectively; seat means were 0.4 and 0.35. Whole-map paired
bootstrap gave a nominal 95% interval of 35–40 panel points. Two maps cannot
support a reliable strength estimate. Reference-normalized score is **null**.

The trusted engine and bots ran on CPU; submitted forwards ran on H100 as a
verified separate unprivileged UID. The receipt confirms isolated inference.
The submitted process received visible arrays and legal masks only, while
private maps, seeds and opponent weights stayed root-only. Complete schedules
were required; policy faults invalidate submissions and simulator/panel faults
abort grading rather than becoming capture losses.

The agent's original notebook incorrectly calls its two *public practice* maps
private. The actual request used `public/practice-maps.npz`; the notebook remains
unchanged as historical evidence. Independent hidden grading happened afterward.

## Verification, retained failures and accounting

- **34 harness tests passed**, including shared training/grading budget ceilings,
  unresolved holds, immutable budgets/submissions, scoring, safe arrays and
  prevention of automatic paid-call duplication.
- **21 full-size/contract tests passed**, including finite PPO parameter updates
  and exact model/optimizer/RNG/environment checkpoint resume. One existing optax
  deprecation warning was emitted.
- Both map-bank audits passed; submitted package verification/source review passed.

Initial offline preparation correctly rejected an old full training checkpoint
whose complete implementation binding had drifted. That failed release directory
is retained. Preparation then verified and copied already sealed opponent bundles
from `preparation-20261108/release-v10`, preserving their original strict-import
provenance and current engine binding. No checkpoint guard was bypassed.

The first launch lost a read-only model-catalog download. The next two launches
failed during CPU isolation probes when the local system proxy stalled Modal
command-router HTTP/2. They made no model calls or GPU allocations. Their
directories and CPU receipts are retained, along with two CPU diagnostics.
The fixed host launcher defaults `MODAL_DISABLE_API_PROXY=1`, guards closed
grpclib connections and polls existing exec IDs. OpenRouter uses HTTP/1.1 curl;
only read-only catalog requests retry automatically. Paid failures retain holds.

| GPU phase | Authoritative allocated seconds |
|---|---:|
| Training, 120-second request | 105.969180 |
| Public practice, 180-second limit | 50.941676 |
| Private grading, 180-second limit | 155.770290 |
| **Total / ceiling** | **312.681146 / 600** |

All 22 model calls completed: provider-reported cost **$0.00744115**, with
**$0.08878607** retained in conservative cumulative reservations, below $2.
CPU sandbox allocations, including failures and diagnostics, total 491.953170
seconds and are reported separately. At retained configured rates, GPU workers
are estimated at $0.370502 and separate CPU sandboxes at $0.043253; provider
billing is unsettled, and local/image-build CPU work is not priced here.
Final provider inventory shows all six owned apps stopped with zero running
tasks; all three H100 receipts have authoritative end timestamps.

## Commands and remaining limits

Run from the repository root, retaining the same campaign for all retries:

```sh
.venv-generals-ar/bin/python -m integrations.generals_autoresearch prepare \
  --out NEW_RELEASE --private-maps 2 \
  --opponent-release evidence/generals-autoresearch/preparation-20261108/release-v10
.venv-generals-ar/bin/python -m integrations.generals_autoresearch audit \
  --release NEW_RELEASE --maps
.venv-generals-ar/bin/python -m integrations.generals_autoresearch.bounded_smoke \
  --release NEW_RELEASE --out NEW_CAMPAIGN --execute
```

Without `--execute`, the bounded launcher prints a plan and starts no workers.
`--resume` retains the original GPU/model ledgers. If only grading needs repair,
add `--resume --grade-session ORIGINAL_CAMPAIGN/attempt-N`; that path cannot
restart training or renew the allowance and writes a new grading directory.
This recovery branch was added after the successful smoke and checked offline;
no extra paid recovery was needed. The actual successful campaign command used
`release-2` and `campaign` under the evidence root above, with `--resume --execute`.

This is workflow qualification, not evidence that weight decay improves play.
The finite 256-map pool, stateless fog interface and unqualified development
opponents remain. Complete current commercial-game parity, a full 128-map policy
assessment and three paired full-budget starter/reference seeds remain untested.
The official normalization gate stays closed. No artifacts were published or
other games' source changed by this work.
