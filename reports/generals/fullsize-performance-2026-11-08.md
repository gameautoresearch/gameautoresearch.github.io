# Full-sized Generals 1v1 environment and throughput

The RL standard is now **`generals-research-1v1-full-v2`**: independently sampled
17–23 rows and columns, fog of war, and a **2,048-joint-tick cap**. Maps are
padded to 24×24 with impassable mountains. The optimized GPU rollout, packed
observations and masks, historical policies, PPO, compact checkpoints and exact
resume are retained. Old 8×8 performance figures and checkpoints are historical
and do not qualify this larger environment.

The [contract](../../integrations/generals/benchmark-full-v2.json),
[commands and implementation notes](../../integrations/generals/README.md), and
[machine-readable results](fullsize-results-2026-11-08.json) define the new task.
All new evidence is in
`evidence/generals-implementation/fullsize-20261108/`. The upstream engine remains
unmodified at MIT commit `954927f508aa47899019d66317cc313af3e9cc19`.

## Rules and implementation

The full-size parameters follow the released
[AverageJoe training configuration](https://github.com/strakam/AverageJoe/blob/main/configs/custom/L_7d_gae90.yaml):
terrain-density input .21, 9–11 neutral castles inclusive, castle armies in
[40,51), and minimum BFS spawn distance 17 with no maximum-distance restriction.
There is no implicit small-board or spawn-distance curriculum. General trade
is enabled; build-castles and deathtouch are disabled. A capture gives ±1;
the cap gives a draw with zero outcome reward. Training bootstraps the actual
final visible state at truncation. Illegal actions and simulator faults are
separate from natural outcomes and cap draws.

Simply setting the native environment's minimum and maximum size would have
silently produced only 23×23 maps: its `init_state()` always uses the maximum.
The wrapper now selects each side independently from 17 through 23 with a
separate folded seed key and invokes the native generator for that exact
rectangle. Grouped eight-key setup batches reuse each shape's compiled generator
and restore original seed order. Dummy batch keys never enter the pool.

The actor and critic still receive only the upstream player-visible observation:
24×24×14, including the intentionally public scoreboard totals and silhouettes.
Each player has 4,609 possible action IDs: pass, or source × direction × split,
filtered by the visible legal mask. Padding
is illegal as a move source or destination. The Pallas/Triton opponent CNN now
uses 64-pixel tiles, including correctly masked partial tiles; it retains the
existing BF16 rounding and float32 parameters. PPO remains Flax/cuDNN.

The current actor is the existing small width-16 CNN, without persistent memory.
This prepares the full-sized environment; it does not reproduce AverageJoe's
15M-parameter transformer or establish a strong playing policy. The unchanged
public partitions contain 256 training, 16 practice and 32 heldout maps, rather
than a refreshed 200k-map training pool. Heldout maps were checked geometrically;
no policies were evaluated on the heldout panel in this campaign.

## Correctness qualification

* **44 integration tests and 145 upstream tests passed.** Full-board fixtures
  cover fog independence, all 4,609 codec choices, padded legal masks, general
  capture and elimination, general trade, territory transfer with ceiling
  rounding, splits, castle/land income, neutral-castle behavior, and a capture
  exactly at tick 2,048 taking precedence over truncation.
* All **49 rectangular shapes** match separately generated native maps and match
  bitwise between CPU and CUDA. All 256 training maps saved by the H100 and L4
  learners also match the local CPU audit in every state field, using the normal
  archived-source checkpoint guard. Each fixture satisfies spawn separation, padding,
  castle-count and castle-army bounds. All **304 public maps** are connected,
  distinct, have valid generals and require no minimum-distance fallback. The
  training pool covers all 49 shapes. Historical seed partitions are unchanged.
* For **100,352 joint ticks** (49 maps × 2,048), the wrapper and native `Env.step`
  match every state field, visible observation, legal mask, reward and episode
  boundary. Only reset-pool metadata is adapted to the same external schedule.
  This trace contains 49 caps and zero natural captures; capture/trade coverage
  comes from the explicit full-board fixtures and upstream suite, rather than
  being inferred from this trace.
* H100 and L4 validation checks packed inputs, float32 baseline/optimized rollout,
  tiled/slice BF16 trajectories, history refresh and cap bootstrapping. Saved
  arrays and resumed simulator/RNG/history are exact; the next PPO update has
  **maximum absolute difference 0** on both tested devices.
* Fresh public CLI train → resume → evaluation → profile and policy-independent
  CUDA/CPU replay succeed. A final local CLI train/resume checkpoint passes the
  normal current-source binding check. No checksum or source guard is bypassed.

Receipts: `release-integration-tests.log`, `upstream-tests.log`, `map-audit/`,
`validate/{h100,l4}-release-3/seed-42/`, and `cpu-cli-release/` beneath the evidence
root. Every delivered remote receipt retains its executed Python/JSON/lock source. The final
launcher-only change selects strict `H100!`; release validation used the same
simulator/learner source before that resource-selection change. Its checkpoint
retains its exact original binding; the final local checkpoint binds the current
complete source.

## Warm complete PPO throughput

One sample is one learner transition per joint tick. There are two player
decisions per tick; neither the opponent's decision nor reuse over PPO epochs
is counted as another learner sample. These are complete rollout + GAE + two
PPO-epoch rates, rather than simulator-only rates.

The recovered H100 profile covers the selected default; the interrupted four-size
H100 sweep yielded no delivered results. The L4 sweep covers all three listed sizes.
Profiles use seven alternating-order trials from the same trained starting
checkpoint. Preconditioning performs 68 real PPO updates and naturally crosses
the 2,048-step cap. Timing does not inject a synthetic clock or artificial armies.
All warm rows exclude cold setup/JIT, which is included in full worker accounting
below. Every worker has one GPU, four physical CPU cores and 16 GiB RAM.

| Actual GPU | Games batched | Slice BF16 samples/s | Tiled BF16 samples/s | Kernel-path gain |
|---|---:|---:|---:|---:|
| H100 | 2,048 (selected) | 224,923 | 334,051 | 1.49× |
| L4 | 128 (selected) | 53,960 | 75,913 | 1.41× |
| L4 | 512 | 48,531 | 71,341 | 1.47× |
| L4 | 2,048 | 47,982 | 68,223 | 1.42× |

At batch 128 on L4, the retained unfused float32 path gives 20,215 samples/s,
the fused float32 path 20,882, slice BF16 53,960 and tiled BF16 75,913. The larger
L4 batches use 4,096-sample minibatches, while its selected 128-game recipe uses
1,024. H100 uses 4,096-sample minibatches at its selected 2,048-game batch.
The tiled/slice gain is **1.49× on H100** and **1.41× on the selected L4 recipe**.
The baseline here is the original trainer path **ported to this full-sized
task**, not the archived 8×8 binary or a claimed equivalent upstream agent.

The initial H100-requested profile actually received an **H200**. Its actual
hardware is retained in every receipt and its results are classified separately:
180,630 / 302,049 / 349,221 / 366,999 warm samples/s at batches
128 / 512 / 2,048 / 8,192. These are not H100 measurements.
Modal documents [automatic H100→H200 upgrades](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s);
the launcher now requests `H100!`, which prevents that substitution.

## Full allocation learning measurements

Three seeds per recipe use 600-second treatments with one GPU plus the same
four-core/16-GiB attachment. Treatment time covers imports, all 49 shape compiles,
map generation, simulation, PPO, development evaluation and checkpoint saving.
The outer function additionally performs a full practice evaluation, exact
action replay and archive creation. The table counts that full function and,
separately, provider-reported GPU allocation including startup/shutdown/idle time.
Unused treatment time is not charged as a fictional reservation; actual
allocated time is charged. Changes to precision and batch size can change
learning trajectories, so these are throughput comparisons, not policy-strength
comparisons.

| Actual GPU / recipe | Games batched | Learner samples (3 seeds) | Full function seconds | Samples/function second | Provider GPU seconds | Samples/provider second |
|---|---:|---:|---:|---:|---:|---:|
| H100 / legacy | 128 | 50,106,368 | 1,689.48 | 29,658 | 1,698.34 | 29,503 |
| H100 / optimized | 2,048 | 317,980,672 | 1,722.01 | 184,657 | 1,729.79 | 183,826 |
| L4 / legacy | 128 | 19,574,784 | 1,717.51 | 11,397 | 1,734.38 | 11,286 |
| L4 / optimized | 128 | 49,397,760 | 1,741.10 | 28,372 | 1,764.24 | 27,999 |

The measured provider-allocation improvement is **6.23× on H100** and **2.48×
on L4** versus the full-size unfused float32 control. Full-size optimized totals
are 317.98M H100 samples and 49.40M L4 samples. These ratios include all three
seeds; they do not reuse or multiply the old 8×8 speedups.

All seeds, including the unusually slow cold-setup L4 baseline seed 43, are kept.
The performance recipe has real optimizer updates, parameter changes and natural
episode completions. All 12 training logs have finite learner metrics and zero
illegal actions; they include capture wins, capture losses and cap draws. The
selected H100 warm preconditioning includes 2 captures and 2,046 caps; selected
L4 includes 1 capture and 127 caps. Exact practice action replays check saved final states,
legal actions and finished boundaries. Outcomes and truncations are recorded
separately; weak scripted practice opponents do not qualify general playing
strength. The optimized recipes each ended with 192/192 practice cap draws,
whereas the baselines had 4/192 (H100) and 9/192 (L4) capture wins and no capture
losses. The optimized configuration therefore demonstrates no strength gain in
this pilot. No short pilot here establishes a strong reference policy.

## Retained failures, source preservation and accounting

The first L4/H100 validation attempts failed an incorrect **validation assertion**
that excluded 11 castles. The native generator already correctly sampled counts
9–11 inclusive; castle-army upper bounds are exclusive. The assertion and rule
description were fixed, and both paid attempts and source archives are retained.
Two optimized launches failed during local mount construction because the README
changed while Modal was hashing it; neither submitted a worker function. A second
unrestricted H100 profile was cancelled after strict `H100!` selection was found;
its allocated usage is retained. The strict four-batch H100 profile was then
preempted after all four child measurements exited successfully, before the
archive was delivered. Its lost warm values are not claimed; the selected
2,048-game configuration was repeated separately. Application-error retries are
configured to zero, but [Modal preemption](https://modal.com/docs/guide/preemption)
can restart a task independently, so the preempted app was explicitly stopped.
All allocated usage is retained. These failures are not silently dropped.

`historical-preservation.json` verifies the old benchmark, seed partitions,
all old recipes and the entire upstream tree are unchanged. The previous
implementation is archived as `previous-standard-source.tgz`, with its original
source binding. The old 8×8 checkpoints require that exact archived source;
there is no implicit migration to the new action/observation geometry.

Provider-reported gross usage for **this full-size campaign is $8.741464**
before credits, as of **2026-11-08T08:27:03.359318+00:00**, covering GPU, attached CPU/RAM and all
reported failed/cancelled/preempted attempts. Reported GPU allocation totals
**11,816.25 seconds (3.2823 GPU-hours)**
across both device classes. These are actual allocation totals, not GPU activity
time or the parallel client wall clock. Resource rates were checked against
[Modal pricing](https://modal.com/pricing). Billing can lag; the two deployment
failures with no submitted functions have no reported resource rows and remain
explicitly listed as missing, rather than silently asserted free. The earlier
Generals campaigns and their ledgers remain separate and unchanged.

Receipts: `evidence/generals-implementation/fullsize-20261108/usage-final-2/summary.json`, `hardware-correction.json`, `preemption.json`,
`deployment-failures.json`, `archive-integrity.json` (21 hash-verified delivered
archives), and `cleanup-final.json`. The final filtered provider inventory at
**2026-11-08T08:25:53.432358+00:00** shows **zero active workers for this campaign**.
No new persistent volume was created. All checkpoints and evidence remain local
and private; no benchmark workers are left running.

Source changes are left uncommitted. No private seeds, checkpoints, traces or
assets were published, and no other game's source or worker was changed by this
campaign.
