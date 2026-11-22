# AverageJoe reference smoke — November 9, 2026

**The actual full AverageJoe model and learning recipe ran successfully on Modal.
This is an engineering pass, not a strong reference bot.** Twenty updates produced
10,485,760 player transitions (5,242,880 joint ticks). Wins increased against
upstream random, but losses increased more: draw-adjusted points fell. The policy
remains unqualified for expert evaluation.

The user authorized a new roughly ten-minute check. All attempts consumed
**563.319 allocated H100 seconds (9m 23s), below the cumulative 600-second cap**.
Both owned workers have finished, independently verified against provider task
metadata. No old campaign allowance was reused. No further paid work was started.

## Recipe and execution

Source: [strakam/AverageJoe](https://github.com/strakam/AverageJoe), pinned at
`5041e635dcf6c87b8b844cb2e244ad3e0abe40d7`, using its unchanged
`configs/custom/L_7d_gae90.yaml`. The vendored source hashes are in
`simulators/generals/averagejoe/upstream.json`; the complete executed source is
snapshotted with each attempt. Upstream has no license file at this pin; the
separate simulator's MIT license is not a license grant for this model source.

The model is the actual **15,351,761-parameter Equinox HistoryTransformer**:
seven blocks, width 448, eight heads, 3×3 patches, BF16 forward with FP32 master
weights, original persistent observations, scoreboard history, action distribution
and categorical value head. It uses 512 environments × 512 rollout ticks,
both player seats, one PPO epoch, the top quarter of absolute normalized
advantages, original GAE/reward/truncation equations, power-law schedules and
EMA decay 0.999. Model and PPO components are imported directly from pinned
upstream source. The frozen 600k-parameter stateless entrant contract is unchanged.

The curriculum is **2–6 → 4–8 → 6–13 → 11–17 → 17–28** spawn distance,
requiring 60% wins in a complete 512-game paired-seat random-opponent evaluation
to advance. This smoke remained at stage 0. Automatic gate checks occur every
50 updates, so GPU promotion was not exercised. Threshold logic passed local tests.

The optimized host framework runs on one H100 with four physical CPU cores and
16 GiB RAM. It removes multi-device replication, keeps training state on device,
fuses update preparation and stops finished evaluation games. The exact native
map generator prepares the initial bank on CPU: 200,000 requested entries become
199,969, balanced across all 49 rectangular shapes. Two CPU preparation attempts
took 146.838 seconds combined and generated identical map bytes. Initial states
are 23×23, as in upstream initialization; resets draw from the mixed 17–23 bank.

An explicit stop at 20 updates was planned before the final assessment to leave
room for export within the original budget. The first cold GPU pool regeneration
at iteration 21 was not exercised. No throughput advantage over the upstream full
transformer was measured. The earlier 6.23× tiny-CNN result does not apply here.

## Assessment

Both assessments use the same 256 public starts in both seats, the same RNG seed,
the current policy with greedy actions and upstream random as opponent. Each
accounts for all 512 games; caps remain separate from natural losses.

| Policy | Wins | Losses | Caps | Win rate | Wins + half caps, /100 |
|---|---:|---:|---:|---:|---:|
| Initialization | 1 | 96 | 415 | 0.20% | 40.72 |
| After 20 updates | 24 | 285 | 203 | 4.69% | 24.51 |

The model changed and received sparse terminal feedback, but these results do
**not** demonstrate overall strength improvement. It is far below the first
curriculum promotion threshold. Training recorded 1,176 natural general captures,
1,733 caps, 2,352 nonzero player rewards, finite metrics and zero illegal sampled
actions. City ownership was nonzero; this does not establish good city strategy.
EMA weights were retained but not assessed separately.

Warm complete updates had a median of 12.272 seconds, or 42,723 player transitions
per second (about 21,362 joint ticks/s). These include rollout and PPO. Full
successful allocation throughput was 19,652 player transitions/s; including the
failed attempt it was 18,614/s. The lower full-allocation figures include setup,
compilation, assessment, saving, transfer and idle time.

## Checkpoints, verification and accounting

Local evidence: `evidence/generals-averagejoe-20261109/`.

- `smoke-02/artifacts/final-state.eqx`: full network, optimizer, EMA, PRNG, game
  states/reset cursors and both observation memories. SHA256
  `9b91a1366f07b4d8e38a0d8147615f7ddeedeb9b25cb670135f9a77157eedefb`.
- `smoke-02/artifacts/final.eqx` and `final-ema.eqx`: standalone model weights.
- Strict source/config/checkpoint admission passed. All 13 exported file hashes
  matched; all 572 full-state array leaves loaded and were finite. Standalone
  current/EMA weights exactly matched their full-state counterparts. The worker
  also verified exact checkpoint tensor readback.
- Four local correctness tests passed, including codec/memory semantics, finite
  learning, exact next-rollout/update restoration on a small CPU fixture and
  complete paired evaluation. A full additional GPU update after restore was not
  run. Stage-0 continuation needs the original cached bank; no resume CLI or
  portable later-stage pool export is implemented yet.
- First attempt: 29.747 allocated seconds, failed because provider start metadata
  was read before it became available. Bounded polling fixed the host launcher.
  The failed allocation remains charged in the new ledger.
- Successful attempt: 533.572 allocated seconds, trainer exit 0, all requested
  final artifacts exported. Provider finished timestamps independently rechecked
  in `cleanup-audit.json`; zero owned active workers or unsettled GPU reservations.
- Estimated GPU plus attached CPU/RAM compute is **$0.767**. Two image builds retain
  a combined $1.00 unpriced reserve. These are estimates/reserves, not a reconciled
  provider invoice. No researcher-model calls were made.

The machine-readable [readout](averagejoe-smoke-2026-11-09.json) retains the exact
assessment counts, allocation receipts, validation and limits. The
[trainer README](../../simulators/generals/averagejoe/README.md) documents the
implementation. Full-distance skill, longer-run stability, GPU curriculum
promotion and refreshed-pool compilation still require a separately authorized
longer experiment. This smoke is not an expert-strength benchmark anchor.
