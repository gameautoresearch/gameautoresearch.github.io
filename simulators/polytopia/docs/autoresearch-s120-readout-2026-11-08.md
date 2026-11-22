# Autoresearch s120 batch: readout, 2026-11-08

Four agent sessions in parallel, each with **120 H100 GPU-minutes** to improve the starter Polytopia PPO recipe
(Terminus-2 via Harbor, driver on Modal, 600k policy-param cap, practice opponents random / starter@5M/10M/20M).
Batch `1107-2204`, launched 22:04 local Nov 7, last session ended 01:00 Nov 8. Plan:
[autoresearch-sample-run-plan-2026-11-07.md](autoresearch-sample-run-plan-2026-11-07.md). Code: `autoresearch/` on
`claude/autoresearch` (uncommitted).

| Session | Model | Route |
|---|---|---|
| `s120-dsflash-1107-2204` | DeepSeek V4 Flash (same model as the 30-min run `smoke-1107-2035`) | OpenRouter |
| `s120-dspro-1107-2204` | DeepSeek V4 Pro | OpenRouter |
| `s120-kimi-1107-2204` | Kimi K3.1 Pro (`moonshotai/kimi-k3.1-pro`, newest Kimi on OpenRouter) | OpenRouter |
| `s120-sol-1107-2204` | gpt-6.1-sol, reasoning effort high | Mac CLI proxy via `proxy_gateway.py` + tunnel + `model_relay.py` |

## Result

Combined round robin (13 players, every pair 32 games on the 16 hidden evaluation maps, Bradley-Terry Elo anchored
at random). Normalized: 0 = starter@25M, 1 = reference@100M (our recipe on 4x the data).

| Player | Elo | 95% CI | Normalized |
|---|---|---|---|
| c3 (older strong bot) | 996 | 895-1126 | 2.64 |
| **DeepSeek V4 Pro final** | **978** | 911-1058 | **2.52** |
| DeepSeek V4 Flash, orphaned 70-min run (never submitted) | 879 | 788-979 | 1.85 |
| **Sol final** | **793** | 733-863 | **1.27** |
| **Kimi final** | **780** | 717-852 | **1.18** |
| reference@100M | 754 | 707-807 | 1.00 |
| starter@100M | 701 | 652-755 | 0.65 |
| earlier 30-min run (Flash, `smoke-1107-2035`) | 677 | 628-735 | 0.48 |
| **DeepSeek V4 Flash final** | **618** | 572-672 | **0.08** |
| starter@25M | 606 | 546-670 | 0.00 |
| reference@25M | 592 | 544-635 | -0.10 |

Head-to-head (32 games): DS Pro final beats reference@100M 0.81, Kimi 0.81, Sol 0.59; loses to c3 0.28.
Sol vs Kimi 0.56. Sol and Kimi each 0.50 vs reference@100M.

Grading points (per-session fits; end point = last submission, see bug 3):

| Session | 1/3 | 2/3 | End |
|---|---|---|---|
| DS Flash | 662 (0.11) | (not recorded: session ended at 51.5 GPU-min) | 662 (0.11) |
| DS Pro | 529 (-0.99) | 529 (-0.99) | 1070 (2.65) |
| Kimi | 654 (0.25) | 554 (-0.36) | 813 (1.22) |
| Sol | 632 (0.37) | 690 (0.79) | 716 (0.98) |

## What each agent did

| | Runs | GPU-min used | Agent steps | Wall clock | Tokens in (cached) / out | Token cost |
|---|---|---|---|---|---|---|
| DS Flash | 4 (1 orphaned) | 51.5 of 120 | 232 | 1 h 05 | 17.0M (98%) / 100k | $0.75 |
| DS Pro | 4 | 118.6 | 224 | 2 h 20 | 15.6M (99%) / 143k | $4.16 |
| Kimi | 6 | 120.0 | 185 | 2 h 44 | 9.5M (88%) / 37k | $3.88 |
| Sol | 5 | 120.1 | 59 | 2 h 55 | 1.6M (89%) / 16k | $0 (subscription) |

No context summarization happened in any session.

- **DS Pro: one big bet, and it won.** Baseline (0.22 vs starter-20m), a 7x48 net that collapsed (0.00), a 4x40 stopped
  after 2 min, then a single 110-min run of the starter's 2x32 net (cap-as-draw, 30% historical pool, entropy 0.02,
  LR decay). The small net trains at ~106k samples/s, so the run saw 352M samples; practice 0.97 vs starter-20m.
  Under a GPU-time budget, "smallest net, longest run" buys the most samples, and our reference anchors (25M / 100M
  samples) are too weak to measure that. No iteration after the bet: zero feedback for 110 of 120 minutes.
- **Sol: steady, hypothesis-driven iteration.** Pooled 3x48 pilot 0.47 -> 2 PPO epochs / LR 4e-4 / entropy 0.001
  0.72 -> attention block 0.59 (dropped) -> same recipe 28 min 0.72 -> 41-min final. Clean notebook; fixed an
  attention head-size error itself. Fewest turns (long `ar wait`s), improving at every grading point.
- **Kimi: careful one-variable ablations.** Gated-pooling 4x48 0.56 -> 3 epochs 0.38 -> entropy 0.003 0.44 ->
  starter's capped-game reward 0.66 -> 60-min scale-up 0.75 (submitted) -> 15-min score-at-cap reserve run.
  Practice and hidden Elo disagreed at the 2/3 point (e4: better practice, worse hidden).
- **DS Flash: quit early on a budget misread.** 2x56 net 0.72 after 10 min, attention variant 0.25, then a 70-min
  final. 36 min into it, it decided the run might "exhaust the budget before evaluation" (it had 64 GPU-min left)
  and called `task_complete` twice. The orphaned run kept training and would have scored 879 (1.85).

## Harness bugs found (all fixed in source; running sessions were repaired by hand)

1. **Counter file stale tail.** `train_entry.py` rewrote `ar_counter.json` in place on the Modal volume; a shorter
   final write left `ing"}` behind and the broker's `json.loads` failed every 6 s for 14 min (Sol blocked, no GPU
   charged). Repaired by rewriting the file; fixed with temp-file + `os.replace` and a tolerant broker parser.
2. **Broker stdin buffer.** The broker pushed each output file through one `sb.exec` stdin write; DS Pro's 110-min
   run had a 17.9 MB `training.jsonl` -> `BufferError` forever. Repaired by thinning the log on the volume (full copy
   kept as `training.full.jsonl`). Fixed: `_sb_run` now writes stdin in 1 MiB pieces with `drain()` (not yet exercised
   on a live session).
3. **End grading point frozen too early.** The 1.0 point was recorded when the budget-spending run finished, before
   the agent could evaluate and submit it (instruction says "at the end"). It hit Sol (graded its previous run) and
   DS Pro (graded its 2-min baseline: 529 instead of 1070). Grader now uses the last submission (keeps
   `recorded_submission`); harness now records the end point only at `_broker finalize`.
4. **Agent can end a session with a run in flight** (DS Flash). Proposed: refuse `task_complete` while a run is active,
   or auto-submit a finished run's checkpoint when it beats the current submission on practice.
5. Grader read an empty `grading_points.json` (artifact copy saved before the verifier) -> fixed to prefer the
   verifier copy.
6. DS Pro's terminal ended mid-evaluation: the agent had sent Ctrl-D while `ar wait` was blocking; the EOF sat in the
   tty buffer behind its queued `ar submit` lines and exited the shell when the wait returned, which killed tmux and the
   trial (found Nov 8 when a smoke session did the same). Fixed: the box's shells set `IGNOREEOF`, and the instruction
   says never to send Ctrl-D or `exit`.

The `task/environment` copy of the harness must be re-staged (`python autoresearch/build_context.py`) before the
next launch so boxes get fixes 1 and 3.

## Proxy path (Sol)

61 calls through Cloudflare tunnel -> gateway -> CLIProxyAPI, all eventually 200 (3 upstream 500s retried by the
relay), p50 12 s, max 26 s, no dropped streams in the session. In pre-launch tests a >100 s call passed the tunnel
via keepalive; two streams were cut mid-call (76 s, 145 s), which led to request-ID re-attach in the gateway.
Gateway stopped and keys revoked after the batch.

## Cost (rough, Modal list prices)

H100: 4 x ~120 GPU-min + orphaned 70 min + idle compile/overhead ~= 9-10 GPU-h ~= $40. Agent CPU boxes and drivers
~$5. Grading + round robin CPU ~$10. Tokens $8.79 (OpenRouter) + subscription. **Total ~ $60-65.**

## Takeaways for the full benchmark

- The anchors must scale with the budget: DS Pro's small-net long run lands near c3. Add reference runs at the
  session's full sample budget, or budget in samples (as the speedrun plan does) so "smallest net" stops buying data.
- n = 1 per model: no ranking claims. DS Pro's win is one high-variance bet; Sol and Kimi are within each other's CIs.
- Practice (32 games vs 4 fixed bots) mis-ranked recipes at least once; more practice maps or a stronger practice
  opponent would give the agents better feedback.
