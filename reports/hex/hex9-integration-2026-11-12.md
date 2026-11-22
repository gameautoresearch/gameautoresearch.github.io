# Hex 9x9 integration — 2026-11-12

Local, offline work. No Modal resource, worker, image build, GPU allocation or model call was made,
and nothing here authorizes spending.

## What was added

Hex is the seventh harness game, modelled on the largest board of Andy Jones, *Scaling Scaling Laws
with Board Games* (arXiv:2104.03113): 9x9, no swap rule, every evaluation pairing played from both
seats, and MoHex 2.0 as the reference that the paper fixes at 0 Elo.

| Part | Where |
|---|---|
| Engine: batched PyTorch, GPU and CPU; BFS reference; shortest-path heuristic; MoHex GTP client and build | `simulators/hex/` |
| Contract, starter, recipes, brief, rulebook, in-box dry run, release constructor, ledger, bundles, arena, grading | `simulators/hex/autoresearch/` |
| Harness backend: `SPEC`, `Backend`, controller, Modal provider | `harness/games/hex/` |
| MoHex 2.0 source (benzene-vanilla-cmake `95614769`), offline git metadata | `vendor/engines/benzene`, `vendor/upstream-git/benzene.tar.gz` |
| Tests | `simulators/hex/tests`, `simulators/hex/autoresearch/tests`, `harness/tests/test_hex.py` |
| Registration | `harness/workspace.py`, `simulators/catalog.json`, `integrations/hex`, tools, navigation docs |

`.venv-hex` (Python 3.12.9, torch 2.15.0, numpy 2.4.2, modal 1.5.2) is both the game and the research
interpreter. Workers install `simulators/hex/requirements-universal-lock.txt` (PyPI torch 2.15.0 with
CUDA 13, as the Pokémon workers already run on Modal) on `nvidia/cuda:13.0.2-devel-ubuntu24.04`.

## Design

- **Engine.** Every stone carries "touches my first edge" and "touches my second edge" flags, as
  boardlaw's CUDA kernel labels groups. A move wins exactly when the new stone's group has both, so the
  win test is a six-neighbour gather; a flag spreads through the mover's group only when the new stone
  gained it. Checked against an independent BFS on 1,200 complete random games, after every move of 60
  more, and against MoHex's own `final_score`.
- **Openings and seats.** Boardlaw's reference evaluation played MoHex against itself from every 2-ply
  opening. Here every game starts from a 2-ply opening drawn without replacement from all 6,480, and is
  played from both seats. Without the swap rule each opening is a theoretical win for one side, so a
  perfect player wins 50% against a perfect anchor: the paper's 0 Elo.
- **Panel**: nine equally weighted opponents spaced about 250 Elo apart from random play to the
  anchor, so the score rises roughly evenly with skill (see "Panel calibration" below). They are built
  from random play, a one-ply shortest-path heuristic and MoHex, some diluted by `epsilon`, the chance of
  a random move instead (boardlaw's MoHex wrapper had the same dial). The anchor is `mohex-anchor`:
  parallel DFPN solver, knowledge threshold 0 (boardlaw's `MoHexAgent()` settings), 200,000 games per
  move instead of MoHex's default 10-second clock, so its strength does not depend on machine load. A
  perfect player's panel ceiling is about 94.
- **Score.** 100 x mean opponent win rate; opening-clustered bootstrap interval. Alongside it,
  `panel_elo`, the candidate's maximum-likelihood Elo against the nine calibrated ratings (anchor = 0),
  and Elo against the anchor alone with the both/one/neither-seat opening counts. No starter/reference normalization:
  the MoHex ladder already gives an absolute scale, so there is no calibration run before official grades.
- **Workers.** Training: L40S, 4 CPU, 16 GiB (the repository default). Practice: 16 public openings, 288
  games, on an 8-CPU, 16 GiB worker without a GPU (MoHex is CPU-only), charged to the allowance at wall
  time. Private grading: 64 openings from a fresh 128-bit host seed, 1,152 games, same shape, not charged.
  The candidate always runs as an unprivileged child where the evaluator can arrange it, practice
  included (Dou Dizhu's practice skips this; Hex does not).
- **Starter.** A 2x256 MLP with policy and value heads over canonical planes (board transposed for white),
  REINFORCE with a learned baseline and entropy bonus, Adam, pure self-play, greedy at evaluation. No
  search, no symmetry augmentation, no replay. The researcher may search at test time inside the 10-second
  per-batch deadline; the engine is importable in the policy process.
- **Dry run.** Because the engine is plain PyTorch, `ar check` runs the researcher's real `train.py`
  against the real engine on CPU in the box for about 45 seconds, then loads the package through
  `create`/`act` and asks it for legal moves in real positions.

## Measurements (Apple M-series laptop, 10 cores)

| What | Result |
|---|---|
| Engine, random play, CPU | 460,000 moves/s at 4,096 games per batch; 117,000 at 64; 12,600 at 1 |
| Starter, CPU, smoke recipe | about 700 self-play games/s; 1,200/s with the full recipe on 2 threads |
| Heuristic vs random | 40 / 40 |
| Heuristic vs MoHex at 1 game | 3 / 32 |
| Heuristic vs `mohex-100` | 0 / 32 |
| `mohex-100` vs `mohex-2k` (24 openings, both seats) | 22.9%; won both seats 0, neither 13 |
| `mohex-2k` vs `mohex-anchor` (24 openings) | 29.2%; won both seats 0, neither 10 |
| MoHex 20k vs 200k games, both with solver (24 openings) | 45.8%; won both 1, neither 3 |
| `mohex-anchor` (200k) vs MoHex 1M games (16 openings) | 43.8%; won both 1, neither 3 |
| Starter after 4 CPU minutes | beats random 96%; 0% against the heuristic and every MoHex level |
| MoHex move cost | `mohex-100` 0.01 s, `mohex-2k` 0.04 s, anchor about 15–22 s of MoHex time per game |
| First panel's practice schedule (160 games, 7 MoHex workers), 4-minute CPU starter | 127 s wall; score 15.6 (random 78%, all others 0%); `elo_vs_anchor` −720, the floor for 32 games |
| Final panel's practice schedule (288 games, 7 workers), same starter | 130 s wall; score 16.0 (random 78%, `heuristic-e90` 59%, `heuristic-e70` 6%, others 0%); `panel_elo` −1800 (95% interval −1869 to −1743); no MoHex restarts |

## Panel calibration

The first panel (random, heuristic, three MoHex levels) had a hole: about 2,000 Elo separate random from
perfect play on 9x9 (the paper's figure), but only random and the heuristic sat below `mohex-100`, at
least 400 Elo apart from each other and from it. Every agent in that range scored the same.

The hole was filled cheaply instead of by a broad round robin with the anchor. One strength dial,
`epsilon`, dilutes an existing player with uniformly random moves. Only the cheap end was calibrated: a
lockstep batch of opponent-vs-opponent games (`simulators/hex/autoresearch/ladder.py`), each player
against its three nearest neighbours on 32 seat-paired openings, in two rounds of 1,536 and 1,728 games
taking 117 and 79 seconds. A Bradley-Terry (Elo) fit of all 3,264 games was pinned to the anchor through
the separately measured `mohex-2k` vs anchor match (29.2%, about -154). The fit recovers known ratings
on simulated data; random play lands at -2032, matching the paper's random-to-perfect gap.

| Player | Elo vs anchor | | Player | Elo vs anchor |
|---|---|---|---|---|
| **random** | **-2032 ± 48** | | heuristic-e10 | -989 ± 24 |
| **heuristic-e90** | **-1786 ± 38** | | **heuristic** | **-943 ± 19** |
| heuristic-e80 | -1602 ± 32 | | mohex-100-e30 | -869 ± 25 |
| **heuristic-e70** | **-1458 ± 29** | | **mohex-100-e15** | **-678 ± 22** |
| heuristic-e60 | -1315 ± 28 | | mohex-100-e10 | -538 ± 30 |
| **heuristic-e50** | **-1226 ± 21** | | **mohex-100-e05** | **-446 ± 26** |
| heuristic-e25 | -1063 ± 26 | | mohex-100 | -285 ± 29 |
| | | | **mohex-2k** | **-154 ± 40** |

Bold rows are the panel, gaps 130 to 330 Elo, plus `mohex-anchor` at 0. The standard errors are within
the fit; the pin adds about ±55 to every rating relative to the anchor. The record, with every pairwise
result, is `reports/hex/panel-calibration-2026-11-12.json`. Rerun `ladder.py` on new seeds to refresh
it, for instance on the worker's CPUs.

While timing practice with this panel, one MoHex process crashed (segfault) with the anchor's
multi-threaded solver running, about once in 300 games; 360 searches replayed in a stress test,
level switching included, did not reproduce it. The pool now replaces a dead process and asks the same
move again (at most twice, counted in `mohex_restarts`); a request that keeps failing is a simulator
fault.

MoHex processes reach about 1.2 GiB resident on macOS regardless of `max_memory`, so the evaluator
keeps one pool of up to seven processes and switches level by re-sending a level's `param_*` lines,
instead of a pool per level.

**The anchor is not perfect play.** In both checks a tenfold or fivefold increase in MoHex search still
won openings from both seats that the weaker setting should have held. The paper's anchor had a 10-second clock with the solver
racing the search, which probably bought more; matching it would cost each grade hours of CPU and make
practice unaffordable within a 2-hour allowance. `elo_vs_anchor` is therefore Elo relative to this fixed
anchor, and the rulebook says a candidate can in principle exceed 50% against it.

## Checks run

- `.venv-hex/bin/python -m pytest -q simulators/hex/tests simulators/hex/autoresearch/tests`: 38 passed,
  1 skipped (CUDA), with a locally built MoHex (MoHex games skip without a binary).
- `.venv-hex/bin/python -m unittest harness.tests.test_hex`: 14 passed, including a whole recipe session
  through the real in-box `ar`, broker, controller and ledger, and provider-receipt recovery.
- `python3 -m unittest harness.tests.test_core`: 15 passed, 1 skipped (it builds every game's fixture).
- `tools/check_benchmark_contracts.py`, `tools/research_tasks.py --check`, `tools/check_layout.py`: pass.
  `tools/audit_research_wikis.py` reports no Hex errors; its overall status remains `incomplete` because
  of two missing Clash provenance files that predate this work.
- `python -m harness prepare / verify / task / run` (plan only) for Hex: a 558-file box with the dry run.

## Not done; gates before any paid or scored work

1. **Worker image.** MoHex has been built only on macOS (clang, Boost 1.87). The image builds it with
   g++ 13 and Boost 1.83 on Ubuntu 24.04; that build, the CUDA 13 torch on L40S, and the engine's GPU
   throughput are untested.
2. **A live smoke** (`harness run hex network --smoke --execute`) exercising training, practice, export,
   submission, private grading and settlement on Modal, with receipts.
3. **Practice grant.** The default is 900 seconds: the 127-second laptop timing above was against a
   weak policy whose games end early. Confirm it on the 8-CPU worker, where each anchor process uses
   three threads, with a strong policy.
4. **Starter at full budget.** How far the starter gets in 3,600 L40S seconds, and whether the ladder
   resolves the range researchers actually reach, is unmeasured.
5. **Anchor strength.** If a near-perfect anchor matters more than cost, a time-based anchor (the
   paper's 10-second clock) can replace `mohex-anchor` in a new contract version.

No run was made, so there is nothing to archive to the private Hugging Face dataset yet.
