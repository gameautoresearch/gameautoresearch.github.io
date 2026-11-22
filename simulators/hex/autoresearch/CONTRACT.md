# Hex 13x13 research contract (`hex13-no-swap-research-v1`)

The machine form is `benchmark.json`; session defaults are `DEFAULT` in `contract.py`.

## Task

13x13 Hex without the swap rule: black moves first and joins top to bottom, white joins left to right, no
passing, no draws. The method (seat-paired openings, a MoHex anchor at 0 Elo) is that of Jones, *Scaling
Scaling Laws with Board Games* (arXiv:2104.03113). The board is larger than the paper's largest, 9x9: on
9x9 the six-researcher suite of 2026-11-13 reached the anchor, which is near perfect play there, so the
score had no room left (`reports/suite-2026-11-13.md`). The 9x9 form (`hex9-no-swap-research-v1`) is in
git history at commit `48f226ee`; its releases, sessions and grades stay as they are and are not
comparable with this contract.

The frozen engine is `simulators/hex/engine.py`, a batched PyTorch implementation
checked against the independent breadth-first reference `simulators/hex/reference.py` on full random
games and on every prefix. The researcher edits only `training/policy.py`, `learner.py`, `train.py` and
the two configs; every run starts from random weights.

## Resources and budget

The researcher trains on workers it leases (there is no `ar run`; `training/launch.py` does the plain case on
one L40S with 4 CPU cores and 16 GiB). The host's fixed training run, used for a recipe's final training, has
that same shape. Every session runs in one of two research budget modes (`harness/budget.py`), chosen when it
starts and recorded in its ledger (`budget_mode`); a session never continues in the other mode:

- **seconds** (the default): **7,200 seconds of the training worker** (`budget_seconds`; smoke 600), one
  job at a time (a practice or a lease), with a wall-clock limit of twice the allocation plus an hour
  (5 hours). A worker of another shape is charged its seconds x its list price relative to the training
  worker (`seconds_weight`), so one second of the training worker is always one second of allocation.
- **usd**: **$10.00 of worker time** (`budget_usd`; smoke $0.50), up to 4 jobs at once, each worker
  second charged at the frozen prices below, with a 5-hour wall-clock limit.

A recipe's fresh final training is 3,600 seconds of the training worker in either mode. Everything
allocated counts: setup, simulation, learning, export, failures, idle time and teardown. Queued and
running work holds its whole grant until it ends.

Session contract `hex13-autoresearch-v1` keeps the leases and both budget modes of the 9x9 contract
`hex9-autoresearch-v3`:

- **Leases.** `ar lease` holds a worker of a chosen shape (0 or 1 L40S, 1-32 CPU cores, 2-64 GiB) on
  which the researcher runs their own commands as an unprivileged user, with no network. All leases of a
  session mount one scratch Volume at `/scratch`; training runs and fresh final training do not.
- **Concurrency.** Seconds mode: one active job. Dollar mode: up to 4 active jobs, at most 2 with a GPU
  and 64 CPU cores in total.
- **Prices.** In dollar mode a worker second is charged `cost(shape)` in US dollars, from Modal's sandbox list prices
  read on 2026-11-13 and frozen in `contract.py` `PRICE`: $0.000542 per GPU second, $0.00003942 per CPU
  core second, $0.00000667 per GiB second. The training worker is about $2.90 an hour, so $10.00 is about
  3.4 hours of it, which one worker can spend within the 5-hour session; workers running side by side
  spend it proportionally faster. Public practice runs on an 8-CPU, 16 GiB worker without a GPU (MoHex is
  CPU-only) at about $1.52 an hour, 1,800 seconds by default. These prices are the contract's unit of account; Modal's invoice can
  differ, and the researcher's box, private grading and tournaments are outside the budget. Seconds
  mode uses the same prices only as ratios: the practice worker is charged about 0.52 seconds per second.
- **Earlier sessions.** A ledger opened under the v2 contract (`gpu_minutes`) is refused by this code:
  it cannot be continued or graded here.
- **MoHex is evaluation-only.** The research image (training runs and leases) contains neither the MoHex
  source nor a binary. The evaluation image compiles MoHex readable by root only, so the unprivileged
  policy process cannot run it.
- **Provenance.** A network packaged from a lease is sealed by the same root-side export as a training
  run. Weights may descend from earlier work in the same session through `/scratch`, never from outside.

Private grading runs on a separate host allocation of the practice shape and is not charged to the
researcher.

## Evaluation

- **Openings.** Every game starts from a 2-ply opening, a black stone and then a white stone, drawn
  without replacement from all 169 x 168 = 28,392. This follows boardlaw's reference evaluation, which
  played all 2-ply openings MoHex against MoHex.
- **Seats.** Every opening is played twice, the candidate as black and as white, against every opponent.
- **Opponents**: nine, equally weighted, 180 to 420 Elo apart from random play to the anchor, so the
  score rises with skill over the whole range. They are built from three players: uniformly random,
  a one-ply shortest-path heuristic (`simulators/hex/heuristic.py`) and MoHex 2.0 (benzene-vanilla-cmake,
  vendored at `vendor/engines/benzene`). Some are diluted by `epsilon`, the chance of a uniformly random
  move instead (the strength dial of boardlaw's MoHex wrapper). MoHex strength is set by simulated games
  per move rather than time, so it does not depend on machine load. Ratings are relative to the anchor,
  fitted by `ladder.py` from opponent-vs-opponent games on 13x13
  (`reports/hex/panel-calibration-13x13-2026-11-14.json`):

| Opponent | Player | epsilon | Elo |
|---|---|---|---|
| `random` | random | | -2580 |
| `heuristic-e90` | heuristic | 0.9 | -2164 |
| `heuristic-e80` | heuristic | 0.8 | -1852 |
| `heuristic-e60` | heuristic | 0.6 | -1486 |
| `heuristic` | heuristic | | -1155 |
| `mohex-100-e15` | MoHex, 100 games, no pre-search | 0.15 | -917 |
| `mohex-100` | MoHex, 100 games, no pre-search | | -498 |
| `mohex-2k` | MoHex, 2,000 games | | -177 |
| `mohex-anchor` | MoHex, 50,000 games, parallel DFPN solver, knowledge threshold 0 (boardlaw's `MoHexAgent()` settings, with a game count in place of MoHex's default 10-second clock) | | 0 |

- **Score.** 100 x mean over opponents of the candidate's win rate across both seats. A 95% interval
  comes from 2,000 bootstrap replicates clustered by opening. The top of the scale is 100: the anchor is
  not perfect play, so nothing caps the win rate against it.
- **Elo against the anchor.** `elo_vs_anchor` is 400 x log10(p / (1 - p)) for win rate p against
  `mohex-anchor` alone. 50% is 0 Elo; a candidate stronger than the anchor is above 0.
- **Candidate Elo.** `panel_elo` is the maximum-likelihood rating against the nine calibrated
  ratings, with an opening-bootstrap interval; at the ends of the scale it is reported as a bound. The grade also reports how
  many openings the candidate won from both seats, one seat and neither.
- **Practice** uses 16 fixed public openings (288 games); **private grading** uses 64 fresh openings from
  a host-only seed (1,152 games) with the same opponents.
- **Inference is capped per move.** The candidate runs in an unprivileged child process with one CPU
  thread. It receives public arrays only. Loading the model has 60 seconds and is not on the move clock.
  After that a call covering n positions (n at most 64) must answer within
  `call_seconds + n x position_seconds` = 0.1 + 0.01 n seconds of wall time: 10 ms per position, plus a
  tenth of a second per call so that a small call is not decided by scheduling noise. That is 0.74 s for
  64 positions and 0.11 s for one, against a flat 4 s per call in the 9x9 contract. On one laptop core a
  64-position forward pass took about 0.1 s for a 48-channel, 4-block residual network, 0.3 s for 64
  channels and 6 blocks, and 0.9 s for 1.3 to 3 million parameters, so the allowance is one pass of a
  network of that size or a few passes of a smaller one; deep search at test time, which carried the top
  9x9 entries, does not fit. A call answered late is replaced by a uniformly random legal move
  in every game of that batch, the policy process is restarted, and play continues. Practice, private
  grading and the tournament use the same clock and report `move_timeouts` (late calls) and
  `random_moves` (moves substituted). The clock is wall time on the evaluation worker, so it is only as
  steady as that worker.
- **Faults.** An illegal or malformed answer, or a model that does not load within 60 seconds, marks
  the policy invalid (no score, not losses). A
  crashed MoHex process is replaced and the same move asked again, at most twice, and counted in
  `mohex_restarts`; a MoHex fault beyond that, or an incomplete schedule, yields no grade. Natural
  results are the only results: no game can be truncated or drawn.
- **Tournament.** When a campaign's sessions are graded, their final networks play a round robin with the
  five strongest panel opponents as references (`tools/hex_tournament.py`, started by `tools/nanny.py`).
  It is reported beside the grades and separates entrants the panel no longer can.

## Submissions and points

As in the other games, the latest explicitly submitted completed run is graded at one third, two thirds
and the end of the allowance. Grades are the raw panel score: there is no starter/reference
normalization, because the MoHex anchor already gives a fixed scale.

## What this contract does not claim

MoHex at 50,000 games is a strong reference on 13x13, not perfect play, and more games buy little: MoHex
at 10,000 games won 14 of 32 games against it. A candidate can exceed 50% against it. The panel ratings
come from 32 to 96 games per pairing and carry about 50 Elo of uncertainty each. The move clock's two
numbers were chosen from forward-pass timings on a laptop, not from sessions or a Modal worker; the practice grant
(1,800 s by default) is likewise unmeasured on a Modal worker. The anchor's parallel solver can make its
choices depend on thread timing. No Modal session, worker image build or paid run has used this contract
yet. See `task/game_rules/scope-and-limitations.md`.
