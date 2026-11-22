# Plan: make the benchmark able to rank models — November 12, 2026

Planning only. Nothing here authorizes spending, launches workers or reopens a campaign; each phase that
spends needs its own go-ahead. Written against `main` at `b1739d7`. The branch
`origin/researcher-feedback-loop` is **not** merged and nothing below depends on it.

## Goal

Be able to say, for a fixed set of models on a fixed set of games: *model X's research improves on the
starter by Δ points (95 % interval), and X is ahead of Y*, with the interval covering every source of
luck. Nothing run so far can say that.

## What is wrong and what fixes it

| Problem | Evidence | Fix | Status on main |
|---|---|---|---|
| No baseline | The starter never played. Clash GLM-5.3's whole submission was `updates 1000000 → 2400`; Clash DeepSeek's was a lighter checkpoint save | Starter anchors in every tournament; score = lift over starter (A, B) | Missing |
| One final seed per recipe | Each submission retrained once; seed noise never measured | Host-only replicate retrain with explicit seed (A); measured in Phase 1 | Missing: `final_training` refuses to run twice and takes no seed |
| One attempt per model | Results flip between runs (DeepSeek Pro best at Polytopia, worst at Pokémon; Opus first, then last, at Pokémon) | 3 attempts per model and game (Phase 2), sized by the Phase 1 gate | Process only |
| Rosters change every run | No two runs share models and games | One fixed roster on two fixed games (Phase 2) | Process only |
| Infrastructure confounds | GLM selection bug; 9 never-started workers; finalize races | Worker queue/settle fixes | **Fixed and live-verified** (`38e14e6`, `d676324`, `b1739d7`) |
| Unequal effort | DeepSeek V4.1 Flash quit after 113–191 of 330 min, using 7–36 % of its GPU | A declared early-finalize rule; allocation used reported beside every score (D) | Missing |
| Capacity starvation | ~12 concurrent L40S wanted; some queued ~50 min | Launch waves under a concurrency cap; every model in every wave (D) | Queue now waited out; no cap |
| Practice signal saturates | Dou Dizhu practice = `random` + `cooperating-shedder` (still true in `releases-20261112`): GLM-5.3 Prime 91.7 vs DeepSeek 87.0 in practice, then lost head-to-head 40–60. Clash entrants score 0 vs standard/teacher; Generals practice ≤ 1.0 | Trained-starter practice opponent (C) | Missing |
| Draw-dominated scoring | Generals: 221/256 tournament games hit the 2048-turn cap at 0.5 | Cap tiebreak (Phase 3) | Missing; Generals stays out of Phase 2 |

A model's result on a game varies for three reasons: the **research attempt** (a new session writes a
different recipe), the **final-training seed** (the same recipe retrained), and the **evaluation games**
(already inside tournament intervals). Phase 1 measures the seed part cheaply; the gate uses it to size
Phase 2, which is the only phase that measures the attempt part.

## Phase 0 — build (code only, no spend)

**A. Replicate retrain.** New host command:

```sh
$PY -m harness retrain GAME --release R (--session S | --starter) --seed N --out NEW_DIR
```

Trains the session's last submitted recipe, or the release's own starter, from scratch on a fresh worker
with the game's full fresh-final allowance, and writes a sealed bundle and a `final-train.json`-shaped
record (recipe sha, seed, allowance, job, budget) into `NEW_DIR`. It reuses `session.final_training`'s
backend path, never writes into a session, and refuses an existing `NEW_DIR`. The seed is set by the host
in every game: through the run seed in Generals and Dou Dizhu, by overriding the config's seed field in
Clash, Polytopia and Tavern, and by making Pokémon's host-chosen final seeds explicit. A recipe that does
not apply to the release fails before any worker starts.
*Done when:* each backend's suite has a stubbed-Modal test that two seeds of one recipe produce two
records with different seeds and identical recipe sha, and that the starter path trains the release's
unmodified workspace.

**B. Tournament anchors and analysis.**
- `tools/tournament/<game>.py` accepts `retrain` directories as entrants, tags every entrant
  `model / attempt / seed` (starter: `starter / - / seed`), and still refuses mixed releases.
- New `tools/tournament/analyze.py` over one or more `result.json`: per model, **lift over starter** (mean
  points against the starter anchors, minus 50) with a hierarchical bootstrap (attempts, then seeds within
  an attempt, then maps/deals/seeds within a pair); secondary Bradley–Terry strengths over all entrants;
  per-seed and per-attempt spreads.
- Pokémon tournaments use 256 seeds instead of 64 (CPU only).
*Done when:* a synthetic `result.json` with known spreads recovers them; a smoke tournament with one
session final and two retrains runs end to end.

**C. Stronger practice opponent (built here, used from Phase 2).** A `prepare` option for Pokémon and Dou
Dizhu, e.g. `practice_bundle=RETRAIN_DIR`, adds one trained-starter network to the public practice panel
and records its bundle hash in the release. It needs a starter retrain, so the Phase 2 releases are
prepared after Phase 1, using a starter seed that is **not** a tournament anchor.

**D. Campaign rules, written into the Phase 2 plan before launch.**
- Early finalize: allowed and reported, or a minimum share of the allowance before `finalize` is accepted
  (owner decision).
- Infrastructure loss: an attempt that loses its submission or final to infrastructure is re-run from
  scratch under a new attempt number; the lost one is kept and reported, never dropped or rescued.
- Concurrency: at most 6 L40S workers at once, in waves, with every model represented in every wave and
  launch order randomized.
- Reporting: allocation used, wall time, model cost and re-runs beside every score (extend
  `collect_stats.py`).

**E. Freeze the analysis** (metric, bootstrap, gate, decision rules below) in a committed plan before
Phase 1 spends anything.

**F. Releases and smoke.** A and B change code identity, so prepare new Pokémon and Dou Dizhu releases
after they land; smoke one `retrain` per game and one tournament with a starter anchor.

## Phase 1 — calibration (≈ 18 L40S-hours, ≈ $45, plus CPU tournaments)

- **Games:** Pokémon and Dou Dizhu, on the Phase 0 releases.
- **Recipes:** the starter and the four existing campaign submissions (Pokémon: Opus 5.5, DeepSeek V4.1
  Flash; Dou Dizhu: GLM-5.3 Prime, DeepSeek V4.1 Flash). Recipes are source/config, so they retrain under
  the new release if they still apply; `retrain` checks that before starting a worker.
- **Seeds:** 3 per recipe → 9 retrains per game, 18 in all, 1 h each.
- **Tournament:** one round-robin per game over all 9 entrants.

Outputs: σ_seed per game (pooled across recipes); lift over starter for each submission; whether the game
separates the starter from trained recipes at all.

**Optional 1b (≈ 12 L40S-hours, ≈ $30, ≈ $2 models):** two more DeepSeek V4.1 Flash research attempts per
game at the Phase 2 research budget, for a first, rough σ_attempt. Flash quits early, so treat it as a lower
bound for stronger models.

**Gate.** From σ_seed (and σ_attempt if 1b ran), compute attempts × seeds needed to detect a 10-point
difference in lift between two models at 95 %.
- Fits in ≤ 4 attempts per model and game → Phase 2 with that design.
- Seed noise alone ≥ the gap → use 2–3 final seeds per attempt, or drop to fewer models.
- A game does not separate starter from trained recipes → that game is not used.

## Phase 2 — the comparison (≈ 150–180 L40S-hours, fixed at the gate)

- **Roster (owner choice), named by served model ID:** e.g. Fable 5.2, GPT-6.1 Sol, Opus 5.5, DeepSeek V4.1
  Flash, GLM-5.3 Prime.
- **Games:** Pokémon and Dou Dizhu; releases from C with the trained-starter practice opponent.
- **Attempts:** 3 independent `recipe` sessions per model and game (or as the gate says); same research
  budget (4 h allocation), brief, practice panel and rules from D.
- **Final seeds:** 1 per attempt, or 2 if the gate says so (via A).
- **Anchors:** the 3 Phase 1 starter retrains play in every tournament (same release required, so if the
  Phase 2 release differs, retrain 3 starter anchors on it).
- **Tournament:** one round-robin per game, scored and analyzed by B.
- **Report:** per model and game, lift over starter with its interval; a two-game ranking as the mean of
  per-game lifts; effort, cost and re-runs beside every number. No claim beyond these two games.

Cost: 5 models × 2 games × 3 attempts × (≤ 4 h research + 1–2 h final) = 150–180 L40S-hours
(≈ $375–450 at ≈ $2.5/h), plus model calls and CPU tournaments.

## Phase 3 — more games, later

Generals returns once the 2048-turn cap is decided by land, then army, then a draw (cap outcomes kept as
their own category, never mixed with captures). Clash, Polytopia and Tavern return once each has a
non-saturating practice opponent. Each game repeats Phase 1 before it joins a comparison.

## Not in this plan

The researcher-side improvements on `origin/researcher-feedback-loop` (Pokémon CPU dry run, live Pokémon
logs, per-game `ar` options, CPU torch in the box) reduce wasted attempts and would lower σ_attempt. They
are a separate decision before Phase 2, not a dependency. If merged, keep the integrity rules the branch
trims from the instructions, and prepare the Phase 2 releases after the merge. Without it,
`ar run --seed` stays accepted but ignored in Pokémon, Clash, Polytopia and Tavern; say so in the brief.

## Every phase

Archive the full retained context to the private dataset (`docs/huggingface-run-archives.md`): new run
path, verified sizes and checksums, recorded in the readout. Keep ledgers and failed attempts. Modal
environment `game-autoresearch-project` only; L40S unless the owner overrides a specific run.

## Owner decisions

1. Early-finalize rule (D).
2. Phase 2 roster and research hours per attempt.
3. Whether to run 1b.
4. Spending go-ahead for Phase 1, separately from Phase 2.
