# Tavern small form (`tavern-four-small-v1`) — 2026-11-13

Tavern is now a smaller benchmark so a researcher's learner can gain strength inside a 2–4 h budget. The
eight-player form (`tavern-eight-mixed-v2`, artifact ABI `tavern-eight-implemented-v1`) is preserved at git tag
`full-size-20261113`. Its releases, networks and receipts do not bind to the new form. Everything here ran
locally on CPU, except the live Modal smoke
(≈ $0.81, no model calls) described below.

## The form

| | Eight-player form | Small form |
|---|---|---|
| Players | 8 | **4** |
| Regular tavern tiers | 1–6 | **1–4** (`max_regular_tier` 4) |
| Minion types per lobby | 5 | **3** |
| Hero choices, board, hand, recruit actions, start turn, 100-turn guard | 2, 7, 10, 30, 1, yes | unchanged |
| Hero damage cap | while more than 4 players are alive | never applies (at most 4 alive) |
| Observation / actions | 3,227 + 124 cards / 245 | unchanged; the four absent opponents' slots are zero (checked on every observation below) |
| Score | `100*(8-placement)/7`, top-4 rate | `100*(4-placement)/3`, top-half rate |
| Practice | 2 groups × 8 seats × 3 profiles = 48 lobbies | 4 × 4 × 3 = 48 lobbies |
| Private final | 12 × 8 × 3 = 288 lobbies | 24 × 4 × 3 = 288 lobbies |
| Opposition | 3 profiles × 7 refs (retained 8-player bots) | 3 profiles × 3 refs (new pool, below) |

No engine code changed. AutoTavern already supports `num_players` 2–8, `max_regular_tier` and `lobby_races`,
and its local modifications are untouched. The native fingerprint is still `217577b83fc4525e`.

## Game length, before and after

The measurement uses 64 lobbies per row on CPU (seed 7) with `simulators/tavern/autoresearch/game_length.py`.
The learner is seat 0 against the native scripted bots, as in starter training. "Whole lobby" means every seat is
played by a public visible script, as in evaluation.

| | 8 players | 4 players |
|---|---|---|
| Random learner: turns to its own exit, mean (median) | 7.5 (7) | 6.9 (7) |
| Random learner: decisions per episode, mean (p90) | 79 (116) | 69 (92) |
| Tempo-script learner: turns, mean (median) | 13.3 (14) | 12.3 (12) |
| Tempo-script learner: decisions per episode, mean (p90) | 166 (234) | 169 (254) |
| Whole lobby: last turn, mean (p90) | 15.4 (16.9) | 13.7 (15.9) |
| Whole lobby: decisions, all seats, mean (p90) | 1,163 (1,300) | 661 (816) |
| Whole lobby: CPU seconds per lobby (scripts only) | 0.194 | 0.107 |
| Random-learner simulation throughput, decisions/s | 34,258 | 52,157 |

A learner's own episode is about as long as before, because players are knocked out at a similar turn. What
shrinks is the lobby: there are half as many seats and ~43% fewer decisions. Evaluation lobbies therefore cost
about half as much, and training simulates fewer bots per learner decision.

## Does the starter learn faster?

Here the unmodified starter (64 lobbies per update, one learning seat vs scripted bots) was trained for 300 s of
CPU in each form, with the same seed. Evidence: `evidence/tavern-four-small-20261113/game-length/curve-{8,4}.json`.

| Window | 8p mean placement | 8p points | 4p mean placement | 4p points |
|---|---|---|---|---|
| 0–15 s | 7.91 | 1.3 | 3.96 | 1.4 |
| 60–90 s | 6.64 | 19.4 | 3.20 | 26.6 |
| 120–150 s | 5.85 | 30.7 | 2.96 | 34.5 |
| 270–300 s | 4.79 | 45.8 | 2.81 | 39.6 |
| Samples/s | 11,201 | | 16,060 | |

The small form simulates 43% more samples per second, and the starter gains points faster early on. By 300 s the
eight-player curve has caught up. Points against a different number of scripted bots are not on a common scale,
so this does **not** show that the small form is easier to improve in over 2–4 h on the benchmark's own
opposition. That would need a GPU recipe run against each form's evaluator, which was not done.

## New opponents and banks

`evaluation_v2/build_four_pool.py` trained three development PPO opponents on CPU from the public starter in the
four-player rules. Each used 64 lobbies per update and a distinct seed. They join the three public visible
scripts (tempo, economy, synergy).

| Opponent | Seed | CPU time | Updates | Last-update mean placement vs 3 scripted bots |
|---|---|---|---|---|
| ppo-1m | 41101 | 60 s | 91 | 3.98 (≈ random) |
| ppo-5m | 41102 | 300 s | 231 | 3.09 |
| ppo-15m | 41103 | 900 s | 618 | 2.23 |

The profiles are `scripts` = tempo, economy, synergy; `mixed` = ppo-5m, economy, ppo-1m; and `learned` = ppo-15m,
ppo-5m, synergy. The pool and its full training runs (with resume checkpoints) are private local evidence at
`evidence/tavern-four-small-20261113/pool-v1/`. The committed receipt is
`evaluation_v2/selected_registry-four-small-v1.json` (path and manifest hash only). No seeds are published. The
public practice seeds are 50101–50104, and the extended set is 50111–50126. The private final bank is still
drawn per session, only after selection closes (`harness/games/tavern/final_bank.py`, now 24 groups).

## End-to-end CPU smoke

The smoke prepared a release (`python -m harness prepare tavern`, release `evidence/…/release-local-v1`) and then
ran `simulators/tavern/autoresearch/local_smoke.py`. It trained with the trusted `train_entry.py` on the release
starter. It practised on the release's 48 public lobbies through the evaluator, with terminal-state recording and
the independent terminal replay of 3 predeclared lobbies. It then graded on a fresh 24-group private bank (288
lobbies) screened like the final bank. Evidence: `evidence/tavern-four-small-20261113/local-smoke-v2/report.json`.

| Network | Training | Practice (48 lobbies) | Private-bank grade (288 lobbies) |
|---|---|---|---|
| Starter after 1 update | 2.5 s | 2.8 [0.7, 4.9], mean placement 3.92 | — |
| Starter after 120 s CPU (168 updates) | 122 s | **38.2** [29.2, 47.2], mean placement 2.85 | **32.9** [29.6, 36.1], mean placement 3.01 |

Every lobby ended naturally for all four players: 0 faults, 0 illegal actions, 0 rejections. The terminal
replays matched exactly. Per-profile grade: scripts 24.3, mixed 49.0, learned 25.3. Grading the 288 lobbies
took 180 s of local CPU. A first smoke attempt (`local-smoke-v1`) failed before practice because my script
did not copy the run's source into the artifact the way the controller does. It is kept as is.

**After merging `38c791a`** (the `small-tasks` harness commit, which adds `run_seed`/`eval_seconds` SPEC fields
and changes `ar` and the box), `release-local-v1` no longer verifies, as expected. A new release
`evidence/…/release-local-v2` (binding `a050aa8c…`) verifies, and the smoke was repeated on it as
`local-smoke-v3`. The 120 s run reached 274 updates this time because the CPU was not shared. Practice scored
41.7 [31.9, 51.4] and the 288-lobby private-bank grade 38.2 [34.5, 41.7]; the one-update starter again scored
2.8. Every lobby was complete and natural, and the terminal replays were exact. All checks below pass again on
the merged source.

## Checks

- Tavern game suites: `pytest simulators/tavern/tests simulators/tavern/autoresearch/tests`: 76 passed. The
  7 skips need CUDA or Linux, as before.
- `harness.tests.test_tavern`: 20 passed. The release-constructor test now runs against the real pool and
  `.venv-tavern`; it was skipped before. `harness.tests.test_core`: 23 passed. `tools/tests/test_tournament_tavern`:
  8 passed, including a new 4-player seating test.
- `tools/check_local.py --game tavern --game harness --game harness-tavern`: all PASS. `tools/check_layout.py`
  and `tools/check_benchmark_contracts.py` both pass. `tools/audit_research_wikis.py` reports no Tavern errors;
  its 2 Clash errors are pre-existing and caused by generated files missing locally.

## What changed in code

- Rules: `recipes/{starter,smoke}.json` now set `num_players` 4, `lobby_races` 3 and `max_regular_tier` 4.
  `contract.py` gains the new `BENCHMARK_ID`, `PLAYERS` and `points()`, and learning seats are limited to 0–3.
- The evaluator follows the rules' `num_players` (8-player behaviour unchanged): `evaluation_readiness/lobby.py`,
  `screen.py`, `replay.py`, `five_hour/recording.py` and `five_hour/terminal_replay.py`.
- Release: `evaluation_v2/release.py` uses the new evaluator id. Profiles must be N−1 wide, name declared
  references and include a learned one; this replaces the requirement for the old `deepseek-r3` incumbent.
  `rules_enabled/release.py` sets 4 practice seeds and 24 final groups.
- Harness backend: `final_bank.py` draws 24 groups and takes its seat count from the registry. `controller.py`
  takes the result's `benchmark_id` from the release. `__init__.py` changes only its mode text, the receipt name
  and the `top_half_rate` summary key. The main checkout's uncommitted 500k parameter-cap edits touch other lines
  of these files and of the contracts, brief and rulebook, so they should merge cleanly.
- Tournament: A and B now sit at seats k and k+N/2 (k and k+2 at N = 4) with N−2 script seats. A full run is
  still 288 lobbies per pair: 72 groups of 4.
- `grade.py` (the v1 path), contracts, brief, rulebook (and its `wiki_maintenance` copies),
  `docs/benchmark-contracts.md` and the catalog readout pointer.

## Live Modal smoke — 2026-11-14

The user approved one paid `--smoke` network session, relayed from the coordinating session: "Yes, run the
smokes". It was launched after a direct "Yes, launch it" here. It used release `release-local-v2` (prepared
after merging `38c791a`), the `game-autoresearch-project` environment and L40S. The researcher was a script on
the real `ar` path (`tools/live_smoke_check.py`, which reuses `tools/live_stop_check.py`), so there were no model
calls. Records: `evidence/tavern-four-small-20261113/live-smoke-v{1,2,3}`, `diag-v1`.

**Result: passed on the third attempt.** Bootstrap, GPU training, cloud practice, submit, finalize and the
private grade all worked in the four-player form.

| | Training run | Cloud practice (48 lobbies) | Private grade (288 lobbies, fresh 24-group bank) |
|---|---|---|---|
| Attempt 3 (`live-smoke-v3`) | `complete`, 93 updates on L40S (60 s of training), 199.7 s charged | 48/48 natural, raw score 5.6 [2.8, 8.3], mean placement 3.83 | 288/288 natural, raw score 2.8 [1.9, 3.7], mean placement 3.92 |

Both evaluations had 0 faults, 0 illegal actions, 0 rejections and 0 truncations. The candidate ran isolated as
uid 65534 with 3 of 3 private paths denied and no credentials in its environment. Every lobby's terminal state
was retained, and the 3 predeclared terminal replays were exact. The grade's `benchmark_id` is
`tavern-four-small-v1`. The low scores are expected: the smoke trains for only 60 s with 32 lobbies per update.
The 120 s local CPU run above reached 38.

**Two failed attempts, kept.** Neither was caused by the form.

- `live-smoke-v1`: the GPU worker ran to its full 320 s lifetime and the run failed with "Sandbox is
  unavailable" (319.3 s charged). The host's Modal client was going through this machine's HTTP proxy
  (`127.0.0.1:7897`), which was dropping connections: heartbeats timed out about 2 minutes in, and afterwards the
  Modal CLI could not connect at all through it.
- `live-smoke-v2` (with `MODAL_DISABLE_API_PROXY=1`, as the tournament docs do): the training command hit its
  245 s limit with no export to package, and the run failed (307.2 s charged).
- `diag-v1`: one L40S worker (84.8 s) on the same image ran the same trainer with its output streamed. It
  finished 82 updates in 60 s, exported every update and exited 0, on a host with driver 620.70.06. So the trainer
  is fine. The host's single blocking `stdout.read()` never returned on a command that is silent for ~190 s
  (180 s of training), but did on a ~64 s one. Attempt 3 therefore used a 60 s training run.

**Harness follow-ups (shared code, not changed here).**

1. `harness/games/tavern/provider.py` `command()` reads a long silent exec stream in one blocking call. On this
   network such a stream can stall until the command's own timeout, failing a healthy run. Polling the process
   and reading output written to a file would avoid this.
2. Copying every lobby recording back one file at a time outlived both CPU workers. The practice worker
   stopped after 270 files at its 900 s cap and the grade worker at its 1,800 s cap. Both runs recorded
   `retention-blocker.json` (`archive_complete: false`), and the recordings left on the workers are lost. Scored
   results, episodes and terminal replays were downloaded first and are complete.
3. `simulators/tavern/rollout.py` hard-codes `tuple(range(8))` as the controlled seats for `opponent: selfplay`.
   A researcher who switches the four-player starter to self-play gets "seat 4 out of range". Fixing it changes
   `rollout.py`, which is in the source binding, so the opponent pool must be retrained and the release prepared
   again.

**Cost** (provider receipts × Modal list prices: L40S $0.000542/s; 4 cores + 16 GiB $0.0000879/s): 911 s of L40S
across 4 GPU workers (319.3 + 307.2 + 199.7 + 84.8 s) ≈ $0.57, plus 2,699 s of CPU-only workers (practice 899 s,
grade 1,800 s) ≈ $0.24. **Total ≈ $0.81** plus small image builds. No model cost.

## Archive

The original session archive is retained privately.

- the opponent pool, including full training runs with resume checkpoints;
- both releases;
- the game-length and learning-curve measurements;
- local smokes v1–v3, the failed v1 included;
- all three live attempts with their boxes and logs;
- the diagnostic worker and its script.

All 6 uploaded files match their local copies (remote size and SHA-256 or git blob id):
`evidence/archive-tavern-four-small-20261114/verify.json`. The secret-pattern scan found nothing.

Excluded with receipts (39 entries):

- 18 withheld private evaluation files: the private seed panels of the local smokes, the releases'
  qualification panels, and the releases' copies of registry metadata. Identical metadata is archived under
  `pool-v1/registry`.
- 17 lock and transient files.
- 4 raw dependency images.

The live grade's private final bank (`live-smoke-v3/research/grading-private/`) is in this private grader archive,
as the archive policy allows. Missing: the per-lobby recordings that stayed on the two CPU evaluation workers
(follow-up 2).

## Not verified

- No Terminus-2 (model) session and no tournament have run on this form. The live smoke used a scripted
  researcher and a 60 s training run instead of the smoke's largest (180 s; see harness follow-up 1).
- Some card effects were written for six tiers ("Discover a Tier 6 minion", tier-7 unlocks, some Buddies and hero
  powers that hard-code tiers 6–7). They run as the engine implements them against the tier 1–4 pool. No fault,
  rejection or invalid lobby appeared in the opponents' 60,160 training lobbies, the 384 smoke-evaluation lobbies
  or the length measurements, but these cards were not audited one by one.
- The opponents are weak development policies with no strength qualification. ppo-1m is about random.
- The local smoke ran the candidate in-process (trusted local code), not in the isolated process.
- The private pool, release and smoke evidence are archived (above), but locally they exist only in this
  worktree's ignored `evidence/` folder. The receipt path resolves only where that folder exists, so copy it to
  the main checkout's `evidence/` (or restore it from the archive) before `prepare` works there.
- `HANDOFF.md` was not updated, to avoid conflicts with the parallel sessions; add a line when merging.
