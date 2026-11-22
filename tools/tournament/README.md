# Head-to-head tournament

The final verifier for research campaigns: the networks the research sessions produce for one game play
each other, and the pairwise results form the leaderboard. It replaces the host-private grading panels
as the campaign score (owner decision, 2026-11-11). Host-only; no language model is involved.

## Every campaign ends with one

A campaign is not finished until its networks have played each other (owner decision, 2026-11-14): the
private panels saturate, and the head-to-head result is what separates the top entrants.
`tools/nanny.py` does this by itself. When every session of a campaign is done and graded, the nanny box
runs each game's tournament (`TOURNAMENTS` in `tools/nanny.py`) between the sessions that submitted a
network and writes it to `tournament/<game>/` in the campaign folder. A campaign run any other way must
run the game's tournament by hand before it is reported or archived.

## Entrants

An entrant is a finished session directory:

- a **recipe** session (`.../<seat>/main`) whose fresh final training completed (`final-train.json`
  status `complete`); its network is the sealed bundle that training produced under `<session>/final/`;
- a **network** session that is finalized with a submitted network; its network is the last submission
  in `session.json`, the sealed bundle its final grade scored (Hex, Generals, Dou Dizhu:
  `research/jobs/<run>/bundle`; Polytopia: `research/submissions/<sha256>`).

All entrants of a game must come from sessions bound to the **same release** (their bundles are sealed
against that release's frozen engine). The release path is in `<session>.harness-run.json` →
`settings.release`.

Hex has its own tool, `tools/hex_tournament.py` (`--session NAME=SESSION_DIR`): one worker plays every
pair in lockstep and can add calibrated panel opponents as references, so its Elo is on the panel's scale.

## Checkpoint Elo (Polytopia)

`tools/tournament/polytopia_elo/` scores every network a campaign produced (research checkpoints, finals, anchors,
references) in one tournament and plots Elo against research GPU-hours. Its default sparse schedule (anchors and
references, random partners, then each network's nearest neighbours) plays about a fifth of a full round robin's
games for the same accuracy, on 1-core Modal Functions or on Daytona CPU sandboxes. See its README.

## One module per game

`tools/tournament/<game>.py`, run with that game's research interpreter from the repository root:

```sh
PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_DISABLE_API_PROXY=1 \
  .venv-<game>-ar/bin/python -m tools.tournament.<game> \
  --entrant NAME=SESSION_DIR [--entrant NAME=SESSION_DIR ...] --out NEW_DIR [--smoke]
```

It must:

1. Resolve and verify each entrant's final bundle with the game's own bundle verifier, and refuse
   entrants from different releases.
2. Play **every unordered pair** of entrants on the release's private evaluation set (maps / deals /
   seeds), **both seats or all roles**, through the game's existing worker evaluator, on Modal workers in
   environment `game-autoresearch-project` (CPU workers where the existing evaluator is CPU-only).
   `--smoke` uses a tiny slice (2–4 maps/deals/seeds) for testing.
3. Treat every entrant's `policy.py` as untrusted code (it was written by a research agent): run it the
   way the evaluator runs a *candidate* (isolated process), never in-process as a trusted panel bot.
4. Write `NEW_DIR/result.json`:
   ```json
   {"game": "...", "release": "...", "release_sha256": "...", "smoke": false,
    "entrants": {"NAME": {"session": "...", "bundle": "...", "bundle_sha256": "..."}},
    "pairs": [{"a": "NAME", "b": "NAME", "games": 0, "a_points": 0.0, "a_score_0_100": 0.0,
               "ci95": [0, 0], "faults": 0, "truncations": 0, "details": "pairs/<a>-vs-<b>/"}],
    "leaderboard": [{"name": "NAME", "score_0_100": 0.0, "games": 0}],
    "worker_seconds": 0.0, "started_at": 0, "finished_at": 0}
   ```
   Points: win 1, draw/tie 0.5, loss 0 (or the game's own placement-based points, stated in the file).
   `leaderboard` = mean score over all of an entrant's games. Keep raw per-game rows under `details`.
5. Never write into a release or a session directory; output only under `--out` (must not exist).

Allowed: new files under `tools/tournament/` and `tools/tests/`. **Forbidden while the campaign
runs:** editing anything under `harness/` (except `harness/tests/`) or `simulators/*/autoresearch/`
— see `evidence/six-game-5h-20261111/FREEZE.md`. To change worker-side behaviour, upload a small script
from `tools/tournament/` into the sandbox and run it there, importing the image's existing modules.
