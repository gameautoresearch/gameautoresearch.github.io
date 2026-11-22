# Polytopia checkpoint Elo

Every network a campaign produced in one tournament: each session's sealed research networks (its `package` jobs),
its fresh final, the anchors (Elo 0) and references from the release's host-private panel. The result is Elo
against research GPU-hours per model (`curves.png`, `checkpoints.html`). Games are played by the release's frozen
CPU grader (`/opt/v1/grade.py`, each network in its own unprivileged `/opt/v1/policy.py`), exactly as
`Provider.evaluate` plays them; `job.py` is one call (one pair, `chunk` maps x both seats).

First used for `polytopia-empty15-20261118` (reports/polytopia/empty15-2026-11-18.md).

## Run

```sh
# 1. a config beside the outputs (see config.py), then the entrants (pull cloud sessions first)
PYTHONPATH=$PWD .venv-polytopia-ar/bin/python -m tools.tournament.polytopia_elo.entrants --config DIR/config.json
# 2. plan, then play (sparse schedule on Modal Functions by default)
PYTHONPATH=$PWD MODAL_PROFILE=duper .venv-polytopia-ar/bin/python -m tools.tournament.polytopia_elo.run \
    --config DIR/config.json --dry-run
PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_ENVIRONMENT=game-autoresearch-project caffeinate -dimsu \
    .venv-polytopia-ar/bin/python -m tools.tournament.polytopia_elo.run --config DIR/config.json --hours 1.5 \
    --report-python .venv-pokemon/bin/python
# 3. the interactive page (report.py already ran at the end of run.py)
PYTHONPATH=$PWD python3 -m tools.tournament.polytopia_elo.make_html --config DIR/config.json
```

`run.py` resumes: a (block, pair) with a complete job line is never played again, so a run that stops (deadline,
lost connection) continues where it was, and new entrants (late finals) can be added by re-running `entrants.py`
and `run.py`.

**Adding a later campaign's networks to a finished tournament** (first used for `polytopia-empty15-cont-20261118`):
copy the finished run's `games.jsonl`, `jobs.jsonl` and `private/schedule.json` into a new folder, list its sessions
in the new config under the same keys plus the new sessions, and run as above: only pairs involving a new network are
played, and one fit covers all games. A session that continues an earlier one (harness `--continue-from`) is named in
`"continues": {"astra+": "astra"}`; its networks are placed after the earlier session's GPU-hours and session hours and
drawn in its model's colour.

## Schedule (`--schedule sparse`, the default)

A **block** is one call: a pair on `--chunk` maps x both seats; block b uses the same maps for every pair.

- **Stage A:** every entrant plays every anchor and reference and `--partners` random entrants (block 0).
- **Stage B:** a provisional Bradley-Terry fit, then every entrant plays its `--neighbours` nearest entrants by
  provisional Elo, up to `--blocks` blocks per pair.

`--schedule full` plays every pair for `--blocks` blocks (the round robin of the first campaign).

Checked offline against that full round robin (753,600 games, 154 entrants; `schedule_sim.py` subsamples its blocks):

| Schedule | Games | Mean abs. error, networks above −800 Elo | Max |
|---|---:|---:|---:|
| sparse, 4 partners, 24 neighbours, 4 blocks (default) | 21 % | 11 Elo | 65 |
| sparse, 4 partners, 16 neighbours, 4 blocks | 15 % | 18 | 58 |
| every pair, 3 of its 4 blocks | 75 % | 16 | 54 |
| every pair, 2 of its 4 blocks | 50 % | 24 | 81 |

The full run's own map sampling (32 maps) moves Elo by 16–27 on average when one block is left out, so the default is
within the reference's own uncertainty, at a fifth of the games. Rank correlation stays above 0.998. Networks far
below everyone (a learner that made play worse than random) have larger absolute errors; their order is unaffected.

## Backends and images

| | Modal (`--backend modal`, default) | Daytona (`--backend daytona`) |
|---|---|---|
| Unit | one 1-core Function container per call; billed max(request, use) | `--workers` sandboxes (2 vCPU, 3 GiB), one call at a time each; billed for the whole allocation |
| Image | `--image cpu` (default): Ubuntu, `/opt/venv` with numpy, CPU JAX 0.11.2 and the release's Rust engine wheel, plus the release's runtime, grader and maps; 2 GiB request. `--image release`: the release's CUDA worker image, 4 GiB | the same CPU image as a Daytona snapshot, one per release (built once, ~6 min) |
| Entrants | baked into the image | a per-tournament volume at `/bundles` |
| Parallelism | up to `--max-containers` (600) | quota-bound: 250 vCPU / 500 GiB in Curiosity Research Inc. → up to ~120 workers |
| Cost per 1,000 games | ≈ $0.06 (CPU image); ≈ $0.08 measured with the release image (754k games, $60.92 billed) | ≈ $0.10 at list, paid from organization credits |
| Isolation | `block_network=True`, project environment only | `network_block_all`, ephemeral, auto-stop off (it fires even while a process runs), TTL backstop, deleted at the end |

**Parity.** The same 3 pairs x 8 maps x both seats (48 games, same host-secret seeds) on the release image, the CPU
image on Modal and the CPU snapshot on Daytona: all 48 games identical in result, final turn, step count and scores
(2026-11-18, `runs/local/polytopia-elo-tooltest-20261118/parity/`). Summed resident memory of a call: 8.5 GB with the
release image (its CUDA JAX build is loaded by the grader and both policy processes even on CPU), 1.7 GB with the CPU
image on Modal, 1.3 GB on Daytona; call time the same (~3.3 s a game, ~1.4 cores).

**What a tournament like polytopia-empty15 (154 entrants) costs:** full round robin on the release image $61 (billed);
sparse schedule (21 % of the games) on the Modal CPU image ≈ $9 in ~10–15 min; on Daytona ≈ $16 of credits in ~1 h
with 120 workers.

Daytona needs its SDK: `uv venv .venv-daytona --python 3.12 && uv pip install --python .venv-daytona/bin/python
daytona==0.232.0 numpy`, and the Daytona CLI logged in (`daytona login`) with the paying organization active
(`daytona organization use <id>`). Run `run.py` with `.venv-daytona/bin/python`. The build-log stream of a snapshot is
refused for CLI logins (403); the backend polls the build instead. The private map seeds go to the workers inside each
call; the entrant networks to the volume (deleted at the end).

## Files

| File | |
|---|---|
| `config.py` | the config format |
| `entrants.py` | the entrant list and per-network metadata (GPU-hours and session hours at sealing, practice, submissions) |
| `schedule.py` | sparse / full schedules and the Bradley-Terry fit |
| `run.py` | the driver: plan, play, stream `games.jsonl` / `jobs.jsonl` / `status.json`, report |
| `job.py` | one call through the frozen grader (shared by both backends) |
| `modal_backend.py`, `daytona_backend.py`, `daytona_worker.py` | the two backends |
| `report.py` | `elo.json`, `curves.png`, `REPORT.md` |
| `make_html.py`, `checkpoints.template.html` | the interactive page (GPU-hours or session hours, linear or log x) |
| `schedule_sim.py` | offline check of schedules against a finished full round robin |
