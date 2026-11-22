# Pokémon benchmark contract and bounded research smoke — 2026-11-08

**The genuine DeepSeek-driven end-to-end smoke passed.** A small agent-authored
policy and PPO learner trained on an actual L4, completed public practice, submitted
an immutable source/config/weights package, and received **66.67/100** on 12 fresh
private games. All private games ended naturally; there were no grading faults or
truncations. This is workflow qualification, not calibrated playing strength.

The machine-readable companion is
[`benchmark-contract-smoke-2026-11-08.json`](benchmark-contract-smoke-2026-11-08.json).
Private receipts remain under `evidence/pokemon-autoresearch/contract-smoke-20261108/`.

## Frozen task and implemented harness

The [contract](../../simulators/pokemon/autoresearch/CONTRACT.md) and
[benchmark configuration](../../simulators/pokemon/autoresearch/benchmark.json)
freeze pinned Showdown `gen9randombattle`: two players, six generated Pokémon per
team, one active per side, switching and Tera, no team building, 14 action IDs and
the existing 700-value player-visible encoding. The horizon is 512 adapter decision
phases, including forced choices/retries. It is not a map dimension or turn count.
The supported simulator remains Showdown CPU; native full-format admission stays closed.

The full research allowance is **7,200 allocated worker seconds**, one L4, four
physical CPU cores and 8 GiB RAM. It was not executed. The new smoke ceiling is
600 cumulative GPU-worker seconds plus $2 charged/reserved model calls. Setup,
simulation, transfer, learning, checkpoints, idle GPU time and teardown count.
No native GPU simulator optimization was restarted.

`autoresearch/` now supplies agent instruction/brief files; a deliberately small
randomly initialized masked PPO self-play starter with terminal reward; editable
policy/learner/train source; prepare/audit/no-spend preview/execute/budget/grade
commands; a DeepSeek file-tool research driver; content-addressed submissions;
source/config/runtime/device-bound complete-batch checkpoints; persistent allocation
holds; finalization and one-third/two-thirds/final submission selection; and isolated
public/private evaluation. The initial starter exposes no population, shaping,
curriculum or strong-reference recipe.

Private grading runs separately on CPU. Trusted root-owned Showdown state and
opponent weights stay under mode-700 `/private`; candidate inference executes as
uid65534 in a separate process with bounded public-observation pipes, no network
and no credentials. Candidate source is never imported into verifier globals.
Full grading is 64 seeds × both seats × three opponents = 384 games. The power
bot and two provenance/source-checked short-trained neural checkpoints receive
equal weight; missing opponents abort rather than reducing the panel.
Natural win/draw/loss points are 1/.5/0; any capped/incomplete panel fails closed
with null score. Reference normalization remains null.

## Executed evidence

DeepSeek V4 Flash inspected the supplied starter, rewrote its policy with LayerNorm
and residual blocks, and added a KL-stop criterion to PPO. It retained the smoke
recipe's 64-hidden/8-dimensional embeddings and 3 × 8 environment topology.
Its 400-second worker grant completed with:

| Measurement | Result |
|---|---:|
| PPO iterations / optimizer steps | 298 / 3,282 |
| Trained samples | 382,136 |
| Natural training games | 7,128 |
| Training truncations | 24 |
| Parameter change L2 | 7.08853 |
| Public practice | 6/8 = 75/100; eight natural games |
| Private final grade | 8/12 = 66.67/100; twelve natural games |

All 297 updated iteration losses were finite; the final allocation-limited
collection was not trained. Its 24 truncations remain distinct from natural games.
Private scores were 2/4 against power and 3/4 against each development checkpoint.
The paired-seed bootstrap interval is [50, 83.33], descriptive only with two seeds.

The explicit submission was made at 375.21 charged seconds. The one-third smoke
milestone correctly records a missing submission; two-thirds and final select the
same immutable package and reuse its independent grade. No early policy was invented.

Provider start/end timestamps settle **375.209156 seconds** across one completed
L4 allocation, with **zero unresolved holds**, below the 600-second ceiling.
Estimated GPU-worker cost is **$0.10962**, not a verified invoice. Twenty-two model
calls report **$0.00285198** charged/reserved, below $2; model calls are closed.
Public practice and private grading used separate CPU-only workers, totaling
47.72 provider-allocated seconds. Source-only image building had no attached GPU.
Provider inventory confirms all three owned research workers stopped, with zero
active workers. A separate CPU-only transport diagnostic also terminated normally.

Ten focused local tests pass, covering actual CPU PPO learning, exact split/resumed
model/Adam/RNG continuation, source drift and package tamper rejection, natural-game
isolated inference, fail-closed scoring, unresolved budget holds, finalization,
editable workspace permissions and deduplicated thirds grading. GPU learning and
serialization were exercised; a separate exact GPU resume comparison was not run.

## Reproduction and limits

Use this game's existing **`.venv-pokemon`** for both ML and Modal; no extra
`.venv-pokemon-ar` or other game's environment is needed. The pinned host dependency
entry point is `simulators/pokemon/autoresearch/requirements-host.txt`.

```sh
.venv-pokemon/bin/python -m unittest discover -s simulators/pokemon/autoresearch/tests -v
.venv-pokemon/bin/python -m integrations.pokemon.autoresearch prepare --out NEW_RELEASE
.venv-pokemon/bin/python -m integrations.pokemon.autoresearch smoke --release NEW_RELEASE --out NEW_SESSION
# The preceding command is a no-spend preview. A newly authorized paid session adds:
# --execute --key-file OWNER_ONLY_OPENROUTER_KEY
```

The executed runtime/public release was immutable `release-4`; its exact source,
config, checkpoint identity, transcript and receipts are retained. The first edits
exposed a copied read-only workspace bug. The operator fixed permissions before
any GPU allocation, and DeepSeek continued on the same ledger. This was genuine
agent-authored research with an assisted harness repair, not a clean untouched run.
Later host reliability/error-branch/strict-runtime checks and milestone grading
were locally checked. Final `release-7`, including an immutable host-code snapshot,
was prepared and audited offline; it was not rerun on paid GPU.

The full 384-game panel, 120-minute campaign and multi-seed stronger-reference
calibration remain unrun. Opponents are development policies; tiny smoke points
are not expert strength. The existing encoder's Illusion/Transform/event limitations
remain, and Showdown execution does not imply commercial-client parity. Historical
records, earlier ledgers, upstream source and the native admission gate were preserved.
