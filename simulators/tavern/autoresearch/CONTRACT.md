# Tavern four-player small form v1 (`tavern-four-small-v1`)

Four players, regular tavern tiers 1–4, three lobby minion types, two hero choices; everything else as
below. The eight-player form (`tavern-eight-implemented-v1`) is at git tag `full-size-20261113`. The harness
evaluates with `evaluation_v2/CONTRACT.md`; this file describes the earlier v1 grading path (`grade.py`),
numbers restated for four seats.

`benchmark.json` is the discoverable contract. Rules and actual native source/data
hashes are frozen by offline preparation. Build freshness is checked before every
training/evaluation. Historical source-bound checkpoints are never relabeled.

The editable starter consists of a one-hidden-layer masked MLP, plain PPO learner,
and full-episode trainer. It starts randomly; no expert recipe or pretrained weights
are supplied. Recipes may change learning hyperparameters and bounded architecture,
while validation locks game rules. Gamma, selected learning seats and scripted/self-play training are research choices; the supplied baseline uses gamma=1 and one seat against three scripted opponents. Evaluation always uses host-owned fixed rules and opposition.

One L40S/four physical CPU cores/16 GiB worker is active at most once. Full research
allowance is 7200 allocated seconds. This campaign has 600 seconds across all GPU
attempts. Before allocation the host persistently reserves its complete timeout plus
startup/shutdown margin. Only authoritative provider end metadata settles it;
unknown ends retain holds and block launches. Image builds without a GPU are
reported separately. Failure/idle/setup/evaluation/IO time on a GPU counts.
The model ledger separately caps charged plus outstanding reserved liability at$2.

Each run exports safe finite NPZ weights, exact editable sources, effective recipe,
and rules/source identity. Host submission copies and hashes the full bundle into a
new readonly directory. Submit only a registered completed artifact. Finalization
closes new runs/submissions. Milestones at 2400/4800 seconds preserve the latest
submitted artifact; unreached milestones remain null. CPU grading is separate compute.

Host-private fresh schedules are committed at preparation, absent from public
context and source-only runtime images. Private grading rotates one candidate through
all four seats against three copies of an independently trained frozen policy.
The smoke uses two fresh seed groups and one lobby per seat, 8 episodes. The full panel is 16 groups×4 seats×4 lobbies against each of three committed weak development PPO opponents (768 episodes). Smoke uses the first opponent only. Preparation trains those opponents for four CPU updates each with distinct seeds and commits their exact weights/identity/summary hashes. Missing or mutated declared opponents fail closed. These are development policies, not full-strength anchors. Public practice uses separate public seeds and scripted
opponents. Candidate source executes only in a separate unprivileged process in a
network-blocked container; it receives visible arrays/masks over a bounded pipe.
The host verifier never imports candidate source or pickle. Frozen opponents and
private seeds remain root-only. Inference deadline 15 s per batch; 500,000 parameter
cap. Untrusted-code rule compliance still requires source review.

Score=mean 100*(4-placement)/3, equal seats and opponents. Report per-seat scores,
mean placement and top-half (first or second) fraction. Bootstrap whole seed groups for intervals;
small smoke intervals are diagnostic. Any fault, cutoff, illegal action or incomplete
schedule invalidates scoring. Natural elimination of the evaluated seat is sufficient;
other bots need not finish their lobby. Reference-normalized score remains null until
three-seed full-budget training and independent calibration show a robust positive gap.

The supported simulator remains Rust CPU. The independent GPU-rules rewrite stays
paused. Current CPU restructuring requires fresh CUDA learner/resume qualification;
this harness's smoke records that evidence rather than borrowing historical identity.
