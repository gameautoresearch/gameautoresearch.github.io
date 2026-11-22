# Clash autonomous research contract v1

> The active Clash benchmark is [short-v1](CONTRACT-short-v1.md): 60 s + 30 s overtime at ten ticks
> (2 decisions) per simulated second. The cadence and timeline below are the pinned form of this
> contract layer.

`benchmark.json` is the versioned machine-readable contract. The game remains
`clash-fixed8-v1`: two players, 36×64 routing cells, 18×32 placement tiles, fixed
level-11 eight-card mirror deck and Princess towers. Each simultaneous decision
advances up to five 50 ms ticks; simulation pauses for inference. Regulation 180 s,
overtime 120 s and natural tiebreak/end delays remain unchanged. No artificial
match cap is used. This task does not measure live reaction deadlines.

Each seat receives only its 7,024-int public observation and 2,307-entry legal mask.
All 200 entity rows remain available. Enemy hand, RNG and engine state cannot
enter policy inference. Two reserved champion actions stay disabled. Rules/data
pins and native CPU/CUDA build receipts remain mandatory. Other decks, levels,
champions and evolutions are outside admission. Commercial-game fidelity remains
unverified; the pinned Java emulator is the differential oracle.

## Research surface and starter

Editable public files are `policy.py`, `learner.py`, `train.py`, `config.json`.
The starter is a small randomly initialized MLP, plain masked PPO/GAE, current
self-play and sparse outcome rewards. It has no historical-opponent toggle,
teacher, shaping, population, recurrence or pretrained weights. Improvements are
agent-authored code. Engine implementation, observation, masks, accounting,
submission verification and private evaluator are protected. Only source/config
and finite float32 parameters are exported; weights are safe NPZ, ≤500,000 parameters for a researcher's policy (the evaluator's own bound is 1 million),
≤16 MiB compressed, with bounded expanded arrays. Editable source is ≤256 KB.
Inference is a stateless greedy forward, no search, public batches ≤256 and a
15-second pipe deadline. CPU-only official grading is standardized separately.

## Resources, clocks and artifacts

Full allowance: 7,200 allocated seconds on one H100 80 GB,32 CPU cores, 128 GiB RAM.
This smoke: 600 cumulative allocated GPU worker seconds and $2 cumulative model
reservations. Source-only image builds use CPU with no attached GPU; any compile
on an attached GPU counts. Setup, transfer, training, development work while
attached, idle, checkpointing, failed attempts and teardown all count. A provider
lease includes setup/cleanup and a retained overhead margin. One active GPU lease,
no warm pool/retries; provider start/end metadata settle elapsed allocation.
Unknown creation/end retains its full reservation and blocks another allocation.
Old campaigns/ledgers remain unchanged. The smoke never launches full calibration.

Run sources/config are copied before launch. Complete-update checkpoints retain
model/Adam, RNG, every raw native world byte using lossless zlib, and exact output
buffers. Resume rejects source/config/rules/data/build/runtime drift. Submission
seals exact source/config/weights with cryptographic hashes; the manifest binds
engine and training checkpoint provenance. Later workspace edits cannot modify a
submitted policy. Finalization requires a submission and resolvedGPU allocations,
closes further research/submissions and selects the latest immutable submission.
Budget milestones preserve the latest selection at ⅓ and ⅔ of the allowance;
no selection at a threshold is represented by null. Final selection is explicit.

## Practice, private grading and score

Offline preparation creates public practice on two documented public seeds,
128 fresh host-private grading seeds and 128 disjoint calibration seeds. Private
schedules use a cryptographic host RNG; seed values never enter agent context.
The smoke takes two fresh private seeds, both seats, against scripted and an
independently randomly initialized frozen policy: 8 natural games. Public practice
has scripted/random: 8 games. The initial policy and scripted bot are development
opponents, with no strong-reference qualification. A full release uses all 128
private seeds: 512 games with this two-opponent panel. Calibration must strengthen
and validate the panel before claims about general strategic ability.

Primary score is 100 × mean game points: win 1, draw 0.5, loss 0, balanced equally by
opponent and seat. Report both breakdowns and a 95% paired seed-cluster bootstrap
interval. Smoke intervals are descriptive. Any illegal action, simulator fault,
entity overflow, truncation or incomplete schedule invalidates the entire grade;
never convert these to natural losses/draws. Reference-normalized score remains
null until independent full-budget multiseed calibration establishes anchors.

Official evaluator runs in a network-blocked CPU sandbox. It never imports
submitted Python in verifier globals. Candidate and checkpoint opponents run as
different unprivileged UIDs under no-new-privileges, bounded pipes and separate
private directories. Only seat observation/mask arrays cross to the candidate.
Private schedules/weights stay in root-owned/private storage; no credentials,
seeds, engine state or opponent weights enter the agent build or model prompts.
Public outputs contain seed indices and commitments, not private seed values.
Process isolation does not prove arbitrary code obeys every research rule;
source review remains necessary. Full-release opponent strength is uncalibrated.
