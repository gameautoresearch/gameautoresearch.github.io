# Clash fast starter successor v4

The frozen game, 200 entity rows, 7024 public observation ints, 2,307 masked
actions, 5x50ms advance and natural 180+120 second battles are unchanged.
The exact v3.1 opponent registry, public practice and protected schedule are
copied from the explicit base release; no evaluation seed bank is regenerated.
Policy inference remains stateless per-seat masked greedy NN. A researcher's policy has at most
500,000 parameters; the evaluator interface keeps its one-million bound. Scoring and natural/illegal/fault/truncation handling
remain the v3.1 contract. No reference-normalized or expert strength is claimed.

The initial editable policy is random Policy64/embedding12. Its default learner
and collector delegate to canonical native Trainer: device rollouts, full live
entity buckets, strict FP32 graph PPO, fused Adam and Triton GAE. No teacher,
reward shaping, pretrained weights or historical-opponent training is supplied.
The default topology is one L40S, 4 physical CPUs, 16GiB and1024 environments;
2048 environments/32CPUs/128GiB remains the separate historical recipe.
A JSON configuration does not allocate resources or grant spending permission.

Only new accepted learner seat decisions count. PPO epochs, ticks, inherited
samples, opponent/teacher rows never inflate that count. Report phase,complete
process and full allocated provider intervals separately. Setup,compilation,
idle time,checkpointing,transfers,failed attempts and shutdown count against
the externally authorized finite ledger. One GPU at a time; no retries.

Sessions run through the single research harness (`harness/README.md`).
`python -m harness prepare clash` creates a new immutable successor from an
explicit base release. A training run snapshots the editable source and seals
the verified result; practice performs complete public practice through
isolated CPU policy processes, in a CPU-only worker limited by count rather
than charged to the GPU ledger; a submission registers a completed sealed run
and records whether it was practiced; finalizing closes that ledger. The
backend makes no model calls. Private grading stays closed for this contract:
the harness backend refuses to grade, and the separate saved hero plan and its
runtime qualification gates are unchanged.

New native checkpoints retain strict native source/data/build/runtime binding
plus entrant source/config and trusted adapter hashes. Legacy training and
bundles retain their original bindings. Reproduction uses the exact retained
source archive rather than changing any historical checkpoint binding.
