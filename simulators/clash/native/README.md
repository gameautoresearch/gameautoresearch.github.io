# Independent fixed8 simulator

This directory contains a purpose-built native port of the frozen `clash-fixed8-v1`
simulator. Its transition code never calls Java. The pinned crforge Java engine remains
an external differential oracle; its source and task pins have not been changed.

`world.hpp` and `world_impl.hpp` assemble the component passes, command queue, placement,
movement, targeting, projectiles, damage, king activation, match endings and public
observations. `engine.cpp` exports the CPU reference. `cuda_engine.cu` launches the same
integer rules on independent GPU-resident worlds. The model consumes only each seat's
original 7,024-int public observation and 2,307-action legal mask.

The default `python -m integrations.clash` entry uses this native trainer. The old
`runner.py` remains the explicit Java reference/baseline. Whole-world CPU comparisons
cover 14,729 frames in the latest four-seed suite; actual L4 checks cover complete
CPU/CUDA outputs, natural endings, raw snapshots, neural features and exact PPO resume.
Current proof is under `native-pathfinding-20261108/` and
`structural-20261108/native-optimized-3/` within `evidence/clash-implementation/`.
The integrated bucket path also has actual CUDA resume evidence, including production
1,024-row minibatches and 48/16-row tails. The latest complete L4/16-core/64-GiB
comparison measured 1.999× original-integration throughput for fully native CUDA
and 2.472× for native CPU simulation with the CUDA learner. Earlier matched workers
measured up to 3.186×; these are separate comparisons, not factors to multiply.
See `reports/clash/native-rewrite-2026-11-08.md` and
`structural-20261108/native-completion-1/` on the private artifact volume.
Failed attempts are retained. The 10× target remains unmet.
The user accepted the measured performance and ended optimization on 2026-11-08.
The environment handoff covers correctness and reproducibility for this frozen task.
No commercial-game fidelity or coverage of other decks is established by this port.

## Building

On a fresh setup, export frozen record/schema constants from the pinned local oracle,
then compile the independent C++ configuration:

```sh
.venv-clash/bin/python simulators/clash/native/export_data.py
.venv-clash/bin/python simulators/clash/native/data_codegen.py
.venv-clash/bin/python simulators/clash/native/build.py
```

The exported JSON, receipts and generated header live in ignored `generated-v2/`.
They contain game-table data and must stay private. The runtime rejects a changed export
receipt, rules pin or stale generated configuration. It needs a C++20 compiler for CPU,
and a CUDA development toolchain with `nvcc` for CUDA. No Java process participates in
native reset, step, observation encoding, policy inference or PPO.

`runtime.NativeCPU` and `runtime.NativeCUDA` share reset/step/output interfaces. CUDA
world state, observations, masks, actions and rewards are Torch device tensors. The
CUDA API launches on the current Torch stream. `gpu_features.py` packs observations
without a host copy or data-dependent CPU shape readback. Its default keeps all 200
schema entity rows; the original alive mask excludes padding.

The CUDA translation unit specializes dimension reads for the unchanged 36×64 arena.
CPU and generic helper dimensions remain dynamic; world layout and serialized bytes
are unchanged. A five-candidate actual L4 experiment qualified every candidate on
the complete trained action tape before timing. The selected constant-dimension
candidate improved raw simulation throughput by 5.55%; this is a component result,
not an additional multiplier on the complete-training comparisons above.

`train.py` implements self-play PPO with device rollouts, GAE, truncation bootstrap,
masked action sampling, source-bound checkpoints and exact environment restoration.
Its `native-cpu-cuda` backend is a transport comparator: CPU native simulation followed
by the same CUDA learner. Full setup, compilation, simulation, training and checkpoint
time belongs to the allocation. A throughput result is not a strength result.

Run locally with `configs/native-cpu-smoke.json`; use `configs/native-l4.json` inside
an already allocated single-L4 worker with 16 CPU cores and 64 GiB RAM. The latter
uses native CPU simulation with CUDA neural training, the faster measured option
at its configured 256 environments. `configs/native-cuda-l4.json` keeps the fully
GPU simulator as an explicit comparison option. Neither provisions compute.
Actual GPU identity, source/data/rules, compiler
flags and loaded binary receipts are checked; checkpoints bind their execution runtime
and any neural dispatch library too. Setup, compile, capture and checkpoint time are
charged by the external allocation controller. The module CLI also retains a
cumulative `allocation-ledger.json` across resume, including final save/cleanup.
It rejects account forks, hardware/budget drift and missing inherited ledgers.
`--allow-moved-ledger` records an explicit checkpoint-floor transfer when the old
ledger is unavailable, with the unknown tail labelled. Saved checkpoint time alone
is a lower bound; the allocation's complete wall-time receipt is authoritative.

`configs/native-h100.json` is the qualified 2,048-environment, 32-update H100
throughput recipe on 32 attached CPU cores and 128 GiB RAM. It uses native CUDA
simulation and strict FP32 CUDA training. Both H100 comparisons use complete
subprocess time; the latest measured 6.448× over the original integration, while
the earlier paired allocation measured 7.045×. This is not a policy-strength claim.
The JSON recipes never provision hardware themselves.

The GPU neural buckets select 32, 64, 96 or all 200 entity rows on the device, with no
CPU count readback. They preserve every live observation row and reject invalid counts.
Sampling stays outside conditional network branches; the learner preserves the same
model, losses and Adam state. Outer rollout capture is disabled for these branches.
`bucket_usage` records actual actor/learner selections. Launch/render settings are part
of the saved recipe/runtime binding and cannot silently change within a Trainer.

`evaluate.py` runs native public-observation policies against wait, random and the
pinned pure-Python scripted policy, and records action tapes for native CPU replay.
CPU and actual CUDA qualification forbid Java startup. The actual CUDA check
replayed 96 public transitions exactly on native CPU and verified schedule-independent
per-game RNG plus complete busy-state restoration. Invalid or incomplete evaluations
have no official score. Evidence: `structural-20261108/native-learner-modes-1/evaluation/`
on the private artifact volume.

For example, evaluate a checkpoint and replay its recorded actions without Java:

```sh
.venv-clash/bin/python simulators/clash/native/evaluate.py evaluate \
  --checkpoint /absolute/path/to/native-run/checkpoint.pt \
  --backend native-cpu --out evidence/clash-implementation/my-native-evaluation
.venv-clash/bin/python simulators/clash/native/evaluate.py replay \
  --manifest evidence/clash-implementation/my-native-evaluation/manifest.json \
  --out evidence/clash-implementation/my-native-replay
```

Use `--backend native-cuda --cuda-launch-threads 1 --cuda-render-mode 4` inside
an already allocated CUDA worker. Evaluation transfers the frozen FP32 master
weights and uses explicit FP32 inference; it does not restore the training
optimizer/simulator or claim a cross-precision policy identity. Both natural
outcomes and incomplete diagnostic games remain separately recorded.

Each native world's finite storage has explicit overflow failure. It never shrinks the
observation or legal-action space to fit a performance target. Natural wins, draws,
artificial limits, illegal actions and simulator faults remain distinct.

## Match forms

`match_forms.py` names each benchmark form: `full` (the pinned Ladder timeline, 5 ticks per
decision), `short-v1` (60 s + 30 s overtime, elixir steps at 40 s and 75 s, 10 ticks per
decision) and `cadence10` (pinned timeline, 10 ticks; for the Java differential only). A form
other than `full` is compiled with explicit `-D` flags (`rules/match_form.hpp`,
`task_spec.hpp`) into its own library whose build identity names the form; the generated data
and the rules lock are untouched. Every loaded library reports the cadence and timeline it
runs (`cn_match_form` / `cn_cuda_match_form`) and the runtime refuses a mismatch. Select a
form with `build(form)`, `build_cuda(form)`, `NativeCPU`/`NativeCPUBatch`/`NativeCUDA(...,
form=...)`, or `"match_form"` in a training config (whose `rules.decision_ticks` must match).
The world layout and serialized world bytes are the same for every form.

```sh
.venv-clash/bin/python -m unittest simulators/clash/native/tests/test_match_forms.py
.venv-clash/bin/python simulators/clash/native/test_world.py OUT --form cadence10   # Java oracle, 10-tick cadence
```

## Experimental modules

Defaults leave `packed_head` disabled. Its actual H100 numeric, graph and separate-
process production/tail resume checks pass, but its latest complete-process timing
was slightly slower than the default. It remains an explicit experimental option.

`cpu_neural.py` is an unintegrated CPU neural prototype with local numeric and
within-mode replay evidence. `entity_encoder*.py` is an unintegrated Triton
prototype with CPU reference evidence; its GPU qualification is pending. Neither
is used by the shipped recipes. The combined compiled/precision and launch-policy
qualifiers are research tools, not additional qualified runtime modes. Rejected
sampler and lazy-route candidates remain in ignored evidence with their limits.

Saved checkpoints bind the entire source/configuration snapshot. To reproduce a
historical checkpoint, restore its retained immutable source archive; do not edit
its binding. Copy recipes outside the integration before customizing a new run.

## Attribution

The port is Apache-2.0. Every ported module names the original crforge source files and
full commit `f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de`. The upstream LICENSE and NOTICE
are preserved here. New constant compilers record their input hashes and compile only
record/schema data, never sampled transitions or action tapes.

## Reuse by the initial autoresearch starter

The explicit [v4 starter](../autoresearch/benchmark-v4.json) delegates to this
same trainer through small editable policy/learner hooks. None of the 71
historically qualified runtime files was changed for this integration. Its
initial network is random Policy64/embedding12, with sparse selfplay rewards.
The current default uses 1024 environments on H100/4 CPUs/16 GiB; the historical
2048-environment/32-CPU/128-GiB measurement is a separate hardware profile.
See the [fresh verification report](../../../reports/clash/fast-starter-requalification-2026-11-09.md)
for complete-process and full-allocation measurements, actual broker workflow,
source identity, exact resume and limits. Old checkpoints require their exact
source/image, not a relaxed source check.
