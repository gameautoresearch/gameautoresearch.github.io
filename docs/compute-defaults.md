# Compute defaults

**NVIDIA L40S is the global GPU for all six games**, expressed as `gpu="L40S"`
in Modal SDK calls. This is the user's explicit project preference, recorded
November 10, 2026 and made global on November 11, 2026: every game's current
benchmark contract and harness backend now requests one L40S for research,
fresh final training and GPU-backed grading. The single source in code is
[`harness/hardware.py`](../harness/hardware.py); each backend checks the
provider-reported GPU type against it and refuses any other. Native kernels
are compiled for the L40S architecture (`sm_89`): Dou Dizhu's CUDA engine and
Polytopia's `engine-cu` image layer.

Before November 11, Generals, Dou Dizhu, Clash and Polytopia contracts declared
H100 and Tavern declared L4; those contracts were switched to L40S, keeping
each game's CPU cores and RAM. Only an explicit user instruction for a specific
run overrides the L40S default. Record the selected GPU and the reason for any
override in the run's configuration and receipts. Do not silently substitute
another GPU because of availability.

Keep the remaining resource profile explicit: CPU cores, host RAM, software,
policy, batch sizes and full allocated-worker budget. A GPU preference does not
grant compute authorization or additional budget. It does not add a GPU to
CPU-only tasks.

## Historical runs and records

Records keep the GPU they ran on: sealed releases, receipts, checkpoint
bindings, accounting ledgers and the superseded contract versions (Clash v3 and
v3.1). Releases built before the switch no longer verify against the current
source; prepare new ones. Results on H100 or L4 are not directly comparable with
new L40S sessions: compare researchers under the same hardware and accounting.

## Measured rationale

The [November 10 matched smoke tests](../reports/gpu-value-smoke-2026-11-10.md)
used four physical CPU cores, 16 GiB RAM and two short trials per GPU/workload.
The full-worker denominator includes startup, compilation, training, checkpoint
export, idle time and teardown. Samples are newly accepted learner decisions;
sample counts are comparable between GPUs within a workload, not across games.

| Workload | H100 decisions/s | L40S decisions/s | L40S saving for equal samples |
|---|---:|---:|---:|
| Pokémon, Opus m1 league trainer | 19,867 | 15,172 | 22.4% |
| Clash, fast starter | 47,596 | 40,251 | 30.0% |
| Battlegrounds, population trainer | 22,291 | 21,202 | 37.7% |

These are short workload-specific measurements, not a qualification of all six
games or evidence about bot strength. In Pokémon's warmed collection/PPO loop,
H100 reached 27,346 decisions/s versus L40S's 18,677; projecting those phases
suggests about 13.3% L40S savings for long leases. H100 can therefore be worth an
explicit override when faster iteration matters. Clash's L40S warm loop was
slightly faster, but its first trial paid for architecture-specific compilation.

The cost estimates use the November 10 Modal Sandbox rates for the same CPU/RAM
profile, plus a 5% margin. They are not reconciled invoices or permanent price
guarantees. Refresh prices and measure a matched workload when materially
changing the policy, backend or batch configuration.

## Reservations: pay for use, keep the cap (November 17, 2026)

Modal bills each second at the larger of a container's **reservation** (request) and its actual use, and
Sandboxes cost 3× the Function rate for CPU and memory. Until commit `5a9f67f0` every container reserved its
whole shape, so idle parts of a session paid full price. In the Pokémon 3v3 relaunch of November 17 (4 sessions,
3.7 h) about 40 % of the $47.62 Modal bill was reserved-but-idle capacity: researcher boxes $13.92, cloud
coordinators $5.30, and public practice panels on L40S workers. Since `5a9f67f0`:

| Container | Reserves | Hard cap (unchanged) | Note |
|---|---|---|---|
| Researcher box (Harbor, all games) | Modal minimum: 0.125 CPU, 128 MB (`--cpus limit --memory limit`) | the task's 4 CPU / 16 GiB | mostly waits on the model |
| Cloud coordinator (`harness/cloud.py`, all games) | 0.25 CPU, 1 GiB | 2 CPU / 8 GiB | still non-preemptible |
| Pokémon CPU jobs: public practice, private grading, tournaments | 1 CPU, 2 GiB | 4 CPU / 16 GiB | an arena uses ~1.4 cores, < 1 GiB |
| Pokémon training runs and fresh finals | 1 L40S, 4 CPU, 16 GiB | same | the fixed contract shape, unchanged |

Public practice no longer holds an L40S: it plays its games on a CPU worker and is still charged to the research
ledger per worker second, so research allowances are unchanged. Provider receipt checks follow the launch kind.

**Verified** with a full smoke on Modal (`poke3v3-lean-smoke-20261117`, release
`evidence/releases-20261117c-poke3v3-empty-lean/pokemon`, DeepSeek V4 Flash, cloud coordinator): the box and
coordinator ran a normal session (164 requests); a training run and the fresh final completed on the exact L40S
shape; two practice panels completed on CPU workers (128 games each, 0 faults, 88 s and 104 s versus ~110 s on
the GPU worker); the cloud private grade came back `complete_valid` with `profile_verified` (receipt: 1,000 /
4,000 milli-CPU, 2,048 / 16,384 MiB). Billed: box $0.02 for ~35 min (≈ $0.55 at the old reservation),
coordinator $0.04 for ~37 min (≈ $0.30). The checkpoint-Elo round robin of the same day ran 366,336 games on
1-CPU / 2 GiB Modal Functions with 0 faults and 0 timeouts.

Trade-off: nothing is held above the reservation. CPU above it is throttled when the host is busy (slower, never
killed). Memory above it is allowed "if the worker has available memory"; Modal does not document what happens
under host memory pressure, so a box or coordinator far above its 128 MB / 1 GiB reservation may be at risk on a
crowded host. Raising those two memory reservations to realistic values (a few GiB) would cost cents per hour.
Projected saving for a 4-model, 4 h Pokémon campaign: Modal ≈ $55 → ≈ $30 (GPU training, ≈ 40 % of the bill,
is unchanged).
