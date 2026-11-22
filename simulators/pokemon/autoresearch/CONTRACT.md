# Pokémon Gen 9 random 3v3, native training (`pokemon-3v3-v1`)

The machine-readable contract is [benchmark.json](benchmark.json). The game is
`gen9randombattle@@@Max Team Size = 3` at Pokémon Showdown pin
`567c1cedddeaa2f35576005d3bfd526cf4f8a13e`: unmodified Gen 9 Random Battle (same
sets, moves, items, levels, one active, switching and Tera) with three generated
Pokémon each instead of six. The 14 move/Tera/switch action IDs and 700
player-visible encoded values are unchanged; own rows 3-5, foe rows 9-11 and switch
actions 11-13 are always empty. There is no team building or content filtering.
The six-Pokémon form (`pokemon-gen9-random-6v6-native-h100-pilot4h-v1`), its
certificate and releases are unchanged history (tag `full-size-20261113`).

Training runs on the admitted native C++ engine in `simulators/pokemon/native/`.
Its three-Pokémon certificate (full Gen 9 admission of this format against the pinned
Showdown rules; plan `reports/pokemon/native-cpu-qualification-3v3.json`) is
verified from the worker's own source tree every time the engine is built, and
it pins the engine sources, the public engine data, `network.py`, `efficient.py`,
`env.py`, `rl.py`, `autoresearch/contract.py` and `autoresearch/starter/learner.py`
byte for byte. Evaluation runs on pinned Showdown through the qualified evaluator.

Every job is one finite, network-blocked worker on the retained runtime image with
an L40S, four physical CPU cores and 16 GiB (private grading: the same CPU and
memory, no GPU). A session's research budget is in one of two modes, chosen per
session. In the default seconds mode one job runs at a time and the research
allowance is 14,400 allocated worker seconds. In dollar mode up to four jobs may
hold an allocation at once within at most two GPUs, so at most two Pokémon jobs
(each one L40S worker) run at once, and the allowance is US$11.61 (smoke: US$0.48,
the seconds allowances at the training worker's frozen list price of US$0.0008064
per second; 18,000 seconds of session wall clock, smoke 4,800). A recipe's fresh
final training is one 3,600-second worker in either mode. A training lease is 180
to 3,600 seconds, of which 120 are kept for setup and publication; a practice lease
is at least 240 seconds (360 by default) and counts against the same allowance. The
whole lease (in dollar mode its whole price) is reserved before the provider is
asked for anything and stays held until the worker settles, so parallel grants never
exceed the allowance. A worker is charged its provider-measured start-to-stop
interval (in dollar mode at that price). A reservation that provably never reached
the provider settles at zero. An interrupted creation, a missing start, a hardware
mismatch or a worker measured beyond its reservation stays held as unknown; in seconds
mode it blocks every further allocation, in dollar mode it keeps its place among the
jobs at once, and a worker measured beyond its reservation blocks every further
allocation in both. There are no automatic retries; seconds mode has no dollar
admission ceiling.

Workers have no network, credentials or private seeds. A trusted bootstrap admits
only the release's packed source, verified byte for byte against its manifest,
quarantines the image's own Python and restores only the image's Showdown engine.
The editable trainer runs as uid 65534 with a public import closure and reaches the
engine through a root-owned environment service that returns projected observations,
rewards and boundary flags. Environment state in a checkpoint is encrypted and
authenticated. The trusted supervisor stops every trainer process before it seals
a policy: tensor-only weights, the exported tensor graph and the source, config and
runtime bindings. A trainer that the supervisor ends itself, on the researcher's
stop request or at the stage time limit, is sealed by the same checks from the last
complete checkpoint, graph, result and progress files it wrote and is recorded as
`stopped_early` or `cut_short`; the host terminates a worker that has not finished
180 seconds after a stop request. A trainer that wrote no complete set, or exits
with an error by itself, is a failed stage. A failed stage stays failed; its last
complete checkpoint is retained unvalidated and can never be submitted.

The scored policy is the exported graph described in
[runtime/GRAPH_CONTRACT.md](runtime/GRAPH_CONTRACT.md); submitted Python is never
imported by inference. A recipe is source and settings only: source files may not
embed encoded payloads or weight tables or import `base64`, `pickle`, `marshal`,
`requests`, `urllib` or `socket`, and a config may not name state files.

The panel is the public max-power heuristic, two frozen learned policies trained by
self-play PPO on three-Pokémon battles for this form (the six-Pokémon panel's
retained policies were trained on six), and a pinned public-information search
opponent (Foul Play, two beliefs, 10 ms per belief; its beliefs sample the opponent's
unrevealed Pokémon up to the published team size). Each seed is played from both
seats against every opponent; all opponents and seats weigh equally. Score is
100 × mean(win=1, natural tie=0.5, loss=0). A panel with any unfinished, capped or
faulted game has no score; illegal actions, policy failures, simulator faults and
infrastructure faults are reported separately and are never favourable outcomes.
Public practice uses 16 fixed public seeds (128 games). Private grading uses 64
fresh seeds (512 games; smoke: 2 seeds, 16 games) that are created only after the
final policy identity is locked, exactly once, and never replayed.

The normalized reference score remains null. The panel's difficulty and the learned
references' strength are unqualified. A new release needs its own exact-source live
qualification on real L40S workers before any scored session.
