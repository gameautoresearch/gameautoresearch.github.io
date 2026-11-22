# Tavern continuation state — 2026-11-08

## Latest user steering — CPU restructuring v1 complete; GPU work paused

At 19:38 the user paused the GPU simulator rewrite. They then requested a bounded,
aggressive CPU restructuring pass, correctness checks and a good stopping point.
That pass is complete. Do not restart GPU experiments without new authorization.

- Read `reports/tavern/cpu-restructuring-2026-11-08.md` and its result JSON.
  Original native `7b257f56e12468e9` becomes selected `15488d9159a6d638`.
  The loaded native build matches source. Full eight-player rules/data stay pinned.
- Kept: compact copy-on-write minion cold fields; shared history/scouting boards;
  direct batch step slices; grouped, fully initialized observation writers;
  parallel indexed initialization; stack masks/readiness arrays. Python API and
  saved JSON schema retained. Bonus-metadata candidate rejected and removed.
- Local M5 Max, four native workers, interleaved under local-performance.lock:
  128-lobby observe+step 1.083×, complete protocol 1.048×, initialization 2.826×.
  1,024-lobby observe+step 1.072×, complete protocol 1.042×, initialization 3.210×.
  A multiple overall simulator speedup was NOT achieved. Do not multiply these
  gains into old CUDA results or call initialization an overall throughput gain.
- Exact replay: 38,400 and 307,200 controlled decisions, 240 and 1,939 natural
  episodes. All observation/mask/result and final full game/RNG digests match.
  Full Rust: 879 passed. Python: 26 passed, seven skipped. Three fresh-policy CPU
  learning pairs (32 lobbies × four updates, 128/8 network) match 29,449 samples
  per arm, parameters, optimizer, full histories/states, counters, losses and RNG.
  Exact production resume passes state, learning, cohort cursors and prior-session
  accounting. This revision has no new CUDA test.
- Evidence root: evidence/tavern-implementation/cpu-restructure-20261108.
  baseline-profile freezes original package/source/profile. selected-v1 freezes
  final package/source/hash manifest/build and test logs. Final paired timing,
  learning and resume directories are immutable. Source-bound checkpoint guards
  unchanged; old-source checkpoints must not be relabeled as current. Migration
  unit test uses a current-source synthetic fixture and rejects dependency AND
  historical actions.rs drift. Historical checkpoints/receipts/manifests untouched.
- Default gpu-throughput recipe remains native CPU simulator + existing CUDA
  learner. The device rewrite remains experimental and paused. No cloud experiments
  after pause. No agents, commits, staging, pushes or shared-doc edits in this CPU
  pass. Stop expanding the implementation here.
- Owned GPU checker ap-iIw24s1fgZqRDPcWwJ2Rsi stopped at 19:38:46; GPU-rewrite build
  ap-w3pFWYiQlmGVM4I58O0L6X stopped at 19:39:08; both statuses observed. Sidecar
  device-port/paused-by-user-1941.json records pivot. Keep cancelled receipts.
- Retention audit complete for 91 files / 1,294,520,705 bytes: device-port/
  retention-audit-device-v30-v44-1931-direct.json matches remote SHA manifest 1925.
  Includes v30/v37/v38/v40/v41/v42/v44; cancelled v46 logs incomplete, KEEP VOLUMES.
  Final billing refresh attempts 1926/1936 failed or unknown. The last confirmed
  billing amount remains historical, not a final total. No new cloud workers for
  reconciliation.

## Historical GPU work (paused; retained for future reference)

The user has repeatedly sharpened the task: a working GPU-resident COMPLETE supported
simulator plus GPU training is now an explicit implementation target. Native wrapper
speedups alone are not completion. No old dollar/hour/attempt caps apply. Preserve old
ledgers and receipts. Own only Tavern integration/report/evidence and necessary engine
source. Polytopia is read-only; no jobs were launched there.

## Completed, retained native/hybrid work

`reports/tavern/structural-performance-2026-11-08.md` needs final consolidation (it has
interim wording). Main code: dynamic native scheduling, bounded idle worker cache,
preencoded PPO inputs, exact actor/learner CUDA graphs, GPU input retention via
`device_rollout.py`, vector bookkeeping, native warmstart, compact ready-seat native
encoder `candidates/AutoTavern/crates/taverngym/src/rl_obs.rs` plus binding API.
Original `obs.rs` and ALL original physics/content/data remain unchanged. New encoder
is pinned in `observation-provenance.json`. Tests include dense/compact view/mask/state/
RNG parity and graph/retained-input exact PPO. Latest full local suite before newest
small pin/boundary tests: 69 passed, 7 CUDA/Linux skips; more recent focused suite passed.

All structural GPU jobs through v21 have finished. Artifacts are under
`evidence/tavern-implementation/structural-20261108/<run>/retained/<run>/`. The two first
recursive downloads failed due nonexistent destination being treated as a file; they
were repaired without retraining and receipt files explain it.

Important receipts:
- `cuda-qualify-v3`: ALL 3 independent 128-update control/graphs pairs, evaluations and
  exact resume completed. Function failed only afterward at optional native profile
  missing `/workspace/evidence` directory. Do not rerun completed training. Reconstruct
  summary from `learning-progress.json`; `summarize_structural.py` already does it.
  1.51–1.55× full process gain, bitwise weights/optimizer/RNG/states; initial mean
  placement 7.96354 -> 6.82292. 1,978,516 unique samples (each arm executed).
- `cuda-refine-v4`: three 8-update pairs and both-graph resume pass. CPU actor null;
  fused Adam null/inconsistent and changes numerics, remains optional.
- v5/v6/v7: failed metadata cgroup path; immutable failures charged.
- v8/v10: old CPU pilot rejected by tightened Cargo pin check; tests also initially
  referenced missing local evidence remotely. Fixed without weakening pin checks.
- v9: queued old-payload 16-core job specifically cancelled (CLI first needed --yes).
  Cancellation logs/attempt preserved. No GPU result. Ensure terminal during cleanup.
- `cuda-massive-4c-v11`, `cuda-massive-16c-v12`, `h100-massive-32c-v13`: complete initial
  scaling and retained-input parity/resume; busy 512 precondition refused naturally
  eliminated warm seats. No invalid score used. Packed follow-up fixes profile-only
  eligibility explicitly, never training/evaluation.
- `cuda-final-4c-v14`: independent current128 210.465s -> fast128 114.955s, same 669,955
  samples and placement 6.53125; final summary full128 parity. 205-second larger cohort
  learning pilots also complete. Native observer later changed, source archives retained.
- v15/v16 packed initial comparisons complete but 8-update busy profile later hit
  warm-state eliminated seat; immutable prefixes retained. v17/v18 correctly complete.
- `cuda-packed-4c-v17`, `h100-packed-32c-v18`: dense/compact exact weights, optimizer,
  CPU/CUDA RNG, states, counts on initial/busy and large cases. Resume passes.
  L4/4-core: 1024 packed 376,317 samples /20.515s loop /25.569s process =18,343 loop/s.
  H100/32-core/128GiB: 4096 packed 1,556,475 /63.773 loop /70.467 process =24,407 loop/s.
  Explicit additional resources and cohort/algorithm changes; not 10× identical-policy.
  Dense H100 4096 loop109.472s -> packed63.773s; bytes69.1GB ->21.26GB.
- `cuda-wall-s12345-v19`, `cuda-wall-s45678-v20`, `cuda-wall-s78901-v21` all complete,
  fixed205s train-window, four public practice seeds ×8 seats×2 lobbies =64 evals/arm.
  Current32 -> packed128 placement: 7.25->6.0625, 6.21875->5.109375, 7.15625->6.21875.
  Samples632114->1597950,953344->3050745,656410->1841492. All faults/rejections zero.
  Same requested L4/4-core/16GiB within each pair, different hosts across pairs. Include
  actual evaluation and outer allocation times, not just training window.

## New standalone full-rule CPU/device port — IN PROGRESS

`integrations/tavern/device/` is a separate workspace, MIT derivative of every pinned
rule/content module. It does not call the original engine for device transitions.
Original per-file hashes and data SHA are in `device/oracle-source-manifest.json`;
LICENSE/NOTICE copied. CPU/device share derived rule source, with target-specific
no_std plumbing. This is not GPU-qualified yet and must not be presented as completed.

- `device/core`: full original mechanics/content, independent crate. Host differential
  test four complete eight-player scripted games matches ALL state JSON, all eight
  views/masks, actions, RNG, placements and checkpoint roundtrips (twice, including
  deterministic map replacement). `core/tests/differential.rs`.
- Device-specific `std` facade, deterministic HashMap wrapper (avoids OS entropy),
  CUDA release/acquire CAS published OnceLock, Rc for per-game single-writer board
  ownership, per-game reentrancy cell for Titanic Guardian/Sinestra, uint64 mulhi PRNG
  path replacing u128 multiply. CPU OS stack diagnostic is not claimed to be a CUDA
  stack guard; device stack capacity/hard-case qualification remains required.
- `device/kernels`: complete Game::new, scripted advance, controlled-seat stepping,
  legal masks, player-visible observations, natural/fault metadata and automatic
  reset from exact original splitmix seed domains. GPU-owned Episode contains Game,
  episode index, cohort seed, finished[8]. JSON export/import explicit checkpoint IO.
- `device_env.py`: Torch CUDA-buffer ownership + direct driver kernel launching,
  one cached module/process, no CPU transitions. Small metadata can return to host;
  game state/observations/actions stay on device. Native integration remains default.
- `device_probe.py`: actual GPU differential against original `taverngym.BatchEnv`,
  controls[0], [3,0], all8, active masks, scripted busy actions, full state/RNG and
  checkpoint import/export, masks/views and outcomes. No neural GPU-rule training yet.
- Rust-CUDA SDK exact revision `3a82c35a7196731fb60eebdac2a40d2b1bf0ba95`, pinned
  nightly-2026-04-02 and LLVM21 backend. SDK clone retained in ignored
  `evidence/tavern-implementation/device-port/rust-cuda-toolchain/`.
- `device_modal.py`: pinned NVIDIA CUDA13.0.2 image digest, Torch2.9.1, original CPU
  oracle editable build, SDK/compiler. Dedicated L4/4-core/16GiB, bounded1800s function,
  zero retries/warm pool. Image builds CPU-only and also charged by provider receipts.
- `device_launch.py` freezes unique payload and attempt before each paid call.
- `device-port-v1` FAILED image compile because libclang missing. No GPU training or
  runtime; preserve build logs/charges. Fixed by adding libclang-dev as a new layer.
- `device-port-v2` is CURRENT running image build, launched via .venv-tavern Python.
  Local process log `/tmp/tavern-device-launch2.log`; full log
  `evidence/tavern-implementation/device-port/device-port-v2/modal.log`.
  Monitor this until completion/failure. Fix actual compile/runtime issues and keep
  pursuing full GPU parity/training, not just a load stub. New build-cache volume
  caches target/CARGO_HOME so failed kernel iterations need not rebuild LLVM backend.

## Pending completion obligations

1. Complete new full-rule GPU core compile, device differential/hard mechanics, save/
   resume, GPU-resident neural collection/PPO and honest CPU-vs-device full-loop tests.
   Can retain CPU production if actual concrete measurements justify it; don't assume.
2. Finish native/packed report + defaults with three-seed fixed-wall useful-learning
   evidence and exact-parity vs cohort/resource-scaling distinctions. No 10× matched
   identical-policy claim established. No strong-reference or commercial-parity claim.
3. `collect_billing.py --out NEWFILE` filters both structural/device logs and provider
   app names. Old billing0.5513 receipt unchanged; fresh observed1401 receipt2.6661 USD
   is partial/lagging, not final. Verify current final billing later.
4. `retain_structural.py` (Modal CLI Python) computes remote/local SHA per file before
   volume cleanup. All existing structural artifacts retained locally; verify checksums,
   confirm no owned active containers, delete ONLY own volumes after retained audit.
   Structural data volume: autoresearch-tavern-structural-evidence-20261108.
   Device data: autoresearch-tavern-device-evidence-20261108.
   Device compiler cache: autoresearch-tavern-device-build-cache-20261108.
   Original historical Tavern volume was already deleted with541-file audit before task.
5. No staging/commit/push, no edits to other games/shared/root docs, no Polytopia changes.

## Continuation at 2026-11-08 15:29 Shanghai — ACTIVE, DO NOT STOP AT REPORT

Latest trusted parent delegation repeats: aim ~10x+ over pinned ORIGINAL upstream;
require TWO explicit baselines (original simulator and earliest runnable complete
trainer), strong matched-resource CPU/GPU controls, full GPU simulation+learning.
No old dollar/hour/attempt caps; retain ALL history. Continue substantial implementation.
No subagents spawned, no messages sent to parent (receiving delegation does not authorize
send_message_to_thread). No commit/stage/push or other games/root shared file edits.

### New fully implemented device stack, still unqualified

- Full shared Rust rule source `device/core` passes independent 4-game host oracle
  differential (from earlier), and actual nvptx/no_std Rust frontend check locally.
  Native nightly `nightly-2026-04-02-aarch64-apple-darwin` + rust-src installed. Check:
  `RUSTFLAGS='-Zunstable-options -Zcrate-attr=feature(register_tool) -Zcrate-attr=register_tool(nvvm_internal) -Zcrate-attr=no_std -Cpanic=immediate-abort' cargo +nightly-2026-04-02-aarch64-apple-darwin check --manifest-path integrations/tavern/device/Cargo.toml -p tavern-device-kernels --features arena --lib --target nvptx64-nvidia-cuda -Zbuild-std=core,alloc`
  Logs `/tmp/tavern-arena-front-check{,2,3}.log`, initial `/tmp/tavern-device-front-check.log`.
- New `core/src/arena.rs`: reusable, per-game, single-writer size-class GPU arenas,
  default1MiB/game, explicit CUDA malloc fallback (never truncate mechanics). Alloc
  headers identify their owner, so frees remain safe across kernels. Persistent once
  constructors enter a separate persistent-CUDA allocation scope, ensuring static
  tables/IDs never dangle when a game arena is freed. CUDA.std optional external
  allocator feature. One ordered stream enforced per module/process. GPU arena bind
  and telemetry kernels, ctor/binder/close in `device_env.py`.
- Arena 10,000 randomized alloc/free/content/alignment host stress PASSED:
  `/tmp/tavern-arena-host-test2.log` (first attempt only type inference test error,
  corrected explicit Vec<(*mut u8,usize,u8)>). No performance claim from that test.
- `device_training.py`: real CUDA-owned full rules+features/masks/actions+retained
  rows, GPU sort/pack/GAE, PPO actor/learner graphs, full source/PTX-bound checkpoints,
  exact resume logic. CPU comparator now uses strong compact native observe_ready,
  transfers once, same GPU GAE/PPO. Only metadata/checkpoint JSON crosses host.
- `device_learning_probe.py`: CPU/device dataset+PPO/RNG identity then actual2 updates
  and split1+resume1 identity. `device_fixtures.py`: 5 hard eight-player states
  (populated deathrattle/Reborn/avenge, capped stats+Titanic guard, end-turn lethal,
  queued triple target menus, full mixed boards); CPU construction and legal action
  progression validated. Hard fixtures are targeted unit states, NOT training data.
- `device_probe.py`: first full state/view/mask/reward/RNG oracle differentials for
  controls[0],[3,0],all8, then hard mechanics, then neural/resume gates. ABI kernel tests
  wide pointers. Saves authoritative PTX before loading module. All still gated.
- `Episode` CURRENT source adds explicit `controlled:[u8;8]`, `k:u8` checkpoint binding.
  GPU import checks seats/k/8players and Game::check_loaded; Python checks explicit
  frozen rules. Fixtures updated. This change is included in v12/v13 onward.

### Toolchain repairs and paid attempts (immutable payloads/logs retained)

- v1 missing libclang; v2 release panic=abort incorrectly applied to host plugin;
  removed workspace panic profile. v3 missing z/zstd linker libs; v4 missing NVVM
  LD_LIBRARY_PATH; v5 SDK broad glam>=.22 selects incompatible0.34; pinned SDK's own
  lock0.30.9. v6 compiler fat-pointer layout panic.
- `device/compiler/` is reusable standalone host compiler project. SDK revision
  `3a82c35a7196731fb60eebdac2a40d2b1bf0ba95`, LLVM21.1.8 prebuilt. Copied required SDK
  crates under compiler/vendor with original per-file SHA/licenses/patch manifest.
  Patches: LLVM21 opaque pointer scalar-pair ABI skips obsolete typed pointer field
  projections; opt_item_name for anonymous libm closures at Rust opt1; CURRENT NEW
  allocator.rs fix defines empty callable __rust_no_alloc_shim_is_unstable_v2 marker
  (confirmed exact rustc LLVM backend source at pinned7e46c5f6f). SDK mem optional
  external-allocator suppresses its global allocator only for arena feature.
- v7 cp failed because separate image RUN lost mounted build outputs; combined build
  and cp into ONE RUN fixes. v8 reached NVVM verifier: modern LLVM21 cannot target L4.
- LLVM21 NVVM modern dialect requires Blackwell (>=compute100), confirmed NVIDIA spec.
  Changed compiler to Compute100 and gpu=B200; modern libNVVM upgraded by pinned
  `cuda-nvcc-13-3=13.3.73-1`, `libnvvm-13-3=13.3.73-1`, cuda-driver-dev13-3 installed
  atop pinned CUDA13.0.2 base. CUDA_PATH/ROOT and LD_LIBRARY_PATH point13.3. SDK CI
  itself uses13.3.1. Resource is B200/4physicalCPU/16GiB, max1800s fn/1700s child,
  no warm pools/retries. Modern13.3 Docker image digest looked up but not used:
  amd64sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e.
- v9 optimized full codegen ran >~18min, predates known atomic bug; cancelled only
  owned app `ap-PmAschEeTFHee9AjuiuGSA`. Server app list at15:28 confirms stopped
 15:22:14/tasks0. cancellation-reason.json preserves reason; cancellation.log local
  CLI oddly hangs/empty, but server status proves stop. Old ta-01M6WZG... was an
  already-stopped builder stage, container-stop.log confirms. No GPU simulator claim.
- v10 low optimization (Rust opt1, NVVM opt0) exposed anonymous libm closure bug;
  fixed opt_item_name. Lowopt codegen is much quicker than opt3.
- v11 arena lowopt FULL COMPLETE RULES COMPILED TO PTX and ran real B200 module load.
  Failed JIT syntax due SDK atomic macros `fence.sc.gl` / `.acqrel`. No mechanics
  stripped. PTX ~160MiB; download might show fixed-size sparse partial contents until
  CLI retrieval completes. v11 local launch session26288, /tmp/tavern-device-launch11.log.
- Replaced TWO own OnceLock operations with explicit PTX acquire ld and acq_rel GPU
  CAS in core/src/device_once.rs, plus asm_experimental_arch feature. Correct PTX
  syntax verified NVIDIA ISA9.2. This removes invalid SDK atomic operations from DCE.
- v12 atomic fix compiled full rules and reached B200 JIT; next failure unresolved
  Rust allocator marker above. v12 session11702; /tmp/tavern-device-launch12.log.
- **CURRENT ACTIVE v13** `device-b200-allocshim-v13`, fixes allocator marker. Rust1 /
  NVVM0 / arena / diagnosticIR. Full codegen succeeded in54.25s, image63.45s. At15:29
  waiting final source image/GPU probe. Local **session64577**, log
  `/tmp/tavern-device-launch13.log`; full
  `evidence/tavern-implementation/device-port/device-b200-allocshim-v13/modal.log`.
  Continue until actual GPU parity/neural/resume, fix real failures, do NOT conclude
  incomplete engine/report is completion. App `ap-tnowFdP7Uabc3oh5Fwo0L6`.
- CLI: `device_launch.py --run-id NEW --arena --rust-opt 1 --nvvm-opt 0 --diagnostic-ir`.
  CURRENT launcher freezes/invokes device_modal.py inside payload too. Future GPU
  target cache dir includes arena/Rust/NVVM controls; active old v10 used old shared
  target path but mount isolation avoided blocking. Host compiler has reusable image.
  Possible next improvement: include compiler source hash in target cache dir so
  backend edits cannot reuse stale dependency objects. IR tag changes currently force
  rebuild in debug attempts. Keep each output new, never change historical payloads.
- Current DeviceEnv stack64KiB, heap2GiB. If stack errors on full rules, test/record
  larger B200 stack capacity; CPU OS64MiB diagnostic is not a GPU stack checker.
- New fixed-time native qualified recipe `recipes/gpu-throughput.json`: n128, compact,
  copyfalse; old gpu-fast n32 remains exact control. Report expanded with completed
  structural results, GPU port remains explicitly unqualified. Need finish README.

### TWO original baselines, now running on fixed L4

- `evidence/tavern-implementation/original-baselines-20261108/{upstream,earliest}`:
  separate clones of exact original HEAD1c2086a...; shared alternates repacked and
  removed so metadata is self-contained. Original never reset/edited. Earliest is
  original clone overlaid with retained `learning-64/source.tar.gz`; EVERY historical
  manifest file SHA verified exact. Captured `upstream-source.json`,`earliest-source.json`.
  Original has no comparable neural learner. Earliest custom full trainer is explicitly
  distinct from original simulator. Historical gamma.99 was NOT compared for strength;
  matched replay declares gamma1/identical network/PPO/rules/seed in both arms.
- New `original_simulation.py`: reproducible real scripted trace300 calls at n32,128,
 1024, full eight-player lobbies and resets; all lobbies active in BOTH arms, no artificially
  bad redundant work in only baseline. Precomputed external player0 actions; internal7
  bots still execute in timed simulator. Hashes ALL ready feature/card/mask/result
  bytes and full final state/RNG; original and compact candidate matched validation.
 3 interleaved native timings per size. Record occupancy/late-game/natural outcomes.
- New `original_baselines.py`: those native tests, then earliest vs current full
  training at n32/64updates, n128/8updates×2interleaved, n512/4updates×2interleaved.
  Identical recipe per size; source/build/config fingerprints and commands retained.
  CURRENT source checks actual metrics.jsonl episode lists and accepted samples; frozen
  v1 script mistakenly checks stats.get('episodes') (None==None), so final analysis MUST
  recompute real episode/sample checks from retained metrics without retraining. Policy/
  optimizer/CUDA-RNG checks in v1 are genuine. Do not claim episode parity from that field.
- `original_modal.py`, `original_launch.py` install upstream/earliest/candidate into
  separate Python venvs sharing Torch2.9.1 on ONE L4/4physicalCPU/16GiB, bounded3600s,
  child3500s, zero retry/warm. Vol `autoresearch-tavern-original-evidence-20261108`.
- **ACTIVE** `matched-l4-v1` app `ap-hWPFAMINLMC8xfV6HimAgP`, local **session83259**,
  `/tmp/tavern-original-launch1.log`; full
  `evidence/tavern-implementation/original-baselines-20261108/matched-l4-v1/modal.log`.
  Image all3 installs built successfully. Worker began15:09. At15:28 app ephemeral1task.
  Subprocess output goes to artifact logs, so app log quiet until final. Let it finish.
- Provisional read-only live native results via container exec saved
  `/tmp/tavern-original-live.log`: n32 original~7.5–8.0k versus current~10.9–11.2k
  decisions/s; n128 original~9.5–11.1k vs13.4–13.8k; n1024first pair11.4k→13.7k.
  This realistic all-scripted populated workload confirms core simulation dominates;
  no10x claim. These are native-only, not neural. Final repeated results still pending.
- Original worker **ta-01M6WZTWSXPD0HSTZMRY885C2R**. First CLI exec ls/progress worked,
  later `container exec python -c ...` hangs with no output; pending sessions62515,
 29927 and logs /tmp/tavern-original-live2.log empty. Do not rely on them. Could be
  task-command router/transport. New volume get of training-progress currently pending
  **session24245**, target `/tmp/tavern-original-training-progress.json`, log
  `/tmp/tavern-original-progress-fetch.log`. Do not kill real benchmark just for silence.
- App list compact snapshot `/tmp/tavern-app-status-1528.json` confirms all old Tavern
  structural/device jobs stopped except current v13+original baseline. Other game
  apps must never be managed.

### Retention, billing and pending cleanup

- Structural ALL20 nonempty run dirs verified remote/local SHA via
  `retention-audit-1455.json`: **2,482 files / 3,429,927,497 bytes**, all match. Audit
  completed in1905.54s (~32min); raw `/tmp/tavern-structural-retain-audit.log`, session11883.
  `cuda-final-4c-v14` was actually missing locally after connection-lost retrieval;
  retried download ONLY (no retraining) successfully, immutable attempt failure kept
  and `retrieval-retry-1455.json` appended. `summary-1455.json` includes that full result.
  Before deleting structural volume, verify no extra loose files outside audited dirs
  and all owning structural apps stopped. Do not remove other volumes/jobs.
- `collect_billing.py` currently filters structural+device logs, but NOT new original-
  baseline log directory! Add those for final receipts. New observed1450 amount
  $4.30562506 is partial, earlier historical total unchanged. Need final after all jobs.
  B200 published.001736/s +CPU.0000131/core/s +RAM.00000222/GiB/s verified via Modal.
- Own volumes now4: structural-evidence, device-evidence, device-build-cache,
  original-evidence (all autoresearch-tavern-*-20261108). Cleanup only after retain/audit.
  For large future audits prefer SHA computed in same worker/CPU remote over re-
  downloading GB again; current structural full download audit already done.
- No memory read/updated, no memory citations. No skills invoked. Keep frequent user
  commentary, no permission questions; all experiment work authorized. No final done
  until working full-task GPU+learning and high-throughput architecture seriously
  validated (or concrete measured failure proof and further implementation as directed).

## Continuation 2026-11-08 15:54 — STILL ACTIVE, USER SAYS DO NOT STOP

User just explicitly reiterated: “yes you need to continue complete rewrite. do not
stop until done”. Continue actual full-task GPU runtime/learning/performance work.

- Original baseline `matched-l4-v1` COMPLETE, downloaded under original-baselines-
  20261108/matched-l4-v1/retained/matched-l4-v1. Native exact full-trace parity passes
  at n32/128/1024, direct medians1.409/1.246/1.217×. Earliest complete trainer vs
  current short n1288updates×2:3.421/3.422×; n5124updates×2:5.147/4.998× process
  sample throughput, identical episodes/shared semantic counts. n3264updates:
  354733→360201samples,243.506→60.774s,4.069× sample throughput; episode divergence
  update18. NO original learning bitwise parity: actor orthogonal init with earliest
  hardcoded4Torchthreads vs current1 differs <=1.862645149230957e-09 in27961 values.
  CPU-only `initialization_modal.py` reproduced each setting twice exactly; log
  original-baselines-20261108/initialization-diagnostic-1538.log. No GPU used there.
- NEW `summarize_original.py` appends corrected actual metrics episode/shared semantic
  count parity. Latest `corrected-analysis-1550.json` authoritative appended analysis;
  `corrected-analysis-1548.json` also retained, but its raw counts equality falsely
  flags candidate-only diagnostic additions. Original frozen summary’s episodes bug
  still unchanged; never cite its true flag. New report original section records
  these caveats and direct ratios, no multiplication. README also has device section.
- Billing collector now includes original-baseline and diagnostic logs. New partial
  `billing-observed-1549.json` observes$5.60104362; NOT final, metering lag/newwork.

### Split device implementation now complete in source
- `device/wire/` reprC shared Episode/layout8-u64 ABI vector; `state-kernels/` GPU
  export/parse/destroy/bind. Main `kernels` default featurestate_io retains monolith;
  split disables default and adds layout/validate/commit kernels. Full check_loaded
  validation stays main. Parse creates temporary handles; any invalid batch frees all
  temporary handles and preserves all old running states. Commit only all-valid.
- `device/compiler/src/main.rs` compiles main+state whenTAVERN_DEVICE_SPLIT=1.
  `device_launch.py --split` freezes it, `device_modal.py` assembles BOTH cubins;
  compiler source SHA in target path prevents stale plugin target reuse.
- `device_env.py` loads/checks/cache main+state module set, preferscubin, sharedstream,
  per-family arena binding, checks GPUlayout. save/loadroute GPUstate module; load
  transactional. `device_training.source_hash` includes bothPTX/cubins+allCargoTomls.
  Probe retains allmodules and future source adds invalid-import transactional tests.
- Local actual nvptx/no_std `cargo check` BOTHcrates with --no-default-features
  --featuresarena PASSED. Local per-game arena stress PASSED10000randomops again.
  PytestTavern25passed7CUDA-skipped, Pythoncompilepasses. No device runtime proof yet.

### Active compiler issue + v16
- v14 `device-b200-cubin-v14` FAILED CPU offline assembly exit137 after~15min,
  client959.544s; no GPUruntime. Attempt/retrievalretained. Don’t rerun sameID.
- v15 splitRust2/NVVM0 compiled both in74.92s but emitted still huge95MB main.
  Read-only CPU inspection `device_inspect_modal.py`, imageim-ULlepvVlW6smIR6RxwBb2t,
  appap-scBfhljNaWcyLGnvNnn4SM completed: main94,865,542bytes/1,747,363lines/
  41,358functions/12kernels; state18,508,262bytes/361,333lines/6,421func/5kernels.
  log `device-port/device-inspection-v15/modal.log`, worker1.495s2CPU8GiB/noGPU.
- ROOT CAUSE SOURCE CONFIRMED: SDK LLVM21 compatibility wrappers for legacy
  LLVMPassManagerBuilderPopulateModulePassManager ONLYadd libraryinfo, and function
  pipeline completelyempty. Rustopt2/3 was not trulyoptimizing LLVM! Explains gigantic
  code and assembly/load memorycliff. Do not keep retrying lowoptassembly unchanged.
- NEW `PassWrapper.cpp` patch adds TavernModernOptimizerPass legacyModulePass adapter
  around LLVM21 PassBuilder defaultpermodulepipeline, honors requestedO1/O2/O3/Os/Oz,
  inliner/vectorize/merge/unrollflags; eageranalysisinvalidation reduces memory.
  Uses originalLLVM21.1.8headersverifiedprimaryrawgithub, nofastmathchange. Original
  hashes/licenses remain; patchedhash+note appendedUPSTREAM.json. Needsrealbuildqual.
- v15 intentionallyCANCELLED during CPUassembly, cancellation-reason.json retained;
  `modal app stop ap-VJgUea3WjW1yOD80uCmMSP --env staging --yes` SUCCESSsession39637.
  Launchsession46109 mayfinishfailure/download; preserveoriginalcancellationcosts.
- **CURRENT ACTIVE v16 `device-b200-modern-opt-v16`**, local **session25869**,
  `/tmp/tavern-device-launch16.log`, ownfullmodal.logunderdevice-port/runID.
  Rust2/NVVM0/arena/split. Compilerbackendbuilding~2min typical. Waitfornewpipeline
  compile/PTXassembly thenrealB200 oracle/hardfixtures/PPO/resume. Don’tclaimqualified.
- v15readonlyCLIcontainerexec stalled as before: **session90481**,
  `/tmp/tavern-v15-assembly-inspect3.log` empty; stoppedapp shouldendit. Don’tuse
  containerexec again; CPU image inspection function worked quickly and safely.
- New v16 includes latest transactionalinvalidimportprobe and stronger sourcehash;
  v15 frozenpayload doesn’t; nevereditfrozenpayloads.

Need actualGPUqualifiedfullmechanics+learning thenmatched originalANDstrong-native
fullprocessspeed comparisons on SAMEGPU/CPU/RAM, sustainedlearning/placement checks,
allattemptbilling/retention/ownedcleanup. User requires~10×goal, do notstopatnative
5×orunqualifiedGPUreport. No root/shareddocs,othergames,staging,commits,pushes.

## Continuation 2026-11-08 16:22 — ACTIVE NATIVE NVPTX ALTERNATE

User confirmed this is AutoTavern/Hearthstone Battlegrounds and explicitly says
proceed with original full rewrite; do NOT end at interim status/report.

- v16 modern LLVM optimization reached NVVM verifier, FAILED on optimizer-created
  `llvm.fabs.f64`. Added LLVMRustLowerNvvmOptimizedIntrinsics after merge/DCE: exact
  scalar/vector integer sign-bit clear, strips unused intrinsic declarations. Verifier
  stays mandatory. C++wrapper+llvm.rs+nvvm.rs hashes/notes appended UPSTREAM.json.
- v17 `device-b200-modern-abs-v17` Rust2/NVVM0/split/arena compiled both in112.50s,
  simulation36,740,225bytes/902877lines/5012func, state9,060,613bytes/264557lines/
  1038func. Then **FAILED ptxas15min timeout124** (noGPUruntime). Image with PTX
  `im-JDvxQVGb7FtkAltPaQUBWK`. CPUinspection authoritative accurateimage receipt:
  `device-inspection-v17-correct-image/modal.log` appap-liiVcy4gAyUD5AEjC5qXLf.
  Earlier `device-inspection-v17/modal.log` measured same bytes but reported old
  defaultimage due missingenvonremote; fixedinspect(image_id) explicitparam.
- v18 `device-b200-optimized-v18`, appap-feWSFntZtanAeD6y9vkP6c, **FAILED NVVM
  optimized compile15min timeout124**. Session75777 mayfinishretention. NoGPUruntime.
- **ACTIVE v19 `device-b200-asm-parallel-v19`**, app IDinownmodal.log, session6041,
  /tmp/tavern-device-launch19.log. Rust2/NVVM0/split/arena; ptxas1,fmadfalse,
  --split-compile=4 (officialNVIDIA docs verified). Assembling since~16:10;
  900s bound each module. Imagepreassemblyim-buYRlUshHdciLkBd2dyMhr. At16:21quiet.
- Launcher now --ptxas-opt0..3, passesTAVERN_DEVICE_ASM_OPT, recordcompilerfield.
  Futureasm uses --fmad=false --split-compile=4; frozenolderpayloadsunchanged.

### Current source improvements after v17
- RulesConfig derivesPartialEq in DEVICECOPYONLY; main validate_import now takes
  currentgamesptr and requires entireeffective cfg==currentgamecfg beforecommit.
  All defaults frozen, not justexplicitPythonkeys. DeviceEnv launchargsupdated.
  Probe rejectsparse_type,card_table,andimplicitbuy_costrule mutations transactionally.
- device_learning_probe now qualify_gae ASTexecutes original rollout.py gae function
  independently against GPU (empty/variablelength,3gamma/lambda settings). Launcher
  copiesrollout.py; source_hash includesit. v18/v19 containthis, notv17.
- Standalone device_training.train now torch.use_deterministic_algorithms(True),
  matching native; v19hasit, v18frozenbeforethischange.
- NEW GPUcheckpointpack kernels `tavern_state_pack`/`tavern_pack_states` onewarp/game,
  DeviceEnv.save startscapacity256K then packsvalidbytes only beforeD2H, noCPU
  transferofrectangularpadding. SourcechangedAFTERv19freeze, needsnextqualification.
- New `device_simulation.py` exactfrozenoriginalscriptedtrace replay onGPU, full
  trace/view/result/state/RNG hashes; protocolincludesinit/H2D/finalcheckpoint,
  noCPUmetadataqueries in timedloop. NeuralPPOtimingsremainseparate. Notrunyet.
- LocalCUDAfrontendcheck4 passeslatestpack+fullcfgeq. Host4full8player differential
  rerun PASS64.77s `/tmp/tavern-host-differential1609.log`. NotGPUevidence.

### Retention audit
- New verify_retention.py CPUworker computesremoteSHAthenlocalcompares, no retraining.
  Firstauditstartupbug ROOT.parents2 onremote/root/verify_retention.py causedstartup
  crashloop; stoppedonlyownappap-pAgNnKXDUILp4wZ9ydDzU2, logsretention1604-modal.log
  and/tmp/tavern-retention1604.log retained. FixedROOT withmodal.is_local guard.
- Successfulnewauditappap-qmqOliM68RDsTmYGAPrmUk, functionfu-HT6yikFJ2TWvtkuv0D293W,
  CPU2/RAM2/noGPU42.061s, **171files/806749142bytes ALLmatch**. Receipt
  evidence/tavern-implementation/retention-audit-1609.json, logretention1609-modal.log.
  Auditsoriginalmatched-l4-v1 + devicev11/v12/v13. Structural2482files alreadyaudited.

### NEW promising alternate: stock Rust LLVM NVPTX (no custom SDK/NVVM backend)
- `device/native/` standaloneworkspace; manifests point at SAMEoriginalshared
  core/kernels/state-kernels/wire/src/lib.rs files. No parallelgamerulecopy.
- Minimal authored runtime namedcuda_std forAPIcompatibility suppliesonlythreadindex
  viaPTXregisters, `mul.hi.u64`, externCUDAmalloc/free, trap panic/OOMhandlers.
  runtime-macros #[kernel] emits extern"ptx-kernel" unsafe/no_mangle.
  NoSDKtypedfatpointer/LLVM21compatibility/override/libm/allocator-marker fixesneeded.
- Toolchain EXACT `nightly-2026-04-02-aarch64-apple-darwin`. Added itsllvm-tools and
  llvm-bitcode-linker components (done). StockRusttargetdocofficialsupports this
  target/linker. Localcompiler runs withoutGPU/CUDAinstallation. Coreusesno ln/sqrt
  orotherfloatlibcalls; serdejson nativeLLVMfabshandlednormally.
- FIRSTnativebuild COMPLETE2m45s, log/tmp/tavern-native-nvptx-build1.log:
```
RUSTFLAGS='-Ctarget-cpu=sm_100 -Cpanic=abort -Zcrate-attr=feature(abi_ptx)' cargo +nightly-2026-04-02-aarch64-apple-darwin build --manifest-path integrations/tavern/device/native/Cargo.toml -p tavern-device-kernels -p tavern-device-state-kernels --features arena --target nvptx64-nvidia-cuda -Zbuild-std=core,alloc --release
```
  PTXat native/target/nvptx64-nvidia-cuda/release/deps/tavern_device_{kernels,state_kernels}.ptx.
  HeaderPTX8.6 SM100, main~86MB,state~89MB unfortunatelylarge. Fullcompileworkslocally.
- Linkerinstalledpath:
  /Users/K.Z./.rustup/toolchains/nightly-2026-04-02-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/self-contained/llvm-bitcode-linker
  --help: -O0|1|2|3|s|z default0; --lto separateboolean. Rustdriveralreadyforwards-O2
  fromprofile, but LlbcLinker.optimize hasnoautomatic--lto (primarypinnedRustsource
  verified). Hypothesis nativehugePTXbecausefinalbitcodelinkerLTOpipelineoff.
- **ACTIVE secondnativebuild session47294**, log/tmp/tavern-native-nvptx-build2.log,
  targetdir native/target-opt; samecommand plus
  `-Clink-arg=--lto -Clink-arg=-O2` inRUSTFLAGS. Extra-O2 mayduplicatealreadyforwarded
  -O2 andClapcouldreject: ifso rerunwithONLY -Clink-arg=--lto. Keepoldoutputs.
  Sourceunchanged; newtargetretainsoriginalPTX. At16:21corebuilt/wirebuilding.
- Addednative/.gitignore /target*/. Launcherignoretarget-*; source_hashskipanypart
  targetortarget-* so generatedbuildfilesneverenterfrozenpayload/checkpointbinding.
- Diagnostic minifier `device/minify_ptx.py` shortensONLYunquotedinternalRustsymbols,
  checksmandatorybyte-exactreversibility, preservesallquoteddata/comments/publicABI.
  Testedoncompletednativefirstmain, receiptsin device-port/native-symbols-1620:
  raw86148778→80847294bytes,7480identifiers, only6% saving. NOTselected/qualified;
  longsymbolnamesarenotprincipalproblem. Do not assumeperfbenefit.
- IfstockbackendLTOproducespracticalmodules, nextstep createboundednativeGPUlauncher
  reusepreassemblybaseimageim-JDvxQVGb7FtkAltPaQUBWK withTorch2.9.1/CPUoracle/CUDA13.3,
  copyfrozenallsource+EXPLICITlocallyemittedPTXs, assembleptxasO1split4fmafalse,
  invokeexistingdevice_probe.py managedworkerclaim1800s4CPU16GiBB200. NOsource
  changesorCPUfallback. Preservelocalcompileflags/toolchain/logs/PTX/SHA aspaidsetup
  accountingcontext; runtimequalificationnotthroughputclaim. Add actualcoldcompile
  CPUsetup toreports,don’tpretendGPUallocationincluded prebuild orfree benchmarktime.

UserrequirescompleteGPUrules+neurallearningand~10×origfull-taskgoal, noearlyfinal.
Othergames/rootfilesuntouched. Noagents/commits/pushes. Fourownedvolumesstillretain.

## Continuation 2026-11-08 16:42 — STANDARD RUNTIME CUDA PASS, FULL RDC ACTIVE

- v19 FAILED ptxas300? actually old900s timeout124, client1047.813s, noGPUruntime.
  AlloldSDK/NVVMpathsremainoptional/unqualified; do notclaimfullGPUdone.
- Nativebuild2 failed because duplicate-O2 forwarded to llvm-bitcode-linker.
  Addednative/linker.py ownadapter removes Rustdriver’sunwanted--debug only,
  forwards actualtoolchainbitcodelinker and recordsargs. Build3 addsONLY--lto;
  COMPLETE2m46s but outputbyte-identical tobuild1 (main86148778/state88953473).
  No actualdebugsections existed, so removing--debug didnotreducefilesize.
- FoundhugeindividualPTXfunctions: RulesConfigdeserialize5.7MB, RawCard2.9MB,
  all_scripts2.6MB; State Playerdeserialize6.5MB etc. Main32bitFPdivisionverified
  div.rn.f32 (56occurrences), noFMA/approxFP. Fullsource/mathunchanged.
- Build4 limitsLLVM inlinethreshold+inlinehint-threshold to5. COMPLETE59.04s;
  main80051069bytes, state33374910bytes. Targetnative/target-small-inline.
- AddedGPU-only inline(never) attributes to57coldScriptregistryfactoryfunctions
  `content::scripts*`/all_scripts (oracleunchanged), preservingorder/content.
- NewOPTIN frozen_rules kernel feature compiles EXACTexistingfrozenbenchmark
  RulesConfig (8players/Implemented/actioncaps30/maxturn100/logfalse/defaultrest),
  avoidingJSONRulesConfigDeserialize. NOT a gameplay simplification. Main always
  exports`tavern_contract(out)`flag. DeviceEnvqueriesandwhenflag1 rejectsallinit
  rulekeys/valuesoutsidethatfixedtask,canonicalizesrequesttoRULES; allactualstate
  loadfullcfgequalitychecksremain. Main/NativeKernelmanifestsfeatureadded.
- Build5 withfrozen_rules+lowinline COMPLETE63s, main78348567/state33374910bytes,
  native/target-frozen. NoGPUfullgamequalificationyet. Schema/fixedtaskexplicit.

### SOURCE-BOUND native builder/launchers
- NEW native_build.py freezesallcore/kernel/state/wire/native sourceinputs (skip
  target*/cache), records exacttoolchain1.96nightly/LLVM22.1.2/Rustflags/linklog,
  runs600sboundcompiler and verifiesinputsunchanged beforecopyingbothrawPTXs.
  **native-build-v6 COMPLETE** evidence/device-port/native-build-v6. Cachebuild0.1s
  (coldbuild5actual63sretain/tmp/tavern-native-nvptx-build5.log). Beforefinalcost
  claims accountcoldlocalcompile+remoteimagebuild/transfer/GPUseparately, nofree
  prebake claims. Inputmetadata includesallnative CargoLock nowABIprojectmember.
- native/.gitignoretarget*/; launcherignores target-*; training.source_hashskips
  anytarget/target-*part. No generatednativebuildoutputsenterpayload/checkpoints.
- Device-sidecheckpointpack/newfullcfgchecks+originalGAEtest includedv6/current.

### REAL B200 primitive/runtime PASS (NOT FULL GAME)
- native/abi-kernels newdiagnostic14,547-bytePTX, SAMEleanruntime+coreallocator/
  deviceOnceLock, noGame simulation. 64threads testthreadidx, Resultwide-sliceABI,
  CUDAmalloc/Vec/drop, unsigned64highmultiply, contendedOnceLockinit.
- native_abi_modal.py appnative-abi-v20 ranactualB200/4CPU16GiB, **ALL5checksTRUE**,
  worker4.415074595s. PTX+actualcubin+result/allocationdownloaded:
  `device-port/native-abi-v20/retained/native-abi-v20/`.
  PTXsha caa08391ed76fd04637b76c16debc6d7f3daa632e6fc2b1591986f0b5926ab50
  cubinsha5d1fb213b71e38d014d701db61e74680a1d49c638ec82e12acbe050d0ecfd61e.
  Stack64KiB,CUDAheap2GiB. CUDA13.3nativecode works withTorch2.9.1+cu128.
- AppCLIENTfailedAFTERsuccess because returnedtorch.__version__TorchVersionobject
  requiredTorch unavailableinModalCLIlocalenv. RemoteJSONresults/allocationgenuine
  retained; DO NOTrerunGPU. Futurecode returnsjson.loads(json.dumps(receipt)).
  Nativefullrunnerreturnsonlyplainints/strings avoidsissue.

### CURRENT ACTIVE full-module separate compilation/linking
- Hypothesis mainptxasWPO getsstuckanalyzingthousandsofrecursive/indirectscript
  callbacks; useCUDAstandardrelocatabledevicecompilation thennvlink instead.
- NEW native_modal.py andnative_launch.py freezesallsource+verifiednative-build-v6
  PTXs+compilerreceipt, baseimage im-JDvxQVGb7FtkAltPaQUBWK containsunchangedCPU
  oracle/Torch2.9.1/CUDA13.3. DoesNOTruncustomSDKcompiler. Fullprobe managedclaim
  B200/4CPU16GiB1800s child1700, min0/retries0/singleuse, allcostsretain.
- **ACTIVE native-rdc-v21**, app **ap-Fhn7ez8sMCRgqbVDaJxklz**, **session48926**,
  /tmp/tavern-native-launch21.log, fullmodal.logunderdevice-port/native-rdc-v21.
  Started16:36:15. RawPTXuploads80/33MBtook~5minutes (slowChina uplink).
  At16:41:10 TinyABI separatecompilation+nvlink SUCCEEDED; nowmaincompiling:
  `ptxas --compile-only --gpu-name=sm_100 --opt-level=1 --fmad=false --split-compile=4`
  bounded300s then nvlink --archsm100 +CUDA13.3libcudadevrt.a bounded120s.
  Thenstate same300s+120s. IfbothworkGPUprobeautoexecutesalloracles/hard/PPO/resume.
  Currentbuildimage im-lXgeq9G8Hz2KZHVQTSMpuK. Preserveallcurrentfrozenpayload.
- Futurelauncher CURRENTworking files nowgzipPTXinputs (timestamp0) toavoidraw111MB
  uploads. `device/unpack_ptx.py` decompressesandverifiesEXACTSHA/sizebeforeassembly,
  preservingallcode/data. NativeModalcopiescompressedinputs+build.json andrestores
  beforeRDCcompile. v21frozenpayloadusesrawuploads; DO NOTeditit. No newrunlaunched.
- FullruntimeprimitivepassmustnotbeconfusedwithcompleteGamequalification. Oncefull
  CUDAworks, doGPUvsindependentCPUaction/view/mask/RNG/state/loadchecks+adversarial
  mechanics+originalGAE+actualPPO/exactresume, thenmatchedB200original+strongnative
  pureSIMANDcompletelearningprocess/fixedwalllearning+eval. Goal~10×original;
  includecoldsourcecompilation/transfers/idle/failures, novaguelooponlyclaim.

## Continuation 2026-11-08 17:15 — RDC works; portable sort qualification active

- Separate relocatable PTX compilation plus nvlink completed both full modules in
  84.59 seconds. Full v21 GPU initialization and transactional invalid imports
  passed, then first step failed CUDA 719. v23 repeated at 256-KiB stack; initial
  complete state/RNG matched the independent CPU oracle exactly, first step failed.
- New shared 64-MiB persistent arena with CUDA CAS mutex retains global registry
  allocations and allows frees without CUDA malloc's initialization cost. v24
  reduced registry prewarm 131.3s to 1.326s, game initialization 42.3s to 0.338s;
  initial full game state/RNG and invalid-checkpoint transaction tests still pass.
  This is initialization evidence, not full-game qualification or a speed claim.
- Native panic handler now reports __assertfail source file/line. v24 first step
  hits Rust core slice/sort/shared/smallsort.rs:860 (order-violation panic), despite
  pure integer sort keys. Larger stack did not resolve it. Keep v21/v23/v24 failures.
- Added device-copy portable stable insertion/merge ordering in core/src/sorting.rs
  and replaced reachable sorts; ties preserve earlier RNG shuffle order. Unit test
  compares original stable CPU ordering over duplicate keys and sizes 0..2049 PASS.
  Original candidates/AutoTavern remains unchanged. Root cause may still be deeper
  backend behavior: full CPU/GPU differential tests remain mandatory.
- ACTIVE full host four-seed differential, session67524, log
  /tmp/tavern-host-sort-differential.log; source-bound native-build-v8 session14973,
  log /tmp/tavern-native-build8.log. Next launch new native-portable-sort-v25 only
  after build/source verification. No successful full GPU rules/PPO/resume yet.
- Latest stock compiler path uses native_build.py then native_launch.py; compressed
  exact-SHA PTX uploads, RDC ptxas O1/fmad=false then nvlink libcudadevrt. Runtime
  B200/4CPU/16GiB, 1800s bounded, zero retries/warm pool, claim committed before child.
- Retention audit1609 verified171 files/806749142 bytes for original/v11-v13; newer
  native attempts still require final SHA retention audit, billing and owned cleanup.
  Original corrected metrics analysis1550 is authoritative; raw receipts unchanged.
- Both v8 checks now PASS: full host differential63.05s, native compile66.02s.
  Copied correctness logs into native-build-v8; main87,063,200bytes, state37,444,662.
  ACTIVE native-portable-sort-v25 session28991, /tmp/tavern-native-launch25.log.
  Native v20-v24 CPU-only remote/local retention audit session93676,
  /tmp/tavern-retention-native-1721.log; verify_retention.py now accepts repeated
  --device-run/--original-run without changing historical default audit scope.
- v20-v24 audit COMPLETE:31files/691855303bytes match remote/local exactly,
  evidence/tavern-implementation/retention-audit-native-1721.json (CPU9.267s).
- v25 FAILED before image/object creation: Modal heartbeat/gRPC connection loss.
  Verified app already stopped and remote v25 worker directory absent. All payload,
  logs/attempt and connection-failure-analysis.json retained. No GPU executed.
- ACTIVE explicit new attempt native-portable-sort-v26, session92287,
  /tmp/tavern-native-launch26.log, appap-hQ0RAhdQf9NrHt9HMunylI. At17:20 building
  image layers normally with same verified native-build-v8. No retry hides v25.
- Prepared device_matched.py / device_matched_modal.py / device_matched_launch.py.
  Syntax checked; NOT launched. Gate requires actual retained differential.json
  passed + full neural qualification, reuses exact qualified PTX/cubin/source.
  Independent original/earliest installs and strongest current native integration,
  one B200/4CPU/16GiB,2 interleaved repeats,300-step realistic full simulator traces,
  then real PPO sizes128/512. Full process and complete protocol separately timed.
  Setup/compile receipts remain separately charged; no multiplied historical ratio.

## Continuation 2026-11-08 17:36 — FULL GPU CORRECTNESS / PPO / RESUME PASS

- **native-portable-sort-v26 SUCCEEDED** on B200/4CPU/16GiB. Worker183.064s,
  full gate180.194s. All180steps each for controlled[0], [3,0], all8 match CPU
  views/masks/results/full Game state/RNG/checkpoint roundtrips; natural counts2/0/7.
  All5adversarialfixtures PASS (180/180/1/180/180steps), originalGAE3casesbitwise,
  realCPU/devicecollection+PPO exactdata/episodes/counters/policy/optimizer/loss/
  CUDARNG. Two full updates =1update+save/resume1 in everypolicy/optimizer/RNG/state.
  Parameters change. **This is genuine complete GPU game + neural learning proof.**
  OriginalCPUoracleunchanged; no CPU transitions called byGPU. Performance NOT done.
- v26 artifacts fully downloaded, retrievalreturn0. Finalimmutable image
  **im-7IaQ7CBLAw20j1HF1Jh3MJ**. MainPTX6d3b062b..., cubin
  dcc19462be8101ce8f52dd88c3339c59f8e81a29832443a2a15275049d81bc9f;
  statePTXcb3407f4..., cubin2ef79ca82d75844cedabc8eef4420fd58aa48fa0f85ec44289a795470ab1cb6c.
  Full SHA audit forv26 stillneeded. README/report updatedboundedcorrectnesspassed.
- IMPORTANT smalln8diagnostic: nativeCPUtransportcollection0.784812s vsGPU10.658247s,
  SAME594samples,8natural,0invalid,129batchsteps. GPUsimulation/metadata9.921s
  dominates. Not a speedup; needwidebatchANDcompileroptimization. Per-gamearena1MiB
  proofhardtraces canhave57–86CUDAspills; neuraln8usesnone, peak317120bytes.
- **ACTIVE matched pureSIM native-device-matched-v27**, session77475,
  /tmp/tavern-device-matched27.log, appap-XOQZ25WmHo5UXQmlcOowCe,
  ownedGPUcontainer **ta-01M6X823E6XKPVQVJ5WPVCQQWR**, started17:33:04.
  sizes128/1024,full300-steptrace, independentoriginal/currentnative/GPU validation
  then2interleavedtimedrepeats. OneB200/4CPU16GiB3600s,zero warm/retry, workerclaim.
  First independently buildsupstream+earliestvenvs inCPUimage. Old source snapshots
  and receipts preserved. Matchedlauncher gatesonpassedfullproof AND actualGPUloaded
  moduleSHAfromv26log, rechecksallPTX/cubinSHAINSIDEworkerbeforemeasurement. It reuses
  immutablequalifiedimage ratherthanuploading260MBofmodulesagain. Parentsetupcharged.
  Original/strongnativeactualsourceidentitiesmustbechecked in retainedbuild_info!
- `device_matched.py` doesnotprintarmprogress(v27alreadyfrozen, donoteditpayload).
  Firstvolumeprogressfetchbeforeautocommitwasmissing. ToinspectliveOWNcontainer:
  `modal container exec --no-pty ta-... -- python -c '...'` (`--`neededbeforepython).
  ACTIVEdiagnosticsession5484, log/tmp/tavern-v27-live-progress.log (readps/list/GPU).
  Owncontainerlistrecord/tmp/tavern-v27-containers.json. Nootherappsmodified.
- **native-build-v9 COMPLETE**:sameunchangedgamesource, rustopt3, inline225,
  localcompile180.013s, main109063145bytes/state107491359bytes. Sourceboundreceipt
  includesflags/toolchain/inputhashes. Strongeroptimizationisnotyetqualified.
- **ACTIVE CPU-ONLY optimized assembly native-opt-build-v28**, session39725,
  /tmp/tavern-native-opt-build28.log, appap-qGHzLv0Of1OOuyUhVIl8jV.
  `native_launch.py --build native-build-v9 --asm-opt3 --build-only`.
  NoGPUwillrun. BuildsRDCptxasO3thenlinks, bounded300s/module+120link. At17:35
  uploadingdefinitionsafterInitialized; mayhaveconnectiondelay. Preserveeveryattempt.
- New `native_reuse_launch.py --compiled-attempt native-opt-build-v28 --run-id NEW`
  requirescompletedCPU-onlybuild, reusesitsimmutableimage/source/compilerreceipt,
  runsfullGPUgatewithoutrecompile. **Onlylaunchafterv27GPUfinishes** tokeeponeGPU.
  `prebuilt_native_modal.py` nowacceptsTAVERN_NATIVE_PREBUILT_IMAGE; propagatesactual
  IMAGEtoremote, TRACE_KERNELS1/stack256KiB/CUDA_LAUNCH_BLOCKING0(graphsafe).
- Nativebuildernew--rust-opt2|3 and--inline-threshold bounded; launchernew--asm-opt
  and--build-only. Defaultsremainqualifiedv8opt2/inline5/assembler1. Buildonlydoesnot
  fetchnonexistentremoteGPUdirectory; explicitlyrecordsgpu_executedfalse.
- Preparedmatchedlearningphase sizes128/512 but NOTlaunched. AfterbaselineSIM,
  inspectrealGPUtimes/occupancy; larger4096cohorts mightamortize GPUunderoccupancy,
  butoriginaltrainer16GiBCPURAMmuststillfit. DonotclaimratiosforOOMbaselines ormix
  workloadsizes. Needactualmatchedoriginal+strongnativefullPPO,3seedlearning/eval,
  finalcosts/retentionaudit/ownedcleanup. Userrequirescontinueuntilcomplete.

## Continuation 2026-11-08 17:49 — WIDE STATE EXACT; OPTIMIZATION CONTINUES

- v27 FAILEDvalidationbeforetimedtrials: sameallviews/masks/results/natural240,
  finalstatehashdiffered. Rootcomparisonbug: GPU Episode additionallyserializes
  batchseed/controlled/k; nativeEpisodehasONLYgame/index/finished (NOseed).
  Oldnormalizationforgotseed. Rawv27receiptsunchanged. Worker207.996s retained.
  ActualindependentCPUbuildsconfirmedupstreamaabaff19d45b25ea, current7b257f56e12468e9.
  DoNOTmistakethisforanactualhiddenGameruledriftoracceptpartialstatehashes.
- Newstrictdevice_state_schema.native_episode validatesexactGPUkeys,seed/seats/k/
  unusedfinishedpadding, projectsONLYwrappermetadata; completeGame/RNGuntouched.
  RegressiontestPASS1 (identitypreservation+badmetadatacases). device_simulation
  usesit. Bothsimulatorsnowretainfullnormalizedvalidation.states.json, so future
  hashfailurescanbedebuggedwithoutrepeatingGPUruns. NochangeGPUcheckpointformat.
- **native-state-schema-v29 PASSED**, session10479,/tmp/tavern-native-state-schema29.log,
  appap-J4QAMXRx9WO4a0GJ2DdIq3, qualifiedv26binaryimage reused,128games300steps,
  38,400decisions240natural,0faults. Alloriginal/native/GPUstatefiles byteEXACT
  in nativeschema, finalhashbedd16809fd371eb3c7fae1a2e39c65b7c1e98b0adf44e640205d54518963dac,
  viewmaskresulthashb75083deaa61a2ed7b20e5ba39222347faa80065012b6e3601ed2226d5599c78.
  Newanalysisdevice-port/state-schema-correction-analysis.json. Retrievalreturn0.
  --trials0 intentionalvalidation-only; no validtimedtrial orspeedupclaim yet.
- **PerformanceisVERYPOORcurrently**:v27validationGPUprotocol198.82s, GPUstepinterval
  180.21s vs originalCPUprotocol1.492s/strongCPU1.107s. Validationintervalsinclude
  validationgaps, so notfairthroughputheadline, butclearlynoreasonclaimfast. v29
  correctedGPUprocess210.57s. Needrealruntimeoptimization, notmerelywiderbatch.
- **native-opt-build-v28 FAILED CPU-only**,843.89sclientsetup, noGPUexecuted.
  Mainopt3/inline225/RDCassembler3SUCCEEDED, huge107MBcheckpointPTXassemblerhit300s
  timeout124. Keepallcosts/failure. DoNOTreuseascompletedbuild-onlyparent.
- NewruntimechangesCURRENTsource(afterqualifiedv26, notyetGPUqualified):
  device_once::get_or_init #[inline(always)]fastacquireread, coldallocation/CAS
  moved#[cold]#[inline(never)]initialize; EXACTsameacquire/releaseprotocol.
  kernelprewarmaddsscript::implemented(CardId0), initializesfullimmutableCACHEon
  oneGPUthreadtoavoid128racingexpensivetext-derivedconstructorcopies. Priorv27
  persistentalloccount2.6M vsn8~106k suggestsCACHEracehugecoldcost. Noplayers/rules
  simplified, CPUoracleunchanged. Profile/performanceeffectnotyetmeasured.
- native-build-v10 COMPLETED177.401s,main118526759/state107491230byteswithnewruntime,
  rustopt3/inline225. NoGPUrunfromthisall-aggressivepair.
- **native-build-v11 COMPLETE**:mainexactv10cached0.190s (39cf407c72bd65ddd5058a46f3fad90b93a1020d7a5906e323b52b31c313f306),
  state rebuiltopt2/inline5 in35.07s,37444662bytes,
  292ed0a4358e37496baee382da4a915107177f47958f95a2099dd87f505613ee.
  SAMEsourceinputsforbothmodules, profile/flags/commandspermoduleinreceipt.
  native_build.py now optional--state-rust-opt/--state-inline-threshold, preserves
  defaultcombinedoldbehavior. Sequentialtwo-phaseoutputs copiedbeforechangingflags.
  CompilationTimeoutExpirednowrecordedreturn124+elapsedratherthanlosingreceipt.
- **ACTIVE native-fast-once-v30**, session **73298**,
  /tmp/tavern-native-fast-once30.log. Newnative_build-v11,
  native_launch --asm-opt3 --state-asm-opt1, fullGPUqualificationautoafterCPUassembly.
  OnlyoneGPUjob;v29alreadycompleted. NativeModalmain/stateasmoptsnowindependent,
  recordedperattempt. B200/4CPU16GiB1800s zero retries/warm claimbeforechild.
- NewDeviceEnv.kernel_resources readsCUDAattribute0/1/3/4 forinit/observe/step/
  stateexport; DeviceProbeprintsresources+actualSMcountonce. Thisisreadonlyprofiling,
  syntaxcheckedbutactualv30runtimewillqualify. Sourceboundcheckpointincludesnewenv.
  Native_reuse_launch overlaysfreshenv/probeaswellascontroller, corestaysparentfrozen.
- ACTIVECPUretentionauditv26/v27/v29 session50212,
  /tmp/tavern-retention-qualified-v26-v29.log, output
  evidence/tavern-implementation/retention-audit-device-qualified-v26-v29.json.
- UpdateREADME/reportagainwithwide128stateproofandhonestpoorperformance. Earlier
  reportstillmentionsboundedn8proofcorrectly, n8GPU10.658vsCPU0.785. OriginalL4
  native/fulltrainerbaselinesremainunchanged. Needfull3seedlearning/eval,matched
  optimizedCPUorigANDstrongnativepureSIM/fullPPO,finalbilling/retention/owncleanup.
  DO NOTstopatfunctionalGPUrewrite;userrequestscompletegoal~10×original.

## Continuation 2026-11-08 18:07 — FAST ONCE PASS; DISPATCH ISOLATION ACTIVE

- native-fast-once-v30 **FULLGATEPASSED**,151.875s (oldv26 180.194s). Independent
  CPU/device594-row8-lobbyneuraldata/optimizer/policy/loss/RNGexact; GPUcollection
  6.823815s (simulation6.110786s),oldv26 10.658s. StillfarbehindnativeCPU.
  Image**im-3fA0G5xbe2kx7Dc0ltuOxj**, appap-FNfDjaeYY7lPMSow5zuF0L.
  MainRust3/inline225/assembler3,newcoldoutlinedOnce+single-cacheprewarm;
  stateRust2/inline5/assembler1. FullsameGame/RNG/state/checkpoint/adversarial/
  GAE/PPO/resumePASS, initialstatessame. Retentionv30downloads maystillbeactive;
  finalSHAauditneeded. ActualCUDAresourcesALL4entries255regs/thread,maxblock256,
  0staticlocalbytes,148SMs. Dynamicstackdoesnotappearinstaticlocalbyteattribute.
- v26/v27/v29 SHAaudit COMPLETE42files/376946429bytesallmatch,CPU12.132s,
  retention-audit-device-qualified-v26-v29.json. Historicalledgersunchanged.
- NewCURRENTDeviceEnv optionalblock_threads/envTAVERN_DEVICE_BLOCK_THREADS accepts
  1/2/4/8/16/32, default32. Logicalglobalindexunchanged, fullcheckpointpackkernels
  remain32threads. DeviceTraining.runtime_options saveresumechecksincludeactual
  blockwidth; DeviceProbe finalproofnowincludesruntime_options. ThesePythonchanges
  areNEWvsfrozenv30, notyetuniversallyqualified. Donotenableunprovenwidths.
- **native-dispatch-v31 FAILED** afterwidth1PASSEDall128games300stepshashesexact,
  diagnosticprotocol56.620974s. Width2thenhad190rejectedactions/0faults. Stopped;
  noselectedwidthfullqualification. CouldbewidthsemanticsORcontextreuse/global
  lifetimebug. Donotassumecompilerbugoracceptfasterinvalidresults. Needisolation.
  Sourcebinarysamequalifiedv30, freshPythonruntimeexplicitexperimentalphase.
- `device_dispatch.py` sweep1,2,4,8,16,32,exactreferencehashes, thenselectedtiming+
  fullProbe/PPO/resumegate; copiesselectedproof/modulesroot forfuturequalification.
  Preparedbutv31stoppedcorrectlyatinvalid2. Futuredispatchcanonlybequalifiedafter
  rootcausefixed. `device_simulation` currentlychecksrejectedBEFOREsavingstate_out
  sofailedmodehasno statedump; isolationdriverbelowremediatesdiagnosticvisibility.
- **ACTIVE native-dispatch-isolation-v33**, session**33110**,
  /tmp/tavern-native-dispatch-isolation33.log, appap-QG8t5uzaL2qCVN8BzYNyAB.
  phase dispatch_debug, samev30immutablebinarysource, newPythonruntime. Driver
  device_dispatch_debug.py runscpu/GPUinitialfullstate,everyview/mask/reward/result,
  finalstateforwidth1-first,width1-repeat,width2-reused-context,thenwidth2-fresh
  childCUDAcontext. FirstmismatchwritesCPU/GPUstates+pathdiffs(case.json). Allcases
  diagnosticsfixedbounded,nottrainingqualification. ParentGPUmemory~82GBstack plus
  child82GBstillfits191GBphysicalB200; onephysicalGPUworker4CPU16GiB.
  casefailureisretained/expecteddiagnostic,drivercancompletewithall_cases_passedfalse;
  allocation.passedreadsactualall_cases_passed,nofalsecompletionproof.
- Matchedlauncherphase dispatch/dispatch_debug skipsunneededoriginal/earliestCPU
  venvbuilds,usesfrozenv29originaltrace/reference. Itscompiledsourcecomesfrompassed
  parent, wrapperPythonnewexplicitexperimentalfrozenmodel/learner/rolloutunchanged.
  Copiescompiler-receiptintooutforfuturequalifieddispatchparentcompatibility.
- **native-reg64-build-v32 FAILED CPU-only**:mainptxasO3withboth--maxrregcount64
  and--device-function-maxrregcount64 hit300s timeout. NoGPU. Shaderinstruction/
  spillregistertradeoff documentedinNVIDIAofficialPTXCompilerAPI12.8 docs:
  https://docs.nvidia.com/cuda/archive/12.8.0/ptx-compiler-api/index.html
  BothentryanddevicefunctioncapflagsneededforRDC; ABIminimumautobumps. No proof
  capactuallyworksuntilassembly+actualcuFuncattributes+fullGPUgate.
- **ACTIVE native-reg64-o1-build-v34**, session**85690**,
  /tmp/tavern-native-reg64-o1-build34.log. Sameverifiednative-build-v11source/PTX,
  assembler1bothmodules,cap64,boundedCPU-onlybuild. NoGPUwillrun;safeparallelv33.
  NativeModal/Launcherrecordmax_registers, passbothentry/devicecapflags. Reuse
  launcherpreservesstateassembleropt/maxregmetadata. OnlyreuseafterCPU-onlySUCCESS
  andafterv33GPUfinishes;keeponeGPUworker.
- NextcandidateideaNOTIMPLEMENTED: onceprewarmfullyconstructsimmutableDB/Registry/
  Implemented-cacheononeGPUthread, canpublishreadonlypointercacheatkernelboundary
  forhotlookups, avoidrepeatedld.acquire.gpu loads. Mustguaranteelifetime/publication
  andresolveexistingcontext/widthcorrectnessfirst. DoNOTclaimthisalreadydone.
- NeedupdateREADME/reportwithv29widehashproof,v30newboundedproofandhonestnegative
  throughput; finalcostsnotreconciledsincebilling-observed-1549. Neverclaim10×yet.
  Noothergames/rootdocs/CPUoraclechanges,commits/pushes,subagentsorcleanupperformed.

## Continuation 2026-11-08 18:40 — THREAD INTRINSICS QUALIFIED; WIDE/CACHE ACTIVE

- v33isolationCOMPLETE:width1-first61.014s andwidth1-repeat58.698s BOTHfull128x300
  CPU/GPUinitial/everyview/mask/result/finalstateexact. Width2reusedANDfreshcontexts
  BOTHfirstdivergebeforeactionstep42, lobby98 (refreshaction1atstep41), sibling99
  buyaction10(ShinyRing). CPU/GPU98next_eid326/331, GPUextra5freshshopIDs. Thusnot
  contextreuse. v33all_cases_passedfalse;diagnosticoperationreturn0notqualification.
  FullCPU/GPUstatedumpsretained; sourceoracle7b257f56e12468e9 verifiedlocalmatches.
- GeneratedindependentCPUone-stepfixture fromfulltracebeforestep41, games98/99:
  evidence/device-port/dispatch-step41-fixture/fixture.json. Confirmedloadedinto
  two-gameCPUenvone-stepafterstateEXACToriginal128-gameCPUafterstate. Bru'kanvs
  Patchwerk, nohelpfulrefreshtrinket. Copiednewvalidationfixtureinto
  device/fixtures/dispatch-step41.json;SHA6bb73677c9f21db5f53bb226bce800ef3daa28f698f62f6aa2be9dc30826e9a0.
  Device_divergence_probe.py testsallwidths1/2/4/8/16/32 onthissnapshot,comparesall
  views/masks/metadata/fullGame/RNG. FullProbe nowrunsitbeforeexistingfullgate.
- NativeRuntime thread::index_1d nowusescore::arch::nvptx::_block_idx_x(),
  _block_dim_x(),_thread_idx_x() withstdarch_nvptx feature instead ofopaque"pure"
  inlineasmmovreads. Hypothesis compilerdivergenceanalysiscouldmisreadoldasm;
  notyetprovenrootcause. Pinnedrust-srcprimarydefinitionsverifiedin
  library/stdarch/crates/core_arch/src/nvptx/mod.rs (llvm.nvvm.read.ptx.sreg.*).
  PTXbra.uniCOUNTSactuallyincreasev11→v12 (108269→110636), soDONOTclaimstatic
  countsprovefix. Neednewfull128width2testbelow. Oldone-stepfixturewasnotrunonv30.
- **native-build-v12 COMPLETE164.763s**, main118185625bytes sha606f7e9e77f8fe69efda95e446d8c700fb7cb5c27606caeb99defa8c9094dafc,
  state36911110bytes sha3827ae54ee880b99942f0338070796b9d1020dd269347cfc0168394ac89c63e0.
  Sameprofilemain3inline225/state2inline5. v36launchconnectionfailedbeforeGPU.
- **native-thread-intrinsics-v37 FULLGATEPASS**, appap-aZX6JjD5OhTIh3Gndhqd4D,
  immutableimage **im-oKgQtlyMWOdVn1fl4ULuRx**. All6widthson2-gameone-stepregression
  PASS, allold180x3/hard/GAE/neural/exactresumePASS. Newcheckpointsource_hashalso
  hashesALLdeviceJSON(includingoriginalcarddata), notjustRust/Cargo/PTX/CUBIN.
  GPUtwo-gamefixturewithsourceGame98seedloadedasgame0/1containsnoreset;independent
  CPUassertprovesone-stepindexindependence. Broadwidth2teststillneeded.
- v37clientconnectionlostAFTERinvocation; **reattachedSAMEcall**, return0,
  fc-01M6XBBXYKE1AAPKZSNGVPVNDQ. Noexperimentresubmitted. call-recovery.json retains
  freshgetresult+8.769sreattachtime. Launcherinitialclientreturn1/effectivereturn0
  saved. Result/largeartifactsdownloadingmaybeongoing; remoteSHAauditstillneeded.
- Newlocalmodal_call_wait.py usespublicFunctionCall.from_id(...).get(timeout),
  parsesrecordedrun-matchingcallid. Native/matched/reusecontrollersnow.spawn(),
  printapp+FCIDs,thenget. Launchers use`modal run --detach` andonlyreattachrecorded
  invocationifclientfails. NOnewGPUattemptonreattach. **Detachedappsneedexplicit
  ownedcleanupafterterminal**;min0/warm0/scaledown2/worker1800stillboundresources.
  Ifclientfailspre-FCID, inspectworkerclaim/appbeforefreshattempt; neverblindrerun.
- v34 andv35reg64O1CPU-onlylaunchesbothfailedconnectionBEFOREassembly/GPU, not
  evidenceO1registercapfailscompilation. v32O3capmainhit300stimeout. Capsourcev11
  nowoldthreadasm; superseded. Preserveallattempts/billing,neverresetledger.
- **ACTIVE native-intrinsic-wide-v38**, session**59149**,
  /tmp/tavern-native-intrinsic-wide38.log, qualifiedv37binary, phase dispatch_debug.
  Testswidth1first+repeat,width2reused,width32reused,thenwidth2freshchildcontext,
  EACHfull128games300stepswithCPUeveryview/mask/result/fullstatechecks. Usesfresh
  wrapperruntime,compiledsourcefrozenv37(doesNOTincludepublished-cachecandidate).
  OneB200/4CPU16GiB3600s, claimbeforechild. Allcasespassneededtoattributewidefix.
- **NEWCANDIDATEimplementedbutNOTGPUQUALIFIED**: read-only Published<T>pointer
  cacheforDB,Registry,implementedBoolVec. Publishedonlysingle-threadmainprewarm
  AFTERcanonicalOnceinit,neverchangesafterward; allGamekernelsoneorderedstream.
  Hotlookupusesld.global.nc.u64 (readonlyL1) ratherthanrepeatedld.acquire.gpu.
  CachedzeroalwaysfallsthroughcanonicalacquireOnce (safe). State-moduledoesnot
  publishandkeepscanonicalfallback. Globaltableownersretainedmodulelifetimeby
  existingpersistentTensor/Once; noGamestate/RNG/ruleschanges.
- Sourcefilesdevice_once.rs Publishedclass,cards.rs dbfastpointer andpublisher,
  script.rs Registry/implementedfastpointers andimplemented_cachehelper,
  mainkernelprewarmpublishcalls. Sourcehost4seedfull8playerdifferentialPASS61.02s,
  /tmp/tavern-host-published-differential.log. AllPythontestsPASS26/7skipped4.34s,
  /tmp/tavern-pytest-after-device-work.log. CPUoracleunchanged.
- **native-build-v13 COMPLETE166.074s**, main117172386bytes shaf4516f47532e9af8983b8122ab74283c325397fc29bb66a20c3c2c22ab5abf9d,
  state36911110bytes shad83a3d003c32cac6470847e0c68e0658af43d7b0dd58a0e5f47bf51e24c35810.
  Inputsunmodified. Sourcecurrentlyv13; native-launchingolderv12fromrootnowfails
  sourceguard(correct); reusev37frozenparentifneeded,donotresetrootnewwork.
- **ACTIVE CPU-ONLY native-published-cache-build-v39**, session**45048**,
  /tmp/tavern-native-published-cache-build39.log. v13mainassembler3/state1,
  --build-only(noGPU). AfterSUCCESSandv38GPUdone, qualifyvia
  native_reuse_launch.py --compiled-attempt native-published-cache-build-v39
  --run-id NEW. REUSElaunchercurrentlyonlyoverlaysenv/probe; parentv39includesnew
  helper+fixturealreadysofine. Olderparentslackdevice_divergence_probe/schema;
  addthem/fixtureoverlayifusingoldparent (donotaccidentallyinvokeincompletepayload).
- NVIDIAprimaryregisterflagdocsretrievedviawebsearch(API12.8), bothentryandfunc
  capsapplyRDCbutremainunmeasured. Fullweb/curlLLVMsourcefetchfailed/hung;terminated
  readonlywebcells. NoLLVMsourceevidencebeyondpinnedRustintrinsicsdefinitions.
- Needupdatehandshake/statusafteractivejobs, end-to-endmatchedorig+strongnative
  pureSIM/fullPPO,3seedlearning/eval,finalbilling(allpost1549jobs),SHAauditsv30+,
  explicitdetachedapp/4ownedvolumecleanupafterretention. Current10×goalUNACHIEVED.
  Userexplicitlywantscontinueuntilcomplete;noearlyfinal.Noagents/commits/pushes.

## Continuation 2026-11-08 19:10 — PUBLISHED CACHE PASSES; SCALING/NVVM ACTIVE

- v38 COMPLETE: proper thread intrinsics did NOT fix width2. Width1 first/repeat
  full128x300 exact63.791/61.002s; width32 exact127.710s; width2 reused/fresh still
  first visible mismatch before action42. Do not claim the thread-index hypothesis
  was established. Diagnostic all_cases_passed=false, return0 only means completed.
- v40 hidden-state watch COMPLETE: first actual Game98 difference AFTER action32
  (end turn), CPU/GPU next_eid319/324; player7 actions6/9 and board differs. Width32
  passes all300 full hidden-state snapshots. A new independently checked CPU two-game
  step32 fixture joins step41. BOTH isolated fixtures pass all six widths in v41;
  history/allocation/static state remains a possible cause, not proven.
- v41 published-cache FULL GATE PASS165.245s /169.629 allocated worker seconds.
  Image im-Encx7d61BHMhCAkY3QyMCv; app ap-nFb3tNiHnDTjP5tiNNyl1h (now stopped).
  All rules/adversarial/GAE/real PPO/exact resume + both regression fixtures pass.
  Runtime width32, stack262144, heap2GiB, arena1MiB. Source stays v13.
  Bulk retrieval FAILED connection and left sparse/zero files. Small authoritative
  files successfully re-fetched separately via Volume.read_file into small-retained/.
  Do not trust bulk retained/ until a SHA audit succeeds.
- device_matched_launch now explicitly recovers a complete printed worker proof if
  downloaded proof JSON is absent/invalid, retaining a new qualification-proof.json
  and proof_origin. It never invents proof, reruns a worker or overwrites history.
- ACTIVE v42 native-published-scaling-v42, session12532,
  /tmp/tavern-native-published-scaling42.log, app ap-2PLKND6iVpdnzexGv4CROU,
  fc-01M6XDA7TXP9EKZG5NQK6R0A27. Phase dispatch_scaling, sizes128/1024/4096,
  width1/32 only,300 steps. 128 every visible field + full final Game/RNG checked;
  larger batches full final state hashes/outcomes exact required. Same B200/4CPU16GiB.
  First128 width1 PASS: native .957221s vsGPU52.227757s (0.01833x). No10x claim.
  Job has3600s bound; no new GPU until terminal. Larger batch RAM/timeout remains risk.
- ACTIVE CPU-ONLY nvvm-rdc-build-v43, session80548,
  /tmp/tavern-nvvm-rdc43.log, app ap-3BjIIE7F859n8gm2oLCKow. New nvvm_rdc_launch.py
  freezes full current source; nvvm_rdc_modal.py reuses pinned patched SDK base
  im-JDvxQVGb7FtkAltPaQUBWK and tries SDK/NVVM PTX with separate ptxas compile-only
  +nvlink (the method solving stock assembly). Rustopt2, main/state assembler1,
  full cfg parser (old compiled driver does not select frozen_rules), CPU compile
  timeout900s, each assembly300s/link120s. No GPU execution until separate full gate.
  SDK source/compiler provenance retained; no claim its ABI/runtime passes yet.
- Modal app list at19:05 confirms all older owned v30-v41 apps stopped; only v42GPU
  and v43CPU build active. No other game resources touched. Billing still needs fresh
  receipt, and postv29 full remote/local artifact SHA audit remains outstanding.

## Continuation 2026-11-08 19:22 — LITERAL IDS / ORDERED TABLES; DIAGNOSTICS

- v42 FAILED after width1 full128x300 exact52.228s vs native0.957s. Width32 reused
  after width1 produced CUDA illegal-address/XID13. No1024/4096 results reached.
  Root source now gates read-only-cache asm under optional published_cache (OFF).
  Preserve v41/v42; bounded v41 proof does not qualify broad reuse or speed.
- v43 CPU-only SDK build lost connection after369.512s before any FCID/GPU; no proof
  compiled/assembled. No retry. App ap-3BjIIE7F859n8gm2oLCKow.
- v44 COMPLETE: old v37 modules,width2,128games first34steps. Compute-sanitizer
  memcheck reports0errors,291.340s. Small authoritative files in small-retained/.
  LOCAL independent CPU replay of exact shortened trace confirms visible digest
  exact but FULL FINAL STATE HASH WRONG (local-independent-analysis/comparison.json).
  Thus checker did not mask bug. It cannot exclude within-arena misuse/uninitialized
  values. v44 old allocation.passed means driver completed, NOT semantic pass (fixed
  in root wrapper). Same FCID recovered, no resubmission. Appap-jlfXyzwZuR5upHbrzVjF01,
  FCIDfc-01M6XDM9E7MWKBE83HQCP4MF8T.
- ACTIVE v46 native-initialization-check-v46, session13923,
  /tmp/tavern-native-initcheck46.log, appap-iIw24s1fgZqRDPcWwJ2Rsi. Oldv37,width2,
  first34steps, compute-sanitizer INITCHECK; compares independent CPU state hash.
  Separates memory_check_passed/state_exact/all_cases_passed. One B2004CPU16GiB3600s.
  NO NEW GPU until v46 terminal. Could later disable per-game arenas while retaining
  persistent arena to let sanitizer track individual mallocs (not implemented).
- New core/build.rs derives sorted literal name/index table +FNV fingerprint from
  exact cards.json (serde_json build dependency). cardid! now const-evaluated binary
  search. Unknown literals fail compile. Additional142 cid(literal/UPPERCASE_CONST)
  and db().id(literal) sites become cardid!; dynamic names keep maps. Oracle untouched.
  Ten host units validate ALL3983 indices/fingerprint. Four-seed full8player host
  differential PASS7.86s release; earlier61s was debug, NOT an8x optimization claim.
- native-build-v14 COMPLETE160.402s: main117978848sha8c53dee623d66a89accf37050d7c28d4b1966a033e9717c1dd0d1090feba2d0d,
  state36911110sha391a429c13e4f36a566d043b3543137bdbf47cd6c10e643103eb9f52bc205a8e.
  Literal IDs on, published_cache OFF, ordered_tables OFF.
- ACTIVE CPU-only native-literal-build-v45, session75104,
  /tmp/tavern-native-literal45.log. v14 mainasm3/state1 --build-only; assembly active.
  Can qualify via native_reuse_launch after success and v46GPUdone. Root advanced;
  do not bypass source guard/relaunch old v14 from changed root.
- CURRENT ROOT optional ordered_tables architecture GPU UNQUALIFIED: Prewarmed<T>
  ordinary raw pointer in UnsafeCell; single-thread prewarm constructs DB first,
  Registry second, implemented cache third. Getters assertnonnull; ordinary global
  loads, NO asm cache, NO constructor/Once fallback. Canonical Once owns objects in
  persistent arena for module lifetime. Host unchanged. Simulation/state each prewarm
  their OWN tables; no callback addresses cross modules. All operations owning ordered
  CUDA stream. Repeated prewarm only republishes SAME pointer. Removes JSON parser/
  factory calls from hot simulation call graph, preserving content/rules.
  Feature propagated root/native core/kernels/state-kernels; builder --ordered-tables.
  tavern_contract bit0 frozen_rules, bit1 ordered_tables (old0/1 compatible); DeviceEnv
  validates0..3 and calls state prewarm if ordered_tables. kernels_only used solely
  ABI/GAE, no table reads. Host ten units+four-seed differential PASS8.48s release.
- ACTIVE LOCAL native-build-v15, session59302,/tmp/tavern-native-build15.log,
  mainRust3inline225/stateRust2inline5,--ordered-tables. DO NOT MODIFY native/core
  inputs until inputs_unchanged receipt written. Then CPU-only assembly NEW attempt,
  then fullGPUgate AFTERv46done. Actual registers/performance not known.
- New device_learning_campaign.py NOT RUN: fixed3seeds12345/45678/78901,
  64updates×32GPUlobbies each, seat0 vs7bots; initial/final paired public-practice
  evaluation all8seats×2seeds×2games=32episodes (CPU oracle+CUDA policy). Requires
  natural/noinvalid/finite changingweights. Needs phase wiring+selected runtime full
  gate BEFORE run. No3seed/learninggain claim yet. Timeout3600 likely enough.
- retain_device.py new atomic/resumable remote SHA+size audit, no remote mutations,
  replaces partial local CLI downloads only after full stream. ACTIVE session33861,
  /tmp/tavern-retain-device1910.log, auditingv30/v37/v38/v40/v41/v42 to
  retention-audit-device-v30-v42-1910.json. v40done59files at19:21. Auditv44+ separately.
- Still need broad GPU correctness, matched samehardware original+strongnative SIM/
  PPO,3seed learning/eval, fresh billing(last1549 $5.601 stale), all audits/ownedapp+
  fourvolume cleanup, final README/report. 10x UNACHIEVED. Noagents/commits/pushes/
  shared docs/othergame resources touched. User wants continue until complete.

## Continuation 2026-11-08 19:27 — CURRENT ACTIVE WORK

- v15 local build SUCCESS207.729s, ordered main/state main116772302bytes
  sha174433e808aa2053eaabd2e4b321528354b24c11432e91d18c168dd18d90e364;
  state113654485bytes (explicit registry prewarm unnecessarily pulls all callbacks).
  Not assembled. Avoid promoting it. Source changed only after immutable receipt.
- v16 local build SUCCESS34.420s (cached main+state rebuild). Same main bytes/SHA;
  state48818858sha e89534772c4163f0846357344f9b06c9adeec88a2ca87611b21744f18d73400a.
  Ordered tables ONLY simulation; native_build always separates phases for this flag.
  State retains canonical Once/getters and state_prewarm initializes DB only (no
  registry callbacks pulled into IO module). Same lifetime/stream rules. Current root.
- v45 CPU-only literal-ID build SUCCESS287.403s, appap-cpFFImDC127zgN1rKzsyiR.
  Parent native-literal-build-v45 ready for native_reuse_launch (full GPU not run).
- v47 CPU-only ordered v16 image build FAILED connection94.966s before FCID/GPU;
  all failures retained. No assembly proof. Appap-RK14O84zDyvMw24YBjrZPf.
- ACTIVE CPU-only retry NEW native-ordered-build-v48, session90847,
  /tmp/tavern-native-ordered48.log. Same v16 source-bound PTX, mainasm3/state1,
  --build-only. Do not launch GPU from it before SUCCESS and v46 terminal.
- v46 INITCHECK GPU remains active (oneGPU), appap-iIw24s1fgZqRDPcWwJ2Rsi,
  container ta-01M6XE99GPZMY1BS3BKBGN24QR (observed19:24), session13923.
  Remote readonly tail attempt session34344 pending. /tmp/tavern-native-initcheck46.log.
- Fresh DeviceEnv adds optional TAVERN_DEVICE_GAME_ARENA=0: keeps persistent table
  arena but leaves per-game bindings null, forcing individual CUDA malloc/free with
  owner headers. Module cache rejects changing this mode inside one context. Stats
  handle no per-game arena. Checkpoint runtime_options now includes game_arena flag.
  NOT GPU TESTED yet; use ONLY fresh-process diagnostics if needed, not production.
  Idea: memcheck/initcheck can then see allocation boundaries hidden by1MiB arenas.
- New device_wide_probe.py extends full device_probe gate with selected runtime's
  full128games300steps, every ready view/mask/result and final COMPLETE Game/RNG.
  Generates public scripted actions online with unchanged CPU oracle; no dependency
  on private/frozen volume traces. New parent/reuse launch copies helper. Future
  full qualification cannot pass only2/8game gates and miss known wide failures.
  Helper/expanded gate NOT GPU RUN yet. Current Python py_compile passes.
- Prior atomic retention session33861 FAILED connection after v40complete59files,
  before final receipt. All copied bytes remain. New retain_device_modal.py CPU-only
  read-only SHA manifest avoids repeatedly transferring117MBPTX. ACTIVE manifest
  session28691,/tmp/tavern-remote-manifest1925.log, appap-DSAvOzkOokQMcqAITX9pN5,
  fc-01M6XEDXSR5BQC81YA5KNA483Y. Writes new local device-remote-manifest-1925.json
  for v30,v37,v38,v40,v41,v42,v44. No remote mutations. Then run retain_device.py
  with same --run-id list + --manifest THATFILE + NEW --out. It can recover exact
  matching bytes from other local compiler/retained copies ONLY after remote SHA
  match; sparse copies never count. Per-run .progress.json now preserves audits if
  later connection fails. Need actual finalreceipt before cleanup.
- ACTIVE fresh billing session39449,/tmp/tavern-billing1926.log; writes NEW
  billing-observed-1926.json. Provider lag still applies. Need final refresh later.
- No new GPU after v46 yet. Still no matched speedup/3seed results. Three-seed helper
  not wired/run; original/native full training comparisons still required. User
  explicitly says persist to completion, do not end just with an incomplete status.
