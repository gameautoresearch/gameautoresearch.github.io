# Clash self-play reference

Separate research track for the exact fixed8 task. Source, rules, data, backend
build, complete optimizer/RNG/world state and frozen league identities are bound
into checkpoints. No released upstream weights are assumed compatible.

The stateless policy consumes only 7024 public ints and the 2307-action mask;
all 200 entity rows contribute. Greedy inference first chooses the largest
card/wait marginal and then the largest tile logit. This avoids treating the
probability of one placement as comparable to the entire wait marginal.

Teacher imitation is labeled separately from PPO. Training alternates learner
seats, keeps scripted and frozen neural opponents assigned until natural match
end, and uses a smoothed PFSP distribution. No engine lookahead is performed.

Local checks: `.venv-clash/bin/python -m pytest simulators/clash/reference_training/tests -q`.
Paid jobs require explicit new-campaign reservations and an evaluator interface
handshake; see the ignored campaign coordination file. No automatic retries.

`v2/` is a separately bound fresh lineage, not a resume of the first proof. It
adds categorical card features and learned entity encoding pooled separately by
visible side, keeps all 200 rows, and learns from the evaluator's exact public-v2
teacher hash. It has 381,902 parameters. `cases.py` precommits paired teacher/PFSP
and teacher-off controls with the same initial seed, model, worker topology and
case process time ceiling. PPO and frozen neural-history play continue after the
finite assisted stage. Live pool win rates are diagnostics, not strength scores.

`export_adapter.py` preserves checkpoint weights and compiles the factorized
hierarchical greedy decoder into a pure neural flat-argmax head. The derived
source is explicitly recorded; the training source/config binding is unchanged.
The exported head's softmax is not the training sampling distribution.

`v3/` is another fresh, separately bound lineage. It removes the exploratory
lightweight script (whose public-coordinate assumptions were wrong) and uses
pinned standard/pressure scripts with explicit fixed tie-breaking, plus public-v2
intermediate/teacher opponents. The older flawed-source runs remain retained.
Independent expert trajectories supply class-balanced wait/deploy imitation;
these rewards/outcomes never enter PPO. The teacher-off control skips unused
imitation/demonstration work and may complete more PPO within the same process
wall-time cap. Complete demonstration worlds join strict checkpoint restoration.
This is a recipe change, not an admission of a strong policy.

`v4/` is one conditional spatial pilot with a paired control, not a replicated
method claim. All valid public entities contribute to a Gaussian raster on the
verified 18×32 public action grid. A small CNN and card-conditioned placement
head replace the MLP placement head, with 151,439 total parameters and strict
FP32 runtime flags bound into the checkpoint. `v4_cases.py` is a separate frozen
controller; the corrected two-seed MLP controller remains unchanged.

`validate_artifacts.py` checks the immutable source archive, checkpoint binding,
every complete native-world payload checksum, frozen histories and exact final
inference tensor bytes. Logical learner counters exclude one extra discarded
update executed by the exact-resume audit; the full allocation remains charged.

`select.py` chooses one public-development stage per family across all training
seeds. An earlier milestone is an inference export, not an earlier full resume
point. Its exact tensors must occur in the validated final checkpoint's frozen
league. Do not combine earlier weights with later optimizer/world state.

`finalize.py` closes selection before creating a private fresh bank. `launch.py`
blocks every new training/recovery allocation after that closure.
`assessment_remote.py` runs only a finite CPU assessment of the closed roster,
with isolated policy processes and the pinned evaluator release.
`analyze_final.py` keeps all seeds, seats, opponents and failure classes separate;
its paired exploratory intervals do not establish human/expert strength.
`cleanup.py` stops only explicitly recorded apps owned by this RL campaign,
after its compute intervals settle. Artifacts and unknown charge holds remain.

The final readout is `reports/clash/selfplay-reference-2026-11-09.md` and its
JSON companion. `report.py` reads accounting without launching a worker;
`readout.py` writes the final readout only after assessment and cleanup finish.
All evidence, checkpoints and private seeds remain in ignored campaign storage.

## New five-hour authorization, November9

`strong/` uses only the separate explicit
`evidence/strong-bots-20261109-0516Z/campaign.sqlite3` OpenCreditBudget ledger.
The old `launch.py`, `finalize.py` and `cleanup.py` above remain attached to the
old capped campaign. Do not use them to create new workers or assessment banks.
Selection closes09:36:36UTC, workers stop10:11:36UTC and reporting ends10:16:36UTC.
The evaluator owns the new private final bank; training never reads its seeds.

`strong.provider` freezes each complete source archive and accepts either an
exact full-state resume or an explicitly declared model-only warm start. Parent
checkpoint SHA is checked before any paid reservation. Actual GPU replay and
legal deterministic inference gates run on every stage. `strong.fetch` verifies
source/config/tensors and closed milestone checkpoint receipts; `strong.validate`
independently checks complete downloaded final artifacts. Every milestone with
a resume receipt includes its own matching optimizer/world/RNG/history state.

`v5/` renews PublicTeacher assistance and a mixed script/neural pool. `v7/` uses
the public Standard fixed-tie demonstrator; teacher-off v5/v7 learning is checked
bitwise equivalent. `v8/` additionally supervises learner-visited states.
`v9/` replaces joint-action entropy with wait/card-marginal entropy. `v10/`
conditions small header/card-cost inputs with binary gains and compensated
weights, requiring exact parent logits/values/actions before learning. It keeps
151,439 trainable parameters and adds224 fixed buffer elements. `v11/` combines
that conditioning with learner-state supervision. `v12/` is an explicitly
supervised-only control: true-prior deploy gate plus conditional card/tile loss;
its PPO/value/entropy coefficients must be zero. Its updates must never be
reported as PPO updates. Code availability is not evidence of a completed run.

Public development uses eight published seeds, both seats and six fixed
opponents, preserving all cells/failures. Local timings take the shared lock.
`strong.dev_remote` creates separately accounted finite8CPU/32GiB isolated
Linux public batches; `development=False` selects isolation, not private seeds.
Only complete panels get an aggregate. `strong.status` and `strong.report` read
the new ledger; `strong.cleanup` defaults to an inventory and its `--stop`
requires every owned compute interval reconciled before stopping recorded apps.
Unpriced builds, observers, storage and transfer holds remain until resolved.

Current readout: `reports/clash/strong-bots-five-hour-2026-11-09-rl.{md,json}`.
All negative runs remain retained; repeated public selection is exploratory,
and no expert/superhuman qualification follows from these development points.
