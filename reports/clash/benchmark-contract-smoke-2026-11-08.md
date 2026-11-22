# Clash benchmark contract and DeepSeek smoke — 2026-11-08

The real research smoke passed: DeepSeek edited the starter and recipe, ran two
actual H100 PPO experiments, evaluated public practice, submitted an immutable
policy, finalized, and received an independent private grade of **50/100**.
The two provider-authoritative GPU leases total **401.191950 seconds** against
the new cumulative 600-second ceiling, with no unresolved GPU reservation.
No full 120-minute experiment or strong-reference calibration was run.

## Delivered contract and interface

`simulators/clash/autoresearch/benchmark.json` and `CONTRACT.md` freeze the full
`clash-fixed8-v1` arena: two players, 36×64 routing cells / 18×32 placement tiles,
level-11 mirror deck, Princess towers, four paused decisions/second and natural
match endings. The native C++/CUDA rules, table pins, 7,024-int seat observation,
200 entity rows and 2,307-entry mask are unchanged. Commercial-game fidelity,
other decks and live reaction deadlines remain unqualified.

The starter has three editable files (`policy.py`, `learner.py`, `train.py`), a
small random MLP, plain masked PPO/GAE, sparse outcome rewards and current-policy
self-play. There are no historical-opponent, imitation or shaping toggles. Agent
instructions and brief are under `autoresearch/task/`. `prepare`, `smoke`,
`session`, `run`, `evaluate`, `submit`, `finalize`, `budget`, `status` and `grade`
provide the control surface. `session` exposes the full 7,200-second allowance;
paid commands require `--execute`. Both smoke/full plan-only commands pass.

The host uses its own `.venv-clash-ar`; `requirements-host.txt` and `setup.sh`
reproduce it. Local simulator checks use `.venv-clash`. Sources, safe float32 NPZ
weights, configs and checkpoint provenance are hash-sealed. Complete-update
resume preserves model, Adam, RNG, every native world byte and public outputs.
Private policy inference runs in a separate unprivileged process, with a distinct
UID for checkpoint opponents. Only public observation/mask arrays cross the pipe.
Private schedules/weights and credentials never enter the model prompt.

## Executed session and evidence

The final resume command was:

```sh
CLASH_AR_BASE_IMAGE_ID=im-8pfiKtMJEgLNOpUg3qFbSS MODAL_DISABLE_API_PROXY=1 \
  .venv-clash-ar/bin/python -u -m simulators.clash.autoresearch smoke \
  --release evidence/clash-autoresearch/release-20261108-v2 \
  --out evidence/clash-autoresearch/campaign-20261108 --execute
```

Every resume used the same campaign/model ledgers. Earlier CPU-only image failures,
transport interruption, response-limit exhaustion and unresolved model-call
reservations remain in logs/ledgers. They did not renew the allowance. The retained
packages-only qualified image avoids reinstalling Torch; source-only native
compilation had no attached GPU. The smoke required operator-assisted setup and
response-limit repairs; a clean unattended rerun is not claimed.

| Agent experiment | PPO updates | Learner rows | Natural training games | Provider GPU seconds | Public points |
|---|---:|---:|---:|---:|---:|
| `smoke` | 357 | 1,462,272 | 920 | 164.551413 | 0/100 |
| `smoke2` | 605 | 2,478,080 | 1,601 | 236.640537 | 50/100 |

Both runs have finite losses, actual Adam updates and nonzero weight changes.
DeepSeek added an entity encoder, widened the network and revised its recipe.
These two unequal experiments do not isolate an algorithmic improvement. Its
second policy beat random in all four public games and lost all four to scripted.
Two later requests were rejected by the remaining allocation ceiling before any
GPU launch. The existing `smoke2` submission survived subsequent workspace edits.
Final identity: `432297c6395878d462f66b1024f01f403f071459fa11ab341a1ed1bc7981c68e`.

Private grading used two fresh private seeds, both physical seats and two
opponents: eight natural games, four wins against the independently initialized
checkpoint and four losses against scripted. Both seat scores are 50/100.
There were no illegal actions, truncations or simulator faults. The initialized
checkpoint is a greedy wait baseline, not a strong reference. The paired seed
bootstrap interval [50,50] is descriptive for this tiny panel. Reference-normalized
score remains **null**. Full release grading has 128 private seeds and a separate,
disjoint 128-seed calibration schedule, but these were not run.

Twelve local contract/session tests pass, including exact PPO/model/Adam/RNG/public
state resume, source rejection, safe bundle tamper rejection, masks, natural versus
truncated GAE, persistent budget enforcement, milestone semantics and plan-only
full-session behavior. A fresh reduced source package excludes legacy root trainer,
model/reference recipes, other games, historical evidence and private schedules.
Local closure verification exactly reproduced all private rows and corrected
public/private labels. The original successful practice artifact mislabeled its
`split` field as private despite using the public schedule; the source fix and
fresh closure verification preserve the original receipt instead of rewriting it.

## Accounting and limits

Model reported charges were **$0.002434753**; all charged/reserved model liability
is conservatively retained as **$0.110330970**, below $2. The model ledger is closed.
GPU compute is approximately **$0.7223** at the recorded historical component
rates; this is an unsettled estimate, not an invoice. CPU-only host grading and
source-image build costs are separate and not invoice-reconciled. The JSON readout
contains final cleanup/provider evidence and any ancillary verification outcome.

The small weak-opponent panel is a plumbing and learning check. Strong policies,
full-budget anchors and statistical strength calibration remain unqualified.
The declared evaluation ABI is stateless greedy; recurrent policy state/search
would require a later contract extension. No files were committed, reset or pushed.

Final cleanup metadata confirms all nine owned apps stopped with zero tasks.
The later reduced package passed local public/private closure replay with exact
private rows, and its remote public check verified candidate UID 65534 and
no-new-privileges. The ancillary remote private repeat remained delayed in
provider control-plane work and was explicitly stopped; its failure and cleanup
receipts are retained. This later check used no additional GPU seconds. The
executed smoke's original independent private grade remains the authoritative
50/100 result; the later packaging check is not represented as a fresh full
unattended GPU smoke.

## Post-smoke milestone settlement correction

Parent review found a real historical control-plane omission: the finalized
401.191950-second ledger records one-third as null but omits two-thirds. The last
process poll was before 400 seconds; teardown crossed that threshold, and the old
settlement path did not tick milestones. No policy had been submitted at 400
seconds. The historical point therefore means **missing/null**, never the later
final policy. The executed ledger, model ledger, grading and provider receipts
remain byte-for-byte unchanged.

Future provider settlement now calls `Ledger.settle`, which charges authoritative
provider start/end time and captures crossed milestones in the same locked
transaction, before a later submit/finalize can run. A focused local regression
recreates teardown-only crossing of 400 seconds and proves that the null point
survives subsequent submission/finalization. All **12 local tests pass**. This
small post-smoke correction had no additional cloud or GPU execution.

Current harness source identity: `c9d4bfd9b5c8d1fc69323aee20f6cba30b47c9b43ccb04c4e2a218f67057d274`. The exact path/hash
manifest and immutable-receipt audit are retained under
`evidence/clash-autoresearch/post-smoke-milestone-settlement-20261108/`; executed
source archives remain separate historical artifacts.
