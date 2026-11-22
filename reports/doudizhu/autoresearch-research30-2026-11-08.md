# Autonomous Doudizhu research trial — 2026-11-08

The prescribed smoke configuration and single-successful-run instruction were removed.
DeepSeek V4 Flash chose its own changes, configurations, allocation lengths, comparisons
and final checkpoint under a **30 allocated H100-minute cap**. The default full benchmark
remains 120 minutes; no full-budget reference calibration was launched.

The session completed with a valid explicit submission. It used **1,658.4184 allocated
H100 seconds (27.6403 minutes)**, including failed work and teardown, leaving 141.5816
seconds unused. Known OpenRouter usage was **$0.429406**; a prior unresolved request retains
a $0.036018 reservation, making charged-or-reserved model liability **$0.465424** under
the unchanged $2 model-call cap. These are not a reconciled GPU/CPU invoice.

## Result

The selected `improved_v2` policy improved public practice performance but showed **no clear
gain on the independent withheld panel**.

| Saved policy | Architecture / parameters | Allocated training worker | Natural games / decisions | Practice quality | Withheld quality |
|---|---|---:|---:|---:|---:|
| `starter_test` | 128 × 3, LSTM-64 / 471,171 | 179.179 s | 555,008 / 21,988,082 | 75.35% | 68.75% |
| `improved_v2` — final | 256 × 4, LSTM-96 / 1,310,211 | 341.671 s | 1,062,912 / 41,912,836 | **80.21%** | **69.01%** |
| `large_v1` | 384 × 5, LSTM-128 / 2,887,299 | 905.000 s | 2,011,136 / 78,352,631 | 76.74% | 66.67% |

Practice used the same 32 deals, three opponents and three controlled roles: 288 games per
policy. The independent evaluation used a precommitted 64-deal prefix of the frozen private
grading schedule, four development opponents and three roles: **768 games per policy**.
The withheld panel includes two full-size neural opponents unavailable to the researcher.
All independent evaluations ran in separate CPU-only Linux sandboxes, with unprivileged
candidate inference and no private feedback supplied to DeepSeek during research.

On that identical withheld schedule, final minus starter quality was **+0.26 percentage
points**, with paired complete-deal bootstrap 95% interval **[−2.73, +3.26] points**.
The larger model's difference was −2.08 points, interval [−5.21, +1.04]. The practice gain
therefore did not establish improvement on the withheld deals/opponents panel.

These intervals compare the saved policies conditionally. The starter used training seed
123; the modified policies used seed 42, different architectures, and longer training.
Training-seed variance and equal-compute algorithm effects are not controlled. This is one
research attempt, not a model ranking or strong-play qualification. Every reported quality
is development win probability; official normalized reward remains null.

## What DeepSeek did

DeepSeek inspected the SDK, proposed larger networks, Adam, replay, target averaging,
historical self-play and scheduling, and wrote replacement learner/trainer implementations.
It later simplified the implementation to **larger networks + Adam + replay + cosine
learning-rate scheduling**, retaining Deep Monte Carlo terminal-return regression. The
policy adapter stayed unchanged; architecture choices were made through its own recipes.

It ran a starter-configuration probe after an infrastructure interruption, evaluated both
the probe and its modified candidate on the same practice schedule, then tested a larger
network. It correctly retained the earlier candidate when the larger model scored worse.
The starter probe used the modified trainer with optional enhancements disabled, not an
untouched source copy; its optimizer and on-policy configuration match the starter behavior.

The final immutable bundle is
`5a53e2f0714bd697576d5f7370ffa99ecd3ad700de1689553238b8235ab4b876`.
Its first explicit submission was retained for the two-thirds and final grading points.
The one-third point had no submission; later submissions did not retroactively fill it.

The recorded reasoning shows real hypothesis generation and model selection, rather than
the prescribed smoke procedure. It also shows weaknesses: speculative diagnosis of a host
transport error as multiprocessing, repeated shell quoting/timeout mistakes, and several
simultaneous algorithm/configuration changes without ablations or equal-compute controls.
The practical conclusion is **successful autonomous experimentation, inconclusive general
playing-strength improvement**.

## Infrastructure and accounting corrections

- The CPU editing shell initially selected a different interpreter from the pinned game
  environment. The shell now selects pinned Python through its own PATH, without replacing
  Modal's service interpreter.
- An initial host transport failure consumed **152.948 H100 seconds** without a recovered
  checkpoint/completed-wave receipt. That failed allocation remains charged. CPU-only
  reconnect attempts added no GPU time. Clearing proxy variables did not disable macOS
  system proxy discovery; the SDK's explicit bypass setting was used for the resumed run.
- The pinned gRPC client's connected check could reuse a connection whose transport had
  already been deleted. A host-only guard rejects closed connections before reuse and
  reports a terminated stream through the client's normal retry path. Installed SDK files
  and the simulator were not modified.
- The old model guard accumulated every worst-case quote, including settled requests. It
  stopped `improved_v2` early despite known model costs of only about $0.16. The checkpoint
  was exported and the same session resumed. Known requests now charge reported cost;
  unresolved requests keep their reservation. The dollar cap and GPU allowance did not grow.
- Resume now retains the conversation and restores source/configuration from the last
  captured run. Lost tool replies are marked unknown. Every previous receipt and failed
  attempt remains intact. Hardware time comes from provider timestamps, not guest logs.

The 900-second large-model request consumed 905 seconds including teardown; all 905 seconds
are charged. The aggregate still remained within the 30-minute cap. All GPU allocations
are in the session ledger; independent grading used CPU only. No reference-training
campaign or off-budget training was started.

## Verification and retained evidence

**75 local tests passed**, including simulator checks and new prompt/budget, conversation
resume, interpreter selection, closed-transport, model reservation and paired-evaluation
tests. Original source/rules/data locks and historical evidence remain unchanged. Private
grading schedules and credentials are excluded from the agent context.

Machine-readable audit: [`autoresearch-research30-summary-2026-11-08.json`](autoresearch-research30-summary-2026-11-08.json).
Private local session evidence is under
`evidence/doudizhu-implementation/autoresearch-deepseek-research30-20261108/`: exact captured
source/recipes, immutable bundles, allocation receipts, provider-returned reasoning and tool
transcript, model-cost ledger, notebook, private evaluation rows and cleanup receipt.
Provider reasoning is a returned model output, not proof of access to every internal process.

The earlier smoke ledgers and 399.1775 H100 seconds remain accounted separately; this new
user-authorized trial did not reset them. No files were committed, pushed or published.
