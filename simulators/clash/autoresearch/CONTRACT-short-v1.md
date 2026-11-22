# Clash short benchmark v1

`clash-autoresearch-short-v1` shortens each match so a learner gets many more complete
games, and so many more terminal rewards, from the same worker time. It replaces fast-v4
as the active Clash benchmark; fast-v4's pinned form is at git tag `full-size-20261113`.

**Match form `short-v1`** (`native/match_forms.py`, `common.MATCH_FORM`):

| | fast-v4 (pinned Ladder form) | short-v1 |
|---|---|---|
| Regulation + overtime | 180 s + 120 s | 60 s + 30 s |
| Elixir x1 / x2 / x3 | 0–120 s / 120–240 s / 240–300 s | 0–40 s / 40–75 s / 75–90 s |
| Card-cycle cooldown 1000 / 500 / 350 ms | same steps as elixir | same steps as elixir |
| Ticks per decision | 5 (4 decisions per simulated second) | 10 (2 per second) |
| Decisions in a wait-only draw | 1,230 | 195 |

The elixir and cycle steps move in proportion: double elixir covers the last third of
regulation, triple elixir the second half of overtime. Elixir per step, the starting 6
elixir, start cooldown, the 1-second deploy delay, tiebreak, end delay and everything
else are the pinned rules unchanged.

Unchanged: the frozen fixed 8-card level-11 mirror deck, the full 36×64 arena, 200 entity
rows, 7,024 public observation ints per seat, 2,307 masked actions, natural match endings
(crowns, overtime sudden death, tiebreak), equal-opponent equal-seat game points, and the
separate handling of natural outcomes, illegal actions, simulator faults and truncation
(any of them voids a schedule's score). The observation's clock fields carry the short
form's values (`decision_ticks` 10, `section_left_ms` from 60 000 / 30 000, `full_bar_ms`
stepping at 40 s and 75 s).

The pinned crforge data and `rules.lock.json` are not edited. The form is compiled into
its own native library by explicit flags that are part of the build identity, and each
loaded library reports the cadence and timeline it runs, which the runtime compares with
the form before use. Bundles and releases bind the form through the engine binding.

**Panel.** standard, pressure, intermediate and teacher: trusted scripted policies that
read only one seat's public observation and need no retraining (the intermediate keeps
its one-second cadence, now every second decision; the teacher's late-game switch reads
the observation's overtime flag). fast-v4's learned opponents retained-deepseek and
incumbent-v4 were trained on the 300-second form and are dropped, not retrained.

**Schedules.** Public practice: 8 new public engine seeds (2111301–2111308) × 4 opponents ×
both seats. Protected development grade: 32 fresh secret seeds drawn at `prepare`, never
shown to the researcher and disjoint from every earlier public seed set. There is no
official private final bank; `grade` is labelled development.

**Starter and budgets.** The starter is fast-v4's (random Policy64/embedding12, sparse
self-play PPO on the canonical native trainer) with `match_form: "short-v1"` and
`rules.decision_ticks: 10` in its configs; the trusted trainer refuses a config naming
any other form. One L40S with 4 CPUs and 16 GiB, 7,200 research worker seconds and a
3,330-second fresh final, as fast-v4. Policy inference is a stateless masked greedy NN.
That research budget is the contract's (seconds) mode, one job at a time. A session may
instead choose the harness's dollar mode: $5.81 of training-worker time ($0.48 smoke) at
$0.0008064 per second, up to two training runs at once (four jobs, two GPUs), each queued
or running run holding its whole grant, on an 18,000-second wall clock. Practice and the
fresh final are the same in both modes.
No reference-normalized or expert strength is claimed.

`python -m harness prepare clash --out NEW_RELEASE` builds a release from this tree
(`short_release.py`); no base release is read.
