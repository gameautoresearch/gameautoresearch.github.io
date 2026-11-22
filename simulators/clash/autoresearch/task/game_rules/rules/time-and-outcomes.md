# Time, phases and natural endings

## Clocks and decision cadence

This benchmark plays the **short match form `short-v1`**: 60 s of regulation and 30 s of overtime at 2 decisions per simulated second. The pinned Ladder timeline of the upstream engine (180 s + 120 s, elixir steps at 120 s and 240 s) is moved in proportion; nothing else in the rules changes.

One engine tick is 50 ms. Each ordinary environment decision asks both seats for one legal action, then advances **up to ten ticks** (500 ms); a natural ending or versioned caller cap can stop early. Inference occurs with the simulated clock paused. No live reaction deadline is implied. The command delay is another 20 ticks (1,000 ms), distinct from troop deployment countdown. No plays are accepted in the first 90 ticks (4,500 ms), although the simulation/production clock still runs.

## Timeline

| Simulated interval | Match section | Elixir full-bar time | Hand refill cooldown |
|---|---|---|---|
| 0–40 s | Regulation | 28,000 ms | 1,000 ms |
| 40–60 s | Regulation | 14,000 ms | 500 ms |
| 60–75 s | Overtime if crowns were equal | 14,000 ms | 500 ms |
| 75–90 s | Overtime if still ongoing | 9,300 ms | 350 ms |

Regulation is 60 seconds. If crowns are unequal at its end, the match ends by crown count. If equal, the next 30-second overtime section is entered. During overtime, unequal crowns immediately decide the match. End-of-section checks use the actual engine timeline and tower listing state, not an external wall-clock timer. Production/hand-refill rates advance independently from section transitions. The observation's clock fields carry these values: `decision_ticks` is 10, `section_left_ms` counts down from 60,000 in regulation and 30,000 in overtime, and `full_bar_ms` steps at 40 s and 75 s. A wait-only match is a draw after 195 decisions (about 97 s of simulated time including the tiebreak and the 4 s ending delay).

## Natural tiebreak

At time-up with equal crowns and a not-yet-ended tiebreak, ordinary hits are held. The native tiebreak initially clears non-tower objects: during the first 1,450 ms boundary window it resumes/removes troops and advances the world only where the clear walk calls for it. Starting when the prior tiebreak timer is at least 3,250 ms, the same HP amount is drained from every crown tower on both sides per 50 ms tiebreak step. The amount depends on the smaller of the two sides' lowest listed tower HP:

| Lowest HP | Drain per step |
|---|---|
| under 21 | 1 |
| 21–199 | 10 |
| 200–499 | 20 |
| 500–999 | 40 |
| 1,000 or more | 50 |

After draining, tiebreak completes when either side's lowest tower HP is below one or when the two lowest HP values are equal. It is not a sum-of-all-health comparison. Queued hits are drained and cleanup runs at the specified boundaries; due commands still receive their tiebreak execution effect rule. The resulting crown/King state then determines winner or draw. This preserves an explicit natural tie rather than choosing an arbitrary seat.

## Ending presentation and failure boundary

Once ending is committed, timeline freezes and a 4,000 ms ending delay completes before the match reports over. King destruction also expands a visual/cleanup kill circle over half that ending-delay duration. The circle's kills are queued, not a replacement victory calculation. A dead tower can remain listed until cleanup. Natural win/loss/tie gives reward +1/−1/0 to each seat and grading points 1/0/½.

Normal mode has `max_ticks=0`: there is no arbitrary training horizon replacing natural match resolution. A caller-declared tick limit is a truncation, separately labeled. Capacity exhaustion, unsupported native state/command conditions, invalid policy actions and incomplete schedules are not natural losses or ties.

[Wiki home](../README.md)
