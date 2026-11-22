# Worked edge cases

[Wiki home](README.md)

All examples state the pre-growth move result; due income is applied afterward only if the game remains active. Counts refer to integer troops, not damage rolls.

| Case | Input | Move result |
|---|---|---|
| Empty expansion | Source 2, whole; neutral empty target 0 | Send 1, source 1, target becomes yours with 1 |
| Half odd stack | Source 9, half | Send 4, reserve 5 |
| Half even stack | Source 10, half | Send 5, reserve 5 |
| Redundant splits | Source 2 | Whole and half both send 1; both action IDs are legal |
| Friendly merge | Source 7 whole into your target 3 | Source 1, target 9 |
| Failed attack | Source 5 whole into enemy 6 | Send 4, source 1, enemy target 2 |
| Equal attack | Source 6 whole into enemy 5 | Source 1; enemy still owns target with 0 |
| Winning attack | Source 8 whole into enemy 5 | Source 1; target yours with 2 |
| Castle chip | Send 5 into neutral castle 12 | Neutral castle 7, still not productive |
| Castle capture | Send 8 into that castle 7 | Castle yours with 1; future due growth applies |
| Transfer rounding | Enemy other tile has 5 when its general falls | Capturer receives that tile with 3 |
| Captured-general surplus | Send 8 into enemy general 3 | New castle has 5; that surplus is not halved again |

## Due growth

Both players pass on the opening tick: time goes 0→1 and each general stays at 1. Pass again: 1→2 and each owned general gains one. On an active tick reaching time 50, each owned plain tile gains one and each owned general/castle gains two total. A neutral castle remains unchanged. If the tick reaching time 50 captures the opponent's general, no growth is paid at all because the game has ended.

## Chasing and stale actions

Suppose B chooses to move from cell X to Y, while A chooses to attack X from Z. Unless B's move is a head-on swap with A, B waits for the incoming move into its source. If A captures X, B's queued action now lacks an owned source and does nothing. If X remains B's with fewer troops, its whole/half amount is calculated from that reduced count. This is legitimate joint-step interaction, not arbitrary source reuse.

## Same destination

If both attack a neutral cell, the first resolved move changes its army/owner. The second fights the resulting state. For example, A sends 4 and captures an empty neutral cell with 4; B then sends 6 and captures it from A with 2. Do not subtract both original attacks from a neutral zero independently and declare both owners.

## General trade versus equal force

Adjacent generals holding 6 each both send whole: 5 against departure reserve 1, so both become the opponent's new general with 4 and the game continues after territory swap. By contrast adjacent generals holding 2 each both send 1 against reserve 1. Neither `1>1` test succeeds; there is no special trade. Normal sequential head-on resolution applies. Exact order matters: the first attack can reduce a general reserve and change the second action. There is no generic “both generals attacked, therefore draw” shortcut.

## Boundary capture

A capture on the step whose new time is 1000 is terminated, not truncated, and scores a win. If no winner exists at that new time, it is a cap draw even when one player has ten times the other's land/army. `transition` returns the true cap board; the next reset map is not the cap observation.
