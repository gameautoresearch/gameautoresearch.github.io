# Natural results, scoring and simulator guards

[Wiki home](../README.md)

Natural placement is 1–4, lower better. The headline per-lobby points are
`100*(4-placement)/3`: first=100, second≈66.667, third≈33.333, fourth=0. Top-half
(first or second) rate and mean placement are separate useful diagnostics. Equal physical-seat/opponent-profile
weighting prevents a favorable seat or composition from dominating the score.
Whole seed groups, retaining their seats/profiles, are the uncertainty unit.
No reward transformation alone demonstrates human/expert strength.

## Identity and evaluation

The source/artifact/rules identity is `tavern-four-small-v1` with native
fingerprint `217577b83fc4525e`, admission-v2; networks from the eight-player form do
not bind to it. Evaluation uses mixed fixed visible-script/learned opposition and
requires all four natural terminals with placements forming 1–4. A later lobby fault
invalidates the cell even if the candidate already died. No private roster, policy or schedule is exposed
by this wiki. Do not apply old grading shortcuts to the new release.

## Safety bounds are not game rules or wins

The recruit cap is thirty ordinary actions per turn in this task; pending choices
can still need resolution. A 100-turn limit is a safety guard. The underlying
engine can rank survivors by health+armor at that forced stop for diagnostics,
but the benchmark treats the guard as a fault/truncation, **not** an official
natural placement. Likewise, a 400-attack combat bound, nested-effect depth 64,
stack guard, no-progress guard, recruit cutoff or infrastructure failure cannot
be converted into a health-ranked win, ordinary loss or valid-only filtered score.

Batch stop codes are 0 ongoing, 1 natural, 2 no-progress, 3 driver guard,
4 turn limit, 5 combat cutoff, 6 recruit cutoff. Policy-illegal actions, visible
rejections, changed-context deferrals, simulator guards and transport/deadline
failures remain distinct categories. A natural zero-point fourth place is valid;
an uncompleted/faulted cell is not a zero-point natural finish.

Full-episode learning excludes every trajectory from a faulted lobby; it does not
keep apparently clean decisions preceding that lobby's eventual fault. Only
actually applied actions are accepted samples. Terminal placement reward received
while a seat waits belongs to its last real decision. Finished seats/lobbies
must not accumulate invented transitions while other cohort members finish.

`BatchEnv` resets after the final natural terminal. Its saved post-step state can
therefore be the next episode, not the terminal state just scored. Placement/stop
records determine which episode ended; replay verification must account for that
boundary. Source/config/checkpoint identity and RNG state remain immutable across
exact resume; reading this wiki grants no permission to rebind an old checkpoint.
