# Player information, observations and assessment

## Visible information

Both players receive only their own structured seat observation. The fixed deck is public by task definition; the enemy's current hand, queue, pending private commands, deck shuffle and engine RNG are not supplied. Visible entities, own hand/costs/queue, clocks, tower health and public executed enemy plays are available. The header's enemy-elixir value is **public reconstruction** from observed play/clock information, not a privileged lookup of actual enemy private elixir. Waiting enemy Archer formation rows are withheld while the owning player can inspect its own waiting unit. Projectile aim points are withheld until their public release/reveal rule permits them.

Schema v4 stores **7,024 int32 values per seat**: 224 header values plus at most 200 rows×34 entity values. A separate 2,307-entry legal mask accompanies it. Entity overflow is counted and logged; it is not an instruction to invent hidden rows. Ordinary native world capacity is larger than the view cap. The [complete observation dictionary](../observations/schema.md) lists every named offset, flag and ID used by this view. Vocabulary IDs identify pinned content rows, not arbitrary learned hash buckets.

Entity IDs in the view are observer-assigned, retained consistently under the declared entity-ID rule. Coordinates are seat-relative. `-1` commonly denotes an absent/unseen ID or target, not a valid unit. Buff slots expose up to three selected effects even though internal buff capacity is eight. The only mechanically reachable normal buff row here is ZapFreeze. Internal state, optimizer/RNG and debug snapshots are not policy inputs.

## Results and grading

Natural win/draw/loss points are 1/½/0, multiplied by 100 for the headline raw score. Both physical seats and each declared opponent receive equal weight; uncertainty resamples paired seed clusters. Natural training rewards are +1/0/−1. Complete valid schedules are required; simulator faults, invalid neural choices, arbitrary truncations and incomplete cells do not become ordinary losses or draws. Reference normalization remains unqualified/null until separately budgeted full-budget multi-seed calibration.

This wiki documents game rules. Later versioned evaluators may change the model parameter cap, inference serialization, opponent panels, worker counts or deadlines; those are read from the selected immutable contract. The baseline public contract uses stateless greedy neural inference, one million float32 parameters and no inference search. Do not borrow a larger/smaller cap from a historical experiment based on this prose.

[Wiki home](../README.md)
