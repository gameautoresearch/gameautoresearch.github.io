# Actions and legal masks

Every action is `(family, source, target)` in action catalog version2. Tile IDs are
`y*11+x`; fixed category slots preserve source-data IDs. Select family, then source,
then target conditionally using full masks, rather than independently sampling
marginals that can compose an illegal path. All values below are zero-based.

| Family ID/name | Shape | Flat interval [start,end) | Meaning |
|---|---|---|---|
|0 move|121×121|0..14641|Source tile→destination tile|
|1 attack|121×121|14641..29282|Source tile→enemy target tile|
|2 summon|121×128|29282..44770|City/Raft tile→unit kind ID|
|3 build|121×128|44770..60258|Tile→structure kind ID|
|4 research|1×128|60258..60386|Tech ID|
|5 harvest|121×1|60386..60507|Tile resource harvest|
|6 forest|121×3|60507..60870|Tile→0 burn /1 clear /2 grow|
|7 reward|121×9|60870..61959|City tile→reward ID0..8 (0 never offered)|
|8 capture|121×1|61959..62080|Unit tile: city/ruin/starfish|
|9 recover|121×2|62080..62322|Unit tile→0 recover /1 disembark in place|
|10 promote|121×1|62322..62443|Unit tile promotion|
|11 disband|121×1|62443..62564|Idle unit tile disband|
|12 heal|121×1|62564..62685|MindBender tile heal neighbors|
|13 destroy|121×1|62685..62806|Improvement tile destruction|
|14 diplomatic|1×18|62806..62824|Opponent block×6 choice; only first block used in1v1|
|15 end|1×1|62824..62825|End current player turn|
|16 resign|1×1|62825..62826|Masked off in this task|

Total logical catalog width is **62826**. Flatten with
`family_offset + source*family_row_width + target`. Move/attack row width is121,
summon/build128; the other widths are shown above. A one-row/one-column native mask
leaf may be represented as a vector/scalar rather than a literal matrix.

The `recover` family's added target1 is disembark-in-place. The diplomatic18-wide
row reserves6 choices for each possible other seat in the generalized runtime;
1v1 has only one opponent, so the last12 entries are always illegal. Explicit
city rewards/peace responses are command triggers: pending peace must be answered
first; pending city reward blocks ordinary commands until selected. Resign is
disabled by task options even though its family exists. End is not a legal escape
from a pending mandatory choice.

Native validation requires the expected state_version. Rejected action state is
unchanged; accepted commands increment version and total action count. Engine
status codes:0 accepted,1 invalid,2 unsupported,3 stale,4 already done. A legal
move into a hidden enemy can reveal it and leave the mover in place without
consuming that unit's movement; do not relabel that accepted reveal as a policy
violation. Masks can expose legality consequences without disclosing hidden state.

[Wiki home](../README.md)
