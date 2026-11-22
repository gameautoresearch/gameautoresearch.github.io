# Mode, clauses and release boundaries

[Wiki home](../README.md)

## The one game mode

Two players receive generated **three**-Pokémon teams; one active Pokémon per side.
The format is `gen9randombattle@@@Max Team Size = 3`: unmodified Gen 9 Random Battle
(same sets, moves, items, abilities, levels, Tera, switching and clauses) with only the
team size reduced from six to three by Showdown's Max Team Size rule. The generator
builds each team with the same procedure, stopping at three Pokémon.
The battle starts with the generated lead, without team preview or team construction.
A player wins when the other side has no usable Pokémon; engine-native ties remain
ties. Benchmarks separately count truncations, illegal choices, rejected unavailable
choices and simulator/infrastructure faults. They are never relabeled as natural losses.

This is Gen 9 singles Random Battle. It excludes doubles, factory/Battle Factory,
BSS, 1v1 factory, Monotype, 9-for-9 spotlight, custom teams, custom bans, arbitrary
competitive tier construction, Mega Evolution, Z-Moves and Dynamax. Seeing a field
for an older-generation mechanic in a shared data object does not enable it here.

## Format rules

The pinned format lists PotD, Obtainable, Species Clause, HP Percentage Mod,
Cancel Mod, Sleep Clause Mod and Illusion Level Mod. Obtainable expands into
validation prerequisites for generated legal species, moves, abilities and items.
Species Clause prevents duplicate base species. There is no Item Clause limiting
one copy of a held item across the team. No ordinary public configuration injects
a Pokémon of the Day; the generator only honors that hook when host configuration
explicitly provides one.

Sleep Clause Mod prevents the opposing side from causing a second living Pokémon
on the same team to be asleep at once. Self-inflicted sleep such as Rest does not
count as a foe-induced sleeper; it does not automatically block inducing another
foe's sleep. This is not Stadium Sleep Clause and not a blanket sleep ban.

HP Percentage Mod publishes opposing HP as percentages. Exact own HP/stats remain
available through that player's request. Illusion Level Mod disguises the level
along with the identity so a mismatching displayed level does not trivially expose
Illusion. Cancel Mod permits revising a choice before the other player has committed;
the benchmark's synchronous request/step machinery controls actual submissions.

## Adapter overlays, not alternate game rules

The currently selected public autoresearch release catalog chooses **v4**. Its
Showdown-backed completion boundary is 4,096 adapter phases with at most sixteen
consecutive unavailable-choice retries per seat. These are adapter submissions,
not a 4,096-battle-turn rule. V4 explicitly dispatches a sole published legal action;
Revival Blessing target requests select the legal fainted own reserve with maximum
sum of its five public non-HP stats, ties to the lowest own-team slot. Such mechanic
choices are logged separately; this does not claim optimal strategic selection.

Historical baseline **v1** has a 512-phase limit. It remains preserved with its own
checkpoints and scoring records. V2/v3/v4 and four-model native pilots have their
own source/task bindings; do not copy their cap, inference restrictions, budgets or
completion convention into another session. The game mechanics and 700-value /
14-action public codec are shared, but the run's declared release is authoritative.

The native C++ CPU backend was subsequently admitted through source-bound
qualification; the public v4 Showdown benchmark declaration still uses Showdown CPU.
Old prose saying that all native work remains incomplete is historical, not the
current general status. A rules wiki does not change backend admission or resume checks.

## Scoring boundary

Natural win/draw/loss points are 1/.5/0, multiplied by 100 and balanced over seats
and the declared opponents. Terminal training rewards are +1/0/−1. Incomplete or
faulted evaluation schedules receive no ordinary complete headline score. Public
practice provides development feedback; private schedules/opponent state do not
belong in this rulebook. Strong-reference, expert and human-strength calibration
are separate evidence requirements.
