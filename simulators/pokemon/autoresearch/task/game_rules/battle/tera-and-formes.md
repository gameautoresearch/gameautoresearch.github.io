# Terastallization and forme changes

[Wiki home](../README.md) · [Species/form index](../species/README.md)

## Ordinary Tera

Each side normally Terastallizes at most once. It is chosen alongside a move when
the request permits it, resolves in its action-queue stage before ordinary moves,
and persists across switching. A fainted already-Tera Pokémon does not restore the
side's once-per-battle allowance. The selected Tera type is generated with the set;
it is not an arbitrary new type chosen every turn.

Ordinary Tera replaces defensive typing with the Tera type. Original-type attacks
retain their ordinary STAB; an attack matching a new Tera type gains STAB; a Tera
type matching an original type normally raises that STAB from 1.5 to 2. Ability
modifiers such as Adaptability and special species rules still apply. Terastallization
does not universally remove every original-type mechanic, change the ability,
heal HP, restore PP or grant a free new move.

An eligible damaging move below 60 power can be raised to 60 when its Tera typing
conditions apply. The pinned engine excludes ordinary positive-priority/multi-hit
cases and variable-power effect rule cases that do not qualify. This is not a global
60-power floor for every move; exact move base power and guards matter.
Tera Blast is normally Normal/special, changes type with Tera, and can choose the
physical attacking stat when its exact comparison says so. It is present as a normal
generated move, not automatically added to all Pokémon on Terastallization.

## Stellar

Stellar retains the user's original defensive typing instead of becoming a single
ordinary defensive type. It tracks boosted attack types: ordinarily the first
attack of each original type receives a 2× STAB boost, while the first attack of
other types receives the pinned fixed-point approximately 1.2× boost; later uses
fall back to ordinary STAB behavior. Terapagos-Stellar has a species exception to
one-use tracking. Stellar Tera Blast and Tera Starstorm have their own target/type/
stat-change rules. Do not insert Stellar as an ordinary column in the type chart.

## Forme mechanics

A set-pool record may describe an eventual battle forme rather than the species
that enters at battle start. The species pages distinguish those records from
additional formes. The catalog also includes relevant appearance variants produced
by getForme, so a hash/name token can legitimately differ while battle statistics
remain the same.

Examples of mechanically meaningful transitions:

- Palafin's Zero to Hero changes its forme after switching out under the ability.
- Meloetta's Relic Song can change its stat distribution and typing.
- Morpeko changes Full Belly/Hangry forme through Hunger Switch and Aura Wheel changes.
- Mimikyu's Disguise blocks a qualifying hit, breaks, and applies its own HP loss.
- Eiscue's Ice Face treats physical and special hits differently and can restore in snow.
- Cramorant changes its gulp/gorge state around specified moves and reacts when hit.
- Minior's Shields Down and Meteor state depend on HP and status prevention rules.
- Ogerpon masks constrain type/forme and Terastallization; Embody Aspect changes
  the relevant stat. This is not an arbitrary ability selection.
- Terapagos has base/Terastal/Stellar states with Tera Shift, Tera Shell and
  Teraform Zero behavior; its max-HP/forme effects differ from ordinary Tera.

Illusion is a disguise, not Transform. Transform changes copied battle properties
under its move/Imposter rules, including available moves and PP conventions, without
simply receiving a fresh randomly generated set. Dynamic typing effects, Roost,
Soak, Protean/Libero and form changes interact with Tera restrictions separately.
The relevant long ability/move descriptions and exact effect rules specify exceptional
guards; this list is an orientation, not a replacement for those complete entries.
