# Team and set generation

[Wiki home](../README.md) · [All species templates](../species/README.md)

## Pool and sampling

The exact singles pool is the 509 records in the pinned `gen9/sets.json`; doubles
and factory JSON files are outside this mode. Each species page transcribes all
of that record's roles, candidate moves, candidate abilities, Tera options and level.
Each record is a collection of templates, not a fixed competitive set. Moveset
selection, culling, ability selection and item selection depend on the whole set
and previous team members. Uniformly choosing four moves from the union is wrong.

The generator groups entries by base species. Base species receive weight one
for one–three formes, two for four–six, and three for seven or more, capped at
three; Squawkabilly has an explicit weight-one exception. It samples without
replacement from this weighted base pool, chooses a forme, rejects incompatible
choices, and stops at the format's team size: three Pokémon in this benchmark (six in
unmodified Random Battle). Duplicate base species are rejected. Other sampled
entries can therefore change the conditional distribution of later teammates.

The limits scale by max(1, round(team size / 6)), which is 1 for both three and six,
so selection normally limits each type to two members,
three weaknesses to an attacking type, one double weakness to an attacking type,
four Freeze-Dry weaknesses, and one level-100 Pokémon. Rock/Fire coverage checks
include the source's special handling of Dry Skin and Fluffy. Team composition
also rejects paired near-duplicate support roles (Blissey/Chansey and
Illumise/Volbeat), duplicate web/screen setter groups, and Toxicroak with a sun
setter. These are set-generation constraints, not restrictions on legal in-battle
moves or replacements. The engine retains retry/failure behavior; no benchmark
should filter to only an easy-to-simulate subset.

Ogerpon/Ogerpon-Hearthflame/Terapagos and the Tera Blast user role share the
team's Tera-user constraint. Zoroark is excluded from the last generated slot (the
third here), and a species that may not lead is skipped when only the last slot remains
and no lead has been placed yet.
The generator places a permitted lead at the front; lead exclusions and lead-sensitive
items/roles are applied before completing the team. Cosmetic/start-form choices
include the generator's Pikachu caps, Basculin colors, Magearna-Original and the
listed near-cosmetic Maushold/Dudunsparce/Polteageist/Sinistcha/Zarude variants.
Battle-only template records may start as their base form and transform later.

## Moveset construction

Choose one compatible role template, then one candidate Tera type. Fast Bulky Setup
is excluded for lead Paradox Pokémon because it generates Booster Energy; another
Tera Blast user is excluded once the team already has one. The generator copies
that template's move pool and builds a move counter describing typing, offense,
recovery, hazards, setup and ability synergies. It repeatedly adds required/enforced
moves and culls incompatible/redundant options until at most four remain.

Examples of explicit enforcement are the Tera Blast user role's Tera Blast;
required forme moves such as Relic Song; Facade when Guts can apply; Night Shade,
Revelation Dance, Revival Blessing and Sticky Web when in the pool; bulky-support
hazard removal when the team lacks it; suitable phazing sets' Stealth Rock; Aurora
Veil when screens are absent; Knock Off on pure Normal/Fighting singles; and
Smeargle's Spore. STAB, priority, reliable recovery and role-appropriate setup get
conditional enforcement. Paired moves, redundant coverage, weather/terrain synergy,
team hazard duplicates and ability compatibility further constrain selection.

The full source has a large conditional culling/enforcement program. Every template
and resulting effect is documented here; its complete branch-distribution semantics
are retained in the pinned generator and [source inventory](../scope-and-limitations.md).
This wiki does not falsely assign a numeric probability to each possible four-move
combination. Once generated, the four moves are shuffled; their request order is
not a persistent canonical ordering across teams.

## Ability assignment

Candidate abilities come from the selected role template, not every ability a
species could have in arbitrary play. A sole option is forced. Hard-coded choices
include Drifblim's Defog→Aftermath (otherwise Unburden), Flash Fire when its proposed
Tera type is Fire-weak, special-only Thundurus/Tornadus→Prankster, Water-coverage
Swampert→Torrent, multi-hit Toucannon→Skill Link, Snowscape→Slush Rush and Golduck
on existing rain→Swift Swim. Remaining candidates are culled for compatibility;
the generator samples an allowed one, falls back preferentially to weather
synergies if none survive, then to the original pool. Initial templates contain
203 distinct abilities. Related forme/effect changes can introduce additional
abilities, identified separately in [the ability index](../abilities/README.md).

## Item assignment

The priority selector runs before the ordinary singles item selector. Required
form items win early; Arceus takes the first required item (a Plate), never a
legacy Z-Crystal. Pikachu gets Light Ball, AV Pivot gets Assault Vest, Regieleki
Magnet, Imposter Choice Scarf, appropriate multi-hit sets Loaded Dice, Shell Smash
White Herb, Meteor Beam or unsupported-weather Electro Shot Power Herb, and
screen sets Light Clay. Rest without Sleep Talk often gets Chesto Berry; bulky
NFE species get Eviolite; Poison Heal/Quick Feet get Toxic Orb; applicable Guts/
Facade sets get Flame/Toxic Orb; Magic Guard/appropriate Sheer Force sets get Life Orb.

Other priority branches include the appropriate required mask/orb/memory items,
role-dependent Lokix, Froslass/Population Bomb/Hustle Wide Lens, Smeargle Focus Sash,
Throat Spray, Weakness Policy, Choice-item Trick/Switcheroo/Healing Wish builds,
Sitrus setup, Unburden/Anger Shell random choices, Heavy-Duty Boots for severe
Rock weakness and Court Change, and deliberately **no item** on eligible Acrobatics.
The five Gluttony random berries are excluded from the documented initial item
candidate set because no pinned initial template selects Gluttony. The separate
doubles-only Dragon Dance→Clear Amulet branch is also excluded.

The ordinary selector prioritizes fully physical or fully special Choice sets
(with conditional Choice Scarf), Assault Vest on suitable status-free roles,
specific Weakness Policy/Custap sets, Substitute→Leftovers, support lead Focus
Sash, Rock-weak hazard avoidance→Heavy-Duty Boots, then recovery/bulk Rocky Helmet,
setup Outrage Lum Berry, Protect Leftovers, special Air Balloon and Pawmot Leppa
Berry branches. Role and move composition choose Life Orb versus Leftovers near
the end; Leftovers is the final fallback. The [59-item catalog](../items/README.md)
is a conservative branch inventory, not a proof that every branch is realizable
for a current initial template, and not a claim of equal item probability.

## Actual stats and metadata

Start with 85 EVs and 31 IVs in each stat; levels are the exact set-table levels
(default 80 only if absent). HP EVs are reduced in steps of four to reach favorable
hazard, Substitute/Sitrus, Belly Drum/Fillet Away/Shed Tail or Minior thresholds.
Attack EV/IV drop to zero when no move needs the user's Attack, with explicit
Transform, Shell Side Arm, physical Tera Blast and other exceptions. Gyro Ball or
Trick Room reduces Speed EV/IV to zero. Thus the base-stat table and level alone
do not determine exact generated HP or Speed. No nature multiplier is specified
by this returned Random Battle set, so ordinary stats are neutral-nature unless
the declared generator changes. Gender follows species restrictions or its random
choice; Greninja is forced male. Shiny chance is 1/1,024 and has no battle-stat bonus.
