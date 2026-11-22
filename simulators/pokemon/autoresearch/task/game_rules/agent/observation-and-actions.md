# Player-visible observation and action contract

[Wiki home](../README.md) · [Mode/version boundaries](../mode/README.md)

The network receives only one player's public protocol plus that player's request.
It never receives the opposing request, hidden team partition, RNG seed/state,
simulator object, submitted opposing action or private opponent policy. Public
species/stat/rule tables are world knowledge; looking up base stats is different
from reading an opponent's generated EV/IV/item before revelation.

## Shapes and categorical IDs

With a leading batch dimension, the dictionary has:

| Field | Shape per observation | Contents |
|---|---|---|
| tokens | 12×12 | species, ability, item, types, Tera type, status, four known moves, last move |
| mons | 12×24 | visible numeric Pokémon fields |
| global | 40 | phase, turn, hazards/conditions and visible flags |
| global_tokens | 4 | weather, terrain, two zero pads |
| action_tokens | 14×3 | move/type/category or switch species/types |
| actions | 14×12 | visible numeric candidate-action information |
| mask | 14 | currently published legal action availability |

The flattened content has 700 values including masks/tokens. Slots 0–5 are the
player's current request order, 6–11 opponent records in reveal order; unseen
opponent slots are zero. Teams have three Pokémon, so own slots 3–5 and opponent
slots 9–11 are always zero; the layout (and every /6 count normalizer) is the same
as in six-Pokémon Random Battle. Categorical zero is missing/padding. Nonempty tokens are
`1 + FNV1a32(prefix + ':' + normalized_name) % 16383`; normalization lowercases and
removes non-alphanumeric characters. This is hashing, not a collision-free ordinal
species ID. Identical numbers in different semantic prefixes need not name the
same concept. Name hash collisions are possible and part of this baseline encoder.

## Pokémon slot fields

Categorical fields in order: species; revealed/current or own base ability; item;
first type; second type; generated own/revealed Tera type; major status; first four
known moves; last move. For an opponent, moves are the first revealed unique moves,
not its secret generated moveset before revelation.

Numeric fields in order: present; active; HP fraction; HP-known flag; fainted flag;
level/100; Attack, Defense, SpA, SpD, Speed each /300; seven stage values (Attack,
Defense, SpA, SpD, Speed, accuracy, evasion) each /6; Tera flag; Substitute;
confusion; Leech Seed; Taunt; own-player flag. Own actual stats are available in
its request. When a visible record lacks a stat, the encoder falls back to public
species base stats; that is not the opposing exact generated stat. An HP value
from the opposing protocol is percentage-based; own HP can be exact.

## Actions

IDs 0–3 choose the current request's four move slots, 4–7 choose the corresponding
move while Terastallizing, 8–13 switch by the current own-request Pokémon order;
with three Pokémon only 8–10 can ever be legal and 11–13 are always masked.
The host applies the mask. Move slots can change under Transform/PP locks; switch
IDs must not be treated as a permanent species identity independent of request order.
Only ready seats act; a wait frame has an all-false mask. Forced switch requests
normally expose reserve choices rather than ordinary attacks.

Move candidate tokens are move/type/category. Numeric fields are base power/150;
accuracy/100 (always-hit represented as one); priority/5; PP/64; max PP/64;
Physical, Special and Status indicators; public type-effectiveness/4; whether the
actor's public original typing matches; Tera-choice flag; zero padding.
These are baseline catalog features, not a complete computed damage simulation:
Weather Ball/Tera Blast/dynamic power, secret abilities/items and other effects can
make them differ from eventual execution. Switch tokens are candidate species and
its two public base types; numeric fields include visible HP, Tera flag and a
switch marker, with zero padding. Revival requests can reuse reserve-action slots
for legal fainted own targets under the declared adapter's handling.

## Global fields and omissions

The forty numeric global fields are turn clamped to 1,000 then /1,000; forced-switch,
trapped, maybe-trapped, own-Tera-available, opponent-Tera-used, revealed-foe-count/6,
public faint-count potential; ten side-condition counters for each side /3; four
field flags; wait, active-own-Tera and six zero pads. The ten counters are Stealth
Rock, Spikes, Toxic Spikes, Sticky Web, Reflect, Light Screen, Aurora Veil, Tailwind,
Safeguard and Mist. Field flags are Trick Room, Gravity, Magic Room, Wonder Room.
Counters follow protocol sidestart/sideend reconstruction; they are not exact hidden
timer values. Last weather/terrain names are tokens, not full duration histories.

This encoder omits many true simulator details: complete volatile history, exact
condition remaining turns, unrevealed opposing moves/items/abilities/stats, some
Illusion/Transform/forme history, and a recurrent memory interface in baseline
releases. The simplified type reconstruction treats an ordinary Tera type as the
visible single type; special Stellar/original-type cases should not be mistaken
for a fully faithful internal type object. A rule existing in this wiki does not
mean its full live state is directly represented in the 700 features. Preserve the
actual codec; do not “fix” these omissions by supplying privileged state.

## Choice failures and outcomes

Visible disabled/trapped/Tera constraints define masks. Hidden trapping can still
cause an upstream unavailable-choice retry; it is neither an illegal masked output
nor a natural loss. Phase caps, unavailable-choice retry limits and special target
policies are release-specific. Natural outcomes, caps, policy illegality,
unsupported requests and simulator faults remain distinct, with no invented
fallback outcome. See the selected session's exact contract before executing it.
