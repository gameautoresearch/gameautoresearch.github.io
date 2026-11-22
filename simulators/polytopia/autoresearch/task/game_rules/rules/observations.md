# Player-visible observation schema

The actor and same-view critic consume `encode_view`, never raw hidden State.
For this11×11 two-player task, observation version3 has **board[11,11,151]** and
**global_state[525]**. Values are float32 after encoding. Model/checkpoint schemas
retain their version; older networks read the compatible prefix/layout, not newly
privileged data.

## Board channel layout

| Channel interval [start,end) | Block | Meaning |
|---|---|---|
|0..8|Terrain IDs0..7|Known-only one-hot; normal generated terrain IDs1..5|
|8..16|Resource IDs0..7|Known-only one-hot; ID8 starfish is not separately represented here|
|16..51|All35 structure table IDs (sorted)|Shared vocabulary includes off-mode entries; they stay zero when unavailable|
|51..97|All46 unit table IDs (sorted)|Shared vocabulary includes off-mode entries|
|97..100|Tile ownership|known neutral /self /other|
|100..103|Unit ownership|known neutral /self /other|
|103..106|Capital ownership|known neutral /self /other|
|106..107|explored|boolean|
|107..108|x|x/13|
|108..109|y|y/13|
|109..110|road|max(raw,0)|
|110..111|structure_level|max(raw,0)/10|
|111..112|unit_health|max(raw,0)/40|
|112..113|unit_veteran|max(raw,0)|
|113..114|detect|own-unit adjacent-hidden-enemy signal|
|114..115|own_unit_moved|boolean|
|115..116|own_unit_attacked|boolean|
|116..117|own_unit_poison|boolean|
|117..118|own_unit_frozen|boolean|
|118..119|own_unit_boosted|boolean|
|119..120|own_unit_invisible|boolean|
|120..121|own_unit_kills|/10|
|121..122|own_unit_home|/121|
|122..123|own_unit_passenger|/128|
|123..124|own_city_level|/20|
|124..125|own_city_progress|/20|
|125..126|own_city_population|/20|
|126..127|own_city_production|/20|
|127..128|own_city_connected|boolean|
|128..129|own_city_riot|boolean|
|129..138|own_city_rewards|IDs0..8 divided by10|
|138..146|unit facing|visible direction1..8 one-hot; missing/unexplored yields0|
|146..149|enemy unit memory|end-of-my-turn presence1/2/3 own turns ago|
|149..150|enemy built|known enemy tile structure/road/level changed since my prior turn|
|150..151|enemy trained|visible enemy unit newly stands on an enemy city tile|

The sorted structure/unit ID sequences used by the categorical blocks are:

- Structures: `0,1,2,3,5,6,8,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33,34,35,37,38,39,41,47,48,50,70,123`.
- Units: `0,2,3,4,5,8,9,10,11,12,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,56,57,59,60,61,62`.

## Global vector

| Interval | Field |
|---|---|
|0..1|round/100|
|1..2|log1p(own stars)/8|
|2..3|physical seat index0 or1|
|3..131|own tech128 flags|
|131..259|own earned monument128 flags|
|259..387|own built unique128 flags|
|387..389|known players,self then opponent|
|389..391|peace,self then opponent|
|391..393|embassies sent,self then opponent|
|393..395|embassies received,self then opponent|
|395..397|pending peace offers,self then opponent|
|397..525|inferred enemy tech128 flags|

## Raw public view and history limitations

The packed two-seat view has5,728 int16 values for121 tiles (`43*n+525`); raw float
HP is cast toward zero during packing. It is decoded before the feature projection.
Unexplored ordinary tile fields are−1. Own-unit and own-city private detail fields
are zero elsewhere; an opponent’s exact progress, unit home, XP/passenger/flags
are not exposed as own detail. Hidden hostile unit type/HP/owner is masked; a
peace partner or own hidden unit remains visible. Crop/metal/starfish visibility
depends on appropriate technologies.

Enemy-tech entries are deductions from actually seen units, structures, roads,
terrain access and prerequisite closure. A seen Dagger implies Diplomacy. This is
player-visible historical inference, not the opponent’s raw researched-tech row.
History is generated only from the seat’s end/start views, and does not provide
an omniscient replay, arbitrary recurrent memory or unseen enemy board snapshots.
Some older public demonstration views lack facing/history and encode those blocks
as zeros; do not mistake those demonstrations for newest runtime observations.

Exact raw packed-field order:

`capital, detect, embassies_received, embassies_sent, enemy_tech, explored, hist_enemy_built, hist_enemy_trained, hist_enemy_unit, known_players, own_built, own_city_connected, own_city_level, own_city_population, own_city_production, own_city_progress, own_city_rewards, own_city_riot, own_monuments, own_unit_attacked, own_unit_boosted, own_unit_frozen, own_unit_home, own_unit_invisible, own_unit_kills, own_unit_moved, own_unit_passenger, own_unit_poison, owner, peace, peace_offers, player, resource, road, stars, structure, structure_level, tech, terrain, turn, unit_direction, unit_health, unit_owner, unit_type, unit_veteran`

[Wiki home](../README.md)
