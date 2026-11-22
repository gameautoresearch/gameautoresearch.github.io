# Stars, cities, population and research

## Population and level progression

Every city has a level, progress toward its next level, cumulative population,
production, border, ruling territory and selected reward counts. Gaining population
adds to both population and progress. While progress≥`level+1`, subtract that
threshold, increment the level and add1 production. This can happen repeatedly.
Cities never lose levels: losing economic contributions can leave negative progress.
Off-turn population changes do not level a city immediately; its next start-of-turn
levels it after income has already read the prior production.

A new captured neutral village becomes level1, progress0, population0, production1,
border1. A captured enemy city preserves its economic state. The score of a city is
`5*progress + 50*(level+1) + 250*Park_count`. With negative progress, add another
`100*progress` and floor total at0. Score changes use differences before/after
population updates. A city progress change of−1 may therefore have a much bigger
score effect below zero than a positive point normally adds.

## City reward decisions

Pending rewards block ordinary actions. The number already chosen is compared with
`level-1`; rewards are offered in level order even when one population grant levels
up multiple times. The following target IDs are fixed:

| Reward slot | Choice IDs | Effects |
|---|---|---|
| First (city reaches level2) |3 Workshop /4 Explorer | +1 production /12-step explorer reveal |
| Second (level3) |1 CityWall /7 Resources | Fortify bonus becomes4 /+5 stars |
| Third (level4) |5 BorderGrowth /8 Population | Radius2 border expansion /+3 population |
| Later (level5+) |2 Park /6 SuperUnit | +1 production and250 score /Giant spawn |

City walls affect a defending Fortify unit, not every unit. BorderGrowth claims
currently unowned radius2 tiles, explores them and updates routes/connections.
A SuperUnit spawn pushes an occupant using its facing-based priority; a trapped
occupant can be removed. The Giant is free as a reward despite nominal cost10.

## Income and turn boundary

At the new owner's start: diplomacy clocks/messages and meetings update, temples
age, work is calculated and stars are paid, multiplier levels refresh, deferred
city level-ups execute, unit flags reset, and observation history updates. Income
is positive only after round0. Owned city work is `production +1` for an original
capital still owned by that seat. An enemy unit standing on the city (even hidden)
or a riot stops that city's work. Negative progress reduces work, floored at0.
Markets pay their own calculated level separately; a besieged city does not also
suppress its market tiles. Embassy income has its own host/builder shares.

## Research

A purchased tech costs `4 + number_of_own_cities * tech_tier`. With Philosophy,
price is `ceil(base * 0.66666)` using the pinned floating multiplier, not a rounded
2/3 shortcut. Research adds100*tier score. Existing unknown technologies adjacent
to any known tech in the native graph are candidates. The graph includes both
`next` edges and reverse prerequisite links, so an out-of-order ruin tech can make
its predecessor available. Basic and Hunting are known from the Bardur start.
New city captures change future prices immediately. All26 native tech records are
listed in [the tech catalog](../technologies/README.md).

## Recruitment and support

An empty own city can recruit an unlocked affordable ordinary unit if the number
of units homed there is≤the city's level, equivalent to capacity `level+1` after
recruiting. Every homed unit counts even if its current owner changed. Recruits pay
their star cost, add5*cost score, begin moved/attacked, and reveal their sight.
Giant, Dagger, boats and infiltration products have acquisition-specific paths;
nominal table cost does not make them ordinary recruits. Converting a non-Independent
unit rehomes it to the converter's current capital. Losing a home city clears homes
of other owners' remaining units. Disband requires FreeSpirit and an idle unit;
it refunds floor((boat cost+passenger cost)/2) while removing the corresponding unit score.

[Wiki home](../README.md)
