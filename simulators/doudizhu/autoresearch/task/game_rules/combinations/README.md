# Complete legal-combination catalog

[Wiki home](../README.md) · [Comparison rules](../rules/turns-and-comparison.md) · [Exact quirks](../rules/composite-quirks.md)

Pass plus fourteen nonempty generator families exhaust the declared action families.
A family is a rule pattern, not a global action ID. The number of instantiated
multisets depends on the hand and live rival. Every available multiset is returned.

| Type | Family | Card count |
|---:|---|---|
| 0 | [Pass](pass.md) | 0 |
| 1 | [Single](single.md) | 1 |
| 2 | [Pair](pair.md) | 2 |
| 3 | [Triple](triple.md) | 3 |
| 4 | [Four-card bomb](bomb.md) | 4 |
| 5 | [Rocket / king bomb](rocket.md) | 2 |
| 6 | [Triple + single](triple-single.md) | 4 |
| 7 | [Triple + pair](triple-pair.md) | 5 |
| 8 | [Single straight](single-straight.md) | 5–12 |
| 9 | [Pair straight](pair-straight.md) | 6–20 in reachable hands |
| 10 | [Bare triple straight](triple-straight.md) | 6–18 in reachable hands |
| 11 | [Airplane + single wings](airplane-singles.md) | 4n, n≥2 |
| 12 | [Airplane + pair wings](airplane-pairs.md) | 5n, n≥2 |
| 13 | [Four + two singles](four-two-singles.md) | 6 |
| 14 | [Four + two pairs](four-two-pairs.md) | 8 |

Minimum sequence ranks are5 for singles,3 for pairs and2 for triples. Only3..A
are consecutive; 2 and jokers cannot extend them. All composites require enough
physical copies in hand. The pinned generator's single-wing permissiveness and
its classification/filter corner cases are essential parts of this exact mode.
`TYPE_15_WRONG` is a detector label, not another allowed action family.
