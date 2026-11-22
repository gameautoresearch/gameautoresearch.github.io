# Meditation

Enables the pacifist-turn counter and AltarOfPeace task.

| Property | Value |
|---|---|
| ID | 21 |
| Tier | 2 |
| Ordinary predecessor | Climbing |
| Next technologies | Philosophy |
| Units named by table | none |
| Relevant structures named by table | MountainTemple |

Purchase cost is `4 + own_city_count*tier`, or `ceil(base*0.66666)` with Philosophy known. A tech is legal only when existing, unknown, adjacent to a known tech in the native reachability graph and affordable; no spending during a pending reward/peace trigger. The graph includes reverse prerequisite links to support technologies gained out of normal order from ruins. Purchased/rewarded technology grants100*tier score. Basic and Hunting are already known at the Bardur start.

[Catalog](README.md) · [Wiki home](../README.md)
