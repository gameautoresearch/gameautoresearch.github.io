# Glossary

[Wiki home](README.md)

| Term | Meaning in this mode |
|---|---|
| Army / stack | Integer troop count held on one cell; not an independent moving unit object |
| Land | A player's owned tile count, irrespective of troop count |
| General | Live defeat-critical structure; owned, productive, one per player |
| Castle / city alias | Productive capturable structure; neutral castles have fixed initial defenders and do not grow until captured |
| Whole | Move all but one troop from current source |
| Half / split | Move floor(source army/2) |
| Joint tick | One decision by both players followed by deterministic sequential resolution and clock/growth |
| Defensive priority | Destination is held by mover's own team; a technical sort key |
| Chasing dependency | Pending move into another pending move's source makes that source's move wait, except head-on swaps |
| General trade | Both mutually capturing general moves resolve as an exchange with no elimination |
| Reserve | Troops left at the source after departing stack is removed |
| Surplus | Attacker's sent army minus defender's garrison after a winning attack |
| Fog | Currently unseen non-castle/non-mountain location; not necessarily historically unvisited |
| Silhouette | Unseen castle-or-mountain marker; hides which type, owner and army |
| BFS distance | Shortest number of orthogonal passable steps; castles are geometrically passable |
| Room | Number of other cells reachable within seven steps for spawn fairness |
| Padded area | Fixed 16×16 tensor including impassable outside-rectangle mountains |
| Natural termination | Winner exists after ordinary general capture in this mode |
| Cap / truncation | Reached 1000 without winner; draw for grading, zero outcome reward |
| Public scoreboard | Global owned army/land totals, intentionally visible even for enemy hidden tiles |
| Hidden state | Unobserved ownership/army, generator/RNG/seed/state handles; excluded from every policy input |
