# Terminology and quick reference

[Wiki home](README.md)

| Term | Exact meaning here |
|---|---|
| Landlord | Lone team, first actor, initial20-card hand |
| Peasant/farmer | Either17-card role, shares outcome with the other peasant |
| Down | First role after landlord clockwise |
| Up | Last role before landlord clockwise |
| Free lead | No live nonempty rival; choose any generated nonempty combination |
| Live rival | Last action, or second-last if last was a pass |
| Pass | Empty action, available only while responding |
| Backbone/body | Triple/quad/sequence portion of a composite |
| Wing/kicker | Cards attached to a triple, airplane or quad |
| Straight | Consecutive ordinary ranks from3..A, fixed multiplicity |
| Airplane | Consecutive triples with one or two attachment cards per triple |
| Bomb | Exact four-of-one-rank action |
| Rocket/king bomb | Exact small-joker + big-joker action |
| WP | Per-role team win target ±1 |
| ADP | Per-role target ±2^bombs |
| logADP | Per-role target ±(bombs+1) |
| Economic score | Landlord magnitude twice each peasant's; zero-sum over three seats |
| Canonical action index | Current-frame lexicographic index into unique card multisets |
| Unseen-card union | Inferable combined cards of both other players, not their partition |
| Thermometer encoding | Count c is c leading ones in a rank's four slots |
| TYPE15 | Detector's wrong/unclassified label; not an additional playable family |

Notation in examples: `33344` means three3s plus two4s, not one numeric card;
`X` is small joker, `D` big joker, `2` has engine code17. All encoded offsets are
zero-based and intervals `[start:end]` exclude the end.
