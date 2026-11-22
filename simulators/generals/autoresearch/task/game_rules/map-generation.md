# Map generation

[Wiki home](README.md) · [Mode](mode.md)

This chapter describes the pinned algorithm, including fallback behavior. A desired spawn-distance parameter is not a proof that every conceivable seed satisfies it; the generator stays total rather than rejecting indefinitely.

## Dimensions and reproducibility

For each public map seed, the wrapper makes a JAX key, folds in `0x53495a45` for dimension selection, and draws two integers with upper bound 16 exclusive and lower bound 12 inclusive. Height and width are independent. A separate original `PRNGKey(seed)` generates the chosen rectangle's terrain. There are exactly 4×4 = **16 shapes**. The wrapper explicitly calls the fixed-size generator for each rectangle; directly calling upstream `init_state` would silently choose the maximum size rather than this mixture.

Maps are grouped by shape for compilation in eight-key chunks, then returned in original input seed order. Repeated padding keys used to complete a setup batch are discarded; they are not extra maps or a changed distribution. Public bank size and private schedules are [benchmark metadata](benchmark-boundary.md), not hidden information supplied to an acting policy.

## Terrain and castles, before either general

1. Compute playable area `A = height*width`, excluding 16×16 padding.
2. Draw desired mountain count uniformly between `floor(.20*A)` and the same upper bound, inclusive. Here it is therefore fixed to `floor(.20*A)`.
3. Select that many distinct playable cells using Gumbel/top-k sampling and mark them mountains. The implementation reserves a static candidate capacity `A//4`, sufficient for this .20-density task.
4. Draw desired castle count uniformly from 4,5,6.
5. Select distinct existing mountain locations with Gumbel/top-k and convert the selected ones into passable neutral castles. Each castle's army is sampled independently from `[10,16)`, meaning 10–15. Castle conversion occurs **before** spawn BFS, so the paths measured for spawn distance are the ones played on.

Consequently .20 is the **initial mountain-placement input**, not the final fraction of impassable cells: castle carving reduces final mountain count by the number of castles. In this task there are ample initial mountains for 4–6 conversions. Generic tiny/nearly mountain-free inputs would cap actual castles by available mountain sites; those are outside the selected size/density contract. Open ground is not converted into a blocking castle.

## General A candidates

Castles are passable for walking-distance geometry, though traversing one during play may require winning its defending army. A general must be placed on ordinary empty ground, never on a mountain or castle.

The generator expands a mask from all castles by six orthogonal passable steps. It samples **eight** distinct candidate A positions by Gumbel/top-k: spawnable positions in this near-castle mask receive logit 20; other spawnable positions logit 0. This is a strong bias, not an absolute guarantee of a castle within six steps.

For each candidate it computes a BFS distance field through all nonmountain cells. Unreachable cells have sentinel `A`; reachable means distance `<A`. A valid far partner must be ordinary spawnable ground and distance at least 10. There is no maximum distance constraint.

## Room fairness and candidate ranking

“Room” of a position is the number of other passable cells reachable within seven orthogonal walking steps; the starting cell itself is excluded. A fair partner has room-count difference at most **5** from candidate A. This is slack, not exact equality; it reduces easy inference of the other general from a uniquely matching room count.

For each of the eight A candidates, compute whether it has any fair far partner, any far partner, the smallest room gap among far partners, and its maximum reachable distance. Its exact priority score is:

`(2*has_fair + has_far)*100000 - best_room_gap*100 + reachable_span - candidate_index`

The largest score wins. This prioritizes existence of fair/far options, then smaller room gap, then larger span, with earlier candidate index breaking remaining ties. The finite geometry bounds make those priority weights meaningful in this mode.

## General B and fallback ladder

For selected A, sample B uniformly by Gumbel-max from the first nonempty candidate set:

1. Far-enough, ordinary-ground reachable cells whose room gap is ≤5.
2. If none, far-enough cells with the **minimum achievable** room gap.
3. If no far-enough cell exists, the farthest reachable cells in the selected allowed component.

The final fallback uses the source's allowed reach mask, so it should not be over-described as a proof of all normal spawn preferences. No new L-shaped repair corridor is carved after spawns, and no rejection loop silently redraws the map. The readout's inspected public pool satisfied the requested distances; that observation does not erase the algorithmic fallback.

A fair coin swaps which selected physical location belongs to seat 0 and seat 1. The generator asserts both general markers, then pads **bottom and right** to 16×16 using `-2`. The playable rectangle's upper-left origin is unchanged.

## What this does not reveal

The policy does not get seed, generator key, unobserved tile arrays or exact enemy general coordinates. The known mode distribution, public dimensions/border structure inferable from views and any information legitimately implied by visible terrain remain available. Do not query private bank layouts or RNG state.
