# Mode, starting position and map distribution

## Exact scope

The task is `polytopia-bardur-1v1-11-drylands-v1`: two Bardur seats, an11×11 board, fog,
`polytopia-2.17.3` native rules, diplomacy enabled by engine default and resignation
disabled. One side takes an arbitrary sequence of atomic actions and explicitly
ends its turn. A network decision is not a complete turn. Two player turns form a
round; the native counter increments when order returns to seat1. A game that
reaches the round limit of30 rounds is cut there and goes to the higher in-game
score; see [caps and outcomes](scope-and-limits.md).

The current Rust engine and its shared exported rule table are authoritative.
The retired simplified `bardur-1v1-v1/v2` research engines are different games.
No other tribe, four-player mode, other board size or current live-game update is
silently admitted. Special-tribe rows in a shared enum/one-hot vocabulary are not
playable content. See [coverage and exclusions](scope-and-limits.md).

## Bardur start

Each seat starts with5 stars, Basic and Hunting researched, one level1 capital
with production1 and border radius1, and a10-HP Warrior homed there. The initial
nominal score is390 before exploration gains. The capital's surrounding radius1
square is owned. The initial city-center marker has the owner's capital ID. The
starting unit is ready for action. Start-of-game exploration then reveals the
capital's radius2 region and appropriate unit sight, awarding5 score per newly
explored tile. No income is paid at round0; do not add another capital payment to
the initial five stars.

## Drylands maps

Every map is of one type, the synthetic `drylands` preset: almost all land (93–99%
of the tiles in a check of3,032 layouts), light mountains and forest, and only
small patches of water. There
is no map mix and no other preset in this task.

The11×11 generator uses minimum capital Chebyshev separation6 and four ruins.
Village spacing is3. Each seat starts on usable land with a surrounding start
region and legal resources. The benchmark's fixed100,000-layout training pool is
separate from public practice and private grading. The pool itself is not copied
into this wiki.

The preset's parameter ranges are in [map parameters](map-parameters.md). They are
the benchmark's distribution, not a claim to match a commercial map preset: this
preset is a synthetic generator, not the real game's map generator.

[Wiki home](../README.md)
