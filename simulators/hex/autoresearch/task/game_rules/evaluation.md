# Evaluation

## Openings and seats

Every evaluation game starts from a 2-ply opening: black's first stone and white's first stone are
placed for both players, and play continues with black's second move. Openings are drawn without
replacement from all 169 x 168 = 28,392 ordered pairs of distinct cells. Each opening is played twice
against each opponent: once with your policy as black and once as white. Without the swap rule, each
opening is a theoretical win for exactly one side, so playing both seats is what makes the result fair.

## Opponents

Nine opponents, weighted equally, 180 to 420 Elo apart from random play to the anchor, so that the
score rises with skill over the whole range. Some are diluted: with probability epsilon (chosen per
move from the game's own seeded generator) they play a uniformly random legal cell instead of their
own move.

| Opponent | What it is | Elo vs anchor |
|---|---|---|
| `random` | Uniformly random legal cells. | -2580 |
| `heuristic-e90` | The heuristic below, but a random cell with probability 0.9. | -2164 |
| `heuristic-e80` | The heuristic, random with probability 0.8. | -1852 |
| `heuristic-e60` | The heuristic, random with probability 0.6. | -1486 |
| `heuristic` | Tries every empty cell and keeps those that maximize the opponent's shortest-path distance minus its own after the move (the distance being the fewest empty cells a colour still needs to join its edges, its own stones free, opponent stones blocking); takes a winning move at once; seeded tie-break. Source: `simulator/simulators/hex/heuristic.py`. | -1155 |
| `mohex-100-e15` | MoHex 2.0 at 100 simulated games per move without pre-search, random with probability 0.15. | -917 |
| `mohex-100` | MoHex 2.0 at 100 simulated games per move without pre-search. | -498 |
| `mohex-2k` | MoHex 2.0 at 2,000 simulated games per move. | -177 |
| `mohex-anchor` | MoHex 2.0 with its parallel DFPN solver at 50,000 simulated games per move: the anchor, fixed at 0. | 0 |

MoHex 2.0 is the Monte Carlo tree search Hex program of the University of Alberta Benzene project.
It is a held-out opponent: you meet it only through `ar eval`, and neither its source nor a binary is
in your workspace or on any worker that runs your code. Beyond its search it uses virtual-connection analysis
and inferior-cell pruning, so even few simulations play sensible connective moves. Its strength is fixed
by simulated games per move, not by time. For a given level and position its choice is reproducible,
apart from the anchor's solver threads.

The ratings come from opponent-vs-opponent games on seat-paired 13x13 openings, fitted with a
Bradley-Terry (Elo) model; each has a standard error of about 30 to 55 Elo (random play about 75).

## Score

The score is 100 x the mean over the nine opponents of your win rate across both seats. Every game ends
in a win or a loss. The report includes a 95% interval from a bootstrap that resamples whole openings
(both seats against all opponents together), the win rate against each opponent and from each seat,
and the mean game length.

## Your Elo

`panel_elo` is the rating that best explains your results against all nine opponents given their
ratings above, with a 95% interval from the same opening bootstrap. It is on the anchor's scale: 0 is
the anchor and about -2600 is random play. If you win or lose nearly every game it reports a bound
(`at_bound`) rather than an estimate. Each opponent is most informative when you are within a few
hundred Elo of it, which is why they are spread as they are.

## Against the anchor

The report gives `elo_vs_anchor` = 400 x log10(p / (1 - p)) for win rate p against `mohex-anchor` alone
(with p kept half a game away from 0 and 1), its interval, and how many openings you won from both
seats, one seat and neither. The anchor is strong but far from perfect on this board (see
[Scope and limitations](scope-and-limitations.md)): nothing caps your win rate against it at 50%, and
the best possible score is 100. Winning an opening from both seats means the anchor erred on it.

## The tournament

When the research sessions of a campaign are over and graded, the final networks of all its researchers
play each other: every pair, the same private openings from both seats, with the five strongest panel
opponents added as references, under the same inference rules as below. Its Elo ranking is reported
beside the panel score. You cannot run or see it during your session.

## Practice and private grading

`ar eval NAME` plays a completed run against all nine opponents on 16 fixed public openings: 288 games.
It runs on an 8-CPU worker without a GPU, and its allocated time is charged to your research allocation at
that worker's rate (about half the rate of a training worker second; `ar budget` gives the number). The
default grant is 1,800 seconds (`ar eval NAME --seconds S` changes it); a schedule that does not finish
in its grant fails with no partial score and the grant stays charged. Most of the time goes to the
anchor's moves, and games against a stronger policy last longer, so practice takes longer as you
improve. Private grading uses 64 fresh openings from a seed you never see (1,152 games), the same
opponents and the same scoring.

## Inference rules and faults

Your policy runs in a separate unprivileged process with one CPU thread and no GPU or network. The
evaluator sends batches of up to 64 positions where your policy is to move, from many games at once.
Loading the model (`create` and the weights) has 60 seconds and is not on the move clock, so do any
warm-up there. After that, inference is capped per move: a call with n positions must answer within
0.1 + 0.01 x n seconds of wall time, that is 10 milliseconds per position plus a tenth of a second per
call (0.74 seconds for 64 positions, 0.11 for one). On one laptop core a 64-position forward pass took
about 0.1 seconds for a 48-channel, 4-block residual network, 0.3 seconds for 64 channels and 6 blocks,
and 0.9 seconds for a network of 1.3 to 3 million parameters; the evaluation CPU is a different one, so
measure with `ar check` and `ar eval` and leave a margin. A call answered late is replaced by a
uniformly random legal move in every game of that batch, your policy process is restarted (its
in-memory state is lost), and play continues; this is the same in practice, in private grading and in
the tournament. The report counts late calls
(`move_timeouts`) and substituted moves (`random_moves`, also per game). An answer of the wrong
length, not an integer cell, or an occupied cell, or a model that does not load in 60 seconds,
invalidates the whole evaluation: there is no score, and no losses are awarded in its place. If a MoHex process crashes, the same move is
asked of a fresh process (at most twice; the report counts `mohex_restarts`); a MoHex failure beyond
that is a simulator fault and also
yields no score. Games cannot be truncated or drawn.
