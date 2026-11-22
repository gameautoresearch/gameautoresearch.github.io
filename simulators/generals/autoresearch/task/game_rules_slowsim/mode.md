# Mode and objective

[Wiki home](README.md)

## What is fixed

| Setting | This benchmark |
|---|---|
| Environment | `generals-research-1v1-medium-v2` |
| Players | Two independent players, physical seat 0 and seat 1 |
| Teams | One player per team; no allies |
| Playable height | Uniform integer 12 through 15 inclusive |
| Playable width | Independently uniform integer 12 through 15 inclusive |
| Neural board | 16×16; unused bottom/right cells are impassable mountain padding |
| Sight | Fog of war; 3×3 neighborhood of owned cells, public totals and structure silhouettes |
| Neutral castles | Enabled; 4–6 inclusive, starting army 10–15 inclusive |
| Terrain-density parameter | Exactly .20; see generation for mountain-to-castle conversion |
| Spawn separation | Minimum requested BFS walking distance 10, no maximum; explicit fallback exists |
| General trades | Enabled |
| Castle construction | Disabled |
| Deathtouch/endgame kill modifier | Disabled (`null`) |
| Decision unit | Both players submit one action per joint tick |
| Horizon | 1,000 joint ticks |
| Cap result | Half-point draw, never a land/army tiebreak |

## Winning

Capture the opponent's live general with a strictly winning army attack. Its tile becomes a castle and the captured player's other territory transfers to the capturer with armies halved, rounded up. In this two-player game ordinary capture ends the game. A special simultaneous mutual-general capture is instead a **general trade**, where both players survive and exchange territories/generals; read [the exact trade rule](joint-order-and-trades.md).

You can expand over empty ground, concentrate armies, capture productive castles, defend your general, raid the enemy's income and seek its general under fog. These are consequences of the rules, not mandatory research strategies. A larger scoreboard army or territory is never by itself a victory.

## Excluded modes and APIs

The upstream engine can express teams, more players, build-castles, deathtouch, surrender/AFK neutralization and perfect-information views. Those are not policy choices or enabled modifiers here. A deprecated `cities` code alias means castles; it does not introduce another city type. The historical 8×8 environment has a different source/rules binding and must not be used as this task's specification.

Hardware, wall-time allowance and checkpoint-resume permissions belong to the selected research session, not to game rules. This wiki grants no compute authorization.
