# Growth clock and outcomes

[Wiki home](README.md) · [Worked examples](worked-examples.md)

## Tick sequence

A game starts at `time=0`, with one troop on each general. One joint step resolves moves/general trades, increments time by one if the game was not already over, settles a last-team-standing winner, then performs growth only if `winner<0`.

The game clock is simulation ticks, not wall-clock seconds. The frozen benchmark does not grant a second decision midway through a tick or permit extra actions for a faster inference process. Do not import a website UI's “turn” or animation-frame convention without converting to these integer ticks.

## Exact growth

| New time after actions | Every owned ordinary cell | Every owned castle/general |
|---|---:|---:|
| Odd nonmultiple of 50 | +0 | +0 |
| Even nonmultiple of 50 | +0 | +1 |
| Multiple of 50 | +1 | +2 (land + structure increments) |

Neutral cells, including neutral castles, receive no growth. Mountains/padding are unowned and receive none. Growth follows **post-action ownership and flags**, so a newly captured castle grows for its new owner if this ongoing tick reaches an even time. A general transformed into a castle during a final capture does not grow because the game has ended.

The first structure increment is at `time=2`, after two joint decisions; none occurs at time 1. The reset state at time 0 does not get a free initial `time%50==0` payment: step increments to time 1 before checking growth.

## Natural ending

An ordinary opponent-general capture gives a natural win/loss. The deciding tick advances time, but **no income** is paid anywhere once the winner is set. This matters when the capture happens on time 2,50 or another growth boundary. Winner is a player/team ID; in 1v1 it is seat 0 or 1. The policy cannot choose a surrender action from this task's action space.

The upstream state has external surrender/neutralize APIs for replay/server events. They are not admitted gameplay controls here. No AFK rule or disconnect loss should be invented for a normal policy wait/pass.

## Horizon

After a step, the wrapper sets:

- `terminated = engine winner exists`.
- `truncated = (post-action time >= 1000) AND NOT terminated`.

Thus a genuine capture **on the 1,000th joint tick wins**, taking precedence over cap truncation. Without capture, that step reaches a cap draw. Final land and army totals are diagnostics only; they do not break the tie. There is no deathtouch mechanism to force a result.

The low-level engine does not own this benchmark horizon. The wrapper preserves the true final state before reset, so the last board of a capped game can still be read. Reading an autoreset observation as the terminal frame would be wrong.

## Distinct outcomes

Natural capture win/loss, cap draw, illegal policy output, simulator exception, incomplete evaluation and budget-aborted work are separate categories. Do not substitute a loss for infrastructure failure, award a draw for a crash or turn a truncated/incomplete schedule into a confident policy score. See [benchmark validity](benchmark-boundary.md).
