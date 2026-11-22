# Capture, ruins, exploration rewards and score

## Village and city capture

An idle own unit (neither moved nor attacked) on a foreign unprotected village/city
can capture it. Moving onto a city and capturing it are separate atomic actions.
A neutral village initializes a level1 city and claims unowned radius1 territory.
An enemy city keeps its level/progress/production/rewards/border; its city score and
owned territory/improvement score transfer to the capturer. New ownership updates
building population, routes, connections and exploration. Capture exhausts/reveals
the unit and assigns that city as home unless Independent. Enemy units still homed
to the captured city lose that home.

Losing one's last city causes elimination: remaining units are disbanded/wiped,
with the native refund/accounting effects. In two-seat mode this is a natural
terminal win/loss. Capturing only the original capital does not necessarily win
if the opponent still owns another city. Original-capital ownership remains
important for income, embassy and network hub behavior.

## Ruin choices and outcomes

Ruin reward candidates are Resources, Population, Swordsman, Battleship, Explorer
and FreeTech. The list is Fisher–Yates shuffled using the pinned per-tile/seed hash
and the first valid reward is chosen; this is not a uniform unconditional draw
from every displayed reward type. The agent never sees the hidden seed.

| Reward | Result/eligibility |
|---|---|
| Resources | +10 stars |
| Population | +3 population to current-capital fallback city, if any |
| FreeTech | One native unlockable unknown tech; gains100*tier score, no star charge |
| Explorer |12-step reveal if an accessible neighbor and nearby unexplored tile exist |
| Swordsman | Land ruin: push examiner, create veteran Swordsman with no home |
| Battleship | Water ruin with Sailing: push examiner, create veteran Rammership carrying Warrior, no home |

Examination consumes the ruin, exhausts and reveals the examiner. Reward-unit pushes
can move/remove the original occupant. Starfish is a separate idle capture action
requiring Navigation: remove starfish and gain8 stars. Neither outcome is a city.

## Explorer behavior

City/ruin explorers walk up to12 steps, not12 arbitrary radius reveals. Every step
scores an11×11 local window for fog distance, accessible terrain, lighthouse and
mountain reveal effects. Candidate neighbor order is x-outer; equal scores use
the pinned hash. Already visited path tiles receive increasing backtracking
penalties. Every step reveals radius1, or2 on a mountain. The FastPath relaxation
has a0x1f2 queue-index bound. This is deterministic engine behavior, not a policy
search privilege.

## In-game score components

- Newly explored tile: +5.
- Territory claim: +20 per gained tile plus its improvement score; subtract from
  a prior foreign owner where appropriate.
- City: `50*(level+1)+5*progress+250*Park_count`, with the special negative-progress
  adjustment described in the economy chapter; captured enemy city score transfers.
- Purchased/free research: +100*tier.
- Unit creation: +5*nominal unit cost; removal loses that amount, including passenger
  cost for boats. Transformations/upgrades adjust the difference rather than creating
  a second full passenger score. Conversion transfers the full unit score.
- Temple:100 per age level, maximum500; Park250; task monument400.

The benchmark uses this current score only at the explicit round cap. Scores are
not used to fabricate natural combat wins before that boundary. Natural results,
round-cap score adjudication, action-cap truncation and simulator failures remain
separate records.

[Wiki home](../README.md)
