# Outcomes, bomb multipliers and evaluation

[Wiki home](../README.md) · [Turn rules](turns-and-comparison.md)

Let s=+1 if landlord wins and s=−1 if either peasant wins. Let b be the number of
**actually played exact four-card bombs or two-joker rockets**, and u=2^b.
A bomb merely held in hand does not count. A four-card backbone inside a six/eight
card composite is not an exact bomb action and does not count. Rocket used as two
single attachments is not an exact rocket action. The final shedding bomb/rocket
counts before terminal targets are computed.

## DMC terminal targets

| Objective | Landlord return | Each peasant return |
|---|---:|---:|
| `wp` | s | −s |
| `adp` | s×2^b | −s×2^b |
| `logadp` | s×(b+1) | −s×(b+1) |

Nonterminal reward is zero. The selected learner's decisions in a completed game
receive its role's terminal target. Peasants share a target; it does not matter
which peasant emptied. The return is not divided between them and no bidding or
spring multiplier is applied. A terminal-return regression target is distinct
from the economic score below; the ADP training convention does not double the
landlord's scalar target relative to each peasant.

## Economic payoff, reported separately

| Role | Economic score |
|---|---:|
| Landlord | s×2×2^b |
| landlord_down | −s×2^b |
| landlord_up | −s×2^b |

The three individual scores sum to zero. If the landlord wins with two played
bomb/rocket actions, returns under WP are (+1,−1,−1); ADP returns (+4,−4,−4);
logADP returns (+3,−3,−3); economic scores (+8,−4,−4). These are four different
reported quantities, not interchangeable score labels.

## Benchmark candidate control and weighting

The candidate controls one role in a game. The panel controls both other seats;
when the candidate is a peasant, this includes its teammate. The candidate does
not jointly control both peasants. Candidate roles and opponent panels receive
equal weight on matched deals. Primary raw score is100 times balanced individual-role
win probability in[0,100], not a 1v1 Elo or economic average pooled across seats.
Per-role win probability and economic scores remain available separately.

Public practice is development data. Private grading uses separate host-owned
deals and opponent policies and does not return feedback for further research.
Uncertainty resamples complete deals, retaining linked roles/opponents together.
A complete valid schedule is required; illegal outputs, inference deadlines,
engine faults and incomplete schedules are typed failures, not natural losses or
unreported dropped games. Reference normalization remains unqualified/null until
separately authorized matched full-budget multi-seed anchors pass calibration.
