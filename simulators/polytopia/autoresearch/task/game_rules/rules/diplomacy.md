# Meetings, peace, embassies and war

Diplomacy is enabled in this benchmark's default native options. There is only one
opponent, so only the first six-entry diplomatic block is legal; unused blocks
exist in the fixed catalog for other seat counts but are masked here.

A player meets another upon exploring its visible unit/city, receiving a peace
request or the applicable embassy interaction. The meeting gift uses the encountered
player's score: `1 + 2*min(ceil(max(score,0)/1000),5)`, and in this two-player version123 mode, meeting player1 grants2 extra away
stars. The gift is paid once per meeting direction. Do not infer hidden score from a guessed
fixed meeting reward. Researching Diplomacy reveals met players' original capitals.

| Block target | Command | Prerequisites/effects |
|---|---|---|
|0 | Break peace | Existing peace; mutual cooldown, remove breaker's units in victim territory and exhaust other breaker units |
|1 | Decline request | A currently pending request from this opponent |
|2 | Accept request | Pending request; mutual peace and cooldown cleared |
|3 | Establish embassy | Diplomacy, known living opponent retaining original capital, no existing outgoing embassy, affordable |
|4 | Offer peace | Strategy, no current peace and no pending bilateral offer |
|5 | Destroy embassy | Existing outgoing embassy |

A pending peace message must be answered before ordinary actions, including city
rewards. Offers arrive at the receiver's next start-of-turn. Cooldown protection
uses seat/round clocks; it is not simply one arbitrary atomic action. Toward a lower
seat ID, cooldown ends when break_round<current_round; toward a higher ID when
break_round+1<current_round. Protection blocks attacks/capture into the protected
relationship and finishing movement on the partner's city.

Establishing an embassy costs `4 + 2*number_of_existing_outgoing_embassies`, meets
the builder at the host, and reveals radius1 around the host's capital. Income is2
per applicable share, doubled to4 at peace, and suppressed while the relation is
at war. The host share is capital work; the builder share depends on the host's
capital not being occupied by another player's unit. Attacking a player destroys
both bilateral embassies and stamps war until the attacker's next turn; it does
not wipe unrelated pending requests. Capturing a capital also invalidates relevant
embassies. Accepted peace/broken peace updates routes and city connections.

[Wiki home](../README.md)
