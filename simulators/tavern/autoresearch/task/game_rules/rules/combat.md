# Combat, hero damage, pairing and elimination

[Wiki home](../README.md)

## Combat setup and first attacker

Combat snapshots both recruit warbands. Ordinary damage, deaths and summons affect
these combat instances rather than deleting the next recruit board. Specific
persistent effects can carry stat/effect changes back, as their scripts say.
The first attacker is the side with more minions; if counts tie, the higher tavern
tier attacks first; if that also ties, the engine draws a random side. This pinned
rule is relevant even if other implementations treat a tie differently.

Start-of-combat effects run by group: (1) trinkets, rewards and quests; (2) hero
powers; (3) Tavern-spell effects; (4) minions in hand; (5) minions on board. Within
each group, the first attacker's side resolves first. Death resolution occurs at
the appropriate intermediate boundaries. A summon created during a stage is not
retroactively an original minion in an earlier snapshot stage. “Attack immediately”
is an attack outside ordinary rotation, not necessarily the first normal turn.

## Attack rotation and targets

Sides alternate attack turns. The attacker is the leftmost living minion with
positive attack that has not attacked in the current cycle; when all eligible
minions have attacked, its side's cycle resets from the left. Zero-attack minions
can still receive attacks but do not initiate an ordinary attack. Target selection
is random among visible legal enemy minions. If any legal Taunt exists, target
one of those; Stealth minions cannot be attacked. Other scripted targeting rules
may narrow or alter the ordinary selection.

A normal attack turn makes one attack, Windfury two, Mega-Windfury four. It picks a
new target for each attack. Windfury gained during the attack turn can increase
remaining attacks; losing it does not retract the already determined allowance.
A dead/removed/zero-attack attacker cannot complete later attacks.

## One attack's resolution

The attacker drops Stealth, then Rally and attack-event effects fire before the
exchange. If either participant no longer exists/alive, the exchange can stop.
Otherwise the engine captures the relevant attack values, deals attacker damage
to the defender (and Cleave neighbors), and defender damage to the attacker, with
damage-trigger dispatch deferred until the simultaneous exchange is complete.
Immune-while-attacking prevents the attacker's return damage. Divine Shield,
immunities and damage modifiers apply to each hit.

Cleave hits the defender's current adjacent neighbors. Cleave carries Venomous to
all targets it actually hits during that exchange and then consumes it after
positive damage, rather than stopping after the first neighbor. Wildfire Elemental's
excess-damage follow-up is separate from Cleave: normal selects one eligible
neighbor; golden can hit both. Shielded hits do not create ordinary excess damage.
After the exchange, deferred events flush, AttackDone/after-attack effects run, and
the death/effect engine processes the resulting chain. Deathrattle, Reborn,
Avenge, summons, auras and newly lethal damage can recursively cause further effects.

## Keywords and survival

See [keyword definitions](keywords-and-events.md) for Shield, Reborn, Poisonous,
Venomous, Taunt, Stealth, Magnetic and immunity. Board space limits summons.
A summon failing for no space is not automatically delayed until a later death
unless that effect explicitly queues it. The exact source chain matters for
cross-card corner cases; this wiki describes the stages and every card's own text,
not every combinatorial interaction as an independently tested theorem.

## End of a combat

If one side has survivors and the other has none, the surviving side wins. If both
are empty or both survive without a legal future attack, the engine can produce
a natural tie. In particular, a Stealth-only targeting deadlock can be a rule tie;
it must not be confused with a safety cutoff. If neither board can initiate an
attack, combat also stops according to that rule handling.

Uncapped hero damage equals winner tavern tier plus the sum of surviving minions'
tavern tiers, not attack or HP. The engine caps damage only while **more than four
players are alive** (turns 1–3: 5; turns 4–7: 10; turn 8 onward: 15). A four-player
lobby never has more than four alive, so **hero damage is uncapped from turn one**. Armor absorbs applicable damage before health; scripts can
modify/prevent hero damage. A ghost combat uses its own no-living-opponent handling
and cannot eliminate an already dead seat again.

## Pairings and information

The engine publishes next opponents during recruit. It searches randomized living
seats for pairings avoiding the last two opponents/ghosts (no rematch within three
combats); if impossible it relaxes to one previous combat, then none. With two
living players it has no avoidance window. Odd counts use a ghost. Recruit-phase
death repairs the announced pairs: unaffected pairs remain; remaining free seats
are paired or assigned a ghost. It does not blindly reshuffle every match.

A player remembers an opponent board only from a combat it personally saw. A live
opponent's board for effects is the board shown in its last finished combat, not
the current secret recruit board. See [agent interface](agent-interface.md).

## Elimination and placement

A hero at zero health during recruit is eliminated immediately. At combat-round
deaths, lower remaining health+armor ranks worse; exact ties are broken by seat
in this emulator (lower seat ranks better). This is a declared server-unknown
assumption, not a claim about Blizzard's tie rules. Dead players' actual pool parts
in board, hand, queue and shop return to the pool. Their final warband can become
a ghost. The last living seat gets first place. See the distinction between natural
placement and safety-ranked forced stops in [outcomes](outcomes-and-guards.md).

