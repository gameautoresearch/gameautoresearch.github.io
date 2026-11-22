# Ports, transports, naval upgrades and disembarkation

Ordinary land units entering a usable port become Raft43 and carry their original
kind in the passenger field. A Giant becomes Juggernaut45, Cloak becomes Dinghy39,
and Dagger becomes Pirate40; these special forms preserve their land identity
implicitly, not through an ordinary passenger field. Any embark/disembark operation
exhausts movement and attack; Dash does not override this transformation rule.

Raft upgrades are represented as Summon-family choices on an own Raft in friendly
(own/peace) territory: Scoutship42 requires Sailing/cost5, Rammership44 requires
Ramming/cost5, Bomber41 requires Navigation/cost15. The city recruitment capacity
check does not turn these into city recruits. Upgrading preserves damage taken,
with new health `max(new_maxHP-(old_maxHP-oldHP),1)`, pays the full upgrade-row cost,
updates unit score by the cost difference and reveals the new form's sight. A
passenger's HP/veterancy determines ordinary boat max HP; do not set every boat
max HP to10 because the generic boat row says10.

A Carry unit landing on land becomes its passenger; special forms map to Giant,
Cloak or Dagger. On a Bridge, Recover-family target1 is Disembark-in-place. It
requires an own Carry unit with a passenger/special land form, but has no own
moved/attacked prerequisite. It keeps the boat's current health and facing, clears
passenger, exhausts the unit, removes the boat part of score, and triggers neither
arrival exploration nor Stomp/hiding. This is different from moving onto shore.

Juggernaut's Stomp applies half-combat damage to adjacent hostile units on eligible
arrival and when a Giant first embarks into a new Stomp form. A forced push uses
MoveReason2 and suppresses arrival abilities. Bomber Splash applies around its
attack target, not around the boat. Boats and Static units gain no kill XP for
promotion; the passenger retains XP earned before embarking. A non-Static ordinary
passenger with3 kills may still be promoted while at sea.

Ports/Bridges also participate in economic route connections, which are separate
from unit path reachability. A sea path legal for a ship is not automatically an
income connection, and every water layout in the map mix is not claimed to reproduce
a commercial water-map generator.

[Wiki home](../README.md)
