# Turn order, leading, responding and comparison

[Wiki home](../README.md) · [Composite quirks](composite-quirks.md)

## One decision

The active role acts, its played card multiset is removed from its hand, public
history/played counts/last-action fields update, bomb count updates if applicable,
and an empty hand ends the game. Otherwise the active role advances cyclically:
landlord → landlord_down → landlord_up → landlord. There are no simultaneous plays,
reaction interrupts or automatic teammate decisions in the rules layer.

## Leading and responding

At the start, or after both opponents of the last nonpass player have passed, that
player has a free lead. A free leader must play a **nonempty** generated combination;
pass is absent from the legal set. When a nonempty rival move is live, a player can
beat it or pass, even when its teammate owns the live lead. Cooperation is a policy
choice; the rules do not compel a peasant to pass to a teammate.

The engine derives the current rival from the last action, or the second-last when
the last action was a pass. Therefore after two consecutive passes the derived
rival is the most recent pass (empty), and the original nonpass player gets a new
lead. `last_pid` separately retains the last **nonpassing** role through passes.
`last_move_dict` instead records each role's latest action, including empty pass.
The observation's `last_move` is the engine-derived rival, so it is empty on a new
free lead even though an older nonpass action exists in the full history.

Example: landlord plays 55; down passes; up passes; landlord can now lead a single,
straight, bomb or any other generated combination. Landlord cannot pass on this
new lead. A player beating 55 with 77 becomes the prospective next free leader.

## Ordinary comparison

A nonbomb response normally has the same family, total card count and sequence
length as the rival, and a strictly stronger comparison rank. Equality never beats.
Single/pair/triple/bomb compare the rank. Triple+single, triple+pair and four+two
compare their backbone rather than their attachment ranks. Straights/pair-straights/
triple-straights compare starting rank with the same number of ranks. Airplanes
use the highest rank with exactly three occurrences in the concrete multiset;
for ordinary unambiguous wings this is the highest backbone rank. For four+two
pairs, the filter uses the last encountered rank with count four (the higher quad
if a detector-only two-quad multiset were supplied). See the quirks chapter for
cases where the simple backbone description is insufficient.

The response generator is called for the **rival detector type and length**. It
then filters its generated candidate multisets; it does not reclassify every
candidate as a prerequisite to that filtering. This distinction matters for
ambiguous wing multisets. The exact supplied canonical legal set is authoritative.

## Bomb and rocket override

A four-of-a-kind bomb beats any nonbomb/nonrocket combination. A higher-rank bomb
beats a lower bomb. The two-joker rocket beats any ordinary move or four-card bomb.
A played rocket has no beating response; other players can only pass unless the
rocket itself already ended the game. A bomb does not become weaker because the
rival contains more cards. A pair of 2s does not beat a triple of 3s: they are
different families, while a bomb of 3s does beat that triple.

## Canonical response algorithm

1. Classify the live rival; an empty rival selects all fourteen generated families.
2. Otherwise generate the rival family at its required sequence length and filter
   strictly higher comparison ranks; rocket has no same-family successor.
3. Against any live type other than pass, bomb or rocket, add all held bombs/rocket.
4. Add pass whenever the live rival is nonempty.
5. Sort each multiset, remove duplicates, and lexicographically sort the legal set.

The detector's special `TYPE_15_WRONG` label is not a fifteenth playable family.
The permissive generator can emit an unusual wing multiset receiving that label;
its subsequent response rule still adds bombs/rocket/pass. That pinned behavior
is documented rather than repaired by this wiki.
