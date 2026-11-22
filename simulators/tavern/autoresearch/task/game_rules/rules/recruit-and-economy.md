# Recruit economy, card play and triples

[Wiki home](../README.md)

## Gold and tavern tiers

At turn `t >= 1`, natural gold is `min(3 + (t-1), 10)`. It refills rather than
carrying the previous unused balance. Source start-turn logic adds maximum-gold
bonuses (natural amount plus bonus capped at 99), then adds banked next-turn gold;
gold gained by later start-turn effects has its own semantics. Card effects may
change prices, spend health or provide free operations, so current legality and
public quoted prices take precedence over a baseline number.

| Operation | Baseline |
| --- | ---: |
| Buy a shop minion | 3 Gold |
| Sell a minion | 1 Gold |
| Refresh | 1 Gold |
| Upgrade to tiers 2 / 3 / 4 | 5 / 7 / 8 Gold base (tier 4 is the regular cap in this form) |
| Future tier-7 upgrade base for special systems | 10 Gold |

Upgrade cost reduces by one at each turn start, floor zero. After an upgrade,
the next tier's base cost becomes the applicable cost. Costs are not the sum
of all previous upgrades. Millhouse and other scripted price overrides are real
exceptions; individual power text and the visible quoted price describe them.
A standard refresh offers 3/4/4/5 minion slots at tiers 1–4 plus one spell slot,
with a maximum of seven total tavern cards. It never overfills the fixed action
space. Tier, tribe, pool depletion and active effects determine which offers appear.

## Shared pool and generated copies

Normal minion copies per tier 1–7 are 15/15/13/11/9/7/5. Ordinary drawing is weighted
by copies left, not uniform over distinct card names. Buying holds the copy; selling,
refreshing unwanted offers and elimination return actual pool parts according to
the engine. Golden/merged/magnetic entities retain the accounting of their parts.
A generated pool minion consumes a copy if one remains; if none remains it is an
extra copy, and selling that extra does not fabricate a pool copy. A generated
golden entity can hold one pool copy rather than automatically three.

Tavern spell copies per tier 1–6 are 5/7/9/11/9/7, counted against copies currently
in taverns and hands. Spell eligibility also uses lobby subset requirements. A
shop spell's cost comes from its own card/effects, not the three-Gold minion price.

## Buying, selling, playing and board order

Buying needs a legal available slot, the current price/payment resource and room
in hand. A buy is different from playing a card. Playing a minion chooses an
insertion position, obeys the board limit, then executes its Battlecry/other play
sequence. Selling removes its board instance, updates pool accounting, grants the
applicable sale value and runs sale/listener effects. Moving a minion changes board
order without selling or replaying its Battlecry. This order affects next combat.
A full hand blocks normal buys; generated cards and queued hand additions follow
the specific overflow/queue effect rather than silently granting extra ordinary
slots. Do not assume a returned entity is a normal collectible purchase.

A spell played from hand resolves targeting and its spell effect. By default,
spells cast only by effects do not count as “you cast” events (`effect_casts_count=false`);
hand casts do. Spellcraft creates temporary special spells, with each card's
restrictions and permanence exceptions in its text. Blood Gems have base +1/+1,
then the player's Gem bonuses and relevant spell/effect bonuses. “Set stats” and
“gain stats” effects are distinct, and temporary recruit buffs expire at their
specified boundary.

## Pending menus and committed payment

Choose One, Discover, spell targets, Battlecry targets, hero-power targets and
Activate targets may open a pending menu. While one is pending, only legal
Choose actions and possibly Cancel are exposed. Choice indices refer to the
current menu only. Some operations commit payment when initiated (for example
the Dark Gift button); cancellable target entry defers resource/effect commitment
according to the operation. Cancel is not a universal refund action. Resolving one
menu may open another; do not presume a single button press completes an effect.

## Magnetic

A hand Magnetic minion can attach only to a legal host rather than taking a new
board slot. Its stats, keywords and attached effects merge as the pinned engine
specifies. Ordinary `Play` and `Magnetize` are separate actions. Host/attachment
normal/golden identities affect trigger numbers. In the default Volumizer rule,
attached Volumizer stats are a fixed enchantment at attachment time; later
Volumizer-specific bonuses do not automatically keep growing that attachment.

## Triples and golden forms

Three eligible normal copies on board and in unlocked hand combine into a golden
minion. Locked hand cards do not count. Cards owned longest merge first, using
entity creation order, so a newly generated extra copy can remain behind. The
result collects the source's stats/effect data; do not reconstruct it as “double
current HP” or infer all golden effects by simply multiplying text by two. Exact
golden definitions are linked in the catalogs. Playing a triple-reward golden
minion awards the appropriate Discover reward at one tier above the applicable
current tavern tier, bounded by the source pool/reward rules.

Triples are held while Discover/target/Choose One menus need their source or target
to remain valid; they are checked again after resolution/cancellation. They do not
form mid-combat or after a recruit phase has already ended. Elemental of Surprise
is a wildcard for eligible Elemental merges, including the source's golden-wildcard
case; it is not a wildcard for every tribe. Hero/card effects may reduce the required
copies or make a minion golden directly, so exact effect text remains authoritative.

## Freeze and recruit ending

Freeze toggles retention of applicable current shop offers. Refresh and unfreeze
interactions operate per offer. The default button limit is ten freeze/unfreeze
presses per turn. The benchmark's 30-action recruit cap prevents an endless policy
loop: after it is reached, ordinary recruit actions close and EndTurn remains;
pending choices can still need resolution. EndTurn runs end-turn effects, expires
appropriate temporary/hand states and snapshots the warband. Combat begins after
all living players finish, subject to immediate recruit-phase deaths.

