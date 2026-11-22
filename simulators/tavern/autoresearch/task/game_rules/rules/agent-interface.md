# Player-visible interface and complete action encoding

[Wiki home](../README.md)

Each ready seat receives **3,227 float32 numeric features**, **124 int32 card
indices** and a **245-entry boolean legal-action mask**. Card index zero means an
empty slot; otherwise it is dense database index + 1. The catalogs give that exact
embedding index and the stable string ID. An entity ID identifies a particular
instance; it is not the card ID or its position. The same slot can hold a new
entity after a buy/sale/summon/choice.

## Flat actions

All bounds below are inclusive.

| IDs | Meaning | Formula |
| --- | --- | --- |
| 0 | EndTurn | one action |
| 1 | Refresh | one action |
| 2 | Freeze/unfreeze | one action |
| 3 | Upgrade | one action |
| 4–5 | HeroPower | slot `id-4` (up to two powers) |
| 6 | DarkGift | one action |
| 7 | Cancel | only cancellable pending menus |
| 8–15 | Buy | shop slot `id-8` |
| 16–22 | Sell | board slot `id-16` |
| 23–29 | Activate | board slot `id-23` |
| 30–45 | Choose | current option `id-30` |
| 46–125 | Play | hand `(id-46)//8`, insertion position `(id-46)%8` |
| 126–195 | Magnetize | hand `(id-126)//7`, host board `(id-126)%7` |
| 196–244 | Move | source `(id-196)//7`, destination `(id-196)%7` |

The action constants, not an older overlapping range comment in `actions.rs`, define
these IDs. Shop capacity is eight for representation, while normal tavern maximum
is seven offers. The mask closes absent slots and illegal targets. Spells normally
use the meaningful play position determined by the mask. Move source/destination
and insertion positions refer to the current board, not a permanent body identity.
Pending choices override ordinary economy actions; a prior choice index is invalid
once the menu changes. Only ready seats act. Nonready placeholder outputs are ignored.

## Numeric feature layout

| Zero-based offsets | Contents | Width |
| --- | --- | ---: |
| 0–52 | 42 global values + 11 lobby-type flags | 53 |
| 53–58 | Three quest present/progress pairs | 6 |
| 59–67 | Pending-kind one-hot | 9 |
| 68–284 | Seven board minions × 31 | 217 |
| 285–684 | Ten hand items × 40 | 400 |
| 685–1004 | Eight shop items × 40 | 320 |
| 1005–1644 | Sixteen pending options × 40 | 640 |
| 1645–1707 | Seven opponent public summaries × 9 (only the first three are used; the rest are zero) | 63 |
| 1708–3226 | Seven remembered opponent boards × seven minions × 31 (first three used; rest zero) | 1519 |

A minion's 31 numeric fields are: present; attack/10; HP/10; maxHP/10; tier/6;
golden; attachment count/3; shield-hit count/3; twelve keyword flags in this order: TAUNT, DIVINE_SHIELD, WINDFURY, REBORN, VENOMOUS, POISONOUS, STEALTH, MAGNETIC, MEGA_WINDFURY, CLEAVE, IMMUNE, IMMUNE_ATTACKING; eleven type flags in this order: BEAST, DEMON, DRAGON, ELEMENTAL, MECHANICAL, MURLOC, PIRATE, QUILBOAR, UNDEAD, ABERRATION, NAGA. An item's nine-field header is
present, is-minion, quoted cost/5, locked, temporary, combined Choose-One,
frozen/doomed-this-turn, kept-from-last-turn, marked/linked. Hand play costs are
encoded as zero; the shop cost is the current quoted purchase price with overrides.
Special spell bodies encode instance data/tier rather than inventing minion HP.

The 42 globals in order are: turn/20, hero-select, game-over, alive-count/8,
self-alive, health/40, armor/20, gold/10, max-gold-bonus/5, next-turn-gold/5,
tier/6, upgrade-price/10, refresh-price/2, minion-price/3, free-refreshes/3,
frozen, first-power-usable, first-power-price/5, first-power-game-uses/5,
second-power-usable, second-power-price/5, second-power-game-uses/5,
DarkGift-usable, DarkGift-uses/3, trinket-count/2, Deity-attack/10, Deity-HP/10,
won-last, lost-last, spell-attack-bonus/5, spell-HP-bonus/5, Gem-attack-bonus/5,
Gem-HP-bonus/5, board-count/7, hand-count/10, has-Deity, hand-queue-count/5,
next-is-ghost, tavern-attack-bonus/10, tavern-HP-bonus/10, effect-count/5,
current-or-saved-Chronum/4. Values are normalized, not clipped probabilities.
The scales are the eight-player encoding and are unchanged in this four-player form
(alive-count/8 is at most 0.5, tier/6 at most 4/6 without a tier unlock, opponent
placement/8 at most 0.5).

Pending-kind order is none, HeroSelect, Discover, ChooseOne, Target,
HeroPowerChoice, Trinket, Quest, Timewarp. Representing a Quest field does not
enable the globally disabled experimental mode.

## Card slot layout

| Zero-based slots | Meaning |
| --- | --- |
| 0 | Own hero |
| 1–2 | Hero powers |
| 3–4 | Deity; pending-menu source |
| 5–13 | Three quest/reward/owned-reward triples |
| 14–18 | Up to five own secrets |
| 19–25 | Board |
| 26–35 | Hand |
| 36–43 | Shop |
| 44–59 | Pending choices |
| 60–67 | Owned trinkets |
| 68–74 | Opponent heroes, clockwise relative to candidate |
| 75–123 | Opponent remembered boards |

## Visibility and known encoder limitations

The candidate sees its own board/hand/shop/options and public summaries. Opponent
hands and unseen current recruit boards are excluded. Board memory is the warband
the candidate saw in its last combat with that opponent and its age, not all
opponents' latest secret boards. During hero selection, opposing hero/private
selection details are masked. `sees_all_boards` is a diagnostic engine capability,
not a permitted benchmark replacement view.

Choose-One branches can be represented by the same source card ID without full
option text, some target distinctions are compressed, and action-use counters
are absent. The offline wiki explains the cards, but does not expand the actual
policy observation. A researcher must not read privileged state to compensate.
References receive the same admitted arrays. Submitted inference has at most
500,000 parameters, finite safe NPZ export (mixed-v2:32MB), and a 15-second complete
input/response deadline without search, external calls or LLM gameplay decisions.

## Admission-v2 simultaneous recruit handling

Several seats can be ready at once. Another seat's earlier applied action can
eliminate a player or change a later seat's pending menu before that later response
is executed. Admission-v2 validates the observed frame, then reports **applied**
versus **deferred** inputs when context changes. A deferred response is not an
illegal policy decision and not an executed learning transition. Pending-menu
identity is checked in addition to a coincidentally unchanged action mask. Only
actual applied ready-seat decisions enter learner counts/losses. Rewards while
waiting credit the prior real action according to the trainer boundary.
