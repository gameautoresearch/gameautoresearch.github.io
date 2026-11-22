# Composite-action conventions and exact pinned quirks

[Wiki home](../README.md) · [Combination catalog](../combinations/README.md)

This benchmark preserves DouZero's actual generator, detector and selector behavior.
The supplied legal set is safer than reconstructing rules from a familiar online
variant. Examples below are rank multisets in ascending order; repeated digits
mean separate copies, and J/Q/K/A/2 map to 11/12/13/14/17.

## Ordinary wing rules

- A triple attachment must have a different rank from its triple. The fourth copy
  of the triple's rank is not a legal single attachment to that same triple.
- For airplane+single wings, remove **every card of every chosen backbone rank**
  from wing candidates, then choose as many remaining physical cards as backbone
  ranks. Wings need not have distinct ranks: a pair can supply two single wings,
  and a triple can supply three. Both jokers can be single wings together.
- Airplane+pair wings choose distinct outside ranks with at least two copies;
  each supplies exactly two cards. A quad can supply a pair, leaving two behind,
  but it cannot supply two pair-wing ranks by itself.
- Four+two singles chooses two physical cards outside the quad's rank. They can
  be two distinct ranks, an ordinary pair, or the two different jokers. Playing
  a quad with the rocket as its two wings is a six-card composite, not a rocket
  or four-card bomb for bomb-count scoring.
- Four+two pairs chooses two **distinct** outside ranks, each with at least two
  copies; a rank with four copies can supply one pair, not both required pairs.
- The generator may produce duplicate wing selections from physically identical
  copies; the adapter removes duplicates so there is only one candidate per multiset.

## Examples and counterexamples

| Multiset | Canonical free-lead behavior |
|---|---|
| 33344455 | Legal airplane+single wings: pair 55 supplies two single wings |
| 333344 | Legal four+two singles: pair 44 is allowed |
| 3333 X D | Legal four+two singles: both jokers are attachments |
| 3333 as triple+single | Not emitted by that family; it is the four-card bomb |
| 33334455 | Legal four+two pairs: two distinct pair ranks |
| 33334444 | Detector recognizes type14, but not emitted by normal family14 generation |
| 10 J Q K A 2 | Not a straight: 2 cannot extend A |
| 333444555666 | Bare four-rank triple straight is detected before wing interpretations |

## Detector accepts a two-quad shape the generator does not emit

`33334444` contains two quads. The detector's eight-card branch classifies it as
four+two pairs (`type14`, rank 4). But the type14 generator selects a four-card
backbone and **two distinct outside pair ranks**, so one other quad is insufficient.
No normal free-lead generator family emits this two-quad eight-card action. A wiki
that lists every detector-recognized shape as playable would therefore be wrong.

## Single-wing generator can emit a detector-wrong multiset

Take `3334445556667777` (16 cards). The generator can choose backbone triples
333/444/555/666 and all four copies of 7 as four single wings. It emits that
multiset as a legal free-lead action. The detector sees a rank with four copies
when examining airplane structure and returns `TYPE_15_WRONG`.

The response code has no same-family branch for type15. It nevertheless adds held
four-card bombs and rocket because type15 is not pass/bomb/rocket, and adds pass
because the rival is nonempty. Against that live multiset, a hand with 8888, X, D,
9 has precisely pass, bomb 8888 and rocket as canonical legal responses. If the
16-card action emptied its player's hand, the game would already have ended.
The native CPU port mirrors these branches. This is an exact pinned convention,
not a suggestion to invent a new family or tamper with the simulator.

## Triple wings and comparison rank ambiguity

`333444555777` can be generated from three consecutive triples 333/444/555 and
three single wings all of rank 7. The detector recognizes an airplane+single
shape with length three using its four-triples special case. However the type11
selector compares the **maximum rank whose multiplicity is three**, so its rank
comparison sees 7, including the wing triple.

A candidate `333444555666` can be generated in the response family using backbone
444/555/666 and three 3 wings, but it fails that comparison (maximum triple 6 is
not greater than 7). The ordinary slogan “higher backbone always wins” does not
capture this special case. Four fully consecutive triple ranks are also detected
as a bare triple straight before the special airplane branches are considered.

## What the documentation validation covers

moves, these composite edge cases, free-lead/pass reset, bomb/rocket overrides,
strict single/bomb comparison, sequence length and lexicographic canonicalization.
It imports only the pinned pure local rule modules. It does not sample private
deals, run a game campaign, compile a simulator or assert all possible hands were
exhaustively enumerated.
