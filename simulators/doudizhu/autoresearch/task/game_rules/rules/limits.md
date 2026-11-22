# Guards, backend identity and excluded variants

[Wiki home](../README.md) · [Pinned quirks](composite-quirks.md)

## Natural termination versus fault

Any emptied hand is a natural terminal. Every nonpass removes at least one of the
54 total cards, and at most two passes occur between nonpasses. The adapter uses
a conservative162-decision guard. Crossing it raises a simulator fault; it is
not a draw, peasant loss, landlord loss or health/hand-size tiebreak. A step after
terminal, index outside the current legal list, invalid full-deck multiset,
negative public card accounting or illegal bottom cards is refused.

Budget-aborted collection waves, process crashes, failed device operations,
unsupported serialization/source identity and incomplete evaluation are distinct
infrastructure/contract failures. Their records and resource charges remain;
none supplies an invented terminal outcome. There is no wall-clock reaction phase
or elapsed-time-based natural win condition in this turn-based game.

## Scope and implementation identities

The Python oracle, native CPU port and resident CUDA rules preserve the frozen
fixed-role task and canonical legal candidate meanings. This wiki uses source
inspection and pure Python rule examples; it does not rerun compiled-native/CUDA
trajectory qualification. Earlier backend qualification belongs to its retained
source/binary receipts. A later implementation change does not retroactively
validate old checkpoints or permit source-bound resume with arbitrary code.

Device sampling precision, actor RNG, optimizer state, model architecture and
throughput are training implementation choices, not extra game rules. The offline
wiki does not provide a trained reference or private opponent recipe. The public
benchmark's ordinary target uses terminal WP; alternative ADP/logADP semantics
are documented because the frozen game API exposes them, not because economic
score replaces the benchmark's balanced win-probability headline.

## Explicitly excluded content

No bidding or robbing-landlord phase; no dynamic role choice; no bid multiplier;
no spring/anti-spring multiplier; no suit ordering; no wild cards; no team voice/
private-hand sharing; no alternative airplane-wing restrictions; no fixed truncated
legal-action catalog; no hidden information for critics or teachers. Another app's
four-with-two/airplane conventions are not evidence for this benchmark's rules.

## Documentation limits

All thirteen ordinary ranks, both jokers, pass and fourteen generated nonempty
families are documented, along with all encoded feature slices and key pinned
composite ambiguities. Representative examples are executable and source-bound.
This is not an exhaustive proof over every possible 54-card partition or every
response sequence, nor a claim that all commercial Dou Dizhu variants agree.
The legal candidates delivered in a real frame remain the exact authority.
