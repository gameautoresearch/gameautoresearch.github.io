# Canonical legal indices and policy I/O

[Wiki home](../README.md) · [Observations](observations.md)

## Actions are local rank multisets

The frozen rules generate every legal card multiset. The integration sorts cards
inside each action, removes duplicate multisets, then lexicographically sorts the
result. An action is the **index into that current list**, not a permanent global
class like “play pair of 7.” If pass is legal, its empty tuple sorts first and is
index zero. On a free lead pass is absent, and index zero is a nonempty move.
A former index can denote a different move in the next observation.

No fixed vocabulary, global 10k-action table, candidate pruning or legal-action
truncation is used. `max_candidates` and chunk sizes bound a transfer/head batch,
not the number of legal choices. Candidate selection must consider the complete
canonical list. Physical suits/copy identities do not create duplicate action rows.

## Public neural interface

Each decision receives the role, static features, public history, legal action rows and the number of candidates for each observation. It must select one local action index per observation. Role is a string (`landlord`, `landlord_down`, `landlord_up`).
Static inputs are int8 `[batch,319]` or `[batch,430]`; history `[batch,5,162]`.
All candidates concatenate to int8 `[sum(counts),54]`, with int32 `counts` giving
each observation's segment length. If counts=[2,3], the first observation owns
candidate rows0..1 and the second rows2..4. Return local indices in ranges[0,2)
and[0,3), not concatenated absolute row numbers.

Three role networks/heads can share a chosen architecture, but the landlord and
peasant static widths differ. The starter uses terminal-return regression and
factors history features once, then scores every legal candidate. A learner can
change its algorithm within the public information contract; the fixed rules,
features and inference restrictions remain host-owned.

## Submission/evaluation boundary

The declared entrant contract caps complete tensor state at 500,000 elements;
weights are tensor-only state_dict up to128MiB. Standardized evaluation runs CPU,
one Torch thread, batches up to32, with at most10seconds per batch including
process initialization. Submitted source/config/weights are bound to immutable
registered artifacts. Exact policy and architecture configuration travel together.
Only public tensors enter the isolated inference process; simulator state and
other policies are unavailable. The actual session/release contract supersedes
nominal metadata; this wiki is not authorization for new compute or continuation.

Illegal index/type, malformed output, timeout and simulator fault remain failures.
They must not be replaced by a convenient fallback heuristic or counted as a
natural loss. CPU-native and CUDA-resident execution preserve the same public
canonical actions; device arithmetic/throughput does not redefine the game.
