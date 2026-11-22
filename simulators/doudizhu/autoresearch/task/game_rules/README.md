# Dou Dizhu fixed-role — offline game wiki

This wiki describes **douzero-fixed-role-v1**, the three-role/no-bidding 54-card
benchmark used by doudizhu-fixed-role-research-v2. Its rules authority is vendored
DouZero commit `718a5c920bf3361e34178a38f3b80458e176b351`, including its unusual
composite-action conventions. All gameplay explanations, tables and examples are
local. There is no dependence on a website or knowledge of another Dou Dizhu variant.

## Read the game

1. [Deck, roles, deal and objective](rules/mode-and-deal.md)
2. [Turn order, leading, responding, passes and natural endings](rules/turns-and-comparison.md)
3. [Every combination family](combinations/README.md)
4. [Composite-action edge cases and pinned implementation quirks](rules/composite-quirks.md)
5. [Every visible field and encoded feature slice](rules/observations.md)
6. [Canonical legal-action indices and policy I/O](rules/action-interface.md)
7. [Win targets, bomb multipliers, economic scores and evaluation](rules/scoring-and-evaluation.md)
8. [Guard failures, backend identity and excluded variants](rules/limits.md)
9. [Terminology](glossary.md)

## Scope

See [scope and limitations](scope-and-limitations.md) for catalog counts and known limits. This folder contains the readable game reference and lookup indexes.
