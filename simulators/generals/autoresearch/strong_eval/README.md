# Exact-task cross-play evaluator

A library for playing registered Generals policies against each other on the
full `generals-research-1v1-full-v2` task: fog, independent 17–23 sides, general
trade and the 2,048-tick cap. It makes no model API calls and is not part of a
research session; `python -m harness` does not use it.

| File | Purpose |
|---|---|
| `contract.py` | The two inference classes: the stateless 600k-parameter entrant and the trusted 16M-parameter AverageJoe reference with bounded history |
| `registry.py` | Immutable registry entries binding a class, source and weights; `verify_entry` fails closed on drift |
| `codec.py` | Native public observation and direction mask to each class's input, and the entrant's 4,609 action IDs |
| `client.py`, `worker.py` | One isolated inference process per player; only public frames cross the pipe |
| `arena.py` | `evaluate(candidate, panel, pool_path, map_indices, ...)`: every map, both seats, each opponent; no partial scores |
| `geometry.py` | Map-family diagnostics (walking separation, area, shape) for a map bank |

Official isolation needs Linux root (`development=False`); a development run
uses the caller's user. The AverageJoe class imports `simulators/generals/averagejoe`
and `vendor/engines/AverageJoe`.

The operator workflow that ran this evaluator for the November 9 strong-bots
campaign (release selection, launch, collection, finalization, reports) is in
git history at `ba5c0d5`. That campaign's results are in
`reports/generals/strong-bots-five-hour-2026-11-09-evaluator.md`.
