# Mixed eight-player Tavern benchmark (mixed-v2)

`benchmark.json` and [CONTRACT.md](CONTRACT.md) are the current contract. The artifact task binding
remains `tavern-eight-implemented-v1`; the evaluation identity is `tavern-eight-mixed-v2`.

Three frozen profiles each supply seven explicit opponents. A policy plays every physical seat and
each lobby must finish naturally for all eight players with placements 1-8. Points are
`100*(8-placement)/7`; any required fault or unfinished lobby makes the primary aggregate null.

| File | Purpose |
|---|---|
| `release.py` | Release paths, hashing and fail-closed `verify` (run before every evaluation) |
| `selected_registry.json` | Receipt naming the exact private reference pool a new release freezes |
| `registry.py` | Adds an entrant bundle to a copy of the frozen pool without importing its code |
| `seed_keys.py`, `seed_exclusions.py` | Native seed arithmetic and the public/training streams a private bank must avoid |
| `freeze_pool.py` | Creates a new immutable reference-pool version from public selection evidence |

Sessions, practice and grading are driven by [harness/games/tavern/](../../../../harness/games/tavern/);
the release constructor is [`../rules_enabled/release.py`](../rules_enabled/release.py).
