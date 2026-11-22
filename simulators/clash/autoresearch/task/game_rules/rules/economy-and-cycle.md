# Elixir, opening hand and deck cycle

Each side starts with **6 elixir**, capped at **10**. The engine stores an elixir point as **10,000 integer units**. A full 10-point bar is produced over 28,000 / 14,000 / 9,300 ms in successive rate phases (0–40 s, 40–75 s and from 75 s in this benchmark's short match form), not by floating-point accumulation.

For one 50 ms king post-hook, the increase is `floor(10 * 500000 / fullBarMs)` internal units: **178**, **357**, **537** respectively. Integer truncation is applied on each visit. Therefore nominal per-elixir times of 2.8 / 1.4 / .93 seconds describe the timeline configuration, while exact accumulation uses the above integer steps. Wasted production above 100,000 is retained in internal diagnostics. The first 4.5 seconds block plays, not the production clock. Production is run in each king's post-hook, even while a dead king is still listed. The plain cards have no production-stop effect.

## Queue and hand

Each player independently shuffles the same eight-card deck; none of these plain cards forces or excludes itself from the opening hand. The first four shuffled entries fill hand slots 0–3, and the last four form the queue. Actual private opening order is not shared with the opponent.

When a command successfully executes, its card is removed from its current hand slot and appended to the back of the queue. The vacated hand slot becomes empty. Hand refill decrements its cooldown by 50 ms, then fills the first empty slot from the front of the queue when ready. It fills at most one slot in a visit and resets its cooldown to the current phase's 1,000 / 500 / 350 ms value. A refill can therefore be delayed while multiple cards are pending/removed; card cycling is not an instantaneous atomic replacement at command acceptance.

The own observation includes the current hand, costs, pending markers, next card and queue. An action addresses a hand slot; it does not permanently identify Knight or Fireball. Re-read current hand after each observation.

## Reservations and affordability

A successfully accepted command promises its elixir, but actual spending/cycling occurs when the command runs. The controller masks a second action that would overspend after sure pending reservations. A card already awaiting execution cannot be submitted again. Promise state can outlast an execution refusal; the exact expiry boundary is a declared unsupported case, not silently treated as extra spendable elixir. See [deployment and command timing](deployment.md).

[Wiki home](../README.md)
