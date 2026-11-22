# Complete observation field dictionary

Schema v4; per-seat int32 vector: 224 header entries followed by 200×34 entity entries. Offsets below are zero-based. Repeated-field index lists are complete; unused reserved positions remain zero/initial template values. A row field at offset `j` for entity row `r` is stored at `224+r*34+j`. [Visibility and scoring](../rules/observation-and-scoring.md).

## Header offsets

| Field | Offsets |
|---|---|
| `tick` | 0 |
| `result` | 1 |
| `elixir` | 2 |
| `elixir_reserved` | 3 |
| `crowns` | 4 |
| `crowns_enemy` | 5 |
| `hand_refill_ms` | 6 |
| `hand_card` | 7, 8, 9, 10 |
| `hand_cost` | 11, 12, 13, 14 |
| `hand_pending` | 15, 16, 17, 18 |
| `next_card` | 19 |
| `deck` | 20, 21, 22, 23, 24, 25, 26, 27 |
| `enemy_seen` | 28, 29, 30, 31, 32, 33, 34, 35 |
| `own_king_hp` | 36 |
| `own_king_max_hp` | 37 |
| `own_left_hp` | 38 |
| `own_left_max_hp` | 39 |
| `own_right_hp` | 40 |
| `own_right_max_hp` | 41 |
| `enemy_king_hp` | 42 |
| `enemy_king_max_hp` | 43 |
| `enemy_left_hp` | 44 |
| `enemy_left_max_hp` | 45 |
| `enemy_right_hp` | 46 |
| `enemy_right_max_hp` | 47 |
| `entity_count` | 48 |
| `entity_overflow` | 49 |
| `decision_ticks` | 50 |
| `play_delay_ticks` | 51 |
| `seat` | 52 |
| `pending_count` | 53 |
| `episode_step` | 54 |
| `time_ms` | 55 |
| `section` | 56 |
| `section_type` | 57 |
| `section_left_ms` | 58 |
| `full_bar_ms` | 59 |
| `overtime` | 60 |
| `play_open_in_ticks` | 61 |
| `last_refusal` | 62 |
| `last_run_refusal` | 63 |
| `illegal` | 64 |
| `pending_card` | 65, 69, 73, 77 |
| `pending_x` | 66, 70, 74, 78 |
| `pending_y` | 67, 71, 75, 79 |
| `pending_ticks` | 68, 72, 76, 80 |
| `deck_slot` | 81, 82, 83, 84, 85, 86, 87, 88 |
| `deck_evolution` | 89, 90, 91, 92, 93, 94, 95, 96 |
| `hand_form` | 97, 98, 99, 100 |
| `ability_champion` | 101, 107 |
| `ability_unit` | 102, 108 |
| `ability_state` | 103, 109 |
| `ability_cooldown_ms` | 104, 110 |
| `ability_charges` | 105, 111 |
| `ability_command_ticks` | 106, 112 |
| `last_ability_run_refusal` | 113 |
| `enemy_play_count` | 114 |
| `enemy_elixir` | 115 |
| `enemy_play_card` | 116, 125, 134, 143, 152, 161, 170, 179 |
| `enemy_play_cast` | 117, 126, 135, 144, 153, 162, 171, 180 |
| `enemy_play_form` | 118, 127, 136, 145, 154, 163, 172, 181 |
| `enemy_play_ticks` | 119, 128, 137, 146, 155, 164, 173, 182 |
| `enemy_play_x` | 120, 129, 138, 147, 156, 165, 174, 183 |
| `enemy_play_y` | 121, 130, 139, 148, 157, 166, 175, 184 |
| `enemy_play_kind` | 122, 131, 140, 149, 158, 167, 176, 185 |
| `enemy_play_flags` | 123, 132, 141, 150, 159, 168, 177, 186 |
| `enemy_play_point_ticks` | 124, 133, 142, 151, 160, 169, 178, 187 |
| `queue` | 188, 189, 190, 191 |
| `deck_level` | 192, 193, 194, 195, 196, 197, 198, 199 |
| `enemy_seen_evolution` | 200, 201, 202, 203, 204, 205, 206, 207 |
| `enemy_seen_evolution_need` | 208, 209, 210, 211, 212, 213, 214, 215 |

## Entity row offsets

| Field | Offset |
|---|---|
| `id` | 0 |
| `kind` | 1 |
| `side` | 2 |
| `row` | 3 |
| `x` | 4 |
| `y` | 5 |
| `hp` | 6 |
| `max_hp` | 7 |
| `shield` | 8 |
| `state` | 9 |
| `flags` | 10 |
| `level` | 11 |
| `dx` | 12 |
| `dy` | 13 |
| `target` | 14 |
| `target_x` | 15 |
| `target_y` | 16 |
| `deploy_ms` | 17 |
| `attack_ms` | 18 |
| `load_ms` | 19 |
| `speed_percent` | 20 |
| `hit_speed_percent` | 21 |
| `buff_count` | 22 |
| `buff_0` | 23 |
| `buff_ms_0` | 24 |
| `buff_1` | 25 |
| `buff_ms_1` | 26 |
| `buff_2` | 27 |
| `buff_ms_2` | 28 |
| `aim_x` | 29 |
| `aim_y` | 30 |
| `time_left_ms` | 31 |
| `radius` | 32 |
| `ability_ticks` | 33 |

## Flags values

| Name | Value |
|---|---|
| `alive` | 1 |
| `hidden` | 2 |
| `invisible` | 4 |
| `underground` | 8 |
| `air` | 16 |
| `building` | 32 |
| `king` | 64 |
| `clone` | 128 |
| `awake` | 256 |
| `no_move` | 512 |
| `no_attack` | 1024 |
| `charged` | 2048 |

## Kinds values

| Name | Value |
|---|---|
| `character` | 1 |
| `tower` | 2 |
| `projectile` | 3 |
| `area` | 4 |

## Info values

| Name | Value |
|---|---|
| `tick` | 0 |
| `episode_step` | 1 |
| `winner` | 2 |
| `crowns_0` | 3 |
| `crowns_1` | 4 |
| `fault` | 5 |
| `reset` | 6 |
| `illegal_0` | 7 |
| `illegal_1` | 8 |
| `truncated` | 9 |
| `first_side` | 10 |
| `not_modelled` | 11 |

## Relevant vocabulary IDs

| Name | ID |
|---|---|
| Archer | 15 |
| ArcherArrow | 16 |
| Arrows | 30 |
| ArrowsSpell | 31 |
| Fireball | 452 |
| FireballSpell | 453 |
| Giant | 527 |
| KingProjectile | 796 |
| KingTower | 797 |
| Knight | 810 |
| MiniPekka | 901 |
| Musketeer | 959 |
| MusketeerProjectile | 961 |
| PrincessTower | 1036 |
| TowerPrincess | 1330 |
| TowerPrincessProjectile | 1331 |
| Zap | 1415 |

## Relevant buff ID

`ZapFreeze` has vocabulary index 244. Other global buff names are outside this mode.

Coordinates use 1,000 world units/tile. HP/damage are integers. Elixir is stored at 10,000 units/point. Timer fields ending `_ms` are milliseconds; tick fields are 50 ms ticks. IDs absent/unrevealed are commonly −1. `result` is 0 ongoing,1 own win,2 own loss,3 draw; the separate `info.winner` uses physical seat 0/1,2 tie,−1 ongoing. Do not confuse the two. The info vector is host outcome metadata rather than an extra private critic input.

[Wiki home](../README.md)
