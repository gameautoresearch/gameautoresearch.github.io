# Terrain and resources

| Terrain ID | Meaning | Relevant behavior |
|---:|---|---|
|0|None/unknown placeholder|Not an ordinary generated traversable tile|
|1|Shallow water|Fishing access; ports, bridges, fish and naval play|
|2|Ocean|Sailing access; naval movement, starfish and water temples|
|3|Field|Ordinary land, fruit/crop and most economic buildings|
|4|Mountain|Climbing access; ends movement, sight2, defense bonus, metal|
|5|Forest|Normally stops movement without road/Creep; animal/LumberHut/temple|
|6|Ice|Shared off-mode enum; not a normal Bardur mix terrain|
|7|Wetland|Shared off-mode enum; not a normal Bardur mix terrain|

| Resource ID | Name | Mode handling |
|---:|---|---|
|0|none|No resource action|
|1|animal|Hunting harvest,2 stars→1 population|
|2|crop|Farming Farm,5 stars→2 population; hidden until relevant tech|
|3|fish|Fishing harvest,2 stars→1 population|
|4|legacy whale|No ordinary native Bardur action in this task slice|
|5|metal|Mining Mine,5 stars→2 population; hidden until relevant tech|
|6|fruit|Organization harvest,2 stars→1 population|
|7|spores|Off-mode poison-death/generic table row, Fungi native action unsupported|
|8|starfish|Visible with Sailing/Navigation; Navigation idle capture→8 stars|
|9|aquacrop|No ordinary native Bardur action in this task slice|

A resource visible in raw table data is not a guarantee of a relevant player action.
Map objects Village1, Ruin2 and Lighthouse47 are structure IDs, not resource IDs.
Tile owner, ruling city and city-center owner are separate fields. A road is a flag,
not merely structure3 replacing the rest of that tile. Improvements can cover an
underlying resource; the mask determines whether harvest/build is available.

[Wiki home](../README.md)
