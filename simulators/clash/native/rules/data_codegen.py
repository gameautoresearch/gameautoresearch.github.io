"""Generate independent fixed8 C++ rules constants from source-bound exported data.

Only table/schema constants are accepted. No transitions, sampled trajectories,
or Java calls enter the generated runtime. The owning C++ file includes rules.hpp
before inserting emit_rules(data)'s namespace block.

Source field mapping: crforge core/src/main/java/org/crforge/core/battle/match/
BattleTimeline.java, MatchCard.java, and battle/deploy/DeployCard.java and unit/
UnitData.java @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

PIN='f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de'
DECK=['Knight','Archer','Musketeer','Giant','MiniPekka','Fireball','Arrows','Zap']

def _i(value):
    if isinstance(value,bool):return 'true' if value else 'false'
    if not isinstance(value,int) or not -2147483648<=value<=2147483647:raise ValueError(f'not int32: {value!r}')
    return str(value)

def rule_values(data):
    """Return exact RulesConfig and CardRule field dictionaries for validation."""
    if data['provenance']['upstream_commit']!=PIN:raise ValueError('unqualified upstream pin')
    if data['rules']['deck']!=DECK or data['rules']['level']!=11 or data['rules']['tower_level']!=11:raise ValueError('not frozen level11 fixed8')
    if data['rules']['decision_ticks']!=5 or data['rules']['mask_version']!=2 or any(data['rules']['slot_flags']):raise ValueError('unsupported cadence/mask/slot flags')
    row=data['timeline'];start=data['start_cooldown_ms']
    if start%50:raise ValueError('start cooldown is not a whole tick')
    config=dict(startingElixir=row['startingElixir'],maxMana=data['max_mana'],startCooldownTicks=start//50,endDelayMs=data['end_delay_ms'],sectionCount=len(row['sectionLengths']),rateCount=len(row['rateLengths']),cooldownCount=len(row['cooldownLengths']),eventCount=len(row['eventTimes']))
    for dest,source,scale,capacity in [('sectionLengths','sectionLengths',1000,8),('sectionTypes','sectionTypes',1,8),('rateLengths','rateLengths',1000,8),('fullBarMs','fullBarMs',1,8),('cooldownLengths','cooldownLengths',1000,8),('cooldownMs','cooldownMs',1,8),('eventTimes','eventTimes',1000,16)]:
        values=[int(v)*scale for v in row[source]]
        if dest=='eventTimes':values.sort()
        if len(values)>capacity:raise ValueError(f'{dest} exceeds qualified capacity')
        config[dest]=values
    ids=data['unit_ids'];cards=[]
    for name in DECK:
        c=data['cards'][name];u=c['unit'];m=data['match_cards'][name]
        if c['secondary'] is not None or c['listed'] or c['searchUnit'] is not None or c['summonWidth']!=0 or c['summonDeployDelaySecondMs']!=0:raise ValueError(f'unsupported fixed8 placement row {name}')
        if u is not None and (c['count'] not in (1,2) or u['spawnAngleShift']!=0 or u['spawnPathfindSpeed']!=0 or u['flyingHeight']!=0 or u['building']):raise ValueError(f'unsupported fixed8 unit {name}')
        result=dict(cost=m['cost'],productionStopMs=m['elixirProductionStopTimeMs'],unitType=-1 if u is None else ids[u['name']],count=c['count'],summonRadius=c['summonRadius'],summonWidth=c['summonWidth'],summonDelayMs=c['summonDeployDelayMs'])
        for dest,source in [('deployTimeMs','deployTimeMs'),('collisionRadius','collisionRadius'),('spawnRadius','spawnRadius'),('spawnAngleShift','spawnAngleShift'),('flyingHeight','flyingHeight'),('spawnPathfindSpeed','spawnPathfindSpeed'),('tileSizeOverride','tileSizeOverride')]:result[dest]=0 if u is None else u[source]
        result.update(margin=c['deployWTileMargin'],startY=c['deployStartY'],endY=c['deployEndY'],enemyOk=c['canDeployOnEnemySide'],onBuildings=c['canPlaceOnBuildings'],onWater=c['canPlaceOnWater'],fullLane=c['fullLaneDeploy'],building=False if u is None else u['building'],spellAsDeploy=c['spellAsDeploy'])
        cards.append(result)
    a=data['arena']
    if a['width']!=36 or a['height']!=64 or len(a['cells'])!=2304:raise ValueError('unqualified arena dimensions')
    return config,cards

def emit_rules(data)->str:
    config,cards=rule_values(data)
    lines=['// SPDX-License-Identifier: Apache-2.0',f'// Generated fixed8 constants from crforge @ {PIN}.', '// Full source paths and mapping: native/rules/data_codegen.py. No episode transitions.', 'namespace clash_native {','CN_HD inline RulesConfig fixed8_rules_config() {','  RulesConfig c{};']
    for key,value in config.items():
        if isinstance(value,list):
            for i,v in enumerate(value):lines.append(f'  c.{key}[{i}]={_i(v)};')
        else:lines.append(f'  c.{key}={_i(value)};')
    lines+=['  c.symmetricalSnap=true;c.laneSequence=true;','  return c;','}','CN_HD inline CardRule fixed8_card_rule(int index) {','  CardRule c{};','  switch(index) {']
    for index,card in enumerate(cards):
        lines.append(f'  case {index}: // {DECK[index]}')
        for key,value in card.items():lines.append(f'    c.{key}={_i(value)};')
        lines.append('    break;')
    lines+=['  default: break;','  }','  return c;','}','inline constexpr int32_t fixed8_arena_cells[2304]={']
    cells=data['arena']['cells']
    for i in range(0,len(cells),36):lines.append('  '+','.join(_i(v&127) for v in cells[i:i+36])+',')
    lines+=['};','inline ArenaView fixed8_arena_view() {return ArenaView{36,64,fixed8_arena_cells};}','} // namespace clash_native','']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'refusing overwrite: {args.output}')
    data=json.loads(args.input.read_text());args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(emit_rules(data))
if __name__=='__main__':main()
