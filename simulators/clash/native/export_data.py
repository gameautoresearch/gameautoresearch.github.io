"""Compile frozen task data/schema once from the pinned oracle (no transition tapes).

Outputs stay in ignored generated/. The native runtime consumes constants, and
never loads Java or queries oracle transitions. This compiler is separately timed
and receipted during qualification/setup.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jpype
from game import HERE,make_vector,resolve,canonical,sha
from runner import setup
NATIVE=Path(__file__).resolve().parent

def convert(value,depth=0):
    if value is None:return None
    if depth>24:raise ValueError('cyclic/unexpected task data record')
    if isinstance(value,(str,bool,int,float)):return value
    cls=value.getClass() if hasattr(value,'getClass') else None
    if cls is None:raise TypeError(type(value))
    name=str(cls.getName())
    if name=='java.lang.String':return str(value)
    if name=='java.lang.Boolean':return bool(value.booleanValue())
    if isinstance(value,jpype.JClass('java.lang.Number')):
        return float(value.doubleValue()) if name in ('java.lang.Double','java.lang.Float') else int(value.longValue())
    if cls.isEnum():return str(value.name())
    if cls.isArray() or isinstance(value,jpype.JClass('java.util.List')) or isinstance(value,jpype.JClass('java.util.Set')):return [convert(x,depth+1) for x in value]
    if isinstance(value,jpype.JClass('java.util.Map')):return {str(x.getKey()):convert(x.getValue(),depth+1) for x in value.entrySet()}
    if cls.isRecord():return {str(component.getName()):convert(getattr(value,str(component.getName()))(),depth+1) for component in cls.getRecordComponents()}
    raise TypeError('uncompiled task data object '+name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=NATIVE/'generated-v2');args=parser.parse_args()
    started=time.perf_counter();out=args.out;out.mkdir(exist_ok=False)
    c=json.loads((HERE/'configs/profile-trimmed.json').read_text());c.update(envs=1,threads=1);setup(c)
    with make_vector(c) as v:
        v.reset([0],c['rules']['deck'],c['rules']['deck'])
        Vector=jpype.JClass('org.crforge.env.VectorEnv');PathJ=jpype.JClass('java.nio.file.Path')
        tables=Vector.tables(PathJ.of(str(resolve(c['tables']))));records=jpype.JClass('org.crforge.core.battle.data.BattleRecords')(tables)
        units={name:convert(records.unit(name)) for name in ('Knight','Archer','Musketeer','Giant','MiniPekka','KingTower','PrincessTower','TowerPrincess')}
        cards={name:convert(records.card(name)) for name in c['rules']['deck']}
        match_cards={name:convert(records.matchCard(name)) for name in c['rules']['deck']}
        projectile_names={str(value['projectile']['name']) for value in units.values() if value.get('projectile')}
        for value in cards.values():
            projectile=value.get('projectile')
            if projectile:projectile_names.add(projectile['name'] if isinstance(projectile,dict) else projectile)
        projectiles={name:convert(records.projectile(name)) for name in sorted(projectile_names)}
        areas={}
        for value in cards.values():
            area=value.get('areaEffect')
            if area:areas[area]=convert(records.areaEffect(area))
        raw_tables={name:json.loads((resolve(c['tables'])/(name+'.json')).read_text()) for name in ('actions','character_buffs','globals','spawn_groups','tilemaps','game_tags','game_object_filters','damage_types') if (resolve(c['tables'])/(name+'.json')).exists()}
        map_object=jpype.JClass('org.crforge.core.pathfinding.grid.TileMap').standard1v1()
        arena={'width':int(map_object.width()),'height':int(map_object.height()),'cells':[int(map_object.bits(x,y)) for y in range(map_object.height()) for x in range(map_object.width())]}
        schema={'header':v.schema.header,'entity':v.schema.entity,'info':v.schema.info,'flags':v.schema.flags,'kinds':v.schema.kinds,'header_ints':v.schema.header_ints,'entity_ints':v.schema.entity_ints,'max_entities':v.schema.max_entities,'seat_ints':v.schema.seat_ints,'actions':v.schema.actions}
        data={'format':1,'task':c['task'],'rules':c['rules'],'metadata':v.metadata,'schema':schema,'vocabulary':v.vocabulary,'buff_names':v.buff_names,'units':units,'cards':cards,'projectiles':projectiles,'areas':areas,'timeline':convert(records.gameModeTimeline('Ladder')),'start_cooldown_ms':int(records.battleStartCooldownMs('Ladder')),'end_delay_ms':int(records.endScreenDelayMs()),'max_mana':int(records.globalNumber('MAX_MANA')),'arena':arena,'initial_observation_template':v.observations[0].tolist(),'initial_info_template':v.infos[0].tolist(),'raw_tables':raw_tables,'provenance':{'upstream_commit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','rules_lock_sha256':sha(HERE/'rules.lock.json'),'compiler_sha256':sha(__file__),'purpose':'frozen schema/table constants, no episode-transition lookup'}}
        data['match_cards']=match_cards;data['unit_ids']={name:i for i,name in enumerate(units)};data['format']=2
        (out/'task-data.json').write_text(json.dumps(data,indent=2)+'\n')
        (out/'manifest.json').write_text(json.dumps({'task_data_sha256':sha(out/'task-data.json'),'elapsed_compiler_seconds':time.perf_counter()-started,'provenance':data['provenance'],'full_native_task_qualified':False},indent=2)+'\n')
        print(json.dumps({'out':str(out),'units':list(units),'projectiles':list(projectiles),'areas':list(areas),'schema':{k:schema[k] for k in ('seat_ints','header_ints','entity_ints')},'seconds':time.perf_counter()-started}))

if __name__=='__main__':main()
