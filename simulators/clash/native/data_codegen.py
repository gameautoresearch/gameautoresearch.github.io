"""Compile pinned fixed8 record constants into independent C++ configuration.

Source mapping: crforge core/src/main/java/org/crforge/core/battle/unit/
{WorldEntity,CharacterEntity,TowerEntity,Standard1v1Battle}.java,
core/src/main/java/org/crforge/core/battle/projectile/{ProjectileAmounts,ProjectileLauncher}.java,
core/src/main/java/org/crforge/core/pathfinding/{EntityFlags,combat/LevelScaling}.java
@ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de, Apache-2.0.
This compiles data, never precomputed transitions or oracle action tapes.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent

def scaling(base,row,mode):
    rarity=row['rarity'];steps=10-rarity['relativeLevel']
    if mode in (1,2):return base*rarity['powerLevelMultiplier'][steps-1]//100 if steps else base
    m=100
    for k in range(steps):m=m*(100+([0,0,0,8,7,8,8][mode] if k<8 else 10))//100
    return base*m//100

def emit(data):
    modspec=importlib.util.spec_from_file_location('clash_rules_codegen',HERE/'rules/data_codegen.py');m=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(m)
    lines=['// Generated pinned data, SPDX-License-Identifier: Apache-2.0. See native/data_codegen.py.','#pragma once','#include "../spec.hpp"',m.emit_rules(data),'namespace clash_native {','inline NativeTaskSpec make_fixed8_spec(){NativeTaskSpec s{};s.rules=fixed8_rules_config();for(int i=0;i<8;++i)s.cards[i]=fixed8_card_rule(i);for(int i=0;i<2304;++i)s.arenaCells[i]=fixed8_arena_cells[i];']
    def val(v):
        if isinstance(v,bool):return str(v).lower()
        if isinstance(v,int):return str(v) if -2147483648<=v<2147483648 else str(v & ((1<<64)-1))+'ULL'
        raise ValueError(v)
    def set_(name,v):lines.append(f' s.{name}={val(v)};')
    tags=data['raw_tables']['game_tags']['rows'];tag=lambda name:1<<tags[name]['index']
    for dest,source in [('noMove','NO_MOVE'),('noMoveAllowAttract','NO_MOVE_ALLOW_ATTRACT'),('noAttack','NO_ATTACK'),('noCheckCollisions','NO_CHECKCOLLISIONS'),('noCheckAvoidance','NO_CHECKAVOIDANCE'),('disablePhysical','DISABLE_PHYSICAL_INTERACTIONS_WITH_OBJECTS'),('avoidanceAsObstacle','AVOIDANCE_AS_OBSTACLE'),('noPushedByEnemy','NO_PUSHED_BY_ENEMY'),('noPushedByAlly','NO_PUSHED_BY_ALLY'),('noPushback','NO_PUSHBACK'),('untargetable','UNTARGETABLE'),('lockTarget','LOCK_TARGET'),('ignoreRangeExtensionToKeepTarget','IGNORE_RANGE_EXTENSION_TO_KEEP_TARGET'),('inactive','INACTIVE'),('activating','ACTIVATING'),('combatDisabled','COMBAT_DISABLED'),('unkillable','UNKILLABLE'),('hidden','HIDDEN')]:set_('tags.path.'+dest,tag(source))
    for dest,source in [('inactive','INACTIVE'),('activating','ACTIVATING'),('noDamage','NO_DAMAGE'),('hidden','HIDDEN'),('unkillable','UNKILLABLE'),('attacking','ATTACKING'),('noBuffs','NO_BUFFS')]:set_('tags.'+dest,tag(source))
    set_('tags.keepsTargetingOff',tag('INACTIVE')|tag('ACTIVATING'))
    projectiles=list(data['projectiles']);vocab=data['vocabulary']
    for i,(name,u) in enumerate(data['units'].items()):
        at=f'units[{i}]';tower=u['king'] or u['summonerTower'];p=u['projectile'];pindex=-1 if not p else projectiles.index(p['name'])
        for dest,source in [('speed','speed'),('jumpSpeed','jumpSpeed'),('spawnPathfindSpeed','spawnPathfindSpeed'),('ingamePathfindSpeed','ingamePathfindSpeed'),('chargeSpeedMultiplier','chargeSpeedMultiplier'),('stopMovementAfterMs','stopMovementAfterMs'),('waitMs','waitMs'),('flyingHeight','flyingHeight'),('hovering','hovering'),('ingamePathfindVisible','ingamePathfindVisible'),('flyDirectPaths','flyDirectPaths')]:set_(at+'.movement.'+dest,u[source])
        for dest,source in [('range','range'),('minimumRange','minimumRange'),('collisionRadius','collisionRadius'),('sightRange','sightRange'),('sightClip','sightClip'),('sightClipSide','sightClipSide'),('hitSpeed','hitSpeedMs'),('loadTime','loadTimeMs'),('attackFinishTime','attackFinishTimeMs'),('overrideAttackFinishTime','overrideAttackFinishTime'),('resetHitTimerWhenNoTarget','resetHitTimerWhenNoTarget'),('loadFirstHit','loadFirstHit'),('targetOnlyBuildings','targetOnlyBuildings'),('targetOnlyTroops','targetOnlyTroops'),('keepTargetWithPendingDamage','keepTargetWithPendingDamage')]:set_(at+'.targeting.'+dest,u[source])
        for dest,v in [('isBuilding',u['building'] or tower),('attacksGround',tower or u['attacksGround']),('attacksAir',tower or u['attacksAir']),('hasProjectile',p is not None),('sightRangeForCharacters',u['sightRange'] if tower else 0),('attackSequenceIndex',0)]:set_(at+'.targeting.'+dest,v)
        set_(at+'.state.deployTimeMs',0 if tower else u['deployTimeMs']);set_(at+'.state.kingTowerMiddle',u['king'])
        # StateVisitGlobals is fixed 250ms, not each row's optional attack-finish override.
        set_(at+'.state.attackFinishTimeMs',250)
        hp=scaling(u['hitpoints'],u,4 if u['king'] else 6 if u['summonerTower'] else 2)
        damage=scaling(u['damage'],u,3 if u['king'] else 5 if u['summonerTower'] else 1) if u['damage'] else (scaling(p['damage'],p,{'NONE':0,'CARD_DAMAGE':1,'KING_DAMAGE':3,'TOWER_DAMAGE':5}[p['damageMode']]) if p else 0)
        for dest,v in [('maximumHp',hp),('damage',damage),('packedLevel',(u['rarity']['relativeLevel']<<8)|(10-u['rarity']['relativeLevel'])),('rowVocab',vocab.index(name)),('projectile',pindex),('rowTags',u['gameTagsToSet']),('king',u['king']),('princess',u['summonerTower']),('building',u['building'] or tower),('hasMovement',not u['building'] and not tower and u['speed']>=1),('ignorePushback',u['ignorePushback']),('occluder',u['occluder'])]:set_(at+'.'+dest,v)
        for dest,source in [('collisionRadius','collisionRadius'),('mass','mass'),('projectileStartRadius','projectileStartRadius'),('projectileStartZ','projectileStartZ'),('projectileYOffset','projectileYOffset'),('noDeploySizeW','noDeploySizeW'),('noDeploySizeH','noDeploySizeH'),('tileSizeOverride','tileSizeOverride'),('crownPercent','crownTowerDamagePercent')]:set_(at+'.'+dest,u[source])
    for i,(name,p) in enumerate(data['projectiles'].items()):
        at=f'projectiles[{i}]'
        for dest,source in [('speed','speed'),('gravity','gravity'),('radius','radius'),('constant_height','constantHeight'),('min_distance','minDistance'),('projectile_range','projectileRange'),('crown_percent','crownTowerDamagePercent'),('pushback','pushback'),('buff_ms','buffTimeMs'),('random_delay_ms','randomDelayMs'),('homing','homing'),('only_enemies','onlyEnemies'),('hits_air','aoeToAir'),('hits_ground','aoeToGround'),('consider_z','considerZDistance'),('buff_before_damage','applyBuffBeforeDamage')]:set_(at+'.'+dest,p[source])
        for dest,v in [('id',i),('damage',scaling(p['damage'],p,{'NONE':0,'CARD_DAMAGE':1,'KING_DAMAGE':3,'TOWER_DAMAGE':5}[p['damageMode']])),('packed_level',(p['rarity']['relativeLevel']<<8)|(10-p['rarity']['relativeLevel'])),('target_buff',-1 if not p['targetBuff'] else data['buff_names'].index(p['targetBuff']))]:set_(at+'.'+dest,v)
        set_(f'projectileVocab[{i}]',vocab.index(name))
    for i,(name,c) in enumerate(data['cards'].items()):
        set_(f'cardVocab[{i}]',vocab.index(name));set_(f'cardProjectile[{i}]',-1 if not c['projectile'] else projectiles.index(c['projectile']));set_(f'cardArea[{i}]',-1 if not c['areaEffect'] else 0)
        for dest,v in [('count',max(1,c['multipleProjectiles'])),('waves',max(1,c['projectileWaves'])),('interval_ms',c['projectileIntervalMs']),('wave_interval_ms',c['projectileWaveIntervalMs']),('radius',c['radius']),('ring',bool(c['multipleProjectiles']))]:set_(f'spells[{i}].'+dest,v)
    a=data['areas']['Zap'];set_('zapVocab',vocab.index('Zap'))
    for dest,source in [('radius','radius'),('max_radius','maxRadius'),('life_ms','lifeDurationMs'),('hit_speed_ms','hitSpeedMs'),('offset_ms','hitSpeedOffsetMs'),('crown_percent','crownTowerDamagePercent'),('pushback','pushback'),('limit','maximumTargets'),('buff_ms','buffTimeMs'),('filter_hits','filterHits'),('ignore_buildings','ignoreBuildings'),('shared','sharedDamage'),('cap_buff_time','capBuffTimeToAreaEffectTime'),('only_own_troops','onlyOwnTroops')]:set_('zap.'+dest,a[source])
    for dest,v in [('damage',scaling(a['typedDamage']['baseDamage'],a,1)),('typed_tower_damage',scaling(a['typedDamage']['towerDamage'],a,1)),('packed_level',10),('buff_id',data['buff_names'].index('ZapFreeze')),('only_enemies',True),('hits_air',True),('hits_ground',True)]:set_('zap.'+dest,v)
    b=data['raw_tables']['character_buffs']['rows']['ZapFreeze']['columns'];set_('stun.id',data['buff_names'].index('ZapFreeze'))
    for dest,source in [('speed_percent','SpeedMultiplier'),('hit_speed_percent','HitSpeedMultiplier'),('spawn_speed_percent','SpawnSpeedMultiplier')]:set_('stun.'+dest,b[source])
    objects=data['raw_tables']['spawn_groups']['rows']['King_PrincessTowers']['columns']['Objects']
    for side in range(2):
        for j,obj in enumerate(objects):
            i=side*3+j;set_(f'towerX[{i}]',obj['x']*500);set_(f'towerY[{i}]',(64-obj['y'] if side else obj['y'])*500);set_(f'towerRow[{i}]',data['unit_ids'][obj['Data']])
    lines+=['return s;}', '}'];return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=HERE/'generated-v2/task-data.json');p.add_argument('--output',type=Path,default=HERE/'generated-v2/config.hpp');a=p.parse_args();data=json.loads(a.input.read_text());a.output.write_text(emit(data));print(a.output)
if __name__=='__main__':main()
