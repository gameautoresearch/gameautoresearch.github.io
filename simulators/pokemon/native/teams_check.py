"""Whole-team lifecycle differential oracle and native replay checks."""
import argparse
import json
import subprocess
import re
import shutil
from pathlib import Path

import numpy as np

from .bridge import HERE, Native, build
from .public_data import intrinsic_flags,BATTLE_FORMES
from .rules import STATS, STATUSES, TYPES, ABILITY_IDS, ITEM_IDS, compile_moves
from .validate import oracle_identity, seed64
from .oracle_stage import stage_oracle


def load_team_fixture(native, data, ids, species_ids=None):
    from .form_data import enrich_fixture,install_fixture_forms
    enrich_fixture(data)
    # The frozen request/protocol calls Greninja-Bond simply Greninja. Older
    # fixture metadata described its internal form ID; leave receipts unchanged.
    if 'metadata' in data and any(row['id']=='greninjabond' for row in data['metadata']['species']):
        from .reset import visible_token
        data['metadata']={**data['metadata'],'species':[{**row,'token':visible_token('species','Greninja')}if row['id']=='greninjabond'else row for row in data['metadata']['species']]}
    if 'metadata' in data:
        # Older retained oracles listed generator forms, before BattleStart's
        # crowned conversion. Complete only those frozen public Dex entries.
        known={row['id']for row in data['metadata']['species']}
        missing={p['species']for side in data['initial']['mons']for p in side}-known
        if missing:
            from .reset import visible_token
            data['metadata']={**data['metadata'],'species':list(data['metadata']['species'])}
            for row in BATTLE_FORMES:
                if row['id']not in missing:continue
                types=[TYPES.index(t)for t in row['types']]
                data['metadata']['species'].append(dict(id=row['id'],token=visible_token('species',row['name']),types=types,
                    type_tokens=[visible_token('type',t)for t in row['types']],stats=[row['stats'][k]for k in STATS[:5]]))
    native.species_names=[r['id']for r in data['metadata']['species']]if species_ids is None else [name for name,i in sorted(species_ids.items(),key=lambda x:x[1])]
    s = native.records[0]
    s.rng = seed64(data['initial']['seed'])
    s.turn = data['initial']['turn'];s.trick_room=data['initial'].get('trick_room',0)
    for i,value in enumerate(data['initial'].get('reviving',[0,0])):s.reviving[i]=int(value)
    s.weather=('', 'raindance','sunnyday','sandstorm','snowscape','hail').index(data['initial'].get('weather',''))
    s.weather_duration=data['initial'].get('weather_duration',0)
    for side,conditions in enumerate(data['initial'].get('side_conditions',[])):
        for j,(layers,duration) in enumerate(conditions):s.side_conditions[side][j]=layers;s.side_durations[side][j]=duration
    for side,names in enumerate(data['initial'].get('hazard_order',[])):
        for name in names:
            s.effect_order+=1;s.side_effect_order[side][('stealthrock','spikes','toxicspikes','stickyweb').index(name)]=s.effect_order
    terrains=('', 'electricterrain','grassyterrain','mistyterrain','psychicterrain')
    s.terrain=terrains.index(data['initial'].get('terrain',''));s.terrain_duration=data['initial'].get('terrain_duration',0)
    s.visible_terrain=terrains.index(re.sub('[^a-z]','',data['initial'].get('public_terrain','').lower().removeprefix('move: ')))
    for side,count in enumerate(data['initial'].get('total_fainted',[0,0])):s.total_fainted[side]=count
    s.limit = data.get('spec',{}).get('phases',512)+1
    s.fault = 0
    categories={m['id']:('Physical','Special','Status').index(m['category']) for m in data['moves']}
    for side in range(2):
        s.active[side] = data['initial']['active'][side]
        for position, slot in enumerate(data['initial']['order'][side]):
            s.order[side][position] = slot
        for slot, p in enumerate(data['initial']['mons'][side]):
            m = s.mons[side][slot]
            m.species = ((species_ids or {r['id']:i for i,r in enumerate(data['metadata']['species'])})[p['species']]
                         if 'metadata' in data else slot + 1)
            m.illusion_slot=p.get('illusion_slot',0);m.level=p['level'];m.base_speed=data['metadata']['species'][m.species]['stats'][4];m.iv_spe=p.get('ivs',{}).get('spe',31);m.ev_spe=p.get('evs',{}).get('spe',0)
            install_fixture_forms(m,p,species_ids or {r['id']:i for i,r in enumerate(data['metadata']['species'])})
            m.ability = ABILITY_IDS[p.get('ability','noability')];m.base_ability=ABILITY_IDS[p.get('base_ability',p.get('ability','noability'))];m.original_ability=ABILITY_IDS[p.get('original_ability',p.get('base_ability',p.get('ability','noability')))];m.embodied=int(p.get('embodied',False));m.tera_resisted=int(p.get('tera_resisted',False));m.trace_seek=int(p.get('trace_seek',False))
            m.item = ITEM_IDS[p.get('item','')]
            m.level = p['level']
            m.type[0] = TYPES.index(p['types'][0])
            m.type[1] = TYPES.index(p['types'][1]) if len(p['types']) > 1 else -1
            m.base_types[:]=m.type[:];m.protean_used=p.get('protean_used',False)
            names=('', 'confusion','flinch','taunt','encore','disable','leechseed','yawn','substitute','healblock','throatchop','protect','stall','roost','lockedmove','protosynthesis','quarkdrive','choicelock','flashfire','curse','saltcure','twoturnmove','solarbeam','meteorbeam','unburden','glaiverush','noretreat','trapped','magnetrise','truant','mustrecharge','trapper','partiallytrapped','chillyreception','charge','focuspunch','beakblast','mirrorcoat','destinybond','attract')
            order=p.get('volatile_order',[])
            for position,name in enumerate(order,1):
                if name in names:m.volatile_order[names.index(name)]=position
            m.form_busted=int(p.get('form_busted',False));m.gender={'M':1,'F':2,'N':3}.get(p.get('gender','N'),3);m.attract_source=p.get('attract_source',0);m.cud_berry=ITEM_IDS.get(p.get('cud_berry',''),0);m.cud_counter=p.get('cud_counter',0);m.destiny_bond=int(p.get('destiny_bond',False));m.mirror_coat=int(p.get('mirror_coat',False));m.mirror_damage=p.get('mirror_damage',0);m.base_attack=p.get('base_attack',100);m.bond_triggered=int(p.get('bond_triggered',False));m.chilly_reception=int(p.get('chilly_reception',False));m.focus_punch=p.get('focus_punch',0);m.beak_blast=int(p.get('beak_blast',False));m.charged=int(p.get('charged',False));m.truant=int(p.get('truant',False));m.recharge=p.get('recharge',0);m.partial_trap=p.get('partial_trap',0);m.partial_source=p.get('partial_source',0)
            for key in ('no_retreat','trapped_volatile','magnet_rise','glaive_rush'):setattr(m,key,int(p.get(key,0)))
            m.slow_start=p.get('slow_start',0)if p.get('ability')=='slowstart'else 0;m.supreme_fallen=p.get('supreme_fallen',0)
            m.unburden=p.get('unburden',False);m.last_item=ITEM_IDS.get(p.get('last_item',''),0);m.unnerved=p.get('unnerved',False)
            m.volatile_serial=len(order)
            m.charge_duration=p.get('charge_duration',0);m.charge_move=ids.get(p.get('charge_move'),0);m.charge_kind={'solarbeam':77,'meteorbeam':78}.get(p.get('charge_move'),0);m.charge_released=p.get('charge_released',False)

            m.tera_type = TYPES.index(p['tera'])
            m.hp, m.maxhp = p['hp'], p['maxhp']
            m.status = STATUSES.index(p['status'])
            m.sleep_source=p.get('sleep_source',side)
            m.times_attacked=p.get('times_attacked',0);m.nfe=int(p.get('nfe',False));m.weight_hg=p.get('weight_hg',1);m.stats_raised=p.get('stats_raised',False);m.heal_block=p.get('heal_block',0);m.throat_chop=p.get('throat_chop',0);m.locked_duration=p.get('locked_duration',0);m.locked_true=p.get('locked_true',0);m.locked_move=ids.get(p.get('locked_move'),-1);m.active_move_actions=p.get('active_move_actions',0)
            for key in ('curse','salt_cure','last_result','current_result','flash_fire','once_boost','species_flags','stats_lowered','newly_switched','hit_by_foe','paradox_stat','paradox_booster','booster_started','paradox_species'):setattr(m,key,int(p.get(key,0)))
            m.species_flags|=intrinsic_flags(p['species']);m.trapping_flags=intrinsic_flags(p['species'])>>16;m.known_type=int(p.get('known_type',not p.get('transformed')or p['terastallized']));m.got_attacked=int(p.get('got_attacked',False))
            m.confusion=p.get('confusion',0);m.flinch=int(p.get('flinch',False))
            for key in ('taunt','encore','disable','active_turns','leech_seed','yawn','substitute'):setattr(m,key,p.get(key,1 if key=='active_turns' else 0))
            for key in ('last_move','encore_move','disable_move'):setattr(m,key,ids.get(p.get(key),-1))
            m.protect=int(p.get('protect',False));m.stall_counter=p.get('stall_counter',0);m.stall_duration=p.get('stall_duration',0)
            m.status_turns, m.toxic_turns = p['statusState']['time'], p['statusState']['stage']
            for j, name in enumerate(STATS[:5]):
                m.stats[j] = p['stats'][name]
            for j, name in enumerate(STATS):
                m.boosts[j] = p['boosts'][name]
            for j, move in enumerate(p['moves']):
                m.move_ids[j] = ids[move['id']]
                m.move_categories[j]=categories[move['id']]
                source=next(r for r in data['moves'] if r['id']==move['id']);m.move_traits[j]=134217728*int(bool(source.get('flags',{}).get('cantusetwice')))|262144*int(bool(source.get('flags',{}).get('heal')))|16*int(bool(source.get('flags',{}).get('sound')))
                m.pp[j], m.maxpp[j] = move['pp'], move['maxpp']
    for side in range(2):
        for slot,p in enumerate(data['initial']['mons'][side]):
            if not p.get('transformed'):continue
            m=s.mons[side][slot];m.transformed=1
            mapping=species_ids or {r['id']:i for i,r in enumerate(data['metadata']['species'])}
            m.original_species=mapping[p['original_species']];m.original_trapping_flags=intrinsic_flags(p['original_species'])>>16;m.original_weight=p['original_weight']
            m.original_types[:]=[TYPES.index(t)for t in p['original_types']]+[-1]*(2-len(p['original_types']))
            for i,key in enumerate(STATS[:5]):m.original_stats[i]=p['original_stats'][key]
            for i in range(4):
                source=p['original_moves'][i]if i<len(p['original_moves'])else None
                m.original_moves[i]=ids[source['id']]if source else -1;m.original_pp[i]=source['pp']if source else 0;m.original_maxpp[i]=source['maxpp']if source else 0
                m.original_categories[i]=categories[source['id']]if source else 0;m.original_traits[i]=0
            from .form_data import FORM_GROUPS
            for family,(base,group)in enumerate(FORM_GROUPS.items(),1):
                group_ids=[''.join(c for c in n.lower()if c.isalnum())for n in group]
                if p['species']in group_ids:m.form_family=family;m.form_current=group_ids.index(p['species']);break
            # These abilities are explicitly inert on transformed Pokemon.
            if p['ability']in ('disguise','iceface','gulpmissile','hungerswitch','zerotohero','protosynthesis','quarkdrive'):
                m.suppressed_ability=m.ability;m.ability=0
    native.initialize_public()
    for side in range(2):
        for slot,p in enumerate(data['initial']['mons'][side]):
            if 'speed_cache' in p:native.records[0].mons[side][slot].speed_cache=p['speed_cache']
    if 'public_initial' in data:
        for side in range(2):
            s.public_count[side]=len(data['public_initial'][side])
            for index,source in enumerate(data['public_initial'][side]):
                slot=source['slot'];s.public_order[side][index]=slot;v=s.public_mons[side][slot]
                for field in ('hp','known_hp','fainted','tera','details_known'):
                    setattr(v,field,int(source[field]))
                v.tera_type=s.mons[side][slot].tera_type
                v.species=(species_ids or {r['id']:i for i,r in enumerate(data['metadata']['species'])}).get(source.get('species') or None,s.mons[side][slot].original_species if s.mons[side][slot].transformed else s.mons[side][slot].species);v.level=source.get('level',100 if s.mons[side][slot].form_current!=s.mons[side][slot].form_permanent else s.mons[side][slot].level);v.status=STATUSES.index(source['status']);v.ability=ABILITY_IDS.get(source['ability'],0)
                v.item=ITEM_IDS.get(re.sub('[^a-z0-9]','',source['item'].lower()),0)
                v.move_count=min(4,len(source['moves']))
                for k,name in enumerate(source['moves'][:4]):v.moves[k]=-2 if name=='struggle' else ids[name]
                name=re.sub('[^a-z0-9]','',source['last_move'].lower())
                v.last_move=-2 if name=='struggle' else ids.get(name,-1)
                for k,name in enumerate(STATS):v.boosts[k]=source['boosts'].get(name,0)


def observation_comparison(native, reference, tables):
    keys=('tokens','mons','global','global_tokens','action_tokens','actions','mask')
    expected=np.asarray([np.concatenate([np.asarray(row[key],dtype=np.float64).ravel() for key in keys])
                         for row in reference['observations']])
    actual=tables.observe(native)[0]
    indices=np.argwhere(actual!=expected)
    if not len(indices):return None
    return dict(observer_mismatch_count=len(indices),observer_differences=[dict(side=int(side),index=int(index),
                actual=float(actual[side,index]),expected=float(expected[side,index])) for side,index in indices[:30]])


def team_comparison(native, reference, index=0, masks=None):
    s = native.records[index]
    actual = dict(rng=s.rng, turn=s.turn, ended=bool(s.terminated), winner=s.winner, fault=s.fault,
                  active=list(s.active),
                  # Oracle `forced` is read from the last private request. Terminal
                  # messages retain that request even after its switch completed.
                  forced=[s.requests[side].forced if s.terminated else s.forced[side] for side in range(2)],
                  masks=(native.masks() if masks is None else masks)[index].tolist(),
                  order=[list(s.order[side])[:len(reference['order'][side])] for side in range(2)], mons=[])
    expected = dict(rng=seed64(reference['seed']), turn=reference['turn'], ended=reference['ended'],
                    winner=0 if reference['winner']=='p1' else 1 if reference['winner']=='p2' else -1, fault=0,
                    active=reference['active'], forced=[int(x) for x in reference['forced']],
                    masks=reference['masks'], order=reference['order'], mons=[])
    if 'unavailable_choices' in reference:actual['unavailable_choices']=list(s.unavailable_choices);expected['unavailable_choices']=reference['unavailable_choices']
    if 'trick_room' in reference:actual['trick_room']=s.trick_room;expected['trick_room']=reference['trick_room']
    if 'reviving' in reference:actual['reviving']=list(s.reviving);expected['reviving']=[int(x)for x in reference['reviving']]
    if 'weather' in reference:
        actual.update(weather=s.weather,weather_duration=s.weather_duration)
        expected.update(weather=('', 'raindance','sunnyday','sandstorm','snowscape','hail').index(reference['weather']),weather_duration=reference['weather_duration'])
    if 'side_conditions' in reference:
        actual['side_conditions']=[[[s.side_conditions[side][j],s.side_durations[side][j]] for j in range(10)] for side in range(2)]
        expected['side_conditions']=reference['side_conditions']
    if 'terrain' in reference:
        actual.update(terrain=s.terrain,terrain_duration=s.terrain_duration)
        expected.update(terrain=('', 'electricterrain','grassyterrain','mistyterrain','psychicterrain').index(reference['terrain']),terrain_duration=reference['terrain_duration'])
    if 'total_fainted' in reference:
        actual['total_fainted']=list(s.total_fainted);expected['total_fainted']=reference['total_fainted']
    if 'future' in reference:
        actual['future']=[[s.future_turn[i],next((name for name,j in getattr(native,'move_names',{}).items()if j==s.future_move[i]),'futuresight'),s.future_source[i],s.future_slot[i]]if s.future_turn[i]else [0,'',0,0]for i in range(2)];expected['future']=reference['future']
    if 'wish' in reference:
        actual['wish']=[[1,s.wish_turn[i],s.wish_hp[i]] if s.wish_pending[i] else [0,0,0] for i in range(2)];expected['wish']=reference['wish']
    statuses = STATUSES + ('fnt',)
    for side in range(2):
        actual['mons'].append([])
        expected['mons'].append([])
        for slot, p in enumerate(reference['mons'][side]):
            m = s.mons[side][slot]
            actual['mons'][side].append(dict(hp=m.hp, pp=list(m.pp)[:len(p['moves'])], status=m.status,
                                             ability=m.suppressed_ability or m.ability,item=m.item,fainted=bool(m.fainted), tera=bool(m.terastallized), boosts=list(m.boosts)))
            expected['mons'][side].append(dict(hp=p['hp'], pp=[x['pp'] for x in p['moves']],
                                               status=statuses.index(p['status']), fainted=p['fainted'],
                                               ability=ABILITY_IDS[p.get('ability','noability')],item=ITEM_IDS[p.get('item','')],
                                               tera=p['terastallized'], boosts=[p['boosts'][name] for name in STATS]))
            if hasattr(native,'species_names'):
                actual['mons'][side][-1].update(species=native.species_names[m.species],stats=list(m.stats))
                expected['mons'][side][-1].update(species=p['species'],stats=[p['stats'][key]for key in STATS[:5]])
            if 'illusion_slot' in p:actual['mons'][side][-1]['illusion_slot']=m.illusion_slot;expected['mons'][side][-1]['illusion_slot']=p['illusion_slot']
            if 'transformed' in p:actual['mons'][side][-1]['transformed']=bool(m.transformed);expected['mons'][side][-1]['transformed']=p['transformed']
            if 'form_busted' in p:actual['mons'][side][-1]['form_busted']=bool(m.form_busted);expected['mons'][side][-1]['form_busted']=p['form_busted']
            if 'attract_source' in p:
                for k in ('attract_source','cud_counter'):actual['mons'][side][-1][k]=getattr(m,k);expected['mons'][side][-1][k]=p[k]if k!='cud_counter'or p.get('ability')=='cudchew'else 0
                actual['mons'][side][-1]['cud_berry']=m.cud_berry;expected['mons'][side][-1]['cud_berry']=ITEM_IDS.get(p['cud_berry'],0)
            if 'destiny_bond' in p:
                for k in ('destiny_bond','mirror_coat','mirror_damage'):actual['mons'][side][-1][k]=int(getattr(m,k));expected['mons'][side][-1][k]=int(p[k])
            if 'bond_triggered' in p:actual['mons'][side][-1]['bond_triggered']=bool(m.bond_triggered);expected['mons'][side][-1]['bond_triggered']=p['bond_triggered']
            if 'focus_punch' in p:
                for k in ('focus_punch','chilly_reception','beak_blast'):actual['mons'][side][-1][k]=int(getattr(m,k));expected['mons'][side][-1][k]=int(p[k])
            if 'charged' in p:actual['mons'][side][-1]['charged']=bool(m.charged);expected['mons'][side][-1]['charged']=p['charged']
            if 'recharge' in p:
                for k in ('recharge','partial_trap'):actual['mons'][side][-1][k]=getattr(m,k);expected['mons'][side][-1][k]=p[k]
            if 'truant' in p:actual['mons'][side][-1]['truant']=bool(m.truant);expected['mons'][side][-1]['truant']=p['truant']
            if 'no_retreat' in p:
                for key in ('no_retreat','trapped_volatile','magnet_rise','glaive_rush'):actual['mons'][side][-1][key]=getattr(m,key);expected['mons'][side][-1][key]=int(p[key])
            if 'slow_start' in p:
                actual['mons'][side][-1].update(slow_start=m.slow_start,supreme_fallen=m.supreme_fallen);expected['mons'][side][-1].update(slow_start=p['slow_start']if p.get('ability')=='slowstart'else 0,supreme_fallen=p['supreme_fallen'])
            if 'unburden' in p:
                actual['mons'][side][-1].update(unburden=bool(m.unburden),last_item=m.last_item,unnerved=bool(m.unnerved));expected['mons'][side][-1].update(unburden=p['unburden'],last_item=ITEM_IDS.get(p['last_item'],0),unnerved=p['unnerved'])
            if 'charge_duration' in p:
                actual['mons'][side][-1].update(charge_duration=m.charge_duration,charge_move=None if not m.charge_duration else {77:'solarbeam',78:'meteorbeam'}.get(m.charge_kind),charge_released=bool(m.charge_released))
                expected['mons'][side][-1].update(charge_duration=p['charge_duration'],charge_move=p['charge_move'],charge_released=p['charge_released'])
            if 'protean_used' in p:
                actual['mons'][side][-1]['protean_used']=bool(m.protean_used);expected['mons'][side][-1]['protean_used']=p['protean_used']
            if 'types_now' in p:
                types=['???'if t==-2 else TYPES[t] for t in m.type if t>=0 or t==-2]
                if m.roost:types=[t for t in types if t!='Flying'] or ['Normal']
                if m.terastallized and m.tera_type!=18:types=[TYPES[m.tera_type]]
                actual['mons'][side][-1]['types_now']=types;expected['mons'][side][-1]['types_now']=p['types_now']
            if 'active_move_actions' in p:
                actual['mons'][side][-1]['active_move_actions']=m.active_move_actions;expected['mons'][side][-1]['active_move_actions']=p['active_move_actions']
            if 'locked_duration' in p:
                for key in ('locked_duration','locked_true'):
                    actual['mons'][side][-1][key]=getattr(m,key);expected['mons'][side][-1][key]=p[key]
            if 'heal_block' in p:
                for key in ('heal_block','throat_chop'):
                    actual['mons'][side][-1][key]=getattr(m,key);expected['mons'][side][-1][key]=p[key]
            for key in ('curse','salt_cure','last_result','current_result','flash_fire','once_boost','stats_lowered','newly_switched','hit_by_foe'):
                if key in p:actual['mons'][side][-1][key]=int(getattr(m,key));expected['mons'][side][-1][key]=int(p[key])
            if 'stats_raised' in p:
                actual['mons'][side][-1]['stats_raised']=bool(m.stats_raised);expected['mons'][side][-1]['stats_raised']=p['stats_raised']
            if 'paradox_stat' in p:
                for key in ('paradox_stat','paradox_booster'):
                    actual['mons'][side][-1][key]=int(getattr(m,key));expected['mons'][side][-1][key]=int(p[key])
            if 'roost' in p:
                actual['mons'][side][-1]['roost']=bool(m.roost);expected['mons'][side][-1]['roost']=p['roost']
            if 'times_attacked' in p:
                actual['mons'][side][-1]['times_attacked']=m.times_attacked;expected['mons'][side][-1]['times_attacked']=p['times_attacked']
            if 'statusState' in p:
                actual['mons'][side][-1]['status_state']=[m.status_turns,m.toxic_turns]
                expected['mons'][side][-1]['status_state']=[p['statusState']['time'],p['statusState']['stage']]
            if 'protect' in p:
                for key in ('protect','stall_counter','stall_duration'):
                    actual['mons'][side][-1][key]=int(getattr(m,key));expected['mons'][side][-1][key]=int(p[key])
            if 'confusion' in p:
                for key in ('confusion','flinch'):
                    actual['mons'][side][-1][key]=int(getattr(m,key));expected['mons'][side][-1][key]=int(p[key])
            if 'taunt' in p:
                for key in ('taunt','encore','disable'):
                    actual['mons'][side][-1][key]=getattr(m,key);expected['mons'][side][-1][key]=p[key]
            if 'substitute' in p:
                for key in ('leech_seed','yawn','substitute'):
                    actual['mons'][side][-1][key]=int(getattr(m,key));expected['mons'][side][-1][key]=int(p[key])
            if 'stellar_types' in p:
                actual['mons'][side][-1]['stellar_used']=m.stellar_used
                expected['mons'][side][-1]['stellar_used']=sum(1<<TYPES.index(name) for name in p['stellar_types'])
    if actual == expected:
        return None
    return dict(actual=actual, expected=expected)


def scenarios():
    rows = []
    for size in (2, 3, 6):
        teams = [[dict(moves=['tackle'], speed_ev=4 if side == 0 else 0) for _ in range(size)] for side in range(2)]
        rows += [dict(teams=teams), dict(teams=teams, switch_every=3), dict(teams=teams, policy='random')]
    rows += [
        dict(teams=[[dict(moves=['tackle'], speed_ev=0) for _ in range(3)] for _ in range(2)]),
        dict(teams=[[dict(moves=['tackle'], speed_ev=0) for _ in range(3)] for _ in range(2)], policy='random'),
        dict(teams=[[dict(moves=['tackle'], status='tox'),dict(moves=['tackle'])], [dict(moves=['tackle']),dict(moves=['tackle'])]], switch_every=3),
        dict(teams=[[dict(moves=['tackle'], status='slp'),dict(moves=['tackle'])], [dict(moves=['tackle']),dict(moves=['tackle'])]], switch_every=3),
        dict(teams=[[dict(species='Blastoise',moves=['watergun','icebeam']),dict(species='Venusaur',moves=['gigadrain'])],
                    [dict(species='Charizard',moves=['ember','wingattack']),dict(species='Pikachu',moves=['thundershock','quickattack'])]], policy='random'),
        dict(teams=[[dict(moves=['tackle'],hp_fraction=.03) for _ in range(3)] for _ in range(2)]),
        dict(teams=[[dict(moves=['takedown'],hp_fraction=.03) for _ in range(3)] for _ in range(2)]),
    ]
    for move in ('uturn','voltswitch','flipturn'):
        for speed in (0,4):
            teams=[[dict(moves=[move],speed_ev=speed),dict(moves=['tackle'],speed_ev=speed)],
                   [dict(moves=['tackle'],speed_ev=2),dict(moves=['tackle'],speed_ev=2)]]
            rows.append(dict(teams=teams))
        rows.append(dict(teams=[[dict(moves=[move],speed_ev=0) for _ in range(3)] for _ in range(2)]))
    rows += [dict(teams=[[dict(moves=['voltswitch']),dict(moves=['tackle'])],
                         [dict(species='Sandshrew',moves=['tackle']),dict(moves=['tackle'])]]),
             dict(teams=[[dict(moves=['uturn']),dict(moves=['tackle'])],
                         [dict(moves=['tackle'],hp_fraction=.02),dict(moves=['tackle'])]])]
    for ability in ('naturalcure','regenerator','pressure','hugepower','purepower','guts','levitate','intimidate','quickfeet','marvelscale'):
        status='par' if ability=='quickfeet' else 'brn' if ability in ('naturalcure','guts','marvelscale') else ''
        own=[dict(moves=['tackle','watergun','uturn'],ability=ability,status=status,hp_fraction=.6),
             dict(moves=['tackle'],ability=ability)]
        foe=[dict(moves=['earthquake','tackle']),dict(moves=['watergun','tackle'])]
        rows += [dict(teams=[own,foe]),dict(teams=[foe,own],policy='random')]
    for item in ('leftovers','choiceband','choicespecs','choicescarf','focussash'):
        rows.append(dict(teams=[[dict(moves=['tackle','watergun','uturn'],item=item),dict(moves=['tackle'],item=item)],
                                [dict(moves=['tackle','watergun']),dict(moves=['tackle'])]],policy='random'))
    rows += [
        dict(teams=[[dict(moves=['tackle'],item='leftovers',status='tox',speed_ev=0) for _ in range(2)] for _ in range(2)]),
        dict(teams=[[dict(moves=['uturn'],ability='intimidate',speed_ev=0) for _ in range(3)] for _ in range(2)]),
        dict(teams=[[dict(moves=['tackle'],ability='guts',item='choiceband',status='brn'),dict(moves=['tackle'])],
                    [dict(moves=['tackle'],ability='marvelscale',status='brn'),dict(moves=['tackle'])]]),
        dict(teams=[[dict(moves=['tackle'],ability='quickfeet',item='choicescarf',status='par'),dict(moves=['tackle'])],
                    [dict(moves=['tackle']),dict(moves=['tackle'])]]),
        dict(teams=[[dict(moves=['tackle'],level=100),dict(moves=['tackle'])],
                    [dict(moves=['tackle'],level=5,item='focussash'),dict(moves=['tackle'])]]),
        dict(teams=[[dict(moves=['tackle'],pp=[0]),dict(moves=['tackle'],pp=[0])] for _ in range(2)],policy='random'),
        dict(teams=[[dict(moves=['tackle'],pp=[1]),dict(moves=['tackle'])],
                    [dict(moves=['tackle'],ability='pressure'),dict(moves=['tackle'])]]),
    ]
    for speed in (0,4):
        rows.append(dict(teams=[[dict(moves=['protect'],speed_ev=speed),dict(moves=['protect'])],
                                [dict(moves=['tackle']),dict(moves=['tackle'])]]))
    rows += [
        dict(teams=[[dict(moves=['protect'],speed_ev=0) for _ in range(2)] for _ in range(2)]),
        dict(teams=[[dict(moves=['protect','tackle'],speed_ev=0) for _ in range(2)] for _ in range(2)],policy='random'),
        dict(teams=[[dict(moves=['protect'],item='leftovers',status='tox',speed_ev=0) for _ in range(2)] for _ in range(2)]),
        dict(teams=[[dict(moves=['protect'],status='par'),dict(moves=['protect'])],[dict(moves=['tackle']),dict(moves=['tackle'])]]),
        dict(teams=[[dict(moves=['protect'],stall_counter=729),dict(moves=['protect'])],[dict(moves=['tackle']),dict(moves=['tackle'])]]),
    ]
    return rows


def run(output, replay=None, observe=False,specs_file=None,library_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    library = Path(library_path)if library_path else build(output / 'build')
    oracle_source=stage_oracle(output,Path(library).parent/'source')
    (output / 'oracle-identity.json').write_text(json.dumps(oracle_identity(), indent=2) + '\n')
    if replay:
        def retained_cases():
            with Path(replay).open()as retained:
                for line in retained:yield json.loads(line)
        fixtures=retained_cases()
    else:
        if specs_file:specs=json.loads(Path(specs_file).read_text())
        else:
            plans=scenarios()
            if observe:plans=[plans[i] for i in (0,2,10,11,12,13,15,16,18,21,24,25,26,27,29,31,33,37,39,41,43,45,47,48,49,50,51,52,53,54,55,56,57,58)]
            specs = [dict(seed=f'{seed},{seed+1},{seed+2},{seed+3}',policy_seed=seed*31,phases=128,observe=observe,**options)
                     for options in plans for seed in range(1, 3 if observe else 9)]
        fixtures = []
        for begin in range(0, len(specs), 8):
            request = json.dumps(dict(cases=specs[begin:begin+8]))
            fixtures.extend(json.loads(subprocess.check_output(['node',str(oracle_source/'team_oracle.cjs')],input=request,text=True)))
    rows = []
    transitions = 0;natural=0
    if replay:shutil.copyfile(replay,output/'oracle-fixtures.jsonl')
    else:
        with (output / 'oracle-fixtures.jsonl').open('x') as file:
            for data in fixtures:
                file.write(json.dumps(data,separators=(',',':')) + '\n')
    for case, data in enumerate(fixtures):
        moves = compile_moves(data['moves'])
        ids = {move['id']: index for index, move in enumerate(data['moves'])}
        with Native(library) as native:
            load_team_fixture(native, data, ids)
            tables=None
            if 'metadata' in data:
                from .visible import PublicTables
                tables=PublicTables(data['metadata'],data['chart'],ids)
            mismatch = team_comparison(native, data['initial'])
            if mismatch is None and tables:mismatch=observation_comparison(native,data['initial'],tables)
            frames = 0
            if mismatch:
                mismatch['frame'] = -1
            else:
                for frame, record in enumerate(data['frames']):
                    native.step([record['actions']],moves,np.asarray(data['chart'],dtype=np.int8))
                    frames += 1
                    mismatch = team_comparison(native, record['state'])
                    if mismatch is None and tables:mismatch=observation_comparison(native,record['state'],tables)
                    if mismatch:
                        mismatch.update(frame=frame,calls=record['calls'])
                        break
        transitions += frames
        natural+=bool(data['frames'][-1]['state']['ended'])
        rows.append(dict(case=case,exact=mismatch is None,frames=frames,spec=data['spec'],mismatch=mismatch))
    result = dict(full_gen9_admission=False,library_sha256=native.binary_sha256,source_signature=native.source_signature,cases=len(rows),exact=sum(row['exact'] for row in rows),
                  transitions=transitions,natural_oracle_games=natural,rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if result['exact'] != result['cases']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--replay')
    parser.add_argument('--observe',action='store_true')
    parser.add_argument('--specs')
    parser.add_argument('--library')
    args=parser.parse_args()
    run(args.output,args.replay,args.observe,args.specs,args.library)
