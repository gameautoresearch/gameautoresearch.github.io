"""Static frozen Dex metadata compiler for native team-generation primitives.

Complete randomMoveset/randomSet/randomTeam assembly is still unfinished. These
tables contain public rules, never pre-generated teams or simulator transitions.
"""
import ctypes as C
import re

from .rules import TYPES


class GenSpecies(C.Structure):
    _fields_=[('base',C.c_int32),('types',C.c_int32*2),('battle_types',C.c_int32*2)]+[
        (name,C.c_int32) for name in ('level','form_begin','form_count','sample_form')]+[('compatibility',C.c_uint32)]+[
        (name,C.c_int32) for name in ('priority_species','raging_bull_type','ivy_cudgel_type','mega_sun','ability_species')]+[('id_tag',C.c_uint32),('required_move',C.c_int32),('stats',C.c_int32*6),
        ('base_tag',C.c_uint32),('nfe',C.c_int32),('required_item_begin',C.c_int32),('required_item_count',C.c_int32),('effectiveness',C.c_int32*19)]+[
        (name,C.c_int32) for name in ('set_begin','set_count','gender','fire_weak_abilities')]


class GenMove(C.Structure):
    _fields_=[(name,C.c_int32) for name in ('type','category','priority','kind')]+[('flags',C.c_uint32),('id_tag',C.c_uint32)]


class GenMoveList(C.Structure):
    _fields_=[('values',C.c_int32*64),('count',C.c_int32),('fault',C.c_int32)]


class GenRange(C.Structure):
    _fields_=[('begin',C.c_int32),('count',C.c_int32)]


class GenSetSpec(C.Structure):
    _fields_=[('role',C.c_int32),('moves',GenRange),('abilities',GenRange),('teras',GenRange),
              ('sample_tera',C.c_int32),('ability_flags',C.c_uint64)]


class GenSet(C.Structure):
    _fields_=[(name,C.c_int32) for name in ('species','forme','ability','item','level','tera','role')]+[
        ('evs',C.c_int32*6),('ivs',C.c_int32*6),('gender',C.c_int32),('shiny',C.c_int32),('moves',GenMoveList)]


class GenTeam(C.Structure):
    _fields_=[('pokemon',GenSet*6),('count',C.c_int32),('fault',C.c_int32)]


class GenCounter(C.Structure):
    _fields_=[('counts',C.c_int32*44),('damaging_moves',C.c_int32*32),('damaging_count',C.c_int32),
              ('base_power_moves',C.c_int32*32),('base_power_count',C.c_int32)]


class GenRules(C.Structure):
    _fields_=[('species',C.c_void_p),('species_count',C.c_int32),('group_count',C.c_int32),
              ('formes',C.c_void_p),('squawkabilly_groups',C.c_void_p),('moves',C.c_void_p),('move_count',C.c_int32),('ability_codes',C.c_void_p),
              ('selector_moves',C.c_void_p),('selector_ranges',C.c_void_p)]
    _fields_ += [('ability_tags',C.c_void_p),('item_tags',C.c_void_p),('item_count',C.c_int32),('required_items',C.c_void_p)]
    _fields_ += [(name,C.c_void_p) for name in ('sets','set_moves','set_abilities','set_teras')]


class GenPool(C.Structure):
    _fields_=[('groups',C.c_int32*1024),('group_count',C.c_int32),('species',C.c_int32*1024),('species_count',C.c_int32)]


ABILITY_FLAGS={name:1<<i for i,name in enumerate(('Grassy Surge','Seed Sower','Psychic Surge','Electric Surge',
    'Chlorophyll','Drought','Mega Sol','Orichalcum Pulse','Drizzle','Aerilate','Galvanize','Pixilate','Refrigerate','Prankster','Guts'))}
ABILITY_CODES=('Chlorophyll','Solar Power','Defiant','Hydration','Swift Swim','Iron Fist','Skill Link','Overgrow',
    'Prankster','Sand Force','Sand Rush','Slush Rush','Swarm','Torrent','Flash Fire','Aftermath','Unburden')
COUNTER_NAMES=list(TYPES)+'''Physical Special Status damage technician skilllink recoil drain strongjaw ironfist sound priority
sheerforce inaccurate recovery contrary physicalsetup specialsetup mixedsetup speedsetup speedcontrol setup hazards stab stabtera'''.split()
ROLE_NAMES=('Bulky Attacker','Bulky Setup','Bulky Support','Fast Attacker','Fast Bulky Setup','Fast Support',
    'Setup Sweeper','Tera Blast user','Wallbreaker','AV Pivot','Bulky Protect','Offensive Protect','Choice Item user')
MOVE_TAGS=[
    'healorder milkdrink moonlight morningsun recover roost shoreup slackoff softboiled strengthsap synthesis',
    'armorcannon closecombat leafstorm makeitrain overheat spinout superpower vcreate',
    'bellydrum bulkup coil curse dragondance honeclaws howl meditate poweruppunch swordsdance tidyup victorydance',
    'calmmind chargebeam geomancy nastyplot quiverdance tailglow takeheart torchsong',
    'clangoroussoul growth happyhour holdhands noretreat shellsmash workup',
    'agility autotomize flamecharge rockpolish snowscape trailblaze',
    'electroweb glare icywind lowsweep nuzzle quash tailwind thunderwave trickroom',
    'acidarmor agility autotomize bellydrum bulkup calmmind clangoroussoul coil cosmicpower curse dragondance flamecharge growth honeclaws howl irondefense meditate nastyplot noretreat poweruppunch quiverdance rockpolish shellsmash shelter shiftgear swordsdance tailglow takeheart tidyup trailblaze workup victorydance',
    'spikes stealthrock stickyweb toxicspikes',
    'acidspray accelerock aquajet bounce breakingswipe bulletpunch chatter chloroblast clearsmog covet dragontail doomdesire electroweb eruption explosion fakeout feint flamecharge flipturn futuresight grassyglide iceshard icywind incinerate infestation machpunch meteorbeam mortalspin nuzzle pluck pursuit quickattack rapidspin reversal selfdestruct shadowsneak skydrop snarl strugglebug suckerpunch trailblaze uturn vacuumwave voltswitch watershuriken waterspout',
]
MOVE_TAG_LISTS=[names.split() for names in MOVE_TAGS]
MOVE_TAGS=[frozenset(names) for names in MOVE_TAG_LISTS]


def generation_tag(name):
    result=2166136261
    for byte in re.sub('[^a-z0-9]','',name.lower()).encode('ascii'):result=((result^byte)*16777619)&0xffffffff
    return result


def generation_move(source):
    target=GenMove();name=source['id'];power=source['basePower'];flags=source.get('flags',{});target.id_tag=generation_tag(name)
    target.type=TYPES.index(source['type']);target.category=('Physical','Special','Status').index(source['category'])
    target.priority=source['priority'];target.kind={'terablast':1,'judgment':2,'multiattack':2,'revelationdance':2,
        'ragingbull':3,'ivycudgel':4,'terrainpulse':5,'weatherball':6}.get(name,0)
    conditions=[bool(source.get('damage') or source.get('damage_callback')),bool(power or source.get('power_callback')),
        name=='lowkick' or bool(power and power<=60 and name not in ('nuzzle','rapidspin')),
        isinstance(source.get('multihit'),list) and source['multihit'][1]==5,
        bool(source.get('recoil') or source.get('hasCrashDamage')),bool(source.get('drain')),
        bool(flags.get('bite')),bool(flags.get('punch')),bool(flags.get('sound')),
        bool(source.get('has_secondary') or source.get('hasSheerForceBoost')),
        bool(source['accuracy'] is not True and source['accuracy'] and source['accuracy']<90)]
    target.flags=sum(1<<i for i,present in enumerate(conditions) if present)
    for i,names in enumerate(MOVE_TAGS):
        if name in names:target.flags|=1<<(11+i)
    if name=='grassyglide':target.flags|=1<<21
    return target


def forme_choices(species):
    name,base=species['name'],species['base']
    if isinstance(species['battle_only'],str):return [species['battle_only']],False
    if species['cosmetic_formes']:return [name,*species['cosmetic_formes']],True
    if name.endswith('-Gmax'):return [name[:-5]],False
    if base in ('Dudunsparce','Maushold','Polteageist','Sinistcha','Zarude'):
        if not species['other_formes']:raise ValueError('Missing frozen forme metadata')
        return [name,*species['other_formes']],True
    if base=='Basculin':return ['Basculin','Basculin-Blue-Striped'],True
    if base=='Magearna':return ['Magearna','Magearna-Original'],True
    if base=='Pikachu':return ['Pikachu'+suffix for suffix in ('','-Original','-Hoenn','-Sinnoh','-Unova','-Kalos','-Alola','-Partner','-World')],True
    return [name],False


def compatibility_flags(name):
    flags={'blissey':1,'chansey':2,'illumise':4,'volbeat':8,'toxicroak':64}.get(name,0)
    if name in ('ariados','smeargle','masquerain','kricketune','leavanny','galvantula','vikavolt','ribombee','araquanid','spidops'):flags|=16
    if name in ('meowstic','grimmsnarl','ninetalesalola','abomasnow'):flags|=32
    if name in ('ninetales','torkoal','groudon','koraidon'):flags|=128
    return flags


class GeneratorTables:
    def __init__(self,records,moves=(),status_moves=(),items=()):
        self.species_ids={r['id']:i for i,r in enumerate(records)}
        move_ids={row['id']:i for i,row in enumerate(moves)}
        self.item_names=[row['name'] for row in items];item_ids={row['id']:i for i,row in enumerate(items)}
        required_items=[]
        self.ability_names=sorted(set(ABILITY_CODES)|{name for row in records for name in row.get('abilities',[])})
        self.ability_ids={name:i for i,name in enumerate(self.ability_names)}
        set_specs=[];set_moves=[];set_abilities=[];set_teras=[]
        self.bases=list(dict.fromkeys(r['base'] for r in records))
        base_ids={name:i for i,name in enumerate(self.bases)}
        self.formes=[];form_ids={};choices=[]
        species=(GenSpecies*len(records))()
        for target,source in zip(species,records):
            target.base=base_ids[source['base']];target.level=source['level']
            for name in ('types','battle_types'):
                values=[TYPES.index(t) for t in source[name]]
                getattr(target,name)[:]=values+[-1]*(2-len(values))
            forms,sample=forme_choices(source);target.form_begin=len(choices);target.form_count=len(forms);target.sample_form=sample
            for name in forms:
                if name not in form_ids:form_ids[name]=len(self.formes);self.formes.append(name)
                choices.append(form_ids[name])
            target.compatibility=compatibility_flags(source['id'])
            target.id_tag=generation_tag(source['id'])
            target.priority_species=source['id'] in ('breloom','brutebonnet','cacturne','honchkrow','mimikyu','ragingbolt','scizor')
            target.raging_bull_type=-1;target.ivy_cudgel_type=-1
            if source['name'].startswith('Tauros-Paldea'):
                for suffix,type in (('Combat',6),('Blaze',1),('Aqua',2)):
                    if source['name'].endswith(suffix):target.raging_bull_type=type
            if source['name'].startswith('Ogerpon'):
                for suffix,type in (('Wellspring',2),('Hearthflame',1),('Cornerstone',12)):
                    if source['name'].endswith(suffix):target.ivy_cudgel_type=type
            target.mega_sun=source.get('is_mega',False) and any(a in ('Chlorophyll','Drought','Mega Sol','Orichalcum Pulse') for a in source.get('abilities',[]))
            target.ability_species={'drifblim':1,'thundurus':2,'tornadus':3,'swampert':4,'toucannon':5,'golduck':6}.get(source['id'],0)
            target.stats[:]=[source['stats'][name] for name in ('hp','atk','def','spa','spd','spe')]
            target.required_move=move_ids[source['required_move']] if source.get('required_move') else -1
            target.base_tag=generation_tag(source['base']);target.nfe=source.get('nfe',False)
            target.effectiveness[:]=source.get('effectiveness',[0]*19)
            target.required_item_begin=len(required_items);target.required_item_count=len(source.get('required_items') or [])
            for name in source.get('required_items') or []:required_items.append(item_ids[re.sub('[^a-z0-9]','',name.lower())])
            target.gender={None:0,'':0,'M':1,'F':2,'N':3}[source.get('gender')]
            target.fire_weak_abilities=any(name in ('Dry Skin','Fluffy') for name in source.get('abilities',[]))
            target.set_begin=len(set_specs);target.set_count=len(source['sets'])
            for row in source['sets']:
                spec=GenSetSpec();spec.role=ROLE_NAMES.index(row['role'])
                spec.moves=GenRange(len(set_moves),len(row['movepool']));set_moves.extend(move_ids[re.sub('[^a-z0-9]','',name.lower())] for name in row['movepool'])
                spec.abilities=GenRange(len(set_abilities),len(row['abilities']));set_abilities.extend(self.ability_ids[name] for name in row['abilities'])
                teras=row['teraTypes'];spec.sample_tera=isinstance(teras,list);teras=teras if isinstance(teras,list) else [teras]
                spec.teras=GenRange(len(set_teras),len(teras));set_teras.extend(TYPES.index(name) for name in teras)
                spec.ability_flags=sum(ABILITY_FLAGS.get(name,0) for name in set(row['abilities']));set_specs.append(spec)
        forme_data=(C.c_int32*len(choices))(*choices)
        squawks=(C.c_uint8*len(self.bases))(*[name=='Squawkabilly' for name in self.bases])
        move_data=(GenMove*len(moves))(*[generation_move(source) for source in moves])
        self.move_names=[row['id'] for row in moves];self.move_ids={name:i for i,name in enumerate(self.move_names)}
        if len({generation_tag(name) for name in self.move_names})!=len(self.move_names):raise ValueError('Generator move tag collision')
        groups=[status_moves,MOVE_TAG_LISTS[7],MOVE_TAG_LISTS[2],
                'chillyreception flipturn partingshot shedtail teleport uturn voltswitch'.split(),
                MOVE_TAG_LISTS[8],MOVE_TAG_LISTS[5],'banefulbunker burningbulwark protect silktrap spikyshield'.split(),
                MOVE_TAG_LISTS[0],MOVE_TAG_LISTS[6]]
        selectors=[];ranges=(GenRange*len(groups))()
        for target,group in zip(ranges,groups):
            ids=[self.move_ids[name] for name in group if name in self.move_ids]
            target.begin=len(selectors);target.count=len(ids);selectors.extend(ids)
        selector_data=(C.c_int32*len(selectors))(*selectors)
        codes=(C.c_int32*len(ABILITY_CODES))(*[self.ability_ids[name] for name in ABILITY_CODES])
        ability_tags=(C.c_uint32*len(self.ability_names))(*[generation_tag(name) for name in self.ability_names])
        item_tags=(C.c_uint32*len(items))(*[generation_tag(row['id']) for row in items])
        required_data=(C.c_int32*len(required_items))(*required_items)
        spec_data=(GenSetSpec*len(set_specs))(*set_specs)
        set_move_data=(C.c_int32*len(set_moves))(*set_moves);set_ability_data=(C.c_int32*len(set_abilities))(*set_abilities)
        set_tera_data=(C.c_int32*len(set_teras))(*set_teras)
        if len(set(item_tags))!=len(item_tags) or len(set(ability_tags))!=len(ability_tags):raise ValueError('Generator item/ability tag collision')
        self.buffers=(species,forme_data,squawks,move_data,codes,selector_data,ranges,ability_tags,item_tags,required_data,spec_data,set_move_data,set_ability_data,set_tera_data)
        self.rules=GenRules(C.addressof(species),len(species),len(self.bases),C.addressof(forme_data),C.addressof(squawks),C.addressof(move_data),len(moves),C.addressof(codes),C.addressof(selector_data),C.addressof(ranges),C.addressof(ability_tags),C.addressof(item_tags),len(items),C.addressof(required_data),C.addressof(spec_data),C.addressof(set_move_data),C.addressof(set_ability_data),C.addressof(set_tera_data))

    def bind(self,native):
        lib=native.lib
        for name,kind in (('species',GenSpecies),('rules',GenRules),('pool',GenPool),('counter',GenCounter),('move',GenMove),('move_list',GenMoveList),('set_spec',GenSetSpec),('set',GenSet),('team',GenTeam)):
            function=getattr(lib,f'pkn_gen_{name}_size');function.restype=C.c_size_t
            if function()!=C.sizeof(kind):raise RuntimeError(f'Native generator ABI mismatch: {name}')
        lib.pkn_gen_pool.argtypes=[C.POINTER(GenRules),C.POINTER(GenPool),C.c_int,C.c_void_p]
        lib.pkn_gen_forme.argtypes=[C.POINTER(GenRules),C.c_int,C.POINTER(C.c_uint64)]
        lib.pkn_gen_compatible.argtypes=[C.POINTER(GenRules),C.c_int,C.c_void_p,C.c_int]
        lib.pkn_gen_query_moves.argtypes=[C.POINTER(GenRules),C.c_int,C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.POINTER(GenCounter)]
        lib.pkn_gen_ability.argtypes=[C.POINTER(GenRules),C.c_int,C.c_void_p,C.c_int,C.POINTER(GenCounter),C.c_uint32,C.c_uint32,C.c_int,C.POINTER(C.c_uint64)]
        lib.pkn_gen_cull_moves.argtypes=[C.POINTER(GenRules),C.c_int,C.POINTER(GenMoveList),C.POINTER(GenMoveList),C.c_uint64,C.c_uint32,C.c_int]
        lib.pkn_gen_moveset.argtypes=[C.POINTER(GenRules),C.c_int,C.POINTER(GenMoveList),C.c_uint64,C.c_uint32,C.c_int,C.c_int,C.POINTER(C.c_uint64),C.POINTER(GenMoveList)]
        lib.pkn_gen_set_stage.argtypes=[C.POINTER(GenRules),C.c_int,C.POINTER(GenMoveList),C.c_void_p,C.c_int,C.c_uint64,C.c_uint32,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_uint64),C.POINTER(GenMoveList),C.c_void_p]
        lib.pkn_gen_random_set.argtypes=[C.POINTER(GenRules),C.c_int,C.c_uint32,C.c_int,C.POINTER(C.c_uint64),C.POINTER(GenSet)]
        lib.pkn_gen_random_team.argtypes=[C.POINTER(GenRules),C.POINTER(C.c_uint64),C.POINTER(GenTeam)]
        lib.pkn_gen_random_team_sized.argtypes=[C.POINTER(GenRules),C.POINTER(C.c_uint64),C.POINTER(GenTeam),C.c_int]
        return lib

    def decode_set(self,record):
        source=self.buffers[0][record.species]
        return dict(name=self.bases[source.base],species=self.formes[record.forme],speciesId=list(self.species_ids)[record.species],
            gender=('', 'M','F','N')[record.gender],shiny=bool(record.shiny),level=record.level,
            moves=[self.move_names[i] for i in record.moves.values[:record.moves.count]],ability=self.ability_names[record.ability],
            evs=dict(zip(('hp','atk','def','spa','spd','spe'),record.evs)),ivs=dict(zip(('hp','atk','def','spa','spd','spe'),record.ivs)),
            item=self.item_names[record.item],teraType=TYPES[record.tera],role=ROLE_NAMES[record.role])

    def to_cuda(self):
        import torch
        import numpy as np
        names=('species','formes','squawkabilly_groups','moves','ability_codes','selector_moves','selector_ranges',
               'ability_tags','item_tags','required_items','sets','set_moves','set_abilities','set_teras')
        self.device_buffers={}
        rules=GenRules.from_buffer_copy(bytes(self.rules))
        for name,source in zip(names,self.buffers):
            data=torch.from_numpy(np.frombuffer(bytes(source),dtype=np.uint8).copy()).cuda()
            self.device_buffers[name]=data;setattr(rules,name,data.data_ptr())
        return rules
