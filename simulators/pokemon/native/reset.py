"""Public Dex table lowering for native generated-set construction.

This constructs Pokemon records. Battle startup callbacks and RL reset wiring
are separate obligations and do not receive full-task admission here.
"""
import ctypes as C
import re

from .bridge import Mon,Move,State
from .public_data import intrinsic_flags,BATTLE_FORMES
from .form_data import FORM_GROUPS,FORM_RECORDS
from .rules import ABILITY_IDS,ITEM_IDS,TYPES,compile_moves

class ResetSpecies(C.Structure):
    _fields_=[('id',C.c_int32),('types',C.c_int32*2),('stats',C.c_int32*6)]+[(name,C.c_int32) for name in ('fixed_hp','nfe','weight_hg','paradox_species','species_flags','battle_form','signature_move','form_family','form_count')]+[('forms',C.c_int32*3)]
class ResetMove(C.Structure):
    _fields_=[('category',C.c_int32),('maxpp',C.c_int32),('traits',C.c_uint32),('unsupported',C.c_uint32)]
class ResetRules(C.Structure):
    _fields_=[('formes',C.c_void_p),('forme_count',C.c_int32),('moves',C.c_void_p),('move_count',C.c_int32)]+[(name,C.c_void_p) for name in ('abilities','items','ability_supported','item_supported')]+[('ability_count',C.c_int32),('item_count',C.c_int32),('iron_head_id',C.c_int32)]

def ident(name):return re.sub('[^a-z0-9]','',name.lower())

class ResetTables:
    def __init__(self,generator,metadata,move_records):
        metadata=[*metadata,*[row for row in [*BATTLE_FORMES,*FORM_RECORDS] if row['id']not in {p['id']for p in metadata}]]
        self.species_names=['']+list(dict.fromkeys(row['id'] for row in metadata))
        species_ids={name:i for i,name in enumerate(self.species_names)}
        forme_names=[*generator.formes,*[row['name']for row in [*BATTLE_FORMES,*FORM_RECORDS] if row['name']not in generator.formes]]
        forms=(ResetSpecies*len(forme_names))()
        lookup={row['name']:row for row in metadata}
        for target,name in zip(forms,forme_names):
            source=lookup[name];target.id=species_ids[source['id']]
            target.types[:]=[TYPES.index(t) for t in source['types']]+[-1]*(2-len(source['types']))
            target.stats[:]=[source['stats'][name] for name in ('hp','atk','def','spa','spd','spe')]
            for key in ('fixed_hp','nfe','weight_hg','paradox_species'):setattr(target,key,int(source[key]))
            target.species_flags=intrinsic_flags(source['id'])
            target.battle_form=-1;target.signature_move=-1
            for family,(base,group) in enumerate(FORM_GROUPS.items(),1):
                if name==base or (base=='Minior' and name.startswith('Minior') and name!='Minior-Meteor'):
                    selected=[name,*group[1:]];target.form_family=family;target.form_count=len(selected)
                    for i,n in enumerate(selected):target.forms[i]=forme_names.index(n)

            if name in ('Zacian','Zamazenta'):
                target.battle_form=forme_names.index(name+'-Crowned')
                target.signature_move=generator.move_ids.get('behemothblade'if name=='Zacian'else 'behemothbash',-1)
        self.ability_names=list(ABILITY_IDS);self.item_names=list(ITEM_IDS)
        for name in generator.ability_names:
            if ident(name) not in self.ability_names:self.ability_names.append(ident(name))
        for name in generator.item_names:
            if ident(name) not in self.item_names:self.item_names.append(ident(name))
        abilities=(C.c_int32*len(generator.ability_names))(*[self.ability_names.index(ident(name)) for name in generator.ability_names])
        items=(C.c_int32*len(generator.item_names))(*[self.item_names.index(ident(name)) for name in generator.item_names])
        ability_supported=(C.c_uint8*len(abilities))(*[ident(name) in ABILITY_IDS for name in generator.ability_names])
        item_supported=(C.c_uint8*len(items))(*[ident(name) in ITEM_IDS for name in generator.item_names])
        moves=(ResetMove*len(move_records))();self.move_rejections={};self.battle_moves=(Move*len(move_records))()
        self.metadata=metadata;self.move_records=move_records;self.generator=generator
        for target,source in zip(moves,move_records):
            record={k:v for k,v in source.items() if k not in ('power_callback','damage_callback','has_secondary')}
            target.category=('Physical','Special','Status').index(source['category'])
            target.maxpp=source['pp'] if source.get('noPPBoosts') else source['pp']*8//5
            target.traits=134217728*bool(source.get('flags',{}).get('cantusetwice'))|262144*bool(source.get('flags',{}).get('heal'))|16*bool(source.get('flags',{}).get('sound'))
            index=generator.move_ids[source['id']]
            try:
                value=compile_moves([record])[0];value.id=index;self.battle_moves[index]=value
            except ValueError as error:
                target.unsupported=1;self.move_rejections[source['id']]=str(error)
                value=self.battle_moves[index];value.id=index;value.unsupported=1;value.category=target.category;value.maxpp=target.maxpp

        self.buffers=(forms,moves,abilities,items,ability_supported,item_supported)
        self.rules=ResetRules(C.addressof(forms),len(forms),C.addressof(moves),len(moves),*[C.addressof(buf) for buf in self.buffers[2:]],len(abilities),len(items),generator.move_ids.get('ironhead',-1))

    def bind(self,native):
        from .teamgen import GenSet,GenTeam,GenRules
        native.species_names=self.species_names
        lib=native.lib
        for name,kind in (('species',ResetSpecies),('move',ResetMove),('rules',ResetRules)):
            function=getattr(lib,f'pkn_reset_{name}_size');function.restype=C.c_size_t
            if function()!=C.sizeof(kind):raise RuntimeError(f'Reset ABI mismatch: {name}')
        lib.pkn_reset_seeds_size.restype=C.c_size_t
        if lib.pkn_reset_seeds_size()!=C.sizeof(ResetSeeds):raise RuntimeError('Reset seed ABI mismatch')
        lib.pkn_start_battle.argtypes=[C.POINTER(Mon),C.POINTER(Mon),C.c_uint64,C.c_int,C.POINTER(State),C.POINTER(Move),C.c_int,C.c_void_p]
        lib.pkn_reset_battle.argtypes=[C.POINTER(GenRules),C.POINTER(ResetRules),C.POINTER(ResetSeeds),C.POINTER(State),C.POINTER(Move),C.c_int,C.c_void_p]
        lib.pkn_materialize_set.argtypes=[C.POINTER(ResetRules),C.POINTER(GenSet),C.POINTER(Mon)]
        lib.pkn_materialize_team.argtypes=[C.POINTER(ResetRules),C.POINTER(GenTeam),C.POINTER(Mon)]
        lib.pkn_materialize_team_sized.argtypes=[C.POINTER(ResetRules),C.POINTER(GenTeam),C.POINTER(Mon),C.c_int]
        lib.pkn_start_battle_sized.argtypes=[C.POINTER(Mon),C.POINTER(Mon),C.c_uint64,C.c_int,C.POINTER(State),C.POINTER(Move),C.c_int,C.c_void_p,C.c_int]
        return lib

    def decode(self,record,move_names):
        return dict(species=self.species_names[record.species],types=[TYPES[t] for t in record.type if t>=0],
            level=record.level,tera=TYPES[record.tera_type],hp=record.hp,maxhp=record.maxhp,stats=list(record.stats),
            ability=self.ability_names[record.ability],item=self.item_names[record.item],weight_hg=record.weight_hg,
            nfe=bool(record.nfe),paradox_species=bool(record.paradox_species),
            moves=[dict(id=move_names[record.move_ids[i]],pp=record.pp[i],maxpp=record.maxpp[i]) for i in range(4) if record.move_ids[i]>=0])

    def to_cuda(self):
        import numpy as np
        import torch
        rules=ResetRules.from_buffer_copy(bytes(self.rules));self.device_buffers={}
        for name,source in zip(('formes','moves','abilities','items','ability_supported','item_supported'),self.buffers):
            value=torch.from_numpy(np.frombuffer(bytes(source),dtype=np.uint8).copy()).cuda();self.device_buffers[name]=value;setattr(rules,name,value.data_ptr())
        return rules

class ResetSeeds(C.Structure):
    # team_size fills the old padding; 0 (every schedule built before it) means the six-Pokémon format.
    _fields_=[('battle',C.c_uint64),('teams',C.c_uint64*2),('limit',C.c_int32),('team_size',C.c_int32)]

def visible_token(kind,name):
    if not name:return 0
    value=(kind+':'+ident(name)).encode();result=2166136261
    for byte in value:result=((result^byte)*16777619)&0xffffffff
    return 1+result%16383

def public_tables(tables,chart):
    from .visible import PublicTables
    records={row['id']:row for row in tables.metadata}
    species=[dict(id='',token=0,types=[-1,-1],type_tokens=[0,0],stats=[0]*5)]
    for name in tables.species_names[1:]:
        row=records[name];types=[TYPES.index(t) for t in row['types']]
        species.append(dict(id=name,token=visible_token('species','Greninja' if name=='greninjabond' else row['name']),types=types+[-1]*(2-len(types)),
            type_tokens=[visible_token('type',t) for t in row['types']]+[0]*(2-len(types)),stats=[row['stats'][name] for name in ('atk','def','spa','spd','spe')]))
    moves=[]
    for row in tables.move_records+[dict(id='struggle',type='Normal',category='Physical',basePower=50,accuracy=True,priority=0)]:
        moves.append(dict(id=row['id'],token=visible_token('move',row['id']),type_token=visible_token('type',row['type']),category_token=visible_token('category',row['category']),
            type=TYPES.index(row['type']),category=('Physical','Special','Status').index(row['category']),power=row['basePower'],accuracy=100 if row['accuracy'] is True else row['accuracy'],priority=row['priority']))
    metadata=dict(species=species,moves=moves,ability_tokens=[visible_token('ability',name) for name in tables.ability_names],item_tokens=[visible_token('item',name) for name in tables.item_names],
        type_tokens=[visible_token('type',name) for name in TYPES],status_tokens=[visible_token('status',name) for name in ('','brn','par','slp','frz','psn','tox')])
    return PublicTables(metadata,chart,tables.generator.move_ids)
