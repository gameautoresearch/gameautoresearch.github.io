"""Native/device lowering of the existing 700-value player-visible schema."""
import ctypes as C
import numpy as np


class SpeciesInfo(C.Structure):
    _fields_ = [('token', C.c_int32), ('types', C.c_int32*2),
                ('type_tokens', C.c_int32*2), ('stats', C.c_int32*5)]


class MoveInfo(C.Structure):
    _fields_ = [(name, C.c_int32) for name in
                ('token','type_token','category_token','type','category','power','accuracy','priority')]


class PublicRules(C.Structure):
    _fields_ = [('species', C.c_void_p), ('species_count', C.c_int32),
                ('moves', C.c_void_p), ('move_count', C.c_int32), ('struggle_index', C.c_int32), ('recharge_index', C.c_int32)] + [
                (name, C.c_void_p) for name in ('ability_tokens','item_tokens','type_tokens','status_tokens','chart')]


class PublicTables:
    def __init__(self, metadata, chart, move_ids):
        self.species_ids = {r['id']: i for i, r in enumerate(metadata['species'])}
        species = (SpeciesInfo*len(metadata['species']))()
        for target, source in zip(species, metadata['species']):
            target.token = source['token']
            for name in ('types','type_tokens','stats'):
                getattr(target,name)[:] = source[name]
        records = {r['id']: r for r in metadata['moves']}
        from .reset import visible_token
        records['recharge'] = dict(token=visible_token('move','recharge'),type_token=0,category_token=0,type=-1,category=-1,power=0,accuracy=0,priority=0)
        names = [name for name,index in sorted(move_ids.items(),key=lambda x:x[1])] + ['struggle','recharge']
        moves = (MoveInfo*len(names))()
        for target, name in zip(moves,names):
            for field, _ in MoveInfo._fields_:
                setattr(target,field,records[name][field])
        values = {name:np.ascontiguousarray(metadata[name],dtype=np.int32)
                  for name in ('ability_tokens','item_tokens','type_tokens','status_tokens')}
        # Old retained wrappers predate appended native enum values. Extend
        # their public tokens from the frozen encoder's normalized string hash.
        from .rules import ABILITY_IDS,ITEM_IDS
        for field,kind,ids in (('ability_tokens','ability',ABILITY_IDS),('item_tokens','item',ITEM_IDS)):
            old=values[field];names=list(ids)
            if len(old)<len(names):
                values[field]=np.asarray([int(old[i])if i<len(old)else visible_token(kind,name)for i,name in enumerate(names)],dtype=np.int32)
        values['chart'] = np.ascontiguousarray(chart,dtype=np.int8)
        self.host_buffers = dict(species=species,moves=moves,**values)
        self.rules = PublicRules(C.addressof(species),len(species),C.addressof(moves),len(moves),len(moves)-2,len(moves)-1,
                                 *[values[name].ctypes.data for name in values])

    def observe(self, native):
        native.lib.pkn_public_rules_size.restype=C.c_size_t
        if native.lib.pkn_public_rules_size()!=C.sizeof(PublicRules):
            raise RuntimeError('Native public rules ABI mismatch')
        native.lib.pkn_observe.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(PublicRules)]
        out=np.empty((native.n,2,700),dtype=np.float64)
        native.lib.pkn_observe(native.ptr,out.ctypes.data,C.byref(self.rules))
        return out

    def to_cuda(self):
        import torch
        self.device_buffers={}
        for name,source in self.host_buffers.items():
            data = np.frombuffer(bytes(source),dtype=np.uint8).copy() if isinstance(source,C.Array) else source
            self.device_buffers[name]=torch.from_numpy(data).cuda()
        device=PublicRules.from_buffer_copy(bytes(self.rules))
        for name,value in self.device_buffers.items():
            setattr(device,name,value.data_ptr())
        self.device_rules=device
        return device
