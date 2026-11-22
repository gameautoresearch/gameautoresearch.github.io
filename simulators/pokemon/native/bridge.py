"""Developer interface for the new native kernel. Full-task admission is closed."""
from __future__ import annotations
import ctypes as C
import hashlib,json,platform,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
class Move(C.Structure):
 _fields_=[(x,C.c_int32) for x in ('id','power','accuracy','priority','type','category','maxpp')]+[('effects',C.c_uint32),('unsupported',C.c_uint32),('stat_delta',C.c_int32*7)]+[(x,C.c_int32) for x in ('target_self','heal_num','heal_den','recoil_num','recoil_den','secondary_status','secondary_chance','drain_num','drain_den')]+[('secondary_stat_delta',C.c_int32*7),('secondary_self',C.c_int32),('protectable',C.c_int32)]+[(x,C.c_int32) for x in ('crit_ratio','always_crit','fixed_damage','ignore_immunity')]+[('self_stat_delta',C.c_int32*7),('self_chance',C.c_int32)]+[(x,C.c_int32) for x in ('hit_min','hit_max','multi_accuracy')]+[('after_stat_delta',C.c_int32*7)]+[(x,C.c_int32) for x in ('kind','traits','stellar_boosted','volatile_status','secondary_volatile','weather','side_condition','terrain','attack_stat','defense_stat','attack_target','defense_source','ignore_boosts','pressure_target','additional_flinch_chance')]
class MonForm(C.Structure):
 _fields_=[('species',C.c_int32),('type',C.c_int32*2),('stats',C.c_int32*5)]+[(x,C.c_int32)for x in ('maxhp','base_attack','weight_hg','base_speed')]
class Mon(C.Structure):
 _fields_=[(x,C.c_int32) for x in ('species','ability','item','level')]+[('type',C.c_int32*2)]+[(x,C.c_int32) for x in ('tera_type','hp','maxhp')]+[('stats',C.c_int32*5),('boosts',C.c_int32*7)]+[(x,C.c_int32*4) for x in ('move_ids','pp','maxpp','disabled')]+[('move_categories',C.c_int8*4),('move_traits',C.c_uint32*4)]+[(x,C.c_int16) for x in ('status','status_turns','toxic_turns','terastallized','fainted','choice_lock','protect','stall_counter','stall_duration')]+[('unsupported',C.c_uint32),('speed_cache',C.c_int32),('stellar_used',C.c_uint32),('sleep_source',C.c_int32),('confusion',C.c_int32),('flinch',C.c_int32)]+[(x,C.c_int32) for x in ('taunt','encore','encore_move','disable','disable_move','last_move','active_turns','leech_seed','yawn','yawn_source','substitute','times_attacked','roost','nfe','paradox_stat','paradox_booster','booster_started','paradox_species','ability_ending','weight_hg','stats_raised','heal_block','throat_chop','stats_lowered','newly_switched','hit_by_foe','species_flags','locked_duration','locked_true','locked_move','active_move_actions')]+[('base_types',C.c_int32*2),('protean_used',C.c_int32),('curse',C.c_int32),('salt_cure',C.c_int32),('last_result',C.c_int32),('current_result',C.c_int32),('flash_fire',C.c_int32),('once_boost',C.c_int32),('volatile_order',C.c_int32*64),('volatile_serial',C.c_int32),('suppressed_ability',C.c_int32)]+[(x,C.c_int32) for x in ('charge_duration','charge_move','charge_kind','charge_released','faint_source_side','faint_source_slot','unburden','last_item','unnerved','slow_start','supreme_fallen','no_retreat','trapped_volatile','magnet_rise','glaive_rush','maybe_trapped','truant','recharge','trap_source','partial_trap','partial_source','charged','block_heal_berry','chilly_reception','focus_punch','beak_blast','bond_triggered','destiny_bond','mirror_coat','mirror_damage','base_attack','inherited_substitute','base_ability','trace_seek','gender','attract_source','cud_berry','cud_counter','form_family','form_current','form_permanent','form_count','form_busted','form_regression','original_ability','embodied','tera_resisted','base_speed','iv_spe','ev_spe','illusion_slot','known_type','got_attacked','trapping_flags','original_trapping_flags')]+[('forms',MonForm*3)]+[(x,C.c_int32)for x in ('transformed','original_species','original_weight')]+[('original_types',C.c_int32*2),('original_stats',C.c_int32*5)]+[(x,C.c_int32*4)for x in ('original_moves','original_pp','original_maxpp')]+[('original_categories',C.c_int8*4),('original_traits',C.c_uint32*4),('original_species_flags',C.c_int32)]
class QueuedAction(C.Structure):
 _fields_=[(x,C.c_int32) for x in ('kind','side','actor','action','order','priority','speed','sub_order','effect_order','move_id','fractional_priority')]
class PublicMon(C.Structure):
 _fields_=[(x,C.c_int32) for x in ('hp','known_hp','fainted','status','tera','ability','item','details_known')]+[('moves',C.c_int32*4)]+[(x,C.c_int32) for x in ('move_count','last_move')]+[('boosts',C.c_int32*7),('species',C.c_int32),('level',C.c_int32),('confusion',C.c_int32),('taunt',C.c_int32),('substitute',C.c_int32),('tera_type',C.c_int32)]
class RequestCache(C.Structure):
 _fields_=[('order',C.c_int32*6)]+[(x,C.c_int32) for x in ('actor','active','waiting','forced','can_tera','trapped','maybe_trapped')]+[(x,C.c_int32*4) for x in ('moves','pp','maxpp','disabled')]+[(x,C.c_int32*6) for x in ('hp','status','tera','appended_move','abilities','species')]+[('stats',(C.c_int32*5)*6),('mon_moves',(C.c_int32*4)*6),('levels',C.c_int32*6),('hp_ratio',C.c_double*6),('unannounced_item_loss',C.c_int32*6)]
class State(C.Structure):
 _fields_=[('rng',C.c_uint64),('team_rng',C.c_uint64*2),('mons',(Mon*6)*2),('order',(C.c_int32*6)*2)]+[(x,C.c_int32*2) for x in ('active','reviving','forced','trapped','tera_used')]+[('revealed',(C.c_int32*6)*2),('move_revealed',((C.c_int32*4)*6)*2),('pivot',C.c_int32*2)]+[(x,C.c_int32) for x in ('phases','turn','limit','terminated','truncated','winner','fault')]+[('queue',QueuedAction*64),('queue_length',C.c_int32),('mid_turn',C.c_int32),('pending_recoil',C.c_int32*2),('last_faint_side',C.c_int32),('public_mons',(PublicMon*6)*2),('public_order',(C.c_int32*6)*2),('public_count',C.c_int32*2),('requests',RequestCache*2),('weather',C.c_int32),('weather_duration',C.c_int32),('side_conditions',(C.c_int32*10)*2),('side_durations',(C.c_int32*10)*2),('side_effect_order',(C.c_int32*10)*2),('effect_order',C.c_int32),('terrain',C.c_int32),('terrain_duration',C.c_int32),('visible_terrain',C.c_int32),('total_fainted',C.c_int32*2)]+[(x,C.c_int32*2) for x in ('wish_turn','wish_hp','wish_pending','used_move','used_move_hit','after_move','drag')]+[('trick_room',C.c_int32),('ignore_ability_side',C.c_int32),('pending_mask',C.c_int32),('pending_actions',C.c_int32*2),('unavailable_choices',C.c_int32*2),('last_successful_move_kind',C.c_int32),('visible_side_conditions',(C.c_int32*10)*2),('shed_tail',C.c_int32*2),('healing_wish',C.c_int32*2)]+[(x,C.c_int32*2)for x in ('future_move','future_turn','future_source','future_slot','future_order','public_active')]

def build(out,*,cpu_chunk=None,admission=None):
 if cpu_chunk is not None and cpu_chunk not in (0,1,4,8,16,32):raise ValueError('Unsupported diagnostic CPU worker chunk')
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 library=out/('libpokemon_native.dylib' if platform.system()=='Darwin' else 'libpokemon_native.so')
 paths=sorted(p for p in HERE.iterdir() if p.suffix in ('.cpp','.hpp','.cu','.py','.cjs'))
 payloads={p.name:p.read_bytes() for p in paths}
 sources={name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}
 certificate=None
 if admission is not None:
  from .admission import verify_certificate
  certificate=verify_certificate(json.loads(Path(admission).read_bytes()),sources)
 snapshot=out/'source';snapshot.mkdir()
 for name,data in payloads.items():
  target=snapshot/name;target.write_bytes(data);target.chmod(0o444)
 command=['c++','-std=c++17','-O3','-ffp-contract=off','-fPIC','-shared','-pthread',str(snapshot/'engine.cpp'),'-o',str(library)]
 if certificate is not None:command.insert(1,'-DPKN_FULL_GEN9_ADMITTED=1')
 if cpu_chunk is not None:command.insert(1,f'-DPKN_CPU_CHUNK={cpu_chunk}')
 subprocess.run(command,check=True)
 manifest=dict(source_sha256=sources,command=command,binary_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),gen9_complete=certificate is not None)
 if certificate is not None:
  receipt=out/'admission.json';receipt.write_text(json.dumps(certificate,indent=2)+'\n');receipt.chmod(0o444);manifest['admission_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
 (out/'build.json').write_text(json.dumps(manifest,indent=2)+'\n');return library

class Native:
 def __init__(self,library,n=1,threads=0):
  self.binary_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest()
  manifest_path=Path(library).parent/'build.json'
  if manifest_path.exists():
   manifest=json.loads(manifest_path.read_text())
   if manifest['binary_sha256']!=self.binary_sha256:raise ValueError('Native build binary changed')
   compiled_sources=manifest['source_sha256']
  else:
   # Ad hoc diagnostic libraries have no production admission or resume rights.
   compiled_sources={'diagnostic_binary':self.binary_sha256}
  self.build_manifest=manifest if manifest_path.exists()else {}
  runtime_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.cpp','.hpp','.cu','.py','.cjs')}
  self.runtime_sources=runtime_sources
  self.source_signature=hashlib.sha256(json.dumps(dict(version=2,compiled=compiled_sources,runtime=runtime_sources),sort_keys=True).encode()).hexdigest()
  self.lib=C.CDLL(str(Path(library).resolve()));l=self.lib
  l.pkn_create.argtypes=[C.c_int,C.c_int];l.pkn_create.restype=C.c_void_p
  l.pkn_destroy.argtypes=[C.c_void_p];l.pkn_states.argtypes=[C.c_void_p];l.pkn_states.restype=C.c_void_p
  for name,cls in [('pkn_state_size',State),('pkn_mon_size',Mon),('pkn_move_size',Move)]:
   f=getattr(l,name);f.restype=C.c_size_t
   if f()!=C.sizeof(cls):raise RuntimeError(f'Native ABI mismatch: {name}')
  l.pkn_step.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
  l.pkn_masks.argtypes=[C.c_void_p,C.c_void_p]
  l.pkn_init_public.argtypes=[C.c_void_p]
  l.pkn_random.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int]
  self.ptr=l.pkn_create(n,threads)
  if not self.ptr:raise RuntimeError('Native arena allocation failed')
  self.n=n;self.records=(State*n).from_address(l.pkn_states(self.ptr))
  for s in self.records:
   s.winner=-1;s.last_faint_side=-1;s.limit=512;s.fault=2
   for side in range(2):
    for i in range(6):
     s.order[side][i]=i;s.mons[side][i].level=100
     s.mons[side][i].choice_lock=-1;s.mons[side][i].last_move=-1
     for j in range(4):s.mons[side][i].move_ids[j]=-1
 def require_full_task(self):
  if not self.lib.pkn_gen9_complete()or not self.build_manifest.get('gen9_complete'):raise RuntimeError('Native Gen 9 mechanics admission is incomplete; production use is forbidden')
  if self.runtime_sources!=self.build_manifest['source_sha256']:raise ValueError('Native runtime source changed after admission')
  from .admission import verify_certificate
  receipt_path=Path(self.lib._name).parent/'admission.json'
  if hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=self.build_manifest['admission_sha256']:raise ValueError('Native admission receipt changed')
  self.admission=verify_certificate(json.loads(receipt_path.read_bytes()),self.build_manifest['source_sha256'])
 def initialize_public(self):self.lib.pkn_init_public(self.ptr)
 def masks(self):
  x=np.zeros((self.n,2,14),dtype=np.uint8);self.lib.pkn_masks(self.ptr,x.ctypes.data);return x
 def step(self,actions,moves,chart):
  a=np.ascontiguousarray(actions,dtype=np.int32);t=np.ascontiguousarray(chart,dtype=np.int8)
  if a.shape!=(self.n,2):raise ValueError('Expected [games,2] actions')
  if t.shape!=(18,18):raise ValueError('Expected [18,18] type chart')
  if not isinstance(moves,C.Array) or moves._type_ is not Move or not len(moves):raise ValueError('Expected nonempty native Move array')
  if not np.all(np.isin(t,[-127,-1,0,1])):raise ValueError('Invalid type chart exponent')
  self.lib.pkn_step(self.ptr,a.ctypes.data,C.addressof(moves),len(moves),t.ctypes.data)
 def snapshot(self):return C.string_at(self.lib.pkn_states(self.ptr),C.sizeof(State)*self.n)
 def restore(self,data):
  if len(data)!=C.sizeof(State)*self.n:raise ValueError('Native snapshot size mismatch')
  C.memmove(self.lib.pkn_states(self.ptr),data,len(data))
 def save_checkpoint(self,path,rules_sha256):
  path=Path(path);data=self.snapshot()
  metadata=dict(source_signature=self.source_signature,binary_sha256=self.binary_sha256,rules_sha256=rules_sha256,state_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),gen9_complete=False)
  path.write_bytes(data);path.with_suffix(path.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
 def load_checkpoint(self,path,rules_sha256):
  path=Path(path);data=path.read_bytes();m=json.loads(path.with_suffix(path.suffix+'.json').read_text())
  if m.get('source_signature')!=self.source_signature or m['binary_sha256']!=self.binary_sha256 or m['rules_sha256']!=rules_sha256:raise ValueError('Native executable/rules checkpoint signature changed')
  if m['state_sha256']!=hashlib.sha256(data).hexdigest():raise ValueError('Native checkpoint state hash mismatch')
  self.restore(data)
 def close(self):
  if self.ptr:self.lib.pkn_destroy(self.ptr);self.ptr=None
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
