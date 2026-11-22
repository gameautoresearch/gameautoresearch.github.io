// SPDX-License-Identifier: MIT
// New packed simulation kernel. The pinned Showdown runtime is never called.
// Admission remains closed until the full Gen 9 catalog is implemented/verified.
#pragma once
#include <cstdint>
#include <cstring>
#ifdef PKN_TRACE_RANDOM
#include <cstdio>
#endif
#ifdef __CUDACC__
#define PK_HD __host__ __device__
#else
#define PK_HD
#endif
#include "ability_flags.hpp"
namespace pkn {
constexpr int TEAM=6,MOVES=4,ACTIONS=14;
constexpr int STELLAR=18;
enum MoveKind:int32_t { ORDINARY_MOVE=0,TOXIC_MOVE=1,ACROBATICS_MOVE=2,FACADE_MOVE=3,TERA_BLAST_MOVE=4,RAIN_ACCURACY_MOVE=5,SNOW_ACCURACY_MOVE=6,WEATHER_BALL_MOVE=7,HYDRO_STEAM_MOVE=8,SPIN_MOVE=9,DEFOG_MOVE=10,TIDY_UP_MOVE=11,BOOST_POWER_MOVE=12,HP_POWER_MOVE=13,HEX_MOVE=14,VENOSHOCK_MOVE=15,BRINE_MOVE=16,KNOCK_OFF_MOVE=17,POLTERGEIST_MOVE=18,RAGE_FIST_MOVE=19,LAST_RESPECTS_MOVE=20,ROOST_MOVE=21,REST_MOVE=22,WEATHER_HEAL_MOVE=23,SUCKER_PUNCH_MOVE=24,WISH_MOVE=25,SLEEP_TALK_MOVE=26,HEAVY_SLAM_MOVE=27,WEIGHT_POWER_MOVE=28,FREEZE_DRY_MOVE=29,BREAK_SCREENS_MOVE=30,ICE_SPINNER_MOVE=31,HAZE_MOVE=32,PAIN_SPLIT_MOVE=33,HALF_HP_MOVE=34,ENDEAVOR_MOVE=35,SHORE_UP_MOVE=36,ALLURING_VOICE_MOVE=37,STRENGTH_SAP_MOVE=38,THROAT_CHOP_MOVE=39,RAMPAGE_MOVE=40,TRIPLE_AXEL_MOVE=41,FIRST_MOVE_ONLY=42,TRICK_MOVE=43,TRI_ATTACK_MOVE=44,DIRE_CLAW_MOVE=45,PARTING_SHOT_MOVE=46,PAYBACK_MOVE=47,AVALANCHE_MOVE=48,LASH_OUT_MOVE=49,PSYBLADE_MOVE=50,EXPANDING_FORCE_MOVE=51,SUPER_EFFECTIVE_POWER_MOVE=52,GRASSY_GLIDE_MOVE=53,STONE_AXE_MOVE=54,CEASELESS_EDGE_MOVE=55,TAKE_HEART_MOVE=56,REVELATION_DANCE_MOVE=57,CLEAR_SMOG_MOVE=58,HEAL_BELL_MOVE=59,BELLY_DRUM_MOVE=60,FILLET_AWAY_MOVE=61,CLANGOROUS_SOUL_MOVE=62,TELEPORT_MOVE=63,FICKLE_BEAM_MOVE=64,BURNING_JEALOUSY_MOVE=65,FAILED_POWER_MOVE=66,CRASH_MOVE=67,EXPLOSION_MOVE=68,TRICK_ROOM_MOVE=69,REVIVAL_BLESSING_MOVE=70,CURSE_MOVE=71,SALT_CURE_MOVE=72,SPARKLING_ARIA_MOVE=73,BLEAKWIND_STORM_MOVE=74,SHELL_SIDE_ARM_MOVE=75,DOUBLE_SHOCK_MOVE=76,SOLAR_BEAM_MOVE=77,METEOR_BEAM_MOVE=78,GLAIVE_RUSH_MOVE=79,NO_RETREAT_MOVE=80,MAGNET_RISE_MOVE=81,JUDGMENT_MOVE=82,PHOTON_GEYSER_MOVE=83,RAGING_BULL_MOVE=84,FUSION_FLARE_MOVE=85,FUSION_BOLT_MOVE=86,HYPERSPACE_FURY_MOVE=87,COURT_CHANGE_MOVE=88,RECHARGE_MOVE=89,SPIRIT_SHACKLE_MOVE=90,WATER_SHURIKEN_MOVE=91,CHILLY_RECEPTION_MOVE=92,FOCUS_PUNCH_MOVE=93,BEAK_BLAST_MOVE=94,BUG_BITE_MOVE=95,DESTINY_BOND_MOVE=96,SHED_TAIL_MOVE=97,HEALING_WISH_MOVE=98,MIRROR_COAT_MOVE=99,BEAT_UP_MOVE=100,AURA_WHEEL_MOVE=101,RELIC_SONG_MOVE=102,SURF_MOVE=103,TRANSFORM_MOVE=104,IVY_CUDGEL_MOVE=105,TERASTAR_STORM_MOVE=106,FUTURE_SIGHT_MOVE=107 };
enum Weather:int32_t { CLEAR_WEATHER=0,RAIN=1,SUN=2,SAND=3,SNOW=4,HAIL=5 };
enum Terrain:int32_t { NO_TERRAIN=0,ELECTRIC_TERRAIN=1,GRASSY_TERRAIN=2,MISTY_TERRAIN=3,PSYCHIC_TERRAIN=4 };
enum SideCondition:int32_t { STEALTH_ROCK=0,SPIKES=1,TOXIC_SPIKES=2,STICKY_WEB=3,REFLECT=4,LIGHT_SCREEN=5,AURORA_VEIL=6,TAILWIND=7,SAFEGUARD=8,MIST=9 };
enum Status:int16_t { NONE=0,BRN=1,PAR=2,SLP=3,FRZ=4,PSN=5,TOX=6,FNT=7 };
enum Fault:int32_t { OK=0,ILLEGAL=1,UNSUPPORTED=2,WAITING=3,SEED_EXHAUSTED=4 };
enum Effect:uint32_t { PROTECT=1,HEAL=2,DRAIN=4,RECOIL=8,STAT=16,STATUS=32,SECONDARY=64,PIVOT=128,SELF_STAT=256,VOLATILE=512,WEATHER=1024,SIDE_CONDITION=2048,TERRAIN=4096,FORCE_SWITCH=8192,TRICK_ROOM=16384 };
enum Volatile:int32_t { NO_VOLATILE=0,CONFUSION=1,FLINCH=2,TAUNT=3,ENCORE=4,DISABLE=5,LEECH_SEED=6,YAWN=7,SUBSTITUTE=8,HEAL_BLOCK=9,THROAT_CHOP=10,CURSE=11,SALT_CURE=12,NO_RETREAT=13,MAGNET_RISE=14,PARTIAL_TRAP=15 };
enum Ability:int32_t { NO_ABILITY=0,NATURAL_CURE=1,REGENERATOR=2,PRESSURE=3,HUGE_POWER=4,PURE_POWER=5,GUTS=6,LEVITATE=7,INTIMIDATE=8,QUICK_FEET=9,MARVEL_SCALE=10,SKILL_LINK=11,OWN_TEMPO=12,INNER_FOCUS=13,OBLIVIOUS=14,INSOMNIA=15,VITAL_SPIRIT=16,IMMUNITY=17,LIMBER=18,WATER_VEIL=19,MAGMA_ARMOR=20,SHIELD_DUST=21,SERENE_GRACE=22,SOUNDPROOF=23,BULLETPROOF=24,INFILTRATOR=25,DRIZZLE=26,DROUGHT=27,SAND_STREAM=28,SNOW_WARNING=29,SWIFT_SWIM=30,CHLOROPHYLL=31,SAND_RUSH=32,SLUSH_RUSH=33,SAND_VEIL=34,SNOW_CLOAK=35,RAIN_DISH=36,ICE_BODY=37,SOLAR_POWER=38,HYDRATION=39,AIR_LOCK=40,CLOUD_NINE=41,TECHNICIAN=42,THICK_FAT=43,ADAPTABILITY=44,TINTED_LENS=45,OVERGROW=46,BLAZE=47,TORRENT=48,SWARM=49,IRON_FIST=50,STRONG_JAW=51,TOUGH_CLAWS=52,FUR_COAT=53,ICE_SCALES=54,FILTER=55,SOLID_ROCK=56,NEUROFORCE=57,SNIPER=58,MULTISCALE=59,SHADOW_SHIELD=60,PROTOSYNTHESIS=61,QUARK_DRIVE=62,CLEAR_BODY=63,WHITE_SMOKE=64,FULL_METAL_BODY=65,DEFIANT=66,COMPETITIVE=67,CONTRARY=68,SIMPLE=69,SHEER_FORCE=70,STURDY=71,BATTLE_ARMOR=72,SHELL_ARMOR=73,WATER_ABSORB=74,VOLT_ABSORB=75,SAP_SIPPER=76,STORM_DRAIN=77,LIGHTNING_ROD=78,MOTOR_DRIVE=79,EARTH_EATER=80,WELL_BAKED_BODY=81,UNAWARE=82,PRANKSTER=83,OVERCOAT=84,STICKY_HOLD=85,HUSTLE=86,COMPOUND_EYES=87,SHARPNESS=88,MEGA_LAUNCHER=89,RECKLESS=90,DRAGONS_MAW=91,TRANSISTOR=92,ROCKY_PAYLOAD=93,STEELWORKER=94,FLARE_BOOST=95,TOXIC_BOOST=96,PROTEAN=97,LIBERO=98,SUCTION_CUPS=99,HEAVY_METAL=100,LIGHT_METAL=101,TRIAGE=102,GALE_WINGS=103,FLAME_BODY=104,STATIC=105,POISON_POINT=106,JUSTIFIED=107,STAMINA=108,WEAK_ARMOR=109,CURSED_BODY=110,SPEED_BOOST=111,STAKEOUT=112,ROCK_HEAD=113,HEATPROOF=114,FLUFFY=115,WATER_BUBBLE=116,PUNK_ROCK=117,PURIFYING_SALT=118,EARLY_BIRD=119,LEAF_GUARD=120,SHED_SKIN=121,FRISK=122,FLASH_FIRE=123,SYNCHRONIZE=124,THERMAL_EXCHANGE=125,WATER_COMPACTION=126,RATTLED=127,TOXIC_CHAIN=128,POISON_TOUCH=129,AFTERMATH=130,ROUGH_SKIN=131,IRON_BARBS=132,WIND_RIDER=133,PSYCHIC_SURGE=134,GRASSY_SURGE=135,ELECTRIC_SURGE=136,MISTY_SURGE=137,DOWNLOAD=138,DAUNTLESS_SHIELD=139,INTREPID_SWORD=140,NO_GUARD=141,POWER_SPOT=142,PLUS=143,MINUS=144,MOLD_BREAKER=145,TERAVOLT=146,TURBOBLAZE=147,SCRAPPY=148,MINDS_EYE=149,KEEN_EYE=150,BIG_PECKS=151,PRISM_ARMOR=152,SURGE_SURFER=153,LIQUID_VOICE=154,PIXILATE=155,GALVANIZE=156,CORROSION=157,UNSEEN_FIST=158,GOOD_AS_GOLD=159,QUEENLY_MAJESTY=160,ANALYTIC=161,GOOEY=162,TANGLING_HAIR=163,TOXIC_DEBRIS=164,SEED_SOWER=165,AROMA_VEIL=166,FLOWER_VEIL=167,POISON_HEAL=168,DRY_SKIN=169,SUPER_LUCK=170,MAGIC_GUARD=171,MOXIE=172,CHILLING_NEIGH=173,GRIM_NEIGH=174,SOUL_HEART=175,UNBURDEN=176,CHEEK_POUCH=177,UNNERVE=178,HARVEST=179,COMATOSE=180,SLOW_START=181,BAD_DREAMS=182,SUPREME_OVERLORD=183,LIQUID_OOZE=184,BEADS_OF_RUIN=185,TABLETS_OF_RUIN=186,VESSEL_OF_RUIN=187,SWORD_OF_RUIN=188,MULTITYPE=189,ARENA_TRAP=190,MAGNET_PULL=191,SHADOW_TAG=192,SAND_FORCE=193,DAMP=194,MIRROR_ARMOR=195,POISON_PUPPETEER=196,HADRON_ENGINE=197,ORICHALCUM_PULSE=198,AS_ONE_GLASTRIER=199,AS_ONE_SPECTRIER=200,TRUANT=201,MYCELIUM_MIGHT=202,EFFECT_SPORE=203,ELECTROMORPHOSIS=204,BERSERK=205,ANGER_SHELL=206,BATTLE_BOND=207,MAGIC_BOUNCE=208,MAGICIAN=209,PICKPOCKET=210,TRACE=211,CUTE_CHARM=212,CUD_CHEW=213,DANCER=214,DISGUISE=215,ICE_FACE=216,HUNGER_SWITCH=217,ZERO_TO_HERO=218,GULP_MISSILE=219,SHIELDS_DOWN=220,IMPOSTER=221,TERA_SHIFT=222,TERA_SHELL=223,TERAFORM_ZERO=224,EMBODY_TEAL=225,EMBODY_WELLSPRING=226,EMBODY_HEARTHFLAME=227,EMBODY_CORNERSTONE=228,ILLUSION=229 };
enum Item:int32_t { NO_ITEM=0,LEFTOVERS=1,CHOICE_BAND=2,CHOICE_SPECS=3,CHOICE_SCARF=4,FOCUS_SASH=5,SITRUS_BERRY=6,LUM_BERRY=7,LOADED_DICE=8,LIFE_ORB=9,ASSAULT_VEST=10,DAMP_ROCK=11,HEAT_ROCK=12,SMOOTH_ROCK=13,ICY_ROCK=14,LIGHT_CLAY=15,HEAVY_DUTY_BOOTS=16,TERRAIN_EXTENDER=17,CHESTO_BERRY=18,EVIOLITE=19,WIDE_LENS=20,BOOSTER_ENERGY=21,SAFETY_GOGGLES=22,FLOAT_STONE=23,WHITE_HERB=24,FLAME_ORB=25,TOXIC_ORB=26,WEAKNESS_POLICY=27,ROCKY_HELMET=28,LIGHT_BALL=29,SOUL_DEW=30,SILK_SCARF=31,CHARCOAL=32,MYSTIC_WATER=33,MIRACLE_SEED=34,MAGNET=35,NEVER_MELT_ICE=36,BLACK_BELT=37,POISON_BARB=38,SOFT_SAND=39,SHARP_BEAK=40,TWISTED_SPOON=41,SILVER_POWDER=42,HARD_STONE=43,SPELL_TAG=44,DRAGON_FANG=45,METAL_COAT=46,BLACK_GLASSES=47,FAIRY_FEATHER=48,AIR_BALLOON=49,SCOPE_LENS=50,POWER_HERB=51,LEPPA_BERRY=52,THROAT_SPRAY=53,FLAME_PLATE=54,SPLASH_PLATE=55,MEADOW_PLATE=56,ZAP_PLATE=57,ICICLE_PLATE=58,FIST_PLATE=59,TOXIC_PLATE=60,EARTH_PLATE=61,SKY_PLATE=62,MIND_PLATE=63,INSECT_PLATE=64,STONE_PLATE=65,SPOOKY_PLATE=66,DRACO_PLATE=67,IRON_PLATE=68,DREAD_PLATE=69,PIXIE_PLATE=70,GRISEOUS_CORE=71,LUSTROUS_GLOBE=72,ADAMANT_CRYSTAL=73,LUSTROUS_ORB=74,ADAMANT_ORB=75,GRISEOUS_ORB=76,RUSTED_SWORD=77,RUSTED_SHIELD=78,WELLSPRING_MASK=79,HEARTHFLAME_MASK=80,CORNERSTONE_MASK=81,EXPERT_BELT=82,CUSTAP_BERRY=83 };
enum Event:int32_t { UPDATE=1,BEFORE_TURN_EVENT=2,SPEED_ORDER_ONLY=3,WEATHER_CHANGE=4,TERRAIN_CHANGE=5 };
struct Move {
  int32_t id,power,accuracy,priority,type,category,maxpp;
  uint32_t effects,unsupported;
  int32_t stat_delta[7],target_self,heal_num,heal_den,recoil_num,recoil_den;
  int32_t secondary_status,secondary_chance,drain_num,drain_den;
  int32_t secondary_stat_delta[7],secondary_self;
  int32_t protectable;
  int32_t crit_ratio,always_crit,fixed_damage,ignore_immunity,self_stat_delta[7],self_chance;
  int32_t hit_min,hit_max,multi_accuracy,after_stat_delta[7];
  int32_t kind,traits,stellar_boosted;
  int32_t volatile_status,secondary_volatile;
  int32_t weather;
  int32_t side_condition;
  int32_t terrain;
  int32_t attack_stat,defense_stat,attack_target,defense_source,ignore_boosts,pressure_target,additional_flinch_chance;
};
struct MonForm {int32_t species,type[2],stats[5],maxhp,base_attack,weight_hg,base_speed;};
struct Mon {
  int32_t species,ability,item,level,type[2],tera_type,hp,maxhp,stats[5],boosts[7];
  int32_t move_ids[4],pp[4],maxpp[4],disabled[4];
  int8_t move_categories[4];
  uint32_t move_traits[4];
  int16_t status,status_turns,toxic_turns,terastallized,fainted,choice_lock,protect,stall_counter,stall_duration;
  uint32_t unsupported;
  int32_t speed_cache;
  uint32_t stellar_used;
  int32_t sleep_source;
  int32_t confusion,flinch;
  int32_t taunt,encore,encore_move,disable,disable_move,last_move,active_turns;
  int32_t leech_seed,yawn,yawn_source,substitute;
  int32_t times_attacked,roost,nfe;
  int32_t paradox_stat,paradox_booster,booster_started,paradox_species,ability_ending;
  int32_t weight_hg,stats_raised,heal_block,throat_chop;
  int32_t stats_lowered,newly_switched,hit_by_foe,species_flags;
  int32_t locked_duration,locked_true,locked_move,active_move_actions;
  int32_t base_types[2],protean_used;
  int32_t curse,salt_cure;
  int32_t last_result,current_result,flash_fire,once_boost;
  int32_t volatile_order[64],volatile_serial,suppressed_ability;
  int32_t charge_duration,charge_move,charge_kind,charge_released;
  int32_t faint_source_side,faint_source_slot;
  int32_t unburden,last_item,unnerved,slow_start,supreme_fallen;
  int32_t no_retreat,trapped_volatile,magnet_rise,glaive_rush,maybe_trapped,truant,recharge,trap_source,partial_trap,partial_source,charged,block_heal_berry,chilly_reception,focus_punch,beak_blast,bond_triggered,destiny_bond,mirror_coat,mirror_damage,base_attack,inherited_substitute,base_ability,trace_seek,gender,attract_source,cud_berry,cud_counter,form_family,form_current,form_permanent,form_count,form_busted,form_regression,original_ability,embodied,tera_resisted,base_speed,iv_spe,ev_spe,illusion_slot,known_type,got_attacked,trapping_flags,original_trapping_flags;
  MonForm forms[3];
  int32_t transformed,original_species,original_weight,original_types[2],original_stats[5];
  int32_t original_moves[4],original_pp[4],original_maxpp[4];int8_t original_categories[4];uint32_t original_traits[4];
  int32_t original_species_flags; // Transform takes the target's species; Raging Bull's and Hyperspace Fury's checks follow it
};
enum ActionKind:int32_t { BEFORE_TURN=1,SWITCH=2,RUN_SWITCH=3,TERA=4,MOVE=5,RESIDUAL=6,REVIVE=7,PREPARE_MOVE=8 };
constexpr int QUEUE_CAPACITY=64;
struct QueuedAction {int32_t kind,side,actor,action,order,priority,speed,sub_order,effect_order,move_id,fractional_priority;};
struct PublicMon {
  int32_t hp,known_hp,fainted,status,tera,ability,item,details_known;
  int32_t moves[4],move_count,last_move,boosts[7],species,level;
  int32_t confusion,taunt,substitute,tera_type;
};
struct RequestCache {
  int32_t order[6],actor,active,waiting,forced,can_tera,trapped,maybe_trapped;
  int32_t moves[4],pp[4],maxpp[4],disabled[4];
  int32_t hp[6],status[6],tera[6],appended_move[6],abilities[6],species[6],stats[6][5],mon_moves[6][4],levels[6];double hp_ratio[6];
  // An item taken with no message naming its holder (Magician) still shows in that player's own view
  // until the next request; 0 when the live item is what the player sees.
  int32_t unannounced_item_loss[6];
};
struct State {
  uint64_t rng,team_rng[2];
  Mon mons[2][6];
  int32_t order[2][6],active[2],reviving[2],forced[2],trapped[2];
  int32_t tera_used[2],revealed[2][6],move_revealed[2][6][4],pivot[2];
  int32_t phases,turn,limit,terminated,truncated,winner,fault;
  QueuedAction queue[QUEUE_CAPACITY];
  int32_t queue_length,mid_turn;
  int32_t pending_recoil[2],last_faint_side;
  PublicMon public_mons[2][6];
  int32_t public_order[2][6],public_count[2];
  RequestCache requests[2];
  int32_t weather,weather_duration;
  int32_t side_conditions[2][10],side_durations[2][10];
  int32_t side_effect_order[2][10],effect_order;
  int32_t terrain,terrain_duration,visible_terrain;
  int32_t total_fainted[2];
  int32_t wish_turn[2],wish_hp[2],wish_pending[2];
  int32_t used_move[2],used_move_hit[2],after_move[2],drag[2];
  int32_t trick_room,ignore_ability_side,pending_mask,pending_actions[2],unavailable_choices[2];
  int32_t last_successful_move_kind,visible_side_conditions[2][10],shed_tail[2],healing_wish[2],future_move[2],future_turn[2],future_source[2],future_slot[2],future_order[2],public_active[2];
};
struct Rules {
  const Move* moves;int32_t move_count;
  // 18x18 exponents, -127 for immune. Indexed by the frozen data compiler.
  const int8_t* chart;
};
// JS volatile insertion order affects the selection-sort swaps even when the
// callbacks themselves have different priorities. Preserve it independently.
PK_HD inline bool volatile_present(const Mon& p,int id){
  switch(id){case 1:return p.confusion;case 2:return p.flinch;case 3:return p.taunt;case 4:return p.encore;case 5:return p.disable;case 6:return p.leech_seed;case 7:return p.yawn;case 8:return p.substitute;case 9:return p.heal_block;case 10:return p.throat_chop;case 11:return p.protect;case 12:return p.stall_duration;case 13:return p.roost;case 14:return p.locked_duration;case 15:return p.paradox_stat&&p.ability==PROTOSYNTHESIS;case 16:return p.paradox_stat&&p.ability==QUARK_DRIVE;case 17:return p.choice_lock>=0;case 18:return p.flash_fire;case 19:return p.curse;case 20:return p.salt_cure;case 21:return p.charge_duration;case 22:return p.charge_duration&&!p.charge_released&&p.charge_kind==SOLAR_BEAM_MOVE;case 23:return p.charge_duration&&!p.charge_released&&p.charge_kind==METEOR_BEAM_MOVE;case 24:return p.unburden;case 25:return p.glaive_rush;case 26:return p.no_retreat;case 27:return p.trapped_volatile;case 28:return p.magnet_rise;case 29:return p.truant;case 30:return p.recharge;case 31:return false;case 32:return p.partial_trap;case 34:return p.charged;case 33:return p.chilly_reception;case 35:return p.focus_punch;case 36:return p.beak_blast;case 37:return p.mirror_coat;case 38:return p.destiny_bond;case 39:return p.attract_source;default:return false;}
}
PK_HD inline void clean_volatile_order(Mon& p){for(int i=1;i<64;i++)if(!volatile_present(p,i))p.volatile_order[i]=0;}
PK_HD inline void mark_volatile(Mon& p,int id){clean_volatile_order(p);if(!p.volatile_order[id])p.volatile_order[id]=++p.volatile_serial;}
PK_HD inline bool has_type(const Mon& p,int type);
PK_HD inline bool trapped(const State& s,int side){const Mon& p=s.mons[side][s.active[side]];return s.trapped[side]==1||((p.trapped_volatile||(p.partial_trap&&s.active[1-side]+1==p.partial_source))&&!has_type(p,13));}
PK_HD inline uint32_t next(State& s) {s.rng=s.rng*0x5D588B656C078965ULL+0x269EC3ULL;return uint32_t(s.rng>>32);}
#ifdef PKN_TRACE_RANDOM
inline uint32_t random(State& s,uint32_t n,const char* caller=__builtin_FUNCTION(),int line=__builtin_LINE()) {
  uint32_t result=uint32_t((uint64_t(next(s))*n)>>32);
  std::fprintf(stderr,"PKN_RANDOM %d %u %u %llu %s %d\n",s.phases,n,result,(unsigned long long)s.rng,caller,line);return result;
}
#else
PK_HD inline uint32_t random(State& s,uint32_t n) {return uint32_t((uint64_t(next(s))*n)>>32);}
#endif
PK_HD inline int minimum(int a,int b){return a<b?a:b;}
PK_HD inline int maximum(int a,int b){return a>b?a:b;}
PK_HD inline int indirect_loss(Mon& p,int amount){if(p.ability==MAGIC_GUARD)return 0;int loss=minimum(p.hp,amount);p.hp-=loss;return loss;}
PK_HD inline int round_ratio(int value,int num,int den){return (value*num*2+den)/(den*2);}
PK_HD inline int chain_modifier(int value,int next){return int((int64_t(value)*next+2048)/4096);}
PK_HD inline int modify(int value,int modifier){return int((int64_t(value)*modifier+2047)/4096);}
PK_HD inline int plate_type(int item){return item>=FLAME_PLATE&&item<=PIXIE_PLATE?item-FLAME_PLATE+1:-1;}
PK_HD inline bool can_take_item(const Mon& holder,const Mon& source){
  int flags=holder.species_flags|source.species_flags,item=holder.item;
  if(item>=WELLSPRING_MASK&&item<=CORNERSTONE_MASK&&(holder.species_flags&32768))return false;
  if((item==BOOSTER_ENERGY&&holder.paradox_species)||(plate_type(item)>=0&&(flags&4)))return false;
  return !((item==GRISEOUS_CORE&&(flags&32))||(item==LUSTROUS_GLOBE&&(flags&16))||(item==ADAMANT_CRYSTAL&&(flags&8))||(item==RUSTED_SWORD&&(flags&64))||(item==RUSTED_SHIELD&&(flags&128)));
}
PK_HD inline bool choice_item(const Mon& p){return p.item==CHOICE_BAND||p.item==CHOICE_SPECS||p.item==CHOICE_SCARF;}
PK_HD inline bool valid_choice_lock(const Mon& p){if(p.choice_lock<0)return false;for(int i=0;i<4;i++)if(p.move_ids[i]==p.choice_lock)return true;return false;}
PK_HD inline bool available(const Mon& p,int i){return p.move_ids[i]>=0&&p.pp[i]>0&&!p.disabled[i]&&(!(p.move_traits[i]&134217728)||p.last_move!=p.move_ids[i])&&((p.item!=ASSAULT_VEST&&!p.taunt)||p.move_categories[i]!=2)&&(!p.encore||p.encore_move==p.move_ids[i])&&(!p.disable||p.disable_move!=p.move_ids[i])&&(!p.heal_block||!(p.move_traits[i]&262144))&&(!p.throat_chop||!(p.move_traits[i]&16))&&(!choice_item(p)||!valid_choice_lock(p)||p.choice_lock==p.move_ids[i]);}
PK_HD inline int boost_stat(int stat,int stage) {
  stage=maximum(-6,minimum(6,stage));return stage>=0?stat*(2+stage)/2:stat*2/(2-stage);
}
PK_HD inline int effective_weather(const State& s){for(int side=0;side<2;side++){const Mon& p=s.mons[side][s.active[side]];if(p.hp>0&&!p.fainted&&!p.ability_ending&&(p.ability==AIR_LOCK||p.ability==CLOUD_NINE))return 0;}return s.weather;}
PK_HD inline int calculate_speed(const Mon& p,bool tailwind=false,int weather=0,bool trick_room=false,int terrain=0) {
  int x=boost_stat(p.stats[4],p.boosts[4]),modifier=4096;
  if(p.ability==QUICK_FEET&&p.status)modifier=modifier*3/2;
  if(p.item==CHOICE_SCARF)modifier=modifier*3/2;
  if(p.paradox_stat==5)modifier=chain_modifier(modifier,6144);
  if(tailwind)modifier*=2;
  if(p.unburden&&!p.item)modifier*=2;
  if(p.ability==SLOW_START&&p.slow_start)modifier=chain_modifier(modifier,2048);
  if(p.ability==SURGE_SURFER&&terrain==ELECTRIC_TERRAIN)modifier*=2;
  if((weather==RAIN&&p.ability==SWIFT_SWIM)||(weather==SUN&&p.ability==CHLOROPHYLL)||(weather==SAND&&p.ability==SAND_RUSH)||((weather==SNOW||weather==HAIL)&&p.ability==SLUSH_RUSH))modifier*=2;
  x=modify(x,modifier);if(p.status==PAR&&p.ability!=QUICK_FEET)x/=2;x=minimum(10000,maximum(1,x));return (trick_room?10000-x:x)&8191;
}
PK_HD inline int speed(const Mon& p){return p.speed_cache;}
PK_HD inline Mon& active(State& s,int side){return s.mons[side][s.active[side]];}
PK_HD inline const Mon& active(const State& s,int side){return s.mons[side][s.active[side]];}
// Ignore Ability lasts through move callbacks and immediate phazing. Keep the
// real ability for End callbacks and restore it before the post-action Update.
PK_HD inline void restore_transform(Mon& p){if(!p.transformed)return;p.species=p.original_species;p.species_flags=p.original_species_flags;p.weight_hg=p.original_weight;for(int i=0;i<2;i++)p.type[i]=p.base_types[i]=p.original_types[i];for(int i=0;i<5;i++)p.stats[i]=p.original_stats[i];for(int i=0;i<4;i++){p.move_ids[i]=p.original_moves[i];p.pp[i]=p.original_pp[i];p.maxpp[i]=p.original_maxpp[i];p.move_categories[i]=p.original_categories[i];p.move_traits[i]=p.original_traits[i];p.disabled[i]=0;}p.transformed=0;p.suppressed_ability=0;p.form_family=0;p.form_current=0;p.form_permanent=0;p.form_count=0;}
PK_HD inline void restore_abilities(State& s){for(int side=0;side<2;side++)for(int i=0;i<6;i++){Mon& p=s.mons[side][i];if(p.suppressed_ability&&!(p.transformed&&notransform_ability(p.suppressed_ability))){p.ability=p.suppressed_ability;p.suppressed_ability=0;}}s.ignore_ability_side=0;}
PK_HD inline void suppress_opponent(State& s){if(!s.ignore_ability_side)return;int actor=s.ignore_ability_side-1;if(active(s,actor).fainted||active(s,actor).hp<=0){restore_abilities(s);return;}Mon& q=active(s,1-actor);if(!q.suppressed_ability&&breakable_ability(q.ability)){q.suppressed_ability=q.ability;q.ability=NO_ABILITY;}}
PK_HD inline void boost_event_order(State& s){if(active(s,0).ability==UNAWARE&&active(s,1).ability==UNAWARE&&!active(s,0).fainted&&!active(s,1).fainted&&speed(active(s,0))==speed(active(s,1)))random(s,2);}
PK_HD inline void update_speeds(State& s){for(int side=0;side<2;side++)if(!active(s,side).fainted){boost_event_order(s);active(s,side).speed_cache=calculate_speed(active(s,side),s.side_conditions[side][TAILWIND],effective_weather(s),s.trick_room>0,s.terrain);}}
PK_HD inline bool struggling(const Mon& p){if(p.locked_duration||p.charge_duration||p.recharge)return false;for(int i=0;i<4;i++)if(available(p,i))return false;return true;}
PK_HD inline int public_hp(const Mon& p){int x=(p.hp*100+maximum(1,p.maxhp)-1)/maximum(1,p.maxhp);return x==100&&p.hp<p.maxhp?99:x;}
PK_HD inline void publish_active(State& s,int side,bool health=false){
  const Mon& p=active(s,side);PublicMon& v=s.public_mons[side][s.public_active[side]];
  if(health){v.hp=public_hp(p);v.known_hp=1;v.fainted=p.hp==0;int ps=s.public_active[side];s.requests[side].hp[ps]=p.hp;s.requests[side].hp_ratio[ps]=double(p.hp)/maximum(1,p.maxhp);}
  // The player's own view (keyed by the shown identity, so an Illusion's disguise) changes status only
  // with a message carrying it: an HP string, or an actual status change.
  int status=p.hp>0&&p.status!=FNT?p.status:NONE;if(health||v.status!=status)s.requests[side].status[s.public_active[side]]=status;v.status=status;

  if(p.hp>0){v.confusion=p.confusion>0;v.substitute=p.substitute>0&&!p.inherited_substitute;}
}
// Explicit protocol ability revelations update both channels, including a
// terminal private request cache that will never receive a fresh request.
PK_HD inline int publish_ability(State& s,int side,int slot,int ability){s.public_mons[side][slot].ability=ability;s.requests[side].abilities[slot]=ability;return ability;}
PK_HD inline bool change_form(State& s,int side,int form,bool permanent=false,bool publish=true){
  Mon& p=active(s,side);if(form<0||form>=p.form_count){s.fault=UNSUPPORTED;return false;}if(p.form_current==form)return false;
  const MonForm& f=p.forms[form];int old_maxhp=p.maxhp;p.species=f.species;p.type[0]=p.base_types[0]=f.type[0];p.type[1]=p.base_types[1]=f.type[1];
  for(int i=0;i<5;i++)p.stats[i]=f.stats[i];p.maxhp=f.maxhp;if(p.hp>0&&p.maxhp!=old_maxhp)p.hp=maximum(1,p.hp+p.maxhp-old_maxhp);p.speed_cache=p.stats[4];p.base_speed=f.base_speed;p.weight_hg=f.weight_hg;p.form_current=form;p.known_type=1;if(permanent)p.form_permanent=form;
  if(publish){s.public_mons[side][s.public_active[side]].details_known=1;s.public_mons[side][s.public_active[side]].species=p.species;s.requests[side].species[s.active[side]]=p.species;s.public_mons[side][s.public_active[side]].level=permanent?p.level:100;s.requests[side].levels[s.public_active[side]]=permanent?p.level:100;publish_active(s,side,p.maxhp!=old_maxhp);}return true;
}
PK_HD inline void begin_illusion(State& s,int side){Mon& p=active(s,side);p.illusion_slot=0;if(p.ability!=ILLUSION)return;int pos=0;for(int i=0;i<6;i++)if(s.order[side][i]==s.active[side])pos=i;for(int i=5;i>pos;i--){int slot=s.order[side][i];const Mon& q=s.mons[side][slot];if(q.maxhp>0&&!q.fainted){if(!p.terastallized||q.form_family<8)p.illusion_slot=slot+1;break;}}}
PK_HD inline void reveal_active(State& s,int side){
  const int slot=active(s,side).illusion_slot?active(s,side).illusion_slot-1:s.active[side];s.public_active[side]=slot;bool found=false;
  for(int i=0;i<s.public_count[side];i++)found|=s.public_order[side][i]==slot;
  if(!found&&s.public_count[side]<6)s.public_order[side][s.public_count[side]++]=slot;
  PublicMon& v=s.public_mons[side][slot];v.details_known=1;v.species=s.mons[side][slot].species;v.level=active(s,side).illusion_slot?s.mons[side][slot].level:pkn::active(s,side).level;for(int k=0;k<7;k++)v.boosts[k]=0;
  publish_active(s,side,true);
}
PK_HD inline void break_illusion(State& s,int side){
  Mon& p=active(s,side);if(!p.illusion_slot)return;const PublicMon previous=s.public_mons[side][s.public_active[side]];p.illusion_slot=0;int slot=s.active[side];s.public_active[side]=slot;
  bool found=false;for(int i=0;i<s.public_count[side];i++)found|=s.public_order[side][i]==slot;if(!found&&s.public_count[side]<6)s.public_order[side][s.public_count[side]++]=slot;
  PublicMon& v=s.public_mons[side][slot];v.hp=previous.hp;v.known_hp=previous.known_hp;v.fainted=previous.fainted;v.status=previous.status;for(int i=0;i<7;i++)v.boosts[i]=previous.boosts[i];v.confusion=previous.confusion;v.taunt=previous.taunt;v.substitute=previous.substitute;v.details_known=1;v.species=p.species;v.level=p.level;
}
PK_HD inline void publish_move(State& s,int side,int move){
  PublicMon& v=s.public_mons[side][s.public_active[side]];v.last_move=move;
  bool found=false;for(int i=0;i<v.move_count;i++)found|=v.moves[i]==move;
  if(!found&&v.move_count<4)v.moves[v.move_count++]=move;
  if(s.public_active[side]!=s.active[side]){int ps=s.public_active[side];bool known=false;for(int k=0;k<4;k++)known|=s.requests[side].mon_moves[ps][k]==move;if(!known)for(int k=0;k<4;k++)if(s.requests[side].mon_moves[ps][k]<0){s.requests[side].mon_moves[ps][k]=move;break;}}
  bool private_known=false;for(int k=0;k<4;k++)private_known|=active(s,side).move_ids[k]==move;
  if(!private_known)s.requests[side].appended_move[s.active[side]]=move;
}
PK_HD inline bool grounded(const Mon& p);
PK_HD inline void update_trapping(State& s){
  for(int side=0;side<2;side++){
    Mon& p=active(s,side);const Mon& foe=active(s,1-side);p.maybe_trapped=0;s.trapped[side]=0;
    if(p.hp<=0||foe.hp<=0||p.fainted||foe.fainted)continue;
    int actual=foe.ability==MAGNET_PULL?1:foe.ability==ARENA_TRAP?2:foe.ability==SHADOW_TAG?4:0;
    if(!has_type(p,13)&&(((actual&1)&&has_type(p,15))||((actual&2)&&grounded(p))||((actual&4)&&p.ability!=SHADOW_TAG)))s.trapped[side]=2;
    int flags=(foe.illusion_slot?s.mons[1-side][foe.illusion_slot-1].trapping_flags:foe.trapping_flags)|actual;
    bool maybe_grounded=grounded(p)||(!p.known_type&&has_type(p,9)&&p.ability!=LEVITATE&&p.item!=AIR_BALLOON&&!p.magnet_rise);
    if((!p.known_type||!has_type(p,13))&&(((flags&1)&&(!p.known_type||has_type(p,15)))||((flags&2)&&maybe_grounded)||((flags&4)&&p.ability!=SHADOW_TAG)))p.maybe_trapped=1;
  }
}
PK_HD inline void refresh_requests(State& s){
  for(int side=0;side<2;side++){
    RequestCache& r=s.requests[side];const Mon& p=active(s,side);
    for(int i=0;i<6;i++){r.unannounced_item_loss[i]=0;r.order[i]=s.order[side][i];r.levels[i]=s.mons[side][i].level;r.hp[i]=s.mons[side][i].hp;r.status[i]=s.mons[side][i].hp>0&&s.mons[side][i].status!=FNT?s.mons[side][i].status:NONE;r.hp_ratio[i]=double(s.mons[side][i].hp)/maximum(1,s.mons[side][i].maxhp);r.tera[i]=s.mons[side][i].terastallized;r.appended_move[i]=-1;r.abilities[i]=s.mons[side][i].suppressed_ability?s.mons[side][i].suppressed_ability:s.mons[side][i].ability;const Mon& q=s.mons[side][i];r.species[i]=q.transformed?q.original_species:q.form_count?q.forms[q.form_permanent].species:q.species;for(int j=0;j<5;j++)r.stats[i][j]=q.transformed?q.original_stats[j]:q.stats[j];for(int j=0;j<4;j++)r.mon_moves[i][j]=q.move_ids[j];}r.actor=s.active[side];
    r.forced=s.forced[side];r.active=!s.forced[0]&&!s.forced[1];
    r.waiting=!r.active&&!s.forced[side];r.can_tera=r.active&&!s.tera_used[side]&&!struggling(p)&&!p.locked_duration&&!p.charge_duration&&!p.recharge&&!(p.transformed&&p.form_family>=8);r.trapped=p.locked_duration>0||p.charge_duration>0||p.recharge>0;r.maybe_trapped=0;for(int i=0;i<6;i++)if(i!=s.active[side]&&s.mons[side][i].hp>0){if(trapped(s,side))r.trapped=1;if(!r.trapped)r.maybe_trapped=p.maybe_trapped;}
    for(int i=0;i<4;i++){r.moves[i]=-1;r.pp[i]=0;r.maxpp[i]=0;r.disabled[i]=1;}
    if(r.active){
      if(p.recharge){r.moves[0]=-4;r.disabled[0]=0;}
      else if(p.locked_duration||p.charge_duration){r.moves[0]=p.charge_duration?p.charge_move:p.locked_move;r.disabled[0]=0;}
      else if(struggling(p)){r.moves[0]=-2;r.disabled[0]=0;}
      else for(int i=0;i<4;i++){r.moves[i]=p.move_ids[i];r.pp[i]=p.pp[i];r.maxpp[i]=p.maxpp[i];r.disabled[i]=!available(p,i);}
    }
  }
}
PK_HD inline void initialize_public(State& s){
  for(int side=0;side<2;side++)for(int j=0;j<10;j++)s.visible_side_conditions[side][j]=s.side_conditions[side][j];
  for(int side=0;side<2;side++){
    for(int slot=0;slot<6;slot++)if(!s.mons[side][slot].protean_used){s.mons[side][slot].base_types[0]=s.mons[side][slot].type[0];s.mons[side][slot].base_types[1]=s.mons[side][slot].type[1];}
    for(int slot=0;slot<6;slot++)s.mons[side][slot].speed_cache=slot==s.active[side]?calculate_speed(s.mons[side][slot],s.side_conditions[side][TAILWIND],effective_weather(s),s.trick_room>0,s.terrain):s.mons[side][slot].stats[4];
    s.public_count[side]=0;
    for(int slot=0;slot<6;slot++){PublicMon& v=s.public_mons[side][slot];v={};v.last_move=-1;for(int k=0;k<4;k++)v.moves[k]=-1;}
    if(active(s,side).ability==ILLUSION)begin_illusion(s,side);reveal_active(s,side);
    if(active(s,side).ability==PRESSURE||active(s,side).ability==INTIMIDATE||active(s,side).ability==AIR_LOCK||active(s,side).ability==CLOUD_NINE)publish_ability(s,side,s.public_active[side],active(s,side).ability);
  }
  update_trapping(s);refresh_requests(s);
}
PK_HD inline void mask(const State& s,int side,uint8_t out[14]) {
  for(int a=0;a<14;a++)out[a]=0;
  if(s.terminated||s.truncated||s.fault||(s.pending_mask&(1<<side)))return;
  const Mon& p=active(s,side);
  if(!s.forced[side]&&!s.reviving[side]) {
    if(s.forced[1-side])return;
    if(p.locked_duration||p.charge_duration||p.recharge){out[0]=1;return;}
    bool has_pp=false;
    for(int i=0;i<4;i++)if(available(p,i))has_pp=true;
    if(!has_pp)out[0]=1; // The frozen Gen 9 request excludes Terastallizing Struggle.
    else for(int i=0;i<4;i++)if(available(p,i)) {
      out[i]=1;out[4+i]=!s.tera_used[side]&&!(p.transformed&&p.form_family>=8);
    }
  }
  for(int i=0;i<6;i++) {
    const int slot=s.order[side][i];const Mon& q=s.mons[side][slot];
    out[8+i]=q.maxhp>0&&(slot!=s.active[side])&&(s.reviving[side]?q.hp==0:q.hp>0)&&(s.forced[side]||s.reviving[side]||!trapped(s,side));
  }
}
PK_HD inline bool can_eat(const State& s,int side){const Mon& foe=active(s,1-side);if(active(s,side).item==SITRUS_BERRY&&active(s,side).block_heal_berry)return false;return !(foe.hp>0&&!foe.fainted&&(foe.ability==UNNERVE||foe.ability==AS_ONE_GLASTRIER||foe.ability==AS_ONE_SPECTRIER)&&foe.unnerved);}
PK_HD inline bool berry_item(int item){return item==SITRUS_BERRY||item==LUM_BERRY||item==CHESTO_BERRY||item==LEPPA_BERRY||item==CUSTAP_BERRY;}
PK_HD inline bool take_item(Mon& p,const Mon* source=nullptr){if(p.item&&p.hp>0&&p.ability==UNBURDEN){mark_volatile(p,24);p.unburden=1;}if(!can_take_item(p,source?*source:p))return false;p.item=NO_ITEM;return true;}
PK_HD inline void use_item(State& s,int side,bool berry=false){
  Mon& p=active(s,side);int item=p.item;if(!item)return;
  if(berry&&p.ability==CUD_CHEW){p.cud_berry=item;p.cud_counter=s.queue_length?2:1;}
  if(berry&&p.ability==CHEEK_POUCH&&!p.heal_block&&p.hp>0){int before=p.hp;p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/3));if(p.hp!=before)publish_active(s,side,true);}
  p.last_item=item;p.item=NO_ITEM;p.booster_started=0;s.public_mons[side][s.public_active[side]].item=0;
  if(p.ability==UNBURDEN){mark_volatile(p,24);p.unburden=1;}
}
PK_HD inline void pop_balloon(State& s,int side){int previous=active(s,side).last_item;use_item(s,side);active(s,side).last_item=previous;}
PK_HD inline void activate_paradox(State& s,int side,bool booster=false){
  Mon& p=active(s,side);if(p.paradox_stat||p.hp<=0||p.fainted)return;
  mark_volatile(p,p.ability==PROTOSYNTHESIS?15:16);
  int best=0;for(int i=1;i<5;i++)if(boost_stat(p.stats[i],p.boosts[i])>boost_stat(p.stats[best],p.boosts[best]))best=i;
  p.paradox_stat=best+1;p.paradox_booster=booster;
}
PK_HD inline void update_paradox_field(State& s,int side,int event){
  Mon& p=active(s,side);if(p.hp<=0||p.fainted)return;
  if((event==WEATHER_CHANGE&&p.ability==PROTOSYNTHESIS)||(event==TERRAIN_CHANGE&&p.ability==QUARK_DRIVE)){
    bool field=p.ability==PROTOSYNTHESIS?effective_weather(s)==SUN:s.terrain==ELECTRIC_TERRAIN;
    if(field)activate_paradox(s,side);else if(!p.paradox_booster)p.paradox_stat=0;
  }
}
PK_HD inline void update_booster(State& s,int side){
  Mon& p=active(s,side);
  if(p.item==BOOSTER_ENERGY&&p.booster_started&&((p.ability==PROTOSYNTHESIS&&effective_weather(s)!=SUN)||(p.ability==QUARK_DRIVE&&s.terrain!=ELECTRIC_TERRAIN))){
    use_item(s,side);activate_paradox(s,side,true);
  }
}
// Even an event with no listeners sorts the active holders in the oracle.
// Keep this scheduler operation explicit so its tie RNG is not optimized away.
PK_HD inline void ability_start(State& s,int side,bool switch_in=false);
PK_HD inline bool transform_into(State& s,int side);
PK_HD inline int each_event(State& s,bool include_fainted=false,int event=UPDATE) {
 const Mon& a=active(s,0);const Mon& b=active(s,1);
 int first=speed(b)>speed(a)?1:0;
 if((include_fainted||(!a.fainted&&!b.fainted))&&speed(a)==speed(b))first=random(s,2);
 if(event==WEATHER_CHANGE||event==TERRAIN_CHANGE){for(int k=0;k<2;k++){int side=k?1-first:first;update_paradox_field(s,side,event);Mon& p=active(s,side);if(event==WEATHER_CHANGE&&p.hp>0&&p.ability==ICE_FACE&&p.form_family==5&&p.form_current==1&&(effective_weather(s)==SNOW||effective_weather(s)==HAIL)){p.form_busted=0;change_form(s,side,0,true);}}return first;}
 if(event!=UPDATE)return first;
 for(int k=0;k<2;k++){
   int side=k?1-first:first;Mon& p=active(s,side);if(p.fainted)continue;if(p.hp<=0){const Mon& foe=active(s,1-side);if(p.ability==TRACE&&p.trace_seek&&foe.hp>0&&traceable_ability(foe.ability))random(s,1);continue;}
   if(p.form_busted&&p.form_current==0&&((p.ability==DISGUISE&&p.form_family==4)||(p.ability==ICE_FACE&&p.form_family==5))){change_form(s,side,1,true);if(p.ability==DISGUISE){p.hp=maximum(0,p.hp-maximum(1,p.maxhp/8));publish_active(s,side,true);}}
   if(p.ability==TRACE&&p.trace_seek){const Mon& foe=active(s,1-side);if(foe.hp>0&&traceable_ability(foe.ability)){random(s,1);if(!unassignable_ability(foe.ability)){p.ability=foe.ability;p.trace_seek=0;publish_ability(s,side,s.public_active[side],p.ability);s.requests[side].abilities[s.active[side]]=p.ability;ability_start(s,side);}}}
   update_booster(s,side);
   if(p.item==LEPPA_BERRY&&can_eat(s,side)){int slot=-1;for(int i=0;i<4;i++)if(p.move_ids[i]>=0&&p.pp[i]==0){slot=i;break;}if(slot>=0){p.pp[slot]=minimum(p.maxpp[slot],p.pp[slot]+10);use_item(s,side,true);}}
   bool cure=((p.ability==WATER_BUBBLE||p.ability==THERMAL_EXCHANGE||p.ability==WATER_VEIL)&&p.status==BRN)||((p.ability==INSOMNIA||p.ability==VITAL_SPIRIT)&&p.status==SLP)||(p.ability==IMMUNITY&&(p.status==PSN||p.status==TOX))||(p.ability==LIMBER&&p.status==PAR)||(p.ability==MAGMA_ARMOR&&p.status==FRZ);
   if(cure){p.status=NONE;p.status_turns=0;p.toxic_turns=0;publish_active(s,side);}
   if(p.attract_source&&(s.active[1-side]+1!=p.attract_source||active(s,1-side).fainted||p.ability==OBLIVIOUS))p.attract_source=0;
   if(p.ability==OWN_TEMPO&&p.confusion){p.confusion=0;publish_active(s,side);}
   if(p.ability==OBLIVIOUS&&p.taunt){p.taunt=0;publish_active(s,side);}
   if(p.item==SITRUS_BERRY&&can_eat(s,side)&&!p.heal_block&&p.hp*2<=p.maxhp){if(!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/4));use_item(s,side,true);publish_active(s,side,true);}
   if(can_eat(s,side)&&((p.item==LUM_BERRY&&(p.status||p.confusion))||(p.item==CHESTO_BERRY&&p.status==SLP))){bool lum=p.item==LUM_BERRY;p.status=NONE;p.status_turns=0;p.toxic_turns=0;if(lum)p.confusion=0;use_item(s,side,true);publish_active(s,side);}
 }
 return first;
}
PK_HD inline bool apply_boosts(State& s,Mon& p,const int32_t delta[7],int source_side=-1,bool reflected=false);
PK_HD inline void sort_action_list(State& s,QueuedAction* list,int length);
PK_HD inline void faint_event(State& s,int dead_side,int dead_slot){
  QueuedAction handlers[2];int count=0;for(int side=0;side<2;side++){const Mon& p=active(s,side);if(!p.fainted&&!(side==dead_side&&s.active[side]==dead_slot)&&p.ability==SOUL_HEART)handlers[count++]={0,side,0,0,0,1,speed(p),7,0};}
  sort_action_list(s,handlers,count);for(int i=0;i<count;i++){int side=handlers[i].side;Mon& p=active(s,side);bool remaining=false;for(int k=0;k<6;k++)remaining|=s.mons[1-side][k].maxhp>0&&!s.mons[1-side][k].fainted&&!((1-side)==dead_side&&k==dead_slot);int32_t delta[7]={0,0,1,0,0,0,0};if(remaining&&apply_boosts(s,p,delta))publish_ability(s,side,s.public_active[side],p.ability);}
}
PK_HD inline void resolve_faints(State& s,int first_side=0){int count=0,last_source=-1,last_slot=-1;for(int position=0;position<2;position++)for(int i=0;i<6;i++){int side=position?1-first_side:first_side;Mon& p=s.mons[side][i];if(p.maxhp>0&&p.hp==0&&!p.fainted){++count;if(p.destiny_bond&&p.faint_source_side>0&&p.faint_source_side!=side+1){Mon& source=s.mons[p.faint_source_side-1][p.faint_source_slot];if(source.hp>0){source.hp=0;publish_active(s,p.faint_source_side-1,true);}}last_source=p.faint_source_side-1;last_slot=p.faint_source_slot;faint_event(s,side,i);if(active(s,1-side).trap_source==i+1){active(s,1-side).trap_source=0;active(s,1-side).trapped_volatile=0;}if(s.ignore_ability_side==side+1&&i==s.active[side])restore_abilities(s);if(p.ability==AIR_LOCK||p.ability==CLOUD_NINE){p.ability_ending=1;each_event(s,false,WEATHER_CHANGE);}if(i==s.active[side]&&!p.transformed&&p.form_current!=p.form_permanent)change_form(s,side,p.form_permanent,false,false);if(p.form_regression&&i==s.active[side]){p.form_permanent=0;change_form(s,side,0,true);p.base_ability=p.original_ability;p.form_regression=0;}if(p.ability!=p.base_ability)p.slow_start=0;restore_transform(p);p.suppressed_ability=0;p.ability=p.base_ability;p.illusion_slot=0;p.trace_seek=0;p.speed_cache=p.stats[4];p.paradox_stat=0;p.paradox_booster=0;p.type[0]=p.base_types[0];p.type[1]=p.base_types[1];p.newly_switched=1;p.got_attacked=0;p.hit_by_foe=0;for(int j=0;j<64;j++)p.volatile_order[j]=0;p.curse=0;p.salt_cure=0;p.flash_fire=0;p.volatile_serial=0;p.last_result=0;p.current_result=0;p.fainted=1;p.maybe_trapped=0;p.no_retreat=0;p.trapped_volatile=0;p.magnet_rise=0;p.glaive_rush=0;p.attract_source=0;p.destiny_bond=0;p.mirror_coat=0;p.mirror_damage=0;p.chilly_reception=0;p.focus_punch=0;p.beak_blast=0;p.charged=0;p.block_heal_berry=0;p.recharge=0;p.trap_source=0;p.partial_trap=0;p.partial_source=0;p.truant=0;p.unburden=0;p.unnerved=0;p.faint_source_side=0;p.faint_source_slot=0;s.total_fainted[side]=minimum(100,s.total_fainted[side]+1);s.last_faint_side=side;p.terastallized=0;p.confusion=0;p.flinch=0;p.taunt=0;p.encore=0;p.disable=0;p.last_move=-1;p.leech_seed=0;p.yawn=0;p.substitute=0;p.inherited_substitute=0;p.roost=0;p.heal_block=0;p.throat_chop=0;p.locked_duration=0;p.locked_true=0;p.charge_duration=0;p.charge_move=0;p.charge_kind=0;p.charge_released=0;p.protect=0;p.stall_counter=0;p.stall_duration=0;for(int j=0;j<7;j++)p.boosts[j]=0;}}
  if(count&&last_source>=0&&last_source<2){Mon& source=s.mons[last_source][last_slot];
    if(source.ability==BATTLE_BOND&&(source.species_flags&8192)&&!source.bond_triggered&&source.hp>0&&s.active[last_source]==last_slot){
      int32_t delta[7]={1,0,1,0,1,0,0};if(apply_boosts(s,source,delta)){source.bond_triggered=1;publish_ability(s,last_source,last_slot,BATTLE_BOND);}
    }
  }
  if(count&&last_source>=0&&last_source<2){bool remaining[2]={false,false};for(int side=0;side<2;side++)for(int i=0;i<6;i++)remaining[side]|=s.mons[side][i].hp>0;
    Mon& source=s.mons[last_source][last_slot];if(remaining[0]&&remaining[1]&&s.active[last_source]==last_slot&&source.hp>0&&!source.fainted&&(source.ability==MOXIE||source.ability==CHILLING_NEIGH||source.ability==GRIM_NEIGH||source.ability==AS_ONE_GLASTRIER||source.ability==AS_ONE_SPECTRIER)){
      int32_t delta[7]={0,0,0,0,0,0,0};delta[source.ability==GRIM_NEIGH||source.ability==AS_ONE_SPECTRIER?2:0]=count;if(apply_boosts(s,source,delta))publish_ability(s,last_source,last_slot,source.ability==AS_ONE_GLASTRIER?CHILLING_NEIGH:source.ability==AS_ONE_SPECTRIER?GRIM_NEIGH:source.ability);
    }
  }
}
PK_HD inline int living(const State& s,int side){int n=0;for(int i=0;i<6;i++)n+=s.mons[side][i].hp>0;return n;}
PK_HD inline void outcomes(State& s) {
  const bool a=living(s,0)>0,b=living(s,1)>0;
  if(!a||!b){s.terminated=1;s.winner=a?0:b?1:s.last_faint_side;}
}
PK_HD inline bool set_weather(State& s,int weather,const Mon& source){
  if(s.weather==weather)return false;s.weather=weather;s.weather_duration=weather?5:0;
  if((weather==RAIN&&source.item==DAMP_ROCK)||(weather==SUN&&source.item==HEAT_ROCK)||(weather==SAND&&source.item==SMOOTH_ROCK)||((weather==SNOW||weather==HAIL)&&source.item==ICY_ROCK))s.weather_duration=8;
  each_event(s,false,WEATHER_CHANGE);return true;
}
PK_HD inline void switch_to(State& s,int side,int request_slot,bool drag=false) {
  const int slot=s.order[side][request_slot];
  if(s.reviving[side]) {
    Mon& revived=s.mons[side][slot];revived.hp=maximum(1,revived.maxhp/2);revived.fainted=0;revived.status=NONE;PublicMon& v=s.public_mons[side][slot];v.hp=public_hp(revived);v.known_hp=1;v.fainted=0;v.status=NONE;s.requests[side].hp[slot]=revived.hp;s.requests[side].hp_ratio[slot]=double(revived.hp)/revived.maxhp;s.reviving[side]=0;s.forced[side]=0;s.pivot[side]=0;return;
  }
  if(active(s,side).ability==ZERO_TO_HERO&&!active(s,side).transformed&&active(s,side).form_family==3&&active(s,side).hp>0)change_form(s,side,1,true);
  int passed_substitute=s.shed_tail[side]?active(s,side).substitute:0;s.shed_tail[side]=0;
  Mon& linked=active(s,1-side);if(linked.trap_source==s.active[side]+1){linked.trap_source=0;linked.trapped_volatile=0;}
  Mon& old=active(s,side);if(old.suppressed_ability){old.ability=old.suppressed_ability;old.suppressed_ability=0;}for(int i=0;i<64;i++)old.volatile_order[i]=0;old.curse=0;old.salt_cure=0;old.flash_fire=0;old.volatile_serial=0;old.last_result=0;if(!old.fainted)old.current_result=0;for(int i=0;i<7;i++)old.boosts[i]=0;
  old.ability_ending=1;s.trapped[side]=0;old.maybe_trapped=0;old.no_retreat=0;old.trapped_volatile=0;old.magnet_rise=0;old.glaive_rush=0;old.attract_source=0;old.destiny_bond=0;old.mirror_coat=0;old.mirror_damage=0;old.chilly_reception=0;old.focus_punch=0;old.beak_blast=0;old.charged=0;old.block_heal_berry=0;old.recharge=0;old.trap_source=0;old.partial_trap=0;old.partial_source=0;old.truant=0;old.unburden=0;old.unnerved=0;if(old.ability!=old.base_ability)old.slow_start=0;
  if(old.hp>0&&(old.ability==AIR_LOCK||old.ability==CLOUD_NINE))each_event(s,false,WEATHER_CHANGE);
  old.paradox_stat=0;old.paradox_booster=0;old.booster_started=0;
  if(old.hp>0&&old.ability==NATURAL_CURE){old.status=NONE;old.status_turns=0;old.toxic_turns=0;s.public_mons[side][s.public_active[side]].status=NONE;}
  if(old.hp>0&&old.ability==REGENERATOR)old.hp=minimum(old.maxhp,old.hp+old.maxhp/3);
  restore_transform(old);if(old.form_current!=old.form_permanent)change_form(s,side,old.form_permanent,false,false);
  old.ability=old.base_ability;old.got_attacked=0;old.known_type=1;old.trace_seek=0;
  old.choice_lock=-1;old.stats_raised=0;old.stats_lowered=0;if(!old.fainted)old.hit_by_foe=0;old.newly_switched=1;old.type[0]=old.base_types[0];old.type[1]=old.base_types[1];old.speed_cache=old.stats[4];
  old.protect=0;old.stall_counter=0;old.stall_duration=0;
  old.confusion=0;old.flinch=0;
  old.taunt=0;old.encore=0;old.disable=0;old.last_move=-1;s.public_mons[side][s.public_active[side]].taunt=0;
  old.leech_seed=0;old.yawn=0;old.substitute=0;old.inherited_substitute=0;old.roost=0;old.heal_block=0;old.throat_chop=0;old.locked_duration=0;old.locked_true=0;old.charge_duration=0;old.charge_move=0;old.charge_kind=0;old.charge_released=0;
  if(old.fainted)old.status=NONE;
  const int old_slot=s.active[side];int old_index=0;
  for(int i=0;i<6;i++)if(s.order[side][i]==old_slot)old_index=i;
  s.order[side][request_slot]=old_slot;s.order[side][old_index]=slot;
  s.active[side]=slot;s.forced[side]=0;s.revealed[side][slot]=1;
  s.pivot[side]=0;
  active(s,side).embodied=0;active(s,side).tera_resisted=0;active(s,side).form_busted=0;active(s,side).cud_berry=0;active(s,side).cud_counter=0;active(s,side).trace_seek=0;active(s,side).choice_lock=-1;active(s,side).slow_start=0;active(s,side).supreme_fallen=0;
  active(s,side).newly_switched=1;active(s,side).active_turns=0;active(s,side).active_move_actions=0;active(s,side).protean_used=0;active(s,side).ability_ending=0;
  if(!drag){boost_event_order(s);int counter=active(s,side).slow_start;active(s,side).slow_start=0;active(s,side).speed_cache=calculate_speed(active(s,side),s.side_conditions[side][TAILWIND],effective_weather(s),s.trick_room>0,s.terrain);active(s,side).slow_start=counter;}
  if(active(s,side).status==TOX)active(s,side).toxic_turns=0;
  if(passed_substitute){active(s,side).substitute=passed_substitute;active(s,side).inherited_substitute=1;mark_volatile(active(s,side),8);}
  begin_illusion(s,side);reveal_active(s,side);
}
PK_HD inline int mon_type(const Mon& p,int index,bool pretera=false){
  if(!pretera&&p.terastallized&&p.tera_type!=STELLAR)return index?-1:p.tera_type;
  if(!p.roost)return p.type[index];
  int t0=p.type[0]==9?-1:p.type[0],t1=p.type[1]==9?-1:p.type[1];
  return index?(t0!=-1?t1:-1):t0!=-1?t0:t1!=-1?t1:0;
}
PK_HD inline bool has_type(const Mon& p,int type);
PK_HD inline bool grounded(const Mon& p);
PK_HD inline void sort_action_list(State& s,QueuedAction* list,int length);
PK_HD inline bool will_move(const State& s,const Mon& p);
PK_HD inline int damage(State& s,Mon& a,Mon& d,Move& m,const Rules& rules,int level) {
  const int rates[5]={24,24,8,2,1};
  bool crit=m.always_crit||random(s,rates[maximum(1,minimum(4,maximum(1,m.crit_ratio)+int(a.item==SCOPE_LENS)+int(a.ability==SUPER_LUCK)))])==0;
  bool barrier=((d.ability==DISGUISE&&d.form_family==4)||(d.ability==ICE_FACE&&d.form_family==5&&m.category==0))&&d.form_current==0&&(!d.substitute||(m.traits&8));
  if(d.ability==BATTLE_ARMOR||d.ability==SHELL_ARMOR||barrier)crit=false;
  int power=m.power,power_modifier=4096;
  if(!a.transformed&&a.form_family>=10&&a.form_family<=12&&a.item==WELLSPRING_MASK+a.form_family-10)power_modifier=chain_modifier(power_modifier,4915);
  if(a.charged&&m.type==4)power_modifier=chain_modifier(power_modifier,8192);
  if(a.ability==SUPREME_OVERLORD&&a.supreme_fallen){const int mods[6]={4096,4506,4915,5325,5734,6144};power_modifier=chain_modifier(power_modifier,mods[minimum(5,a.supreme_fallen)]);}
  if(d.ability==DRY_SKIN&&m.type==1)power_modifier=chain_modifier(power_modifier,5120);
  if(m.kind==SOLAR_BEAM_MOVE&&effective_weather(s)&&effective_weather(s)!=SUN)power_modifier=chain_modifier(power_modifier,2048);
  if(a.ability==TECHNICIAN&&power<=60)power_modifier=chain_modifier(power_modifier,6144);
  if(a.ability==SAND_FORCE&&effective_weather(s)==SAND&&(m.type==8||m.type==12||m.type==15))power_modifier=chain_modifier(power_modifier,5325);
  if((m.kind==FUSION_FLARE_MOVE&&s.last_successful_move_kind==FUSION_BOLT_MOVE)||(m.kind==FUSION_BOLT_MOVE&&s.last_successful_move_kind==FUSION_FLARE_MOVE))power_modifier=chain_modifier(power_modifier,8192);
  if(a.ability==IRON_FIST&&(m.traits&4096))power_modifier=chain_modifier(power_modifier,4915);
  if(m.traits&32768)power_modifier=chain_modifier(power_modifier,5325);
  if(a.ability==TOUGH_CLAWS&&(m.traits&2048))power_modifier=chain_modifier(power_modifier,5325);
  if((a.ability==SHARPNESS&&(m.traits&1048576))||(a.ability==MEGA_LAUNCHER&&(m.traits&2097152))||(a.ability==FLARE_BOOST&&a.status==BRN&&m.category==1)||(a.ability==TOXIC_BOOST&&(a.status==PSN||a.status==TOX)&&m.category==0))power_modifier=chain_modifier(power_modifier,6144);
  if(a.ability==RECKLESS&&((m.effects&RECOIL)||m.kind==CRASH_MOVE))power_modifier=chain_modifier(power_modifier,4915);
  if(a.ability==STRONG_JAW&&(m.traits&8192))power_modifier=chain_modifier(power_modifier,6144);
  if(((a.item>=SILK_SCARF&&a.item<=FAIRY_FEATHER&&m.type==a.item-SILK_SCARF)||(plate_type(a.item)==m.type&&m.type>=0))||(a.item==SOUL_DEW&&(a.species_flags&2)&&(m.type==10||m.type==14)))power_modifier=chain_modifier(power_modifier,4915);
  if((((a.item==GRISEOUS_CORE||a.item==GRISEOUS_ORB)&&(a.species_flags&32))&&(m.type==13||m.type==14))||
     (((a.item==LUSTROUS_GLOBE||a.item==LUSTROUS_ORB)&&(a.species_flags&16))&&(m.type==2||m.type==14))||
     (((a.item==ADAMANT_CRYSTAL||a.item==ADAMANT_ORB)&&(a.species_flags&8))&&(m.type==15||m.type==14)))power_modifier=chain_modifier(power_modifier,4915);
  if(m.traits&8388608)power_modifier=chain_modifier(power_modifier,4915);
  if(a.ability==ANALYTIC){bool pending=false;for(int side=0;side<2;side++){const Mon& foe=active(s,side);if(&foe!=&a&&will_move(s,foe))pending=true;}if(!pending)power_modifier=chain_modifier(power_modifier,5325);}
  if(a.ability==PUNK_ROCK&&(m.traits&16))power_modifier=chain_modifier(power_modifier,5325);
  if(s.terrain==ELECTRIC_TERRAIN&&m.type==4&&grounded(a))power_modifier=chain_modifier(power_modifier,5325);
  if(s.terrain==GRASSY_TERRAIN){if((m.traits&64)&&grounded(d))power_modifier=chain_modifier(power_modifier,2048);else if(m.type==3&&grounded(a))power_modifier=chain_modifier(power_modifier,5325);}
  if(s.terrain==MISTY_TERRAIN&&m.type==14&&grounded(d))power_modifier=chain_modifier(power_modifier,2048);
  if(s.terrain==PSYCHIC_TERRAIN&&m.type==10&&grounded(a))power_modifier=chain_modifier(power_modifier,5325);
  if(m.kind==KNOCK_OFF_MOVE&&d.item&&can_take_item(d,d))power_modifier=chain_modifier(power_modifier,6144);
  if(m.kind==FICKLE_BEAM_MOVE&&random(s,10)<3)power_modifier=chain_modifier(power_modifier,8192);
  if(m.kind==LASH_OUT_MOVE&&a.stats_lowered)power_modifier=chain_modifier(power_modifier,8192);
  if((m.kind==PSYBLADE_MOVE&&s.terrain==ELECTRIC_TERRAIN)||(m.kind==EXPANDING_FORCE_MOVE&&s.terrain==PSYCHIC_TERRAIN&&grounded(a)))power_modifier=chain_modifier(power_modifier,6144);
  if(m.kind==SUPER_EFFECTIVE_POWER_MOVE){int e=0;for(int i=0;i<2;i++){int t=mon_type(d,i);if(t>=0)e+=rules.chart[m.type*18+t];}if(e>0&&!(d.ability==TERA_SHELL&&d.form_family==8&&d.form_current==1&&(d.hp==d.maxhp||d.tera_resisted)))power_modifier=chain_modifier(power_modifier,5461);}
  power=modify(power,power_modifier);
  bool stellar_available=m.type>=0&&!(a.stellar_used&(1u<<m.type));
  if(a.terastallized&&(a.tera_type==STELLAR?stellar_available:a.tera_type==m.type)&&power<60&&(m.id<0||rules.moves[m.id].priority<=0)&&!m.hit_max&&!(m.traits&128))power=60;
  int astat=m.attack_stat>=0?m.attack_stat:m.category==0?0:2,dstat=m.defense_stat>=0?m.defense_stat:m.category==0?1:3;
  const Mon& attacker=m.attack_target?d:a;const Mon& defender=m.defense_source?a:d;
  int ab=attacker.boosts[astat],db=defender.boosts[dstat];
  if((m.ignore_boosts&1)||((crit||(m.ignore_boosts&4))&&ab<0))ab=0;
  if((m.ignore_boosts&2)||((crit||(m.ignore_boosts&8))&&db>0))db=0;
  if(d.ability==UNAWARE&&(astat==0||astat==1||astat==2))ab=0;
  if(a.ability==UNAWARE&&(dstat==1||dstat==3))db=0;
  if(a.ability==UNAWARE&&d.ability==UNAWARE&&speed(a)==speed(d)){random(s,2);random(s,2);}
  int atk=boost_stat(attacker.stats[astat],ab),def=boost_stat(defender.stats[dstat],db);
  if((effective_weather(s)==SAND&&dstat==3&&has_type(d,12))||(effective_weather(s)==SNOW&&dstat==1&&has_type(d,5)))def=modify(def,6144);
  int attack_modifier=4096,defense_modifier=4096;
  if(m.category==0){
    if(a.ability==HUGE_POWER||a.ability==PURE_POWER)attack_modifier*=2;
    if(a.ability==GUTS&&a.status)attack_modifier=attack_modifier*3/2;
    if(a.item==CHOICE_BAND)attack_modifier=attack_modifier*3/2;
  }else if(m.category==1&&a.item==CHOICE_SPECS)attack_modifier=attack_modifier*3/2;
  if((m.category==0&&a.ability==ORICHALCUM_PULSE&&effective_weather(s)==SUN)||(m.category==1&&a.ability==HADRON_ENGINE&&s.terrain==ELECTRIC_TERRAIN))attack_modifier=chain_modifier(attack_modifier,5461);
  if(m.category==0&&a.ability==SLOW_START&&a.slow_start)attack_modifier=chain_modifier(attack_modifier,2048);
  if(a.flash_fire&&a.ability==FLASH_FIRE&&m.type==1)attack_modifier=chain_modifier(attack_modifier,6144);
  if(a.ability==STAKEOUT&&d.active_turns==0)attack_modifier=chain_modifier(attack_modifier,8192);
  if(a.item==LIGHT_BALL&&(a.species_flags&1))attack_modifier=chain_modifier(attack_modifier,8192);
  if(a.paradox_stat==(m.category==0?1:3))attack_modifier=chain_modifier(attack_modifier,5325);
  if(d.paradox_stat==dstat+1)defense_modifier=chain_modifier(defense_modifier,5325);
  if(dstat==1&&d.ability==FUR_COAT)defense_modifier*=2;
  if((dstat==1||dstat==3)&&d.item==EVIOLITE&&d.nfe)defense_modifier=chain_modifier(defense_modifier,6144);
  if(dstat==1&&d.ability==MARVEL_SCALE&&d.status)defense_modifier=defense_modifier*3/2;
  if(dstat==3&&d.item==ASSAULT_VEST)defense_modifier=defense_modifier*3/2;
  if(m.category==1&&a.ability==SOLAR_POWER&&effective_weather(s)==SUN)attack_modifier=attack_modifier*3/2;
  if(m.category==0&&a.ability==HUSTLE)atk=modify(atk,6144);
  if(a.ability==TRANSISTOR&&m.type==4)attack_modifier=chain_modifier(attack_modifier,5325);
  if((a.ability==DRAGONS_MAW&&m.type==14)||(a.ability==ROCKY_PAYLOAD&&m.type==12)||(a.ability==STEELWORKER&&m.type==15))attack_modifier=chain_modifier(attack_modifier,6144);
  if(a.ability==WATER_BUBBLE&&m.type==2)attack_modifier=chain_modifier(attack_modifier,8192);
  if(d.ability==THICK_FAT||d.ability==HEATPROOF||d.ability==WATER_BUBBLE||d.ability==PURIFYING_SALT){
    bool attack_listener=m.category==0?(a.ability==HUGE_POWER||a.ability==PURE_POWER||a.ability==GUTS||a.ability==SLOW_START||a.ability==OVERGROW||a.ability==BLAZE||a.ability==TORRENT||a.ability==SWARM):(a.ability==SOLAR_POWER||a.ability==OVERGROW||a.ability==BLAZE||a.ability==TORRENT||a.ability==SWARM||a.ability==DRAGONS_MAW||a.ability==TRANSISTOR||a.ability==ROCKY_PAYLOAD||a.ability==STEELWORKER||a.ability==STAKEOUT||a.ability==PLUS||a.ability==MINUS);
    bool priority5=a.ability==OVERGROW||a.ability==BLAZE||a.ability==TORRENT||a.ability==SWARM||a.ability==DRAGONS_MAW||a.ability==TRANSISTOR||a.ability==ROCKY_PAYLOAD||a.ability==STEELWORKER||a.ability==STAKEOUT||(m.category==1&&(a.ability==PLUS||a.ability==MINUS))||(m.category==0?(a.ability==HUGE_POWER||a.ability==PURE_POWER||a.ability==GUTS||a.ability==SLOW_START):a.ability==SOLAR_POWER);
    if(speed(a)==speed(d)&&((m.category==1&&attack_listener)||(m.category==0&&d.ability==WATER_BUBBLE&&priority5)))random(s,2);
    if((d.ability==PURIFYING_SALT&&m.type==13)||(d.ability!=PURIFYING_SALT&&m.type==1)||(d.ability==THICK_FAT&&m.type==5))attack_modifier=chain_modifier(attack_modifier,2048);
  }
  if(a.hp*3<=a.maxhp&&((a.ability==OVERGROW&&m.type==3)||(a.ability==BLAZE&&m.type==1)||(a.ability==TORRENT&&m.type==2)||(a.ability==SWARM&&m.type==11)))attack_modifier=chain_modifier(attack_modifier,6144);
  int attack_ruin=m.category==0?TABLETS_OF_RUIN:VESSEL_OF_RUIN;
  int defense_ruin=dstat==1?SWORD_OF_RUIN:dstat==3?BEADS_OF_RUIN:0;
  if((a.ability==attack_ruin||a.ability==WATER_BUBBLE)&&d.ability==attack_ruin&&speed(a)==speed(d))random(s,2);
  if(defense_ruin&&a.ability==defense_ruin&&d.ability==defense_ruin&&speed(a)==speed(d))random(s,2);
  bool field_attack_ruin=false,field_defense_ruin=false;for(int side=0;side<2;side++){const Mon& holder=active(s,side);if(holder.hp>0&&!holder.fainted){field_attack_ruin|=holder.ability==attack_ruin;field_defense_ruin|=defense_ruin&&holder.ability==defense_ruin;}}
  if(field_attack_ruin&&a.ability!=attack_ruin)attack_modifier=chain_modifier(attack_modifier,3072);
  if(field_defense_ruin&&d.ability!=defense_ruin)defense_modifier=chain_modifier(defense_modifier,3072);
  atk=modify(atk,attack_modifier);def=modify(def,defense_modifier);
  int base=((2*level/5+2)*power*atk/maximum(1,def))/50+2;
  if(effective_weather(s)==RAIN&&(m.type==1||m.type==2))base=modify(base,m.type==2?6144:2048);
  if(effective_weather(s)==SUN&&(m.type==1||m.type==2))base=modify(base,m.type==1||m.kind==HYDRO_STEAM_MOVE?6144:2048);
  if(crit)base=base*3/2;
  base=base*(100-int(random(s,16)))/100;
  bool stab=m.type>=0&&(m.type==mon_type(a,0,true)||m.type==mon_type(a,1,true));
  if(a.terastallized&&a.tera_type==STELLAR&&m.type>=0){
    if((a.form_family==8&&a.form_current==2)||stellar_available||m.stellar_boosted){base=modify(base,stab?8192:4915);m.stellar_boosted=1;if(!(a.form_family==8&&a.form_current==2))a.stellar_used|=1u<<m.type;}
    else if(stab)base=base*3/2;
  }
  else {
    int stab_modifier=a.terastallized&&m.type==a.tera_type?(stab?8192:6144):stab?6144:4096;
    if(a.ability==ADAPTABILITY&&m.type>=0&&has_type(a,m.type))stab_modifier=stab_modifier==8192?9216:8192;
    base=modify(base,stab_modifier);
  }
  bool tera_defense=d.terastallized&&d.tera_type!=STELLAR;
  const int t0=mon_type(d,0),t1=mon_type(d,1);
  int exponent=m.type==STELLAR&&d.terastallized?1:0;
  for(int i=0;i<2;i++) {int t=i?t1:t0;if(t<0||t>=18||m.type<0||m.type>=18)continue;int e=rules.chart[m.type*18+t];if(e==-127){if((m.traits&16777216)&&t==13)e=0;else return 0;}if(m.kind==FREEZE_DRY_MOVE&&t==2)e=1;if(barrier)e=0;exponent+=e;}
  if(d.ability==TERA_SHELL&&d.form_family==8&&d.form_current==1&&m.id>=0){if(d.tera_resisted)exponent=-1;else if(d.hp==d.maxhp&&exponent>=0){d.tera_resisted=1;exponent=-1;}}
  if(exponent>0)base<<=exponent;else if(exponent<0)base>>=-exponent;
  if(a.status==BRN&&m.category==0&&a.ability!=GUTS&&m.kind!=FACADE_MOVE)base/=2;
  int defender_side=0;for(int side=0;side<2;side++)for(int i=0;i<6;i++)if(&s.mons[side][i]==&d)defender_side=side;
  QueuedAction modifiers[12];int modifier_count=0;
  int amod=4096,dmod=4096;bool ahook=a.ability==TINTED_LENS||a.ability==NEUROFORCE||a.ability==SNIPER;
  bool dhook=d.ability==FILTER||d.ability==SOLID_ROCK||d.ability==PRISM_ARMOR||d.ability==ICE_SCALES||d.ability==MULTISCALE||d.ability==SHADOW_SHIELD||d.ability==FLUFFY||d.ability==PUNK_ROCK;
  if(a.ability==TINTED_LENS&&exponent<0)amod=8192;
  if(a.ability==NEUROFORCE&&exponent>0)amod=5120;
  if(a.ability==SNIPER&&crit)amod=6144;
  if((d.ability==FILTER||d.ability==SOLID_ROCK||d.ability==PRISM_ARMOR)&&exponent>0)dmod=3072;
  if((d.ability==ICE_SCALES&&m.category==1)||((d.ability==MULTISCALE||d.ability==SHADOW_SHIELD)&&d.hp==d.maxhp))dmod=2048;
  if(d.ability==FLUFFY){dmod=m.type==1?8192:4096;if(m.traits&2048)dmod/=2;}
  if(d.ability==PUNK_ROCK&&(m.traits&16))dmod=2048;
  if(d.glaive_rush)modifiers[modifier_count++]={8192,0,0,0,0,0,speed(d),2,0};
  if(ahook)modifiers[modifier_count++]={amod,0,0,0,0,0,speed(a),7,0};
  if(dhook)modifiers[modifier_count++]={dmod,0,0,0,0,0,speed(d),7,0};
  if(a.item==LIFE_ORB)modifiers[modifier_count++]={5324,0,0,0,0,0,speed(a),8,0};
  if(a.item==EXPERT_BELT)modifiers[modifier_count++]={exponent>0?4915:4096,0,0,0,0,0,speed(a),8,0};
  for(int side=0;side<2;side++)for(int k=REFLECT;k<=AURORA_VEIL;k++)if(s.side_conditions[side][k]){
    bool screen=side==defender_side&&!crit&&!(m.traits&131072)&&(k==(m.category==0?REFLECT:LIGHT_SCREEN)||(k==AURORA_VEIL&&!s.side_conditions[side][m.category==0?REFLECT:LIGHT_SCREEN]));
    modifiers[modifier_count++]={screen?2048:4096,0,0,0,0,0,0,4,0};
  }
  sort_action_list(s,modifiers,modifier_count);int final_modifier=4096;
  for(int i=0;i<modifier_count;i++)final_modifier=chain_modifier(final_modifier,modifiers[i].kind);
  base=modify(base,final_modifier);
  return maximum(1,base);
}
PK_HD inline bool has_type(const Mon& p,int type){return mon_type(p,0)==type||mon_type(p,1)==type;}
PK_HD inline bool add_volatile(State& s,Mon& p,int kind,const Rules& rules,int source_side);
PK_HD inline bool set_status(State& s,Mon& p,int status,int source_side=-1,bool ignore_safeguard=false,bool move_effect=true,bool replace=false,bool allow_sync=true,bool from_yawn=false){
  if((!replace&&p.status!=NONE)||p.status==status||p.hp<=0)return false;
  if(grounded(p)&&(s.terrain==MISTY_TERRAIN||(s.terrain==ELECTRIC_TERRAIN&&status==SLP)))return false;
  if(p.ability==COMATOSE||(p.ability==SHIELDS_DOWN&&!p.transformed&&p.form_family==7&&p.form_current==1))return false;
  if(p.ability==FLOWER_VEIL&&has_type(p,3)&&source_side>=0&&!from_yawn){int side=0;for(int i=0;i<6;i++)if(&s.mons[1][i]==&p)side=1;if(source_side!=side)return false;}
  if(p.ability==PURIFYING_SALT||(p.ability==LEAF_GUARD&&effective_weather(s)==SUN))return false;
  if((status==SLP&&(p.ability==INSOMNIA||p.ability==VITAL_SPIRIT))||((status==PSN||status==TOX)&&p.ability==IMMUNITY)||(status==PAR&&p.ability==LIMBER)||(status==BRN&&(p.ability==WATER_VEIL||p.ability==WATER_BUBBLE||p.ability==THERMAL_EXCHANGE))||(status==FRZ&&p.ability==MAGMA_ARMOR))return false;
  if(status==FRZ&&effective_weather(s)==SUN)return false;
  if((status==BRN&&has_type(p,1))||(status==PAR&&has_type(p,4))||(status==FRZ&&has_type(p,5))||((status==PSN||status==TOX)&&(has_type(p,7)||has_type(p,15))&&!(source_side>=0&&active(s,source_side).ability==CORROSION)))return false;
  int target_side=0;
  for(int i=0;i<6;i++)if(&s.mons[1][i]==&p)target_side=1;
  if(!ignore_safeguard&&source_side>=0&&source_side!=target_side&&s.side_conditions[target_side][SAFEGUARD]&&(!move_effect||active(s,source_side).ability!=INFILTRATOR))return false;
  if(status==SLP&&source_side!=target_side){
    for(int i=0;i<6;i++){const Mon& q=s.mons[target_side][i];if(q.hp>0&&q.status==SLP&&q.sleep_source!=target_side)return false;}
  }
  p.status=status;p.status_turns=0;p.toxic_turns=0;
  if(status==SLP){p.status_turns=2+random(s,3);p.sleep_source=source_side;}
  // Showdown's AfterSetStatus event sorts its handlers: every active Poison Puppeteer (an onAny handler, even the
  // target's own) and a Synchronize target. Their order never changes the outcome, but each group of tied speeds
  // costs the shuffle's draws (k tied handlers: k-1 draws).
  {int speeds[3],count=0;for(int side=0;side<2;side++){const Mon& h=active(s,side);if(h.hp>0&&!h.fainted&&h.ability==POISON_PUPPETEER)speeds[count++]=speed(h);}
   if(p.ability==SYNCHRONIZE&&count<3)speeds[count++]=speed(p);
   for(int i=0;i<count;i++)for(int j=i+1;j<count;j++)if(speeds[j]>speeds[i]){int x=speeds[i];speeds[i]=speeds[j];speeds[j]=x;}
   for(int i=0;i<count;){int k=i;while(k<count&&speeds[k]==speeds[i])k++;for(int r=i;r<k-1;r++)random(s,k-r);i=k;}}
  if(source_side>=0&&source_side!=target_side&&move_effect&&(status==PSN||status==TOX)&&active(s,source_side).ability==POISON_PUPPETEER&&(active(s,source_side).species_flags&4096)){
    Rules unused{};add_volatile(s,p,CONFUSION,unused,source_side);
  }
  if(p.item==LUM_BERRY&&can_eat(s,target_side)){p.status=NONE;p.status_turns=0;p.toxic_turns=0;p.confusion=0;use_item(s,target_side,true);}
  if(allow_sync&&p.ability==SYNCHRONIZE&&source_side>=0&&source_side!=target_side&&status!=SLP&&status!=FRZ)set_status(s,active(s,source_side),status,target_side,false,true,false,false);
  return true;
}
PK_HD inline bool apply_boosts(State& s,Mon& p,const int32_t delta[7],int source_side,bool reflected){
  if(p.hp<=0)return false;
  bool changed=false;int side=0,slot=0;
  for(int a=0;a<2;a++)for(int i=0;i<6;i++)if(&s.mons[a][i]==&p){side=a;slot=i;}
  bool remaining=false;for(int i=0;i<6;i++)remaining|=s.mons[1-side][i].maxhp>0&&!s.mons[1-side][i].fainted;if(!remaining)return false;int public_slot=slot==s.active[side]?s.public_active[side]:slot;
  if(source_side<0)source_side=side;
  bool foe=source_side!=side;
  bool block=foe&&(p.ability==CLEAR_BODY||p.ability==WHITE_SMOKE||p.ability==FULL_METAL_BODY||(p.ability==FLOWER_VEIL&&has_type(p,3))||(s.side_conditions[side][MIST]&&active(s,source_side).ability!=INFILTRATOR));
  int capped[7];for(int i=0;i<7;i++){
    int change=delta[i];
    if(foe&&p.ability==MIRROR_ARMOR&&!reflected&&change<0){
      if(p.boosts[i]>-6){int32_t bounce[7]={};bounce[i]=maximum(-6,p.boosts[i]+change)-p.boosts[i];
        Mon& source=active(s,source_side);if(source.hp>0){publish_ability(s,side,public_slot,p.ability);apply_boosts(s,source,bounce,side,true);}}
      change=0;
    }
    change*=p.ability==CONTRARY?-1:p.ability==SIMPLE?2:1;
    capped[i]=maximum(-6,minimum(6,p.boosts[i]+change))-p.boosts[i];if((block||(foe&&((i==5&&(p.ability==KEEN_EYE||p.ability==MINDS_EYE))||(i==1&&p.ability==BIG_PECKS))))&&capped[i]<0)capped[i]=0;
  }
  for(int i=0;i<7;i++){
    int x=maximum(-6,minimum(6,p.boosts[i]+capped[i])),actual=x-p.boosts[i];changed|=actual!=0;if(actual>0)p.stats_raised=1;if(actual<0)p.stats_lowered=1;
    s.public_mons[side][public_slot].boosts[i]+=actual;p.boosts[i]=x;
    if(foe&&actual<0&&(p.ability==DEFIANT||p.ability==COMPETITIVE)){
      int stat=p.ability==DEFIANT?0:2,before=p.boosts[stat];p.boosts[stat]=minimum(6,before+2);
      s.public_mons[side][public_slot].boosts[stat]+=p.boosts[stat]-before;
      if(p.boosts[stat]!=before){p.stats_raised=1;}
      if(p.boosts[stat]!=before)publish_ability(s,side,public_slot,p.ability);
    }
  }
  return changed;
}
PK_HD inline bool move_boosts(State& s,Mon& p,const int32_t delta[7],int source_side){return apply_boosts(s,p,delta,source_side);}
PK_HD inline bool grounded(const Mon& p){return !has_type(p,9)&&p.ability!=LEVITATE&&p.item!=AIR_BALLOON&&!p.magnet_rise;}
PK_HD inline bool set_terrain(State& s,int terrain,const Mon& source){
  if(s.terrain==terrain)return false;
  if(!terrain){s.terrain=0;s.terrain_duration=0;if(s.visible_terrain!=MISTY_TERRAIN)s.visible_terrain=0;}
  else {s.terrain=terrain;s.terrain_duration=source.item==TERRAIN_EXTENDER?8:5;s.visible_terrain=terrain;}
  each_event(s,false,TERRAIN_CHANGE);return true;
}
PK_HD inline void drain_heal(State& s,int side,int amount){
  Mon& p=active(s,side);const Mon& source=active(s,1-side);int before=p.hp;
  // Heal Block and Liquid Ooze use cached speed, then Condition/Ability order.
  if(source.ability==LIQUID_OOZE&&(!p.heal_block||speed(source)>speed(p)))indirect_loss(p,amount);
  else if(!p.heal_block&&p.hp>0)p.hp=minimum(p.maxhp,p.hp+amount);
  if(p.hp!=before)publish_active(s,side,true);
}
PK_HD inline void remove_side_condition(State& s,int side,int j){if(s.side_conditions[side][j])s.visible_side_conditions[side][j]=0;s.side_conditions[side][j]=0;s.side_durations[side][j]=0;}
PK_HD inline void add_side_condition(State& s,int side,int j){++s.side_conditions[side][j];++s.visible_side_conditions[side][j];}
PK_HD inline void clear_hazards(State& s,int side){for(int j=0;j<4;j++){remove_side_condition(s,side,j);}active(s,side).leech_seed=0;active(s,side).partial_trap=0;active(s,side).partial_source=0;}
PK_HD inline void entry_hazards(State& s,int side,const Rules& rules){
  Mon& p=active(s,side);int ordered[4],count=0;
  for(int j=0;j<4;j++)if(s.side_conditions[side][j])ordered[count++]=j;
  for(int i=0;i<count;i++)for(int j=i+1;j<count;j++)if(s.side_effect_order[side][ordered[j]]<s.side_effect_order[side][ordered[i]]){int t=ordered[i];ordered[i]=ordered[j];ordered[j]=t;}
  for(int k=0;k<count;k++){
    if(p.hp<=0||p.fainted)return;
    int j=ordered[k],before=p.hp,layers=s.side_conditions[side][j];
    if(j==STEALTH_ROCK&&p.item!=HEAVY_DUTY_BOOTS){
      int exponent=0;for(int t=0;t<2;t++){int type=mon_type(p,t);if(type>=0&&type<18)exponent+=rules.chart[12*18+type];}
      int loss=exponent>=0?(p.maxhp<<exponent)/8:p.maxhp/(8<<(-exponent));indirect_loss(p,maximum(1,loss));
    }else if(j==SPIKES&&grounded(p)&&p.item!=HEAVY_DUTY_BOOTS){const int numerators[4]={0,3,4,6};indirect_loss(p,maximum(1,p.maxhp*numerators[layers]/24));}
    else if(j==TOXIC_SPIKES&&grounded(p)){
      if(has_type(p,7))remove_side_condition(s,side,j);
      else if(!has_type(p,15)&&p.item!=HEAVY_DUTY_BOOTS)set_status(s,p,layers==2?TOX:PSN,1-side,false,false,false,false);
    }else if(j==STICKY_WEB&&grounded(p)&&p.item!=HEAVY_DUTY_BOOTS){const int32_t delta[7]={0,0,0,0,-1,0,0};move_boosts(s,p,delta,1-side);}
    publish_active(s,side,p.hp!=before);resolve_faints(s);outcomes(s);if(s.terminated)return;
  }
}
PK_HD inline bool will_move(const State& s,const Mon& p){for(int i=0;i<s.queue_length;i++){const QueuedAction& a=s.queue[i];if(a.kind==MOVE&&&s.mons[a.side][a.actor]==&p)return true;}return false;}
PK_HD inline bool add_volatile(State& s,Mon& p,int kind,const Rules& rules,int source_side){
  if(p.hp<=0)return false;
  if(p.ability==SHIELDS_DOWN&&!p.transformed&&p.form_family==7&&p.form_current==1&&kind==YAWN)return false;
  if(p.ability==AROMA_VEIL&&(kind==TAUNT||kind==ENCORE||kind==DISABLE||kind==HEAL_BLOCK))return false;
  if(p.ability==FLOWER_VEIL&&has_type(p,3)&&kind==YAWN)return false;
  mark_volatile(p,kind==CURSE?19:kind==SALT_CURE?20:kind==NO_RETREAT?26:kind==MAGNET_RISE?28:kind==PARTIAL_TRAP?32:kind);
  if(kind==NO_RETREAT){if(p.no_retreat||p.trapped_volatile||has_type(p,13))return false;p.no_retreat=1;mark_volatile(p,27);p.trapped_volatile=1;return true;}
  if(kind==PARTIAL_TRAP){if(p.partial_trap)return false;mark_volatile(p,32);p.partial_trap=5+random(s,2);p.partial_source=s.active[source_side]+1;return true;}
  if(kind==MAGNET_RISE){if(p.magnet_rise)return false;p.magnet_rise=5;return true;}
  if(kind==CURSE){if(p.curse)return false;p.curse=1;return true;}
  if(kind==SALT_CURE){if(p.salt_cure)return false;p.salt_cure=1;return true;}
  if(grounded(p)&&((kind==CONFUSION&&s.terrain==MISTY_TERRAIN)||(kind==YAWN&&s.terrain==ELECTRIC_TERRAIN)))return false;
  int target_side=0;for(int i=0;i<6;i++)if(&s.mons[1][i]==&p)target_side=1;
  if((kind==CONFUSION||kind==YAWN)&&source_side!=target_side&&s.side_conditions[target_side][SAFEGUARD]&&active(s,source_side).ability!=INFILTRATOR)return false;
  if(kind==HEAL_BLOCK){if(p.heal_block)return false;p.heal_block=2;return true;}
  if(kind==THROAT_CHOP){if(p.throat_chop)return false;p.throat_chop=2;return true;}
  if(kind==CONFUSION){if(p.confusion||p.ability==OWN_TEMPO)return false;p.confusion=2+random(s,4);return true;}
  if(kind==FLINCH){if(p.focus_punch||p.flinch||p.ability==INNER_FOCUS)return false;p.flinch=1;return true;}
  if(kind==TAUNT){if(p.taunt||p.ability==OBLIVIOUS)return false;p.taunt=3+int(p.active_turns>0&&!will_move(s,p));return true;}
  if(kind==LEECH_SEED){if(p.leech_seed)return false;p.leech_seed=1;return true;}
  if(kind==YAWN){if(p.ability==PURIFYING_SALT||(p.ability==LEAF_GUARD&&effective_weather(s)==SUN))return false;if(p.yawn||p.status||p.ability==INSOMNIA||p.ability==VITAL_SPIRIT)return false;p.yawn=2;p.yawn_source=source_side;return true;}
  if(kind==SUBSTITUTE){if(p.substitute||p.maxhp<=1||p.hp*4<=p.maxhp)return false;p.substitute=p.maxhp/4;p.inherited_substitute=0;p.hp-=p.maxhp/4;p.partial_trap=0;p.partial_source=0;return true;}
  if(kind==ENCORE||kind==DISABLE){
    if((kind==ENCORE?p.encore:p.disable)||p.last_move<0)return false;
    if(kind==ENCORE&&(rules.moves[p.last_move].traits&4))return false;
    bool pp=false;for(int i=0;i<4;i++)if(p.move_ids[i]==p.last_move&&p.pp[i]>0)pp=true;if(!pp)return false;
    if(kind==ENCORE){p.encore=3+int(!will_move(s,p));p.encore_move=p.last_move;}
    else {p.disable=5-int(will_move(s,p));p.disable_move=p.last_move;}
    return true;
  }
  return false;
}
PK_HD inline void start_locked(State& s,Mon& p,int move_id){
  if(p.hp<=0)return;
  if(!p.locked_duration){mark_volatile(p,14);p.locked_duration=2;p.locked_true=2+random(s,2);p.locked_move=move_id;}
  else if(p.locked_true>=2)p.locked_duration=2;
}
PK_HD inline void end_locked(State& s,int side,bool confuse=true){
  Mon& p=active(s,side);bool fatigue=p.locked_true<=1;p.locked_duration=0;p.locked_true=0;
  if(confuse&&fatigue){Rules unused{};add_volatile(s,p,CONFUSION,unused,side);publish_active(s,side);}
}
PK_HD inline int mon_weight(const Mon& p){int value=p.weight_hg;if(p.ability==HEAVY_METAL)value*=2;if(p.ability==LIGHT_METAL)value/=2;if(p.item==FLOAT_STONE)value/=2;return maximum(1,value);}
PK_HD inline void prepare_type(State& s,int side,const Move& move){
  Mon& p=active(s,side);
  if((p.ability!=PROTEAN&&p.ability!=LIBERO)||p.protean_used||p.terastallized||move.kind==SLEEP_TALK_MOVE||move.type<0)return;
  if(mon_type(p,0)==move.type&&mon_type(p,1)<0)return;
  p.type[0]=move.type;p.type[1]=-1;p.protean_used=1;
}
PK_HD inline int accuracy_modifier(State& s,const Mon& actor,const Mon& target,const Move& move){
  int value=4096;
  if(actor.ability==HUSTLE&&move.category==0)value=chain_modifier(value,3277);
  if(actor.ability==COMPOUND_EYES)value=chain_modifier(value,5325);
  if((target.ability==SAND_VEIL&&effective_weather(s)==SAND)||(target.ability==SNOW_CLOAK&&(effective_weather(s)==SNOW||effective_weather(s)==HAIL)))value=chain_modifier(value,3277);
  if(actor.item==WIDE_LENS)value=chain_modifier(value,4505);
  return value;
}
PK_HD inline void accuracy_event_order(State& s,const Mon& actor,const Mon& target){
  if((actor.ability==HUSTLE||actor.ability==COMPOUND_EYES)&&(target.ability==SAND_VEIL||target.ability==SNOW_CLOAK)&&speed(actor)==speed(target))random(s,2);
}
PK_HD inline void damaging_hit(State& s,int side,const Move& move,const Rules& rules,bool struggle,int source_slot=-1){
  Mon& p=source_slot<0?active(s,side):s.mons[side][source_slot];Mon& q=active(s,1-side);bool source_active=(source_slot<0||source_slot==s.active[side])&&!p.fainted;QueuedAction handlers[5];int count=0;
  bool reaction=q.ability==GULP_MISSILE||q.ability==CUTE_CHARM||q.ability==EFFECT_SPORE||q.ability==FLAME_BODY||q.ability==STATIC||q.ability==POISON_POINT||q.ability==CURSED_BODY||q.ability==STAMINA||q.ability==JUSTIFIED||q.ability==WEAK_ARMOR||q.ability==THERMAL_EXCHANGE||q.ability==WATER_COMPACTION||q.ability==RATTLED||q.ability==GOOEY||q.ability==TANGLING_HAIR||q.ability==TOXIC_DEBRIS||q.ability==SEED_SOWER;
  if(q.ability==ILLUSION)break_illusion(s,1-side);
  if(q.ability==ELECTROMORPHOSIS){mark_volatile(q,34);q.charged=1;}
  if(q.ability==ROUGH_SKIN||q.ability==IRON_BARBS||q.ability==AFTERMATH)handlers[count++]={1,1-side,0,0,1,0,speed(q),7,0};
  if(q.item==ROCKY_HELMET)handlers[count++]={2,1-side,0,0,2,0,speed(q),8,0};
  if(reaction)handlers[count++]={3,1-side,0,0,2147483647,0,speed(q),7,0};
  if(q.item==AIR_BALLOON)handlers[count++]={6,1-side,0,0,2147483647,0,speed(q),8,0};
  if(q.item==WEAKNESS_POLICY)handlers[count++]={5,1-side,0,0,2147483647,0,speed(q),8,0};
  if(p.ability==TOXIC_CHAIN||p.ability==POISON_TOUCH)handlers[count++]={4,side,0,0,2147483647,0,speed(p),p.ability==POISON_TOUCH?6:7,0};
  // DamagingHit uses stable left-to-right order, ignoring speed and subOrder.
  // Target handlers precede source handlers; this event never shuffles ties.
  for(int i=1;i<count;i++){QueuedAction x=handlers[i];int j=i;while(j>0&&handlers[j-1].order>x.order){handlers[j]=handlers[j-1];--j;}handlers[j]=x;}
  for(int k=0;k<count;k++){
    int kind=handlers[k].kind;
    if(kind==1&&(move.traits&2048)&&(q.ability!=AFTERMATH||(q.hp==0&&p.ability!=DAMP&&q.ability!=DAMP))&&p.hp>0){if(indirect_loss(p,maximum(1,p.maxhp/(q.ability==AFTERMATH?4:8))))publish_active(s,side,true);}
    if(kind==2&&(move.traits&2048)&&p.hp>0){if(indirect_loss(p,maximum(1,p.maxhp/6)))publish_active(s,side,true);}
    if(kind==3){
      if(q.ability==GULP_MISSILE&&q.form_family==6&&q.form_current>0&&p.hp>0&&source_active){int form=q.form_current;indirect_loss(p,maximum(1,p.maxhp/4));if(form==1){int32_t delta[7]={0,-1,0,0,0,0,0};apply_boosts(s,p,delta,1-side);}else set_status(s,p,PAR,1-side,false,true);change_form(s,1-side,0);publish_active(s,side,true);}
      if(q.ability==CUTE_CHARM&&(move.traits&2048)&&random(s,10)<3&&p.hp>0&&!p.attract_source&&p.gender>0&&p.gender<3&&q.gender>0&&q.gender<3&&p.gender!=q.gender&&p.ability!=OBLIVIOUS&&p.ability!=AROMA_VEIL){mark_volatile(p,39);p.attract_source=s.active[1-side]+1;}
      if(q.ability==EFFECT_SPORE&&(move.traits&2048)&&!has_type(p,3)&&p.ability!=OVERCOAT&&p.item!=SAFETY_GOGGLES){int roll=random(s,100);if(roll<30)set_status(s,p,roll<11?SLP:roll<21?PAR:PSN,1-side,false,false);}
      if((q.ability==FLAME_BODY||q.ability==STATIC||q.ability==POISON_POINT)&&(move.traits&2048)&&random(s,10)<3)set_status(s,p,q.ability==FLAME_BODY?BRN:q.ability==STATIC?PAR:PSN,1-side,false,false);
      if(q.ability==CURSED_BODY&&move.kind!=FUTURE_SIGHT_MOVE&&!p.disable&&!struggle&&random(s,10)<3){if(add_volatile(s,p,DISABLE,rules,1-side))p.disable=4;}
      if((q.ability==GOOEY||q.ability==TANGLING_HAIR)&&(move.traits&2048)){publish_ability(s,1-side,s.public_active[1-side],q.ability);int32_t delta[7]={0,0,0,0,-1,0,0};apply_boosts(s,p,delta,1-side);}
      if(q.ability==TOXIC_DEBRIS&&move.category==0&&s.side_conditions[side][TOXIC_SPIKES]<2){if(!s.side_conditions[side][TOXIC_SPIKES])s.side_effect_order[side][TOXIC_SPIKES]=++s.effect_order;add_side_condition(s,side,TOXIC_SPIKES);}
      if(q.ability==SEED_SOWER)set_terrain(s,GRASSY_TERRAIN,q);
      int32_t delta[7]={0,0,0,0,0,0,0};
      if(q.ability==STAMINA)delta[1]=1;
      if((q.ability==JUSTIFIED&&move.type==16)||(q.ability==THERMAL_EXCHANGE&&move.type==1))delta[0]=1;
      if(q.ability==WATER_COMPACTION&&move.type==2)delta[1]=2;
      if(q.ability==RATTLED&&(move.type==16||move.type==11||move.type==13))delta[4]=1;
      if(q.ability==WEAK_ARMOR&&move.category==0){delta[1]=-1;delta[4]=2;}
      if(apply_boosts(s,q,delta)&&q.hp>0)publish_ability(s,1-side,s.public_active[1-side],q.ability);
    }
    if(kind==4&&q.ability!=SHIELD_DUST&&(p.ability==TOXIC_CHAIN||(move.traits&2048))&&random(s,10)<3)set_status(s,q,p.ability==TOXIC_CHAIN?TOX:PSN,side,false,false);
    if(kind==6)pop_balloon(s,1-side);
    if(kind==5&&!move.fixed_damage&&q.hp>0){int e=0;if(move.type==STELLAR)e=q.terastallized?1:0;/* Showdown runEffectiveness: Stellar is super effective on a Terastallized target, neutral otherwise; it has no chart row */else{for(int i=0;i<2;i++){int t=mon_type(q,i);if(t>=0)e+=rules.chart[move.type*18+t];}if(move.kind==FREEZE_DRY_MOVE&&has_type(q,2))e+=2;}if(q.tera_resisted)e=-1;if(e>0){use_item(s,1-side);int32_t delta[7]={2,0,2,0,0,0,0};apply_boosts(s,q,delta);}}
  }
}
PK_HD inline void use_white_herb(State& s,int side){
  Mon& p=active(s,side);if(p.item!=WHITE_HERB||p.hp<=0||p.fainted)return;
  bool ready=false;for(int i=0;i<7;i++)ready|=p.boosts[i]<0;
  if(!ready)return;use_item(s,side);
  // The frozen observer does not parse -clearnegativeboost; retain its shadow.
  for(int i=0;i<7;i++)if(p.boosts[i]<0)p.boosts[i]=0;
}
PK_HD inline void attack_hazard(State& s,int side,const Move& move){
  if((move.traits&32768)||(move.kind!=STONE_AXE_MOVE&&move.kind!=CEASELESS_EDGE_MOVE))return;
  int j=move.kind==STONE_AXE_MOVE?STEALTH_ROCK:SPIKES,target=1-side,cap=j==SPIKES?3:1;
  if(s.side_conditions[target][j]>=cap)return;
  if(!s.side_conditions[target][j])s.side_effect_order[target][j]=++s.effect_order;
  add_side_condition(s,target,j);
}
PK_HD inline void after_move_items(State& s){
  QueuedAction handlers[2];int count=0;
  for(int side=0;side<2;side++)if(active(s,side).item==WHITE_HERB&&active(s,side).hp>0&&!active(s,side).fainted)handlers[count++]={0,side,0,0,0,0,speed(active(s,side)),0,0};
  sort_action_list(s,handlers,count);for(int i=0;i<count;i++)use_white_herb(s,handlers[i].side);
}
PK_HD inline int execute(State& s,int side,int action,const Rules& rules,int level,bool protected_target,int selected_move=-3,bool called=false,int caller_id=-1,int caller_priority=-100,int caller_flags=0);
PK_HD inline int execute_impl(State& s,int side,int action,const Rules& rules,int level,bool protected_target,int selected_move,bool called,int caller_id,int caller_priority,int caller_flags) {
  Mon& p=active(s,side);Mon& q=active(s,1-side);
  if(p.hp<=0)return 0;
  if(action>=8){switch_to(s,side,action-8);return 0;}
  if(p.recharge&&!called){++p.active_move_actions;p.recharge=0;p.truant=0;p.current_result=3;return 0;}
  int slot=action%4;bool external=caller_flags<0;
  bool locked=!called&&!external&&(p.locked_duration>0||p.charge_duration>0);
  const bool struggle=selected_move==-3?struggling(p):selected_move==-2;
  int move_id=selected_move==-3?p.move_ids[slot]:selected_move;
  if(locked)move_id=p.charge_duration?p.charge_move:p.locked_move;
  bool prankster=called?bool(caller_flags&65536):!struggle&&p.ability==PRANKSTER&&move_id>=0&&move_id<rules.move_count&&rules.moves[move_id].category==2;
  if(!called&&!external&&!locked&&!struggle&&p.encore){move_id=p.encore_move;for(int i=0;i<4;i++)if(p.move_ids[i]==move_id)slot=i;}
  s.used_move[side]=struggle?-2:move_id;
  if(!struggle&&(move_id<0||move_id>=rules.move_count)){s.fault=UNSUPPORTED;return 0;}
  Move fallback{};fallback.id=-1;fallback.power=50;fallback.accuracy=101;fallback.type=-1;fallback.protectable=1;fallback.traits=2048;
  fallback.attack_stat=-1;fallback.defense_stat=-1;
  Move move=struggle?fallback:rules.moves[move_id];
  if(move.unsupported||p.unsupported||q.unsupported){s.fault=UNSUPPORTED;return 0;}
  if(!called){
  p.glaive_rush=0;
  ++p.active_move_actions;
  if(caller_priority!=-100)move.priority=caller_priority;else if(prankster)move.priority++;
  if(action>=4){p.terastallized=1;s.tera_used[side]=1;}
  if(p.heal_block&&p.throat_chop)random(s,2);
  if(p.status==SLP){if(p.ability==EARLY_BIRD)--p.status_turns;if(--p.status_turns>0){if(move.kind!=SLEEP_TALK_MOVE)return 0;}else {p.status=NONE;p.status_turns=0;}}
  if(p.status==FRZ&&!(move.traits&1)){if(random(s,5)!=0)return 0;p.status=NONE;}
  if(p.ability==TRUANT){if(p.truant){p.truant=0;return 0;}mark_volatile(p,29);p.truant=1;}
  if(p.flinch)return 0;
  if(p.disable&&move_id==p.disable_move)return 0;
  if((p.heal_block&&(move.traits&262144))||(p.throat_chop&&(move.traits&16)))return 0;
  if(p.taunt&&move.category==2)return 0;
  if(p.confusion){
    --p.confusion;publish_active(s,side);
    if(p.confusion&&random(s,100)<33){
      int atk=boost_stat(p.stats[0],p.boosts[0]),def=boost_stat(p.stats[1],p.boosts[1]);
      int base=(((2*p.level/5+2)*40*atk/maximum(1,def))/50+2)&65535;
      int loss=maximum(1,base*(100-int(random(s,16)))/100);
      p.hp=maximum(0,p.hp-loss);publish_active(s,side,true);return 0;
    }
  }
  if(p.attract_source&&random(s,2)==0)return 0;
  if(p.status==PAR&&random(s,4)==0)return 0;
  if(move.kind==FOCUS_PUNCH_MOVE&&p.focus_punch==2)return 0;
  if(move.kind!=DESTINY_BOND_MOVE)p.destiny_bond=0;
  if(p.choice_lock>=0&&!choice_item(p))p.choice_lock=-1;
  if(!locked&&!struggle&&choice_item(p)&&p.choice_lock>=0&&p.choice_lock!=move_id){publish_move(s,side,move_id);return 0;}
  // Showdown looks up PP by move ID, including constructed duplicate slots.
  if(!external&&!locked&&!struggle){for(int i=0;i<4;i++)if(p.move_ids[i]==move_id){slot=i;break;}if(p.pp[slot]<=0)return 0;}
  if(!external&&!locked&&!struggle){p.pp[slot]=maximum(0,p.pp[slot]-(q.hp>0&&q.ability==PRESSURE&&move.pressure_target&&!(move.kind==CURSE_MOVE&&!has_type(p,13))?2:1));s.move_revealed[side][s.active[side]][slot]=1;if(choice_item(p)){mark_volatile(p,17);p.choice_lock=move_id;}}
  if(!struggle&&choice_item(p)&&p.choice_lock<0){mark_volatile(p,17);p.choice_lock=move_id;}
  if(!external)p.last_move=struggle?-2:move_id;s.after_move[side]=1;p.current_result=0;
  }else{
    move.priority=caller_priority;
    // Called attacks charge Pressure's extra PP to Sleep Talk, never to
    // the selected move (whose PP and disabled state are immaterial).
    if(q.hp>0&&q.ability==PRESSURE&&move.pressure_target&&!(move.kind==CURSE_MOVE&&!has_type(p,13)))for(int i=0;i<4;i++)if(p.move_ids[i]==caller_id){p.pp[i]=maximum(0,p.pp[i]-1);break;}
  }
  if(p.status==FRZ&&(move.traits&1)){p.status=NONE;publish_active(s,side);}
  if(p.heal_block&&p.throat_chop)random(s,2);
  if((p.heal_block&&(move.traits&262144))||(p.throat_chop&&(move.traits&16)))return 0;
  if(called)move.traits=(move.traits&~268435456)|(caller_flags&268435456);
  if((move.traits&268435456)||(p.ability==MYCELIUM_MIGHT&&move.category==2)||p.ability==MOLD_BREAKER||p.ability==TERAVOLT||p.ability==TURBOBLAZE){s.ignore_ability_side=side+1;suppress_opponent(s);}
  if(move.kind==SHELL_SIDE_ARM_MOVE){int physical=((2*p.level/5+2)*90*boost_stat(p.stats[0],p.boosts[0])/maximum(1,boost_stat(q.stats[1],q.boosts[1])))/50;int special=((2*p.level/5+2)*90*boost_stat(p.stats[2],p.boosts[2])/maximum(1,boost_stat(q.stats[3],q.boosts[3])))/50;if(physical>special||(physical==special&&random(s,2)==0)){move.category=0;move.traits|=2048;}}
  if(move.kind==PHOTON_GEYSER_MOVE&&boost_stat(p.stats[0],p.boosts[0])>boost_stat(p.stats[2],p.boosts[2]))move.category=0;
  if(move.kind==JUDGMENT_MOVE&&plate_type(p.item)>=0)move.type=plate_type(p.item);
  if(p.ability==LIQUID_VOICE&&(move.traits&16))move.type=2;
  if((p.ability==PIXILATE||p.ability==GALVANIZE)&&move.type==0&&move.kind!=WEATHER_BALL_MOVE&&move.kind!=REVELATION_DANCE_MOVE&&move.kind!=JUDGMENT_MOVE&&!(move.kind==TERA_BLAST_MOVE&&p.terastallized)){move.type=p.ability==PIXILATE?17:4;move.traits|=8388608;}
  if(p.ability==KEEN_EYE||p.ability==MINDS_EYE)move.traits|=512;
  if((p.ability==SCRAPPY||p.ability==MINDS_EYE)&&(move.type==0||move.type==6))move.traits|=16777216;
  if(p.ability==UNSEEN_FIST&&(move.traits&2048))move.protectable=0;
  if(prankster)move.traits|=65536;
  if(move.kind==IVY_CUDGEL_MOVE){if(p.form_family==10)move.type=2;else if(p.form_family==11)move.type=1;else if(p.form_family==12)move.type=12;}
  if(move.kind==TERASTAR_STORM_MOVE&&p.form_family==8&&p.form_current==2){move.type=STELLAR;if(p.terastallized&&boost_stat(p.stats[0],p.boosts[0])>boost_stat(p.stats[2],p.boosts[2]))move.category=0;}
  if(move.kind==AURA_WHEEL_MOVE&&p.form_family==2)move.type=p.form_current?16:4;
  if(move.kind==RAGING_BULL_MOVE){int form=(p.species_flags>>8)&7;if(form)move.type=form==1?6:form==2?1:2;}
  s.used_move[side]=struggle?-2:move_id;s.used_move_hit[side]=0;
  publish_move(s,side,struggle?-2:move_id);
  if(move.kind==SOLAR_BEAM_MOVE||move.kind==METEOR_BEAM_MOVE){
    if(p.charge_duration&&p.charge_move==move_id)p.charge_released=1;
    else if(move.kind!=SOLAR_BEAM_MOVE||effective_weather(s)!=SUN){
      if(move.kind==METEOR_BEAM_MOVE){int32_t delta[7]={0,0,1,0,0,0,0};apply_boosts(s,p,delta);}
      if(p.item==POWER_HERB)use_item(s,side);
      else {mark_volatile(p,21);p.charge_duration=2;p.charge_move=move_id;p.charge_kind=move.kind;p.charge_released=0;mark_volatile(p,move.kind==SOLAR_BEAM_MOVE?22:23);prepare_type(s,side,move);p.current_result=3;return 0;}
    }
  }
  if(move.kind==EXPLOSION_MOVE&&(p.ability==DAMP||q.ability==DAMP))return 0;
  if(move.kind==AURA_WHEEL_MOVE&&p.form_family!=2)return 0;
  if(move.kind==HYPERSPACE_FURY_MOVE&&!(p.species_flags&2048))return 0;
  if(move.kind==DESTINY_BOND_MOVE){if(p.destiny_bond){p.destiny_bond=0;return 0;}prepare_type(s,side,move);mark_volatile(p,38);p.destiny_bond=1;each_event(s);return 1;}
  if(move.kind==SHED_TAIL_MOVE){if(living(s,side)<=1||p.substitute||p.hp<=(p.maxhp+1)/2)return 0;prepare_type(s,side,move);mark_volatile(p,8);p.substitute=p.maxhp/4;p.inherited_substitute=0;/* its own, announced by -start; a broken passed one must not hide it */p.partial_trap=0;p.partial_source=0;p.hp-=(p.maxhp+1)/2;s.shed_tail[side]=1;s.pivot[side]=1;publish_active(s,side,true);each_event(s);return 1;}
  if(move.kind==HEALING_WISH_MOVE){if(living(s,side)<=1)return 0;prepare_type(s,side,move);s.healing_wish[side]=1;p.hp=0;publish_active(s,side,true);each_event(s);return 1;}
  if(move.kind==FUTURE_SIGHT_MOVE){int target=1-side;if(s.future_turn[target])return 0;s.future_move[target]=move_id;s.future_turn[target]=s.turn+2;s.future_source[target]=side;s.future_slot[target]=s.active[side];s.future_order[target]=++s.effect_order;return 3;}
  if(move.kind==TRANSFORM_MOVE){if(q.ability==GOOD_AS_GOLD||((move.traits&65536)&&has_type(q,16)))return 0;prepare_type(s,side,move);if(q.substitute)return 1;bool success=transform_into(s,side);if(s.fault)return 0;if(!success)return 0;each_event(s);return 1;}
  if(move.kind==MIRROR_COAT_MOVE){if(!p.mirror_coat||!p.mirror_damage)return 0;move.fixed_damage=p.mirror_damage;}
  if(move.kind==NO_RETREAT_MOVE&&p.no_retreat)return 0;
  if(move.kind==DOUBLE_SHOCK_MOVE&&!has_type(p,4)){p.current_result=3;return 0;}
  if(!move.target_self&&q.ability==QUEENLY_MAJESTY&&move.priority>0&&!(move.traits&(33554432|67108864)))return 0;
  if(move.kind==SLEEP_TALK_MOVE){
    if(p.status!=SLP&&p.ability!=COMATOSE)return 0;
    int options[4],count=0;for(int i=0;i<4;i++){int id=p.move_ids[i];if(id>=0&&id<rules.move_count&&!(rules.moves[id].traits&1024))options[count++]=id;}
    if(!count)return 0;
    int chosen=options[random(s,count)];
    int hit=execute(s,side,0,rules,level,false,chosen,true,move_id,move.priority,move.traits);
    s.used_move_hit[side]=hit!=0;
    // Finish the called hit loop, then the caller's primary hit loop.
    if(hit&&hit!=3)each_event(s);
    each_event(s);return 1;
  }
  if(p.ability==SHEER_FORCE&&(move.traits&16384)){move.effects&=~(SECONDARY|SELF_STAT);move.traits|=32768;}
  if(p.ability==SERENE_GRACE){move.secondary_chance*=2;move.self_chance*=2;move.additional_flinch_chance*=2;}
  if(p.ability==INFILTRATOR)move.traits|=8|131072;
  if(move.kind==ACROBATICS_MOVE&&!p.item)move.power*=2;
  if(move.kind==FACADE_MOVE&&p.status&&p.status!=SLP)move.power*=2;
  if(move.kind==BOOST_POWER_MOVE){for(int i=0;i<7;i++)move.power+=20*maximum(0,p.boosts[i]);}
  if(move.kind==HEAVY_SLAM_MOVE){int ratio=mon_weight(p)/mon_weight(q);move.power=ratio>=5?120:ratio>=4?100:ratio>=3?80:ratio>=2?60:40;}
  if(move.kind==WEIGHT_POWER_MOVE){int weight=mon_weight(q);move.power=weight>=2000?120:weight>=1000?100:weight>=500?80:weight>=250?60:weight>=100?40:20;}
  if(move.kind==FAILED_POWER_MOVE&&p.last_result==1)move.power*=2;
  if(move.kind==HP_POWER_MOVE)move.power=maximum(1,move.power*p.hp/maximum(1,p.maxhp));
  if((move.kind==HEX_MOVE&&(q.status||q.ability==COMATOSE))||(move.kind==VENOSHOCK_MOVE&&(q.status==PSN||q.status==TOX))||(move.kind==BRINE_MOVE&&q.hp*2<=q.maxhp))move.power*=2;
  if(move.kind==RAGE_FIST_MOVE)move.power=minimum(350,50+50*p.times_attacked);
  if(move.kind==LAST_RESPECTS_MOVE)move.power=50+50*s.total_fainted[side];
  if(move.kind==PAYBACK_MOVE){bool pending=false;for(int i=0;i<s.queue_length;i++)pending|=s.queue[i].kind==MOVE&&s.queue[i].side==1-side&&s.queue[i].actor==s.active[1-side];if(!q.newly_switched&&!pending)move.power*=2;}
  if(move.kind==AVALANCHE_MOVE&&(p.hit_by_foe&(1<<s.active[1-side])))move.power*=2;
  if(move.kind==RAIN_ACCURACY_MOVE){if(effective_weather(s)==RAIN)move.accuracy=101;else if(effective_weather(s)==SUN)move.accuracy=50;}
  if(move.kind==BLEAKWIND_STORM_MOVE&&effective_weather(s)==RAIN)move.accuracy=101;
  if(move.kind==SNOW_ACCURACY_MOVE&&(effective_weather(s)==SNOW||effective_weather(s)==HAIL))move.accuracy=101;
  if(move.kind==WEATHER_BALL_MOVE&&effective_weather(s)){const int types[6]={0,2,1,12,5,5};move.power*=2;move.type=types[effective_weather(s)];}
  if(move.kind==TERA_BLAST_MOVE&&p.terastallized){
    move.type=p.tera_type;
    if(boost_stat(p.stats[0],p.boosts[0])>boost_stat(p.stats[2],p.boosts[2]))move.category=0;
    if(p.tera_type==STELLAR){move.power=100;move.effects|=SELF_STAT;move.self_chance=100;move.self_stat_delta[0]=-1;move.self_stat_delta[2]=-1;}
  }
  if(!move.target_self&&q.hp>0&&!q.fainted&&q.ability==MAGIC_BOUNCE&&(move.traits&536870912)&&!(caller_flags&1073741824)){
    prepare_type(s,side,move);
    execute(s,1-side,0,rules,level,false,move_id,true,-1,move.priority,1073741824);
    p.current_result=1;return 0;
  }
  if(move.effects&PROTECT){
    bool will_act=false;for(int i=0;i<s.queue_length;i++)will_act|=s.queue[i].kind==MOVE||s.queue[i].kind==SWITCH;
    if(!will_act)return 0;
    if(p.stall_counter&&random(s,p.stall_counter)!=0){p.stall_counter=0;p.stall_duration=0;return 0;}
    prepare_type(s,side,move);
    mark_volatile(p,11);p.protect=1;mark_volatile(p,12);p.stall_counter=p.stall_counter?minimum(729,p.stall_counter*3):3;p.stall_duration=2;
    each_event(s);
    return 1;
  }
  if(move.kind==COURT_CHANGE_MOVE){
    prepare_type(s,side,move);bool success=false;
    for(int j=0;j<10;j++){success|=s.side_conditions[0][j]||s.side_conditions[1][j];
      int x=s.side_conditions[0][j];s.side_conditions[0][j]=s.side_conditions[1][j];s.side_conditions[1][j]=x;
      x=s.side_durations[0][j];s.side_durations[0][j]=s.side_durations[1][j];s.side_durations[1][j]=x;
      x=s.side_effect_order[0][j];s.side_effect_order[0][j]=s.side_effect_order[1][j];s.side_effect_order[1][j]=x;
    }return success?3:0;
  }
  if(move.effects&TRICK_ROOM){prepare_type(s,side,move);s.trick_room=s.trick_room?0:5;return 3;}
  if(move.effects&WEATHER){prepare_type(s,side,move);bool changed=set_weather(s,move.weather,p);if(move.kind==CHILLY_RECEPTION_MOVE){bool can_switch=living(s,side)>1;if(can_switch)s.pivot[side]=1;return changed||can_switch?3:0;}return changed?3:0;}
  if(move.effects&TERRAIN){prepare_type(s,side,move);return set_terrain(s,move.terrain,p)?3:0;}
  if(move.kind==CURSE_MOVE&&!has_type(p,13))move.target_self=1;
  if(move.kind==REVELATION_DANCE_MOVE)move.type=mon_type(p,0);
  if(move.kind==TELEPORT_MOVE&&living(s,side)<=1)return 0;
  if((move.kind==FILLET_AWAY_MOVE&&(p.hp*2<=p.maxhp||p.maxhp==1))||(move.kind==CLANGOROUS_SOUL_MOVE&&(p.hp*100<=p.maxhp*33||p.maxhp==1)))return 0;
  if(move.kind==FIRST_MOVE_ONLY&&p.active_move_actions>1)return 0;
  if(move.kind==POLTERGEIST_MOVE&&!q.item)return 0;
  if(move.kind==SUCKER_PUNCH_MOVE){
    bool attack_pending=false;for(int i=0;i<s.queue_length;i++){const QueuedAction& a=s.queue[i];if(a.kind==MOVE&&a.side==1-side&&a.actor==s.active[1-side])attack_pending|=a.move_id==-2||(a.move_id>=0&&rules.moves[a.move_id].category!=2);}
    if(!attack_pending)return 0;
  }
  if(move.kind==REST_MOVE){if(p.ability==COMATOSE)return 0;
    if(p.hp==p.maxhp||p.status==SLP||p.ability==INSOMNIA||p.ability==VITAL_SPIRIT)return 0;
    prepare_type(s,side,move);
    if(!set_status(s,p,SLP,side,false,true,true))return 0;
    p.status_turns=3;p.hp=p.maxhp;publish_active(s,side,true);each_event(s);return 1;
  }
  if(move.kind==WEATHER_HEAL_MOVE||move.kind==SHORE_UP_MOVE){
    prepare_type(s,side,move);
    int weather=effective_weather(s);int healed=modify(p.maxhp,move.kind==SHORE_UP_MOVE?(weather==SAND?2732:2048):weather==SUN?2732:weather?1024:2048);
    int before=p.hp;p.hp=minimum(p.maxhp,p.hp+healed);publish_active(s,side,p.hp!=before);each_event(s);return 1;
  }
  if(move.kind==HAZE_MOVE){
    prepare_type(s,side,move);
    for(int a=0;a<2;a++)for(int i=0;i<6;i++)for(int j=0;j<7;j++){if(i==s.active[a])s.mons[a][i].boosts[j]=0;s.public_mons[a][i].boosts[j]=0;}return 3;
  }
  if(move.kind==REVIVAL_BLESSING_MOVE){
    prepare_type(s,side,move);bool dead=false;for(int i=0;i<6;i++)dead|=s.mons[side][i].maxhp>0&&s.mons[side][i].fainted;
    if(!dead)return 0;s.reviving[side]=1;s.pivot[side]=1;each_event(s);return 1;
  }
  if(move.kind==WISH_MOVE){
    prepare_type(s,side,move);
    if(s.wish_pending[side])return 0;s.wish_pending[side]=1;s.wish_hp[side]=p.maxhp/2;s.wish_turn[side]=(s.turn-1)&255;each_event(s);return 1;
  }
  if(move.effects&SIDE_CONDITION){
    prepare_type(s,side,move);
    int condition=move.side_condition,target=move.target_self?side:1-side;
    if(condition<4){int maximum_layers=condition==SPIKES?3:condition==TOXIC_SPIKES?2:1;if(s.side_conditions[target][condition]>=maximum_layers)return 0;if(!s.side_conditions[target][condition])s.side_effect_order[target][condition]=++s.effect_order;add_side_condition(s,target,condition);return 3;}
    if(s.side_conditions[target][condition]||(condition==AURORA_VEIL&&effective_weather(s)!=SNOW&&effective_weather(s)!=HAIL))return 0;
    add_side_condition(s,target,condition);s.side_durations[target][condition]=condition==TAILWIND?4:5;
    if(condition==TAILWIND&&active(s,target).ability==WIND_RIDER){int32_t delta[7]={1,0,0,0,0,0,0};if(apply_boosts(s,active(s,target),delta))publish_ability(s,target,s.public_active[target],WIND_RIDER);}
    if(condition>=REFLECT&&condition<=AURORA_VEIL&&p.item==LIGHT_CLAY)s.side_durations[target][condition]=8;
    return 3;
  }
  if(!move.target_self&&q.hp<=0)return 0;
  prepare_type(s,side,move);
  // TryHit order: Psychic Terrain (priority 4) before Protect and its variants (3), then Good as Gold (0).
  if(!move.target_self&&s.terrain==PSYCHIC_TERRAIN&&move.priority>0&&grounded(q))return 0;
  if(q.protect&&move.protectable){p.current_result=3;return 0;}
  if(!move.target_self&&move.category==2&&q.ability==GOOD_AS_GOLD&&!(move.traits&(33554432|67108864)))return 0;
  if(!move.target_self&&((q.ability==SOUNDPROOF&&(move.traits&16))||(q.ability==BULLETPROOF&&(move.traits&32))||(q.ability==OBLIVIOUS&&move.volatile_status==TAUNT)))return 0;
  if(move.kind==CURSE_MOVE){
    if(!has_type(p,13)){move.effects&=~VOLATILE;move.effects|=SELF_STAT;move.self_chance=100;move.self_stat_delta[0]=1;move.self_stat_delta[1]=1;move.self_stat_delta[4]=-1;}
    else if((move.target_self?p:q).curse)return 0;
  }
  if(move.kind==FILLET_AWAY_MOVE||move.kind==CLANGOROUS_SOUL_MOVE){if(!apply_boosts(s,p,move.stat_delta))return 0;move.effects&=~STAT;}
  if(move.kind==AURA_WHEEL_MOVE&&p.form_family!=2)return 0;
  if(move.kind==HYPERSPACE_FURY_MOVE&&q.protect){q.protect=0;q.stall_counter=0;q.stall_duration=0;}
  if(move.kind==BREAK_SCREENS_MOVE||move.kind==RAGING_BULL_MOVE)for(int j=REFLECT;j<=AURORA_VEIL;j++){remove_side_condition(s,1-side,j);}
  if(!move.target_self){
    if(q.ability==FLASH_FIRE&&move.type==1){if(!q.flash_fire){mark_volatile(q,18);q.flash_fire=1;}return 0;}
    if(q.ability==WIND_RIDER&&(move.traits&4194304)){int32_t delta[7]={1,0,0,0,0,0,0};if(apply_boosts(s,q,delta)){publish_ability(s,1-side,s.public_active[1-side],q.ability);s.requests[1-side].abilities[s.public_active[1-side]]=q.ability;}return 0;}
    bool heal_immunity=((q.ability==WATER_ABSORB||q.ability==DRY_SKIN)&&move.type==2)||(q.ability==VOLT_ABSORB&&move.type==4)||(q.ability==EARTH_EATER&&move.type==8);
    bool boost_immunity=(q.ability==SAP_SIPPER&&move.type==3)||(q.ability==STORM_DRAIN&&move.type==2)||((q.ability==LIGHTNING_ROD||q.ability==MOTOR_DRIVE)&&move.type==4)||(q.ability==WELL_BAKED_BODY&&move.type==1);
    if(heal_immunity){int before=q.hp;if(!q.heal_block)q.hp=minimum(q.maxhp,q.hp+maximum(1,q.maxhp/4));publish_active(s,1-side,q.hp!=before);return 0;}
    if(boost_immunity){int32_t delta[7]={};delta[q.ability==SAP_SIPPER?0:q.ability==MOTOR_DRIVE?4:q.ability==WELL_BAKED_BODY?1:2]=q.ability==WELL_BAKED_BODY?2:1;if(apply_boosts(s,q,delta)){publish_ability(s,1-side,s.public_active[1-side],q.ability);s.requests[1-side].abilities[s.public_active[1-side]]=q.ability;}return 0;}
  }
  if(!move.target_self&&(move.traits&524288)&&(has_type(q,3)||q.ability==OVERCOAT||q.item==SAFETY_GOGGLES))return 0;
  if(move.volatile_status==LEECH_SEED&&has_type(q,3))return 0;
  if(!move.ignore_immunity&&move.type>=0&&move.type<18){
    bool tera_defense=q.terastallized&&q.tera_type!=STELLAR;
    const int t0=mon_type(q,0),t1=mon_type(q,1);
    if(move.type==8&&(q.ability==LEVITATE||q.item==AIR_BALLOON||q.magnet_rise))return 0;
    if((t0>=0&&move.type<STELLAR&&rules.chart[move.type*18+t0]==-127&&!((move.traits&16777216)&&t0==13))||(t1>=0&&move.type<STELLAR&&rules.chart[move.type*18+t1]==-127&&!((move.traits&16777216)&&t1==13)))return 0;
  }
  if(!move.target_self&&prankster&&has_type(q,16))return 0;
  if(move.kind==ENDEAVOR_MOVE&&p.hp>=q.hp)return 0;
  if(move.kind==TRICK_MOVE&&q.ability==STICKY_HOLD)return 0;
  const Mon& accuracy_target=move.target_self?p:q;
  accuracy_event_order(s,p,accuracy_target);int accuracy=move.accuracy;
  if(move.accuracy<=100){
    if(p.ability==UNAWARE&&q.ability==UNAWARE&&speed(p)==speed(q)){if(!(move.traits&256))random(s,2);if(!(move.traits&512))random(s,2);}
    int boost=maximum(-6,minimum(6,((move.traits&256)||accuracy_target.ability==UNAWARE?0:p.boosts[5])-((move.traits&512)||p.ability==UNAWARE?0:accuracy_target.boosts[6])));
    int base_accuracy=modify(move.accuracy,accuracy_modifier(s,p,accuracy_target,move));
    accuracy=boost>=0?base_accuracy*(3+boost)/3:base_accuracy*3/(3-boost);
  }
  bool auto_accuracy=(move.target_self&&move.category==2)||(move.kind==TOXIC_MOVE&&has_type(p,7));
  if(!auto_accuracy){
    if(p.ability==NO_GUARD&&q.ability==NO_GUARD&&speed(p)==speed(q))random(s,2);
    if(p.ability!=NO_GUARD&&accuracy_target.ability!=NO_GUARD&&!accuracy_target.glaive_rush&&move.accuracy<=100&&int(random(s,100))>=accuracy)return 0;
  }
  if(protected_target)return 0;
  if(move.volatile_status==DISABLE&&q.last_move<0)return 0;
  if(move.volatile_status==YAWN&&q.status)return 0;
  if((move.effects&FORCE_SWITCH)&&move.category==2&&living(s,1-side)<=1)return 0;
  if(move.kind==SURF_MOVE&&p.ability==GULP_MISSILE&&p.form_family==6&&p.form_current==0)change_form(s,side,p.hp*2<=p.maxhp?2:1);
  if(move.kind==HALF_HP_MOVE)move.fixed_damage=maximum(1,q.hp/2);
  if(move.kind==ENDEAVOR_MOVE)move.fixed_damage=q.hp-p.hp;
  if(move.kind==WATER_SHURIKEN_MOVE&&p.ability==BATTLE_BOND&&(p.species_flags&16384)){move.power=20;move.hit_min=move.hit_max=3;}
  q.tera_resisted=0;
  int beat_power[6],beat_count=0;if(move.kind==BEAT_UP_MOVE){for(int i=0;i<6;i++){const Mon& ally=s.mons[side][s.order[side][i]];if(&ally==&p||(ally.maxhp>0&&!ally.fainted&&!ally.status))beat_power[beat_count++]=5+ally.base_attack/10;}move.hit_min=move.hit_max=beat_count;}
  int hits=move.hit_max?move.hit_max:1,total_damage=0,damage_hits=0,substitute_hits=0;bool any_substitute=false,aria_cure=false,final_target=true;
  if(move.hit_min&&move.hit_min!=move.hit_max&&p.ability!=SKILL_LINK){
    if(move.hit_min==2&&move.hit_max==5){int roll=random(s,20);hits=roll<7?2:roll<14?3:roll<17?4:5;if(hits<4&&p.item==LOADED_DICE)hits=5-random(s,2);}
    else hits=move.hit_min+random(s,move.hit_max-move.hit_min+1);
  }
  if(hits==10&&p.item==LOADED_DICE)hits-=random(s,7);
  for(int hit=0;hit<hits;hit++){
    if(p.hp<=0||(!move.target_self&&q.hp<=0))break;
    if(hit&&p.status==SLP&&!called)break;
    final_target=true;
    if(hit&&move.multi_accuracy&&p.ability!=SKILL_LINK&&p.item!=LOADED_DICE&&move.accuracy<=100){
      const double boosts[7]={1,4.0/3,5.0/3,2,7.0/3,8.0/3,3};double accuracy=move.accuracy;
      accuracy_event_order(s,p,q);accuracy=modify(int(accuracy),accuracy_modifier(s,p,q,move));
      if(p.ability==UNAWARE&&q.ability==UNAWARE&&speed(p)==speed(q)){if(!(move.traits&256))random(s,2);if(!(move.traits&512))random(s,2);}
      int a=((move.traits&256)||q.ability==UNAWARE)?0:maximum(-6,minimum(6,p.boosts[5])),b=((move.traits&512)||p.ability==UNAWARE)?0:maximum(-6,minimum(6,q.boosts[6]));
      accuracy=a>=0?accuracy*boosts[a]:accuracy/boosts[-a];accuracy=b>=0?accuracy/boosts[b]:accuracy*boosts[-b];
      if(p.ability==NO_GUARD&&q.ability==NO_GUARD&&speed(p)==speed(q))random(s,2);
      if(p.ability!=NO_GUARD&&q.ability!=NO_GUARD&&!q.glaive_rush&&double(random(s,100))>=accuracy)break;
    }
    if(move.kind==BEAT_UP_MOVE)move.power=beat_power[hit];
    if(move.kind==TRIPLE_AXEL_MOVE)move.power=20*(hit+1);
    const int before=q.hp,own_before=p.hp;
    bool hit_substitute=!move.target_self&&q.substitute>0&&!(move.traits&8);
    if(hit_substitute){
      any_substitute=true;final_target=false;
      int dealt=move.fixed_damage?(move.fixed_damage<0?p.level:move.fixed_damage):move.power>0?damage(s,p,q,move,rules,p.level):0;
      dealt=minimum(dealt,q.substitute);q.substitute-=dealt;if(move.power>0||move.fixed_damage)substitute_hits++;
      if(struggle)p.hp=maximum(0,p.hp-maximum(1,round_ratio(p.maxhp,1,4)));
      if((move.effects&RECOIL)&&p.ability!=ROCK_HEAD&&dealt)indirect_loss(p,maximum(1,round_ratio(dealt,move.recoil_num,maximum(1,move.recoil_den))));
      if((move.effects&DRAIN)&&dealt>0)drain_heal(s,side,maximum(1,(dealt*move.drain_num+maximum(1,move.drain_den)-1)/maximum(1,move.drain_den)));
      if((move.power>0||move.fixed_damage)&&q.item==AIR_BALLOON){pop_balloon(s,1-side);}
      if(move.kind==RECHARGE_MOVE&&p.hp>0){mark_volatile(p,30);p.recharge=2;}
      if(move.kind==GLAIVE_RUSH_MOVE){mark_volatile(p,25);p.glaive_rush=1;}
      if(move.kind==DOUBLE_SHOCK_MOVE&&!p.terastallized){for(int i=0;i<2;i++)if(p.type[i]==4)p.type[i]=-2;}
    if(move.kind==RAMPAGE_MOVE)start_locked(s,p,move_id);
      if((move.effects&SELF_STAT)&&int(random(s,100))<move.self_chance)apply_boosts(s,p,move.self_stat_delta);
      if(move.effects&SECONDARY){int roll=random(s,100);if(move.secondary_self&&roll<move.secondary_chance){if(move.secondary_status)set_status(s,p,move.secondary_status,side);move_boosts(s,p,move.secondary_stat_delta,side);if(move.secondary_volatile)add_volatile(s,p,move.secondary_volatile,rules,side);}if(move.additional_flinch_chance)random(s,100);}
      if(p.hp>0)attack_hazard(s,side,move);
      if(move.kind==SPIN_MOVE&&p.hp>0)clear_hazards(s,side);
      if(move.kind==ICE_SPINNER_MOVE&&p.hp>0)set_terrain(s,0,p);
      publish_active(s,side,p.hp!=own_before);publish_active(s,1-side);each_event(s);continue;
    }
    if(move.power>0||move.fixed_damage){int dealt=move.fixed_damage?move.fixed_damage<0?p.level:move.fixed_damage:damage(s,p,q,move,rules,p.level);if(q.ability==STURDY&&q.hp==q.maxhp&&dealt>=q.hp){publish_ability(s,1-side,s.public_active[1-side],q.ability);dealt=q.hp-1;}if(q.item==FOCUS_SASH&&q.hp==q.maxhp&&dealt>=q.hp){use_item(s,1-side);dealt=q.hp-1;}if(q.ability==BERSERK||q.ability==ANGER_SHELL)q.block_heal_berry=!move.hit_max&&!(move.traits&32768);if(((q.ability==DISGUISE&&q.form_family==4)||(q.ability==ICE_FACE&&q.form_family==5&&move.category==0))&&q.form_current==0){dealt=0;q.form_busted=1;}q.hp=maximum(0,q.hp-dealt);if(q.hp==0&&before>0){q.faint_source_side=side+1;q.faint_source_slot=s.active[side];}damage_hits++;}
    if(!move.target_self)q.got_attacked=1;
    total_damage+=before-q.hp;if(before>q.hp)q.hit_by_foe|=1<<s.active[side];
    if(move.effects&HEAL){if(p.hp==p.maxhp&&move.category==2)return 0;p.hp=minimum(p.maxhp,p.hp+maximum(1,round_ratio(p.maxhp,move.heal_num,maximum(1,move.heal_den))));if(move.kind==ROOST_MOVE&&!p.terastallized){mark_volatile(p,13);p.roost=1;}}
    if((move.effects&DRAIN)&&before>q.hp)drain_heal(s,side,maximum(1,round_ratio(before-q.hp,move.drain_num,maximum(1,move.drain_den))));
    if(move.kind==PAIN_SPLIT_MOVE){int average=maximum(1,(p.hp+q.hp)/2);p.hp=minimum(p.maxhp,average);q.hp=minimum(q.maxhp,average);publish_active(s,side,true);publish_active(s,1-side,true);}
    if(move.kind==STRENGTH_SAP_MOVE){
      if(q.boosts[0]==-6)return 0;boost_event_order(s);int heal=boost_stat(q.stats[0],q.boosts[0]);
      int32_t delta[7]={-1,0,0,0,0,0,0};bool success=move_boosts(s,q,delta,side);int before=p.hp;drain_heal(s,side,heal);if(!success&&before==p.hp)return 0;
    }
    if(move.kind==TRICK_MOVE){
      int mine=p.item,theirs=q.item;bool your_ok=take_item(q,&p),mine_ok=take_item(p);
      p.item=theirs;q.item=mine;
      bool destination_ok=can_take_item(q,p)&&can_take_item(p,q);p.item=mine;q.item=theirs;
      if(!your_ok||!mine_ok||(!mine&&!theirs)||!destination_ok)return 0;
      p.item=theirs;q.item=mine;p.booster_started=0;q.booster_started=0;
      if(choice_item(p))p.choice_lock=-1;if(choice_item(q))q.choice_lock=-1;
      s.public_mons[side][s.public_active[side]].item=p.item;s.public_mons[1-side][s.public_active[1-side]].item=q.item;
      // Target.setItem runs before Source.setItem, including item Start hooks.
      if(q.item==WHITE_HERB)use_white_herb(s,1-side);
      if(p.item==WHITE_HERB)use_white_herb(s,side);
      if(q.item==BOOSTER_ENERGY){q.booster_started=1;update_booster(s,1-side);}
      if(p.item==BOOSTER_ENERGY){p.booster_started=1;update_booster(s,side);}
      // Trick publishes its transferred item after each item Start callback,
      // even when the callback immediately consumes that item.
      s.public_mons[side][s.public_active[side]].item=theirs;s.public_mons[1-side][s.public_active[1-side]].item=mine;
    }
    if(move.kind==BUG_BITE_MOVE&&p.hp>0&&berry_item(q.item)){int berry=q.item;if(can_take_item(q,p)&&take_item(q,&p)){
      s.public_mons[1-side][s.public_active[1-side]].item=0;
      if(berry==SITRUS_BERRY&&!p.heal_block){int before=p.hp;p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/4));publish_active(s,side,p.hp!=before);}
      if(berry==LUM_BERRY||(berry==CHESTO_BERRY&&p.status==SLP)){p.status=NONE;p.status_turns=0;p.toxic_turns=0;if(berry==LUM_BERRY)p.confusion=0;}
      if(berry==LEPPA_BERRY){int selected=-1;for(int i=0;i<4;i++)if(p.move_ids[i]>=0&&p.pp[i]==0){selected=i;break;}if(selected<0)for(int i=0;i<4;i++)if(p.move_ids[i]>=0&&p.pp[i]<p.maxpp[i]){selected=i;break;}if(selected>=0)p.pp[selected]=minimum(p.maxpp[selected],p.pp[selected]+10);}
      if(p.ability==CHEEK_POUCH&&!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/3));
    }}
    if(move.kind==CLEAR_SMOG_MOVE)for(int i=0;i<7;i++){q.boosts[i]=0;s.public_mons[1-side][s.public_active[1-side]].boosts[i]=0;}
    if(move.kind==TAKE_HEART_MOVE){int32_t delta[7]={0,0,1,1,0,0,0};bool changed=apply_boosts(s,p,delta),status=p.status!=NONE;if(status){p.status=NONE;p.status_turns=0;p.toxic_turns=0;publish_active(s,side);}if(!changed&&!status)return 0;}
    if(move.kind==HEAL_BELL_MOVE){bool changed=false;for(int i=0;i<6;i++){Mon& ally=s.mons[side][i];if(ally.status){changed=true;ally.status=NONE;ally.status_turns=0;ally.toxic_turns=0;s.public_mons[side][i].status=NONE;s.requests[side].status[i]=NONE;}}return changed?3:0;}
    if(move.kind==BELLY_DRUM_MOVE){if(p.hp*2<=p.maxhp||p.boosts[0]>=6||p.maxhp==1)return 0;p.hp-=p.maxhp/2;int32_t delta[7]={12,0,0,0,0,0,0};apply_boosts(s,p,delta);}
    if(move.kind==FILLET_AWAY_MOVE||move.kind==CLANGOROUS_SOUL_MOVE)p.hp-=move.kind==FILLET_AWAY_MOVE?p.maxhp/2:maximum(1,p.maxhp*33/100);
    if(move.kind==PARTING_SHOT_MOVE){int32_t delta[7]={-1,0,-1,0,0,0,0};if(!move_boosts(s,q,delta,side)&&q.ability!=MIRROR_ARMOR)move.effects&=~PIVOT;}
    bool boost_changed=false;
    if(move.effects&STAT)boost_changed=move_boosts(s,move.target_self?p:q,move.stat_delta,side);
    if(move.kind==DEFOG_MOVE){
      bool changed=false;if(!q.substitute||p.ability==INFILTRATOR){const int32_t delta[7]={0,0,0,0,0,0,-1};changed=move_boosts(s,q,delta,side);}
      for(int target=0;target<2;target++)for(int j=0;j<10;j++)if(j<4||(target==1-side&&j!=TAILWIND)){if(s.side_conditions[target][j]&&j<4)changed=true;remove_side_condition(s,target,j);}
      set_terrain(s,0,p);
      if(!changed)return 0;
    }
    if(move.kind==TIDY_UP_MOVE){
      bool changed=false;for(int target=0;target<2;target++){changed|=active(s,target).substitute>0;active(s,target).substitute=0;for(int j=0;j<4;j++){changed|=s.side_conditions[target][j]>0;remove_side_condition(s,target,j);}}
      const int32_t delta[7]={1,0,0,0,1,0,0};changed|=apply_boosts(s,p,delta);if(!changed)return 0;
    }
    if((move.effects&STATUS)&&!set_status(s,move.target_self?p:q,move.secondary_status,side)&&move.category==2)return 0;
    if((move.effects&VOLATILE)&&!add_volatile(s,move.target_self?p:q,move.volatile_status,rules,side)&&!boost_changed&&move.category==2)return 0;
    if((move.effects&STAT)&&!(move.effects&VOLATILE)&&!boost_changed&&move.category==2)return 0;
    if(move.kind==CURSE_MOVE&&(move.effects&VOLATILE))p.hp=maximum(0,p.hp-p.maxhp/2);
    if(move.kind==RECHARGE_MOVE&&p.hp>0){mark_volatile(p,30);p.recharge=2;}
      if(move.kind==GLAIVE_RUSH_MOVE){mark_volatile(p,25);p.glaive_rush=1;}
      if(move.kind==DOUBLE_SHOCK_MOVE&&!p.terastallized){for(int i=0;i<2;i++)if(p.type[i]==4)p.type[i]=-2;}
    if(move.kind==RAMPAGE_MOVE)start_locked(s,p,move_id);
    if((move.effects&SELF_STAT)&&int(random(s,100))<move.self_chance)apply_boosts(s,p,move.self_stat_delta);
    if((move.effects&SECONDARY)&&(q.ability!=SHIELD_DUST||move.secondary_self)&&int(random(s,100))<move.secondary_chance){
      if(move.kind==SPARKLING_ARIA_MOVE)aria_cure=true;
      Mon& target=move.secondary_self?p:q;if(move.kind==TRI_ATTACK_MOVE||move.kind==DIRE_CLAW_MOVE){const int statuses[2][3]={{BRN,PAR,FRZ},{PSN,PAR,SLP}};int chosen=statuses[move.kind==DIRE_CLAW_MOVE][random(s,3)];set_status(s,q,chosen,side);}
      if(move.kind==SPIRIT_SHACKLE_MOVE&&p.hp>0&&!has_type(q,13)&&!q.trapped_volatile){mark_volatile(q,27);q.trapped_volatile=1;q.trap_source=s.active[side]+1;}
      if(move.kind==BURNING_JEALOUSY_MOVE&&q.stats_raised)set_status(s,q,BRN,side);if(move.kind==THROAT_CHOP_MOVE)add_volatile(s,q,THROAT_CHOP,rules,side);if(move.kind==ALLURING_VOICE_MOVE&&q.stats_raised)add_volatile(s,q,CONFUSION,rules,side);if(move.secondary_status)set_status(s,target,move.secondary_status,side);move_boosts(s,target,move.secondary_stat_delta,side);if(move.secondary_volatile)add_volatile(s,target,move.secondary_volatile,rules,side);
    }
    if((move.effects&SECONDARY)&&move.additional_flinch_chance&&q.ability!=SHIELD_DUST&&int(random(s,100))<move.additional_flinch_chance)add_volatile(s,q,FLINCH,rules,side);
    if(move.power>0&&(move.type==1||(move.traits&2))&&q.status==FRZ&&q.hp>0)q.status=NONE; // Showdown cureStatus needs HP: a KO'd target stays frozen
    if(move.kind==EXPLOSION_MOVE)p.hp=0;
    if((move.effects&FORCE_SWITCH)&&p.hp>0&&q.hp>0&&living(s,1-side)>1&&q.ability!=SUCTION_CUPS)s.drag[1-side]=1;
    if(q.mirror_coat&&move.category==1)q.mirror_damage=2*(before-q.hp);
    if(move.category!=2&&q.focus_punch)q.focus_punch=2;
    if(q.beak_blast&&(move.traits&2048))set_status(s,p,BRN,1-side,false,false);
    if((move.power>0||move.fixed_damage)&&q.illusion_slot)publish_active(s,1-side,true);
    if(move.power>0||move.fixed_damage)damaging_hit(s,side,move,rules,struggle);
    if(move.kind==SPIN_MOVE&&p.hp>0)clear_hazards(s,side);
    if(move.kind==ICE_SPINNER_MOVE)set_terrain(s,0,p);
    if(p.hp>0)attack_hazard(s,side,move);
    if(move.kind==KNOCK_OFF_MOVE&&p.hp>0&&q.item&&!(q.ability==STICKY_HOLD&&q.hp>0)){if(take_item(q))s.public_mons[1-side][s.public_active[1-side]].item=0;}
    publish_active(s,side,p.hp!=own_before);publish_active(s,1-side,q.hp!=before);
    each_event(s);
  }
  if(q.ability==BERSERK||q.ability==ANGER_SHELL){
    q.block_heal_berry=0;
    if(!(move.traits&32768)&&damage_hits&&total_damage>0&&q.hp>0&&q.hp*2<=q.maxhp&&(q.hp+total_damage)*2>q.maxhp){
      int32_t delta[7]={0,0,1,0,0,0,0};if(q.ability==ANGER_SHELL){delta[0]=1;delta[1]=-1;delta[3]=-1;delta[4]=1;}
      if(apply_boosts(s,q,delta))publish_ability(s,1-side,s.public_active[1-side],q.ability);
    }
  }
  if(final_target)q.times_attacked+=damage_hits+substitute_hits;
  if(damage_hits){if(q.hp<=0){resolve_faints(s,move.kind==EXPLOSION_MOVE?side:1-side);if(total_damage>0)q.hit_by_foe|=1<<s.active[side];}}
  if((move.effects&RECOIL)&&p.ability!=ROCK_HEAD&&p.ability!=MAGIC_GUARD&&total_damage)s.pending_recoil[side]=maximum(1,round_ratio(total_damage,move.recoil_num,maximum(1,move.recoil_den)));
  if(struggle&&!any_substitute)s.pending_recoil[side]=maximum(1,round_ratio(p.maxhp,1,4));
  if(!(move.traits&32768)&&damage_hits&&move.category!=2){
    int receiver=-1,donor=-1;
    if(q.ability==PICKPOCKET&&q.hp>0&&!q.item&&!s.drag[1-side]&&!s.pivot[1-side]&&(move.traits&2048)&&p.item){receiver=1-side;donor=side;}
    if(receiver>=0){Mon& to=active(s,receiver);Mon& from=active(s,donor);int item=from.item;
      if((from.ability!=STICKY_HOLD||from.hp<=0)&&can_take_item(from,to)&&take_item(from,&to)){to.item=item;if(choice_item(to))to.choice_lock=-1;to.booster_started=0;from.booster_started=0;s.public_mons[donor][s.public_active[donor]].item=0;s.public_mons[receiver][s.public_active[receiver]].item=item;if(to.item==WHITE_HERB)use_white_herb(s,receiver);if(to.item==BOOSTER_ENERGY){to.booster_started=1;update_booster(s,receiver);}}
    }
  }
  // Item listeners are snapshotted after the target Pickpocket event and
  // before the source Magician event; a stolen Orb is not a new listener.
  if(p.item==LIFE_ORB)s.after_move[side]|=2;
  if(!(move.traits&32768)&&damage_hits&&move.category!=2){
    if(p.ability==MAGICIAN&&p.hp>0&&!p.item&&q.item){int item=q.item;if((q.ability!=STICKY_HOLD||q.hp<=0)&&can_take_item(q,p)&&take_item(q,&p)){p.item=item;if(!s.requests[1-side].unannounced_item_loss[s.active[1-side]])s.requests[1-side].unannounced_item_loss[s.active[1-side]]=item;if(choice_item(p))p.choice_lock=-1;p.booster_started=0;q.booster_started=0;s.public_mons[side][s.public_active[side]].item=item;if(p.item==WHITE_HERB)use_white_herb(s,side);if(p.item==BOOSTER_ENERGY){p.booster_started=1;update_booster(s,side);}}}
  }
  if(aria_cure&&p.hp>0&&!p.fainted&&q.hp>0&&q.status==BRN){q.status=NONE;q.status_turns=0;q.toxic_turns=0;publish_active(s,1-side);}
  if((move.effects&PIVOT)&&p.hp>0&&living(s,side)>1)s.pivot[side]=1;
  return 1;
}
PK_HD inline int execute(State& s,int side,int action,const Rules& rules,int level,bool protected_target,int selected_move,bool called,int caller_id,int caller_priority,int caller_flags){
  Mon& p=active(s,side);bool had_target=active(s,1-side).hp>0;p.current_result=called?0:1;
  int hit=execute_impl(s,side,action,rules,level,protected_target,selected_move,called,caller_id,caller_priority,caller_flags);
  if(!called&&!s.after_move[side])p.destiny_bond=0;
  if(!called&&p.charge_duration&&!s.after_move[side]){p.charge_duration=0;p.charge_move=0;p.charge_kind=0;p.charge_released=0;}
  int id=s.used_move[side];if(id>=0&&id<rules.move_count&&(called||s.after_move[side])){
    int kind=rules.moves[id].kind;
    if((kind==CRASH_MOVE&&had_target&&!hit&&p.ability!=MAGIC_GUARD)||(kind==EXPLOSION_MOVE&&p.ability!=DAMP&&active(s,1-side).ability!=DAMP)){int before=p.hp;p.hp=kind==EXPLOSION_MOVE?0:maximum(0,p.hp-maximum(1,p.maxhp/2));if(p.hp!=before)publish_active(s,side,true);}
  }
  if(s.after_move[side]&&id>=0&&id<rules.move_count&&rules.moves[id].kind==BEAK_BLAST_MOVE)p.beak_blast=0;
  if(p.charged){int mid=s.used_move[side];
    if(mid>=0&&mid<rules.move_count&&(rules.moves[mid].type==4||(rules.moves[mid].kind==TERA_BLAST_MOVE&&p.terastallized&&p.tera_type==4)))p.charged=0;
  }
  if(hit&&p.hp<=0&&!p.fainted)resolve_faints(s,side);
  if(!p.current_result&&(!p.fainted||called||s.after_move[side]))p.current_result=hit?2:1;
  return hit;
}
PK_HD inline void sort_action_list(State& s,QueuedAction* list,int length);
PK_HD inline void future_hit(State& s,int target,const Rules& rules){
  int side=s.future_source[target],slot=s.future_slot[target],id=s.future_move[target];s.future_turn[target]=0;
  Mon& source=s.mons[side][slot];Mon& q=active(s,target);if(q.fainted||q.hp<=0||(&source==&q))return;
  Move move=rules.moves[id];move.kind=FUTURE_SIGHT_MOVE;move.ignore_immunity=0;move.protectable=0;move.traits=0;
  bool onfield=slot==s.active[side]&&!source.fainted;Mon actor=source;if(!onfield){actor.ability=NO_ABILITY;actor.item=NO_ITEM;actor.paradox_stat=0;}
  if(actor.ability==INFILTRATOR)move.traits|=8|131072;
  if(has_type(q,16))return;
  accuracy_event_order(s,actor,q);
  double accuracy=100.0*accuracy_modifier(s,actor,q,move)/4096.0;int a=actor.boosts[5],d=q.boosts[6];const double scales[7]={1,4.0/3,5.0/3,2,7.0/3,8.0/3,3};
  if(q.ability==UNAWARE)a=0;if(actor.ability==UNAWARE||actor.ability==KEEN_EYE||actor.ability==MINDS_EYE)d=0;
  accuracy=a>=0?accuracy*scales[a]:accuracy/scales[-a];accuracy=d>=0?accuracy/scales[d]:accuracy*scales[-d];
  if(actor.ability!=NO_GUARD&&q.ability!=NO_GUARD&&!q.glaive_rush&&random(s,100)>=accuracy)return;
  q.tera_resisted=0;int before=q.hp;bool sub=q.substitute&&!(move.traits&8);int dealt=damage(s,actor,q,move,rules,actor.level);source.stellar_used=actor.stellar_used;
  if(sub){q.substitute=maximum(0,q.substitute-dealt);if(q.item==AIR_BALLOON)pop_balloon(s,target);publish_active(s,target);each_event(s);if(onfield&&source.item==LIFE_ORB&&source.hp>0&&source.ability!=MAGIC_GUARD){source.hp=maximum(0,source.hp-maximum(1,source.maxhp/10));publish_active(s,side,true);}return;}
  if(q.ability==STURDY&&q.hp==q.maxhp&&dealt>=q.hp){publish_ability(s,target,s.public_active[target],q.ability);dealt=q.hp-1;}
  if(q.item==FOCUS_SASH&&q.hp==q.maxhp&&dealt>=q.hp){use_item(s,target);dealt=q.hp-1;}
  if(((q.ability==DISGUISE&&q.form_family==4)||(q.ability==ICE_FACE&&q.form_family==5&&move.category==0))&&q.form_current==0){dealt=0;q.form_busted=1;}
  q.hp=maximum(0,q.hp-dealt);if(q.hp==0&&before>0){q.faint_source_side=side+1;q.faint_source_slot=slot;}
  if(before>q.hp)q.hit_by_foe|=1<<slot;publish_active(s,target,true);each_event(s);
  // The reaction event uses the retained source rather than the current teammate.
  int saved_ability=source.ability,saved_item=source.item;if(!onfield){source.ability=NO_ABILITY;source.item=NO_ITEM;}
  damaging_hit(s,side,move,rules,false,slot);if(!onfield){source.ability=saved_ability;source.item=saved_item;}
  resolve_faints(s,target);if(before>q.hp)q.hit_by_foe|=1<<slot;++q.times_attacked;q.got_attacked=1;each_event(s);
  if(q.hp>0&&(q.ability==BERSERK||q.ability==ANGER_SHELL)&&q.hp*2<=q.maxhp&&before*2>q.maxhp){int32_t delta[7]={q.ability==ANGER_SHELL?1:0,q.ability==ANGER_SHELL?-1:0,1,q.ability==ANGER_SHELL?-1:0,q.ability==ANGER_SHELL?1:0,0,0};if(apply_boosts(s,q,delta))publish_ability(s,target,s.public_active[target],q.ability);}
  if(onfield&&source.item==LIFE_ORB&&source.hp>0&&source.ability!=MAGIC_GUARD){source.hp=maximum(0,source.hp-maximum(1,source.maxhp/10));publish_active(s,side,true);}
  int left[2]={};for(int a=0;a<2;a++)for(int i=0;i<6;i++)left[a]+=s.mons[a][i].maxhp>0&&!s.mons[a][i].fainted;
  if(!left[0]||!left[1]){s.terminated=1;s.winner=left[0]?0:left[1]?1:s.last_faint_side;}
}
PK_HD inline void residual(State& s,const Rules& rules) {
  update_speeds(s);
  QueuedAction handlers[QUEUE_CAPACITY];int count=0;
  if(s.weather)handlers[count++]={11,-1,0,0,1,0,0,0,0};
  if(s.trick_room)handlers[count++]={25,-1,0,0,27,0,0,1,0};
  if(s.terrain)handlers[count++]={14,-1,0,0,27,0,0,7,0};
  for(int side=0;side<2;side++)if(s.future_turn[side])handlers[count++]={38,side,0,0,3,0,speed(active(s,side)),3,0};
  const int side_suborders[10]={0,0,0,0,1,2,10,5,3,4};
  for(int side=0;side<2;side++)for(int j=4;j<10;j++)if(s.side_durations[side][j])handlers[count++]={12,side,0,j,26,0,0,side_suborders[j],0};
  for(int side=0;side<2;side++){
    const Mon& p=active(s,side);int spd=speed(p);
    if(p.status==BRN||p.status==PSN||p.status==TOX)handlers[count++]={1,side,0,0,p.status==BRN?10:9,0,spd,2,0};
    QueuedAction volatiles[25];int vc=0;
    if(p.protect)volatiles[vc++]={2,side,11,0,2147483647,0,spd,2,0};
    if(p.stall_duration)volatiles[vc++]={3,side,12,0,2147483647,0,spd,2,0};
    if(p.flinch)volatiles[vc++]={5,side,2,0,2147483647,0,spd,2,0};
    if(p.taunt)volatiles[vc++]={6,side,3,0,15,0,spd,2,0};
    if(p.encore)volatiles[vc++]={7,side,4,0,16,0,spd,2,0};
    if(p.disable)volatiles[vc++]={8,side,5,0,17,0,spd,2,0};
    if(p.leech_seed)volatiles[vc++]={9,side,6,0,8,0,spd,2,0};
    if(p.locked_duration)volatiles[vc++]={20,side,14,0,2147483647,0,spd,2,0};
    if(p.heal_block)volatiles[vc++]={18,side,9,0,20,0,spd,2,0};
    if(p.throat_chop)volatiles[vc++]={19,side,10,0,22,0,spd,2,0};
    if(p.magnet_rise)volatiles[vc++]={32,side,28,0,18,0,spd,2,0};
    if(p.yawn)volatiles[vc++]={10,side,7,0,23,0,spd,2,0};
    if(p.mirror_coat)volatiles[vc++]={35,side,37,37,2147483647,0,spd,2,0};
    if(p.chilly_reception)volatiles[vc++]={35,side,33,33,2147483647,0,spd,2,0};
    if(p.focus_punch)volatiles[vc++]={35,side,35,35,2147483647,0,spd,2,0};
    if(p.beak_blast)volatiles[vc++]={35,side,36,36,2147483647,0,spd,2,0};
    if(p.recharge)volatiles[vc++]={33,side,30,0,2147483647,0,spd,2,0};
    if(p.partial_trap)volatiles[vc++]={34,side,32,0,13,0,spd,2,0};
    if(p.charge_duration)volatiles[vc++]={28,side,21,0,2147483647,0,spd,2,0};
    if(p.curse)volatiles[vc++]={26,side,19,0,12,0,spd,2,0};
    if(p.salt_cure)volatiles[vc++]={27,side,20,0,13,0,spd,2,0};
    if(p.roost)volatiles[vc++]={16,side,13,0,25,0,spd,2,0};
    for(int i=1;i<vc;i++){QueuedAction x=volatiles[i];int j=i;while(j>0&&p.volatile_order[volatiles[j-1].actor]>p.volatile_order[x.actor]){volatiles[j]=volatiles[j-1];--j;}volatiles[j]=x;}
    for(int i=0;i<vc;i++)handlers[count++]=volatiles[i];
    if(p.ability==HYDRATION)handlers[count++]={13,side,0,0,5,0,spd,3,0};
    if(p.ability==SLOW_START)handlers[count++]={30,side,0,0,28,0,spd,2,0};
    if(p.ability==BAD_DREAMS)handlers[count++]={31,side,0,0,28,0,spd,2,0};
    if(p.ability==HARVEST)handlers[count++]={29,side,0,0,28,0,spd,2,0};
    if(p.ability==HUNGER_SWITCH||p.ability==SHIELDS_DOWN||p.suppressed_ability==HUNGER_SWITCH||p.suppressed_ability==SHIELDS_DOWN)handlers[count++]={37,side,0,0,29,0,spd,3,0};
    if(p.ability==CUD_CHEW)handlers[count++]={36,side,0,0,28,0,spd,2,0};
    if(p.ability==SPEED_BOOST)handlers[count++]={23,side,0,0,28,0,spd,2,0};
    if(p.ability==SHED_SKIN)handlers[count++]={24,side,0,0,5,0,spd,3,0};
    if(p.item==LEFTOVERS)handlers[count++]={4,side,0,0,5,0,spd,4,0};
    if(p.item==FLAME_ORB||p.item==TOXIC_ORB)handlers[count++]={21,side,0,0,28,0,spd,3,0};
    if(p.item==WHITE_HERB)handlers[count++]={22,side,0,0,29,0,spd,8,0};
    if(s.wish_pending[side])handlers[count++]={17,side,0,0,4,0,spd,0,0};
    if(s.terrain==GRASSY_TERRAIN)handlers[count++]={15,side,0,0,5,0,spd,2,0};
  }
  sort_action_list(s,handlers,count);
  for(int k=0;k<count;k++){
    if(handlers[k].kind==38){int side=handlers[k].side;if(((s.turn-1)%256)>=s.future_turn[side]-1)future_hit(s,side,rules);if(s.terminated)return;resolve_faints(s);outcomes(s);if(s.terminated)return;continue;}
    if(handlers[k].kind==25){--s.trick_room;continue;}
    if(handlers[k].kind==14){if(--s.terrain_duration==0)set_terrain(s,0,active(s,0));continue;}
    if(handlers[k].kind==12){int side=handlers[k].side,j=handlers[k].action;if(--s.side_durations[side][j]==0)remove_side_condition(s,side,j);continue;}
    if(handlers[k].kind==11){
      if(s.weather_duration>0&&--s.weather_duration==0){s.weather=0;each_event(s,false,WEATHER_CHANGE);continue;}
      if((s.weather==SAND||s.weather==SNOW||s.weather==HAIL)&&!effective_weather(s))continue;
      int first=speed(active(s,1))>speed(active(s,0))?1:0;
      if(!active(s,0).fainted&&!active(s,1).fainted&&speed(active(s,0))==speed(active(s,1)))first=random(s,2);
      for(int j=0;j<2;j++){
        int side=j?1-first:first;Mon& p=active(s,side);if(p.fainted||p.hp<=0)continue;
        int weather=effective_weather(s),before=p.hp;
        bool weather_immune=p.ability==OVERCOAT||p.item==SAFETY_GOGGLES;
        if(!weather_immune&&((weather==SAND&&!has_type(p,8)&&!has_type(p,12)&&!has_type(p,15)&&p.ability!=SAND_RUSH&&p.ability!=SAND_VEIL&&p.ability!=SAND_FORCE)||(weather==HAIL&&!has_type(p,5)&&p.ability!=ICE_BODY&&p.ability!=SNOW_CLOAK)))indirect_loss(p,maximum(1,p.maxhp/16));
        if(!p.heal_block&&((weather==RAIN&&p.ability==RAIN_DISH)||((weather==SNOW||weather==HAIL)&&p.ability==ICE_BODY)))p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/16));
        if(weather==RAIN&&p.ability==DRY_SKIN&&!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/8));
        if(weather==SUN&&(p.ability==SOLAR_POWER||p.ability==DRY_SKIN))p.hp=maximum(0,p.hp-maximum(1,p.maxhp/8));
        publish_active(s,side,p.hp!=before);
      }
      each_event(s);resolve_faints(s,first);outcomes(s);if(s.terminated)return;continue;
    }
    int side=handlers[k].side;Mon& p=active(s,side);
    if(handlers[k].kind==17){if(((s.turn-1)&255)>s.wish_turn[side]){s.wish_pending[side]=0;if(p.hp>0&&!p.fainted){int before=p.hp;if(!p.heal_block)p.hp=minimum(p.maxhp,p.hp+s.wish_hp[side]);publish_active(s,side,p.hp!=before);}}continue;}
    if(p.fainted)continue;int before=p.hp;
    if(handlers[k].kind==1){
      if(p.status==BRN)indirect_loss(p,maximum(1,p.maxhp/(p.ability==HEATPROOF?32:16)));
      else if(p.status==PSN&&p.ability==POISON_HEAL){if(!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/8));}
      else if(p.status==PSN)indirect_loss(p,maximum(1,p.maxhp/8));
      else if(p.status==TOX){p.toxic_turns=minimum(15,p.toxic_turns+1);if(p.ability==POISON_HEAL){if(!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/8));}else indirect_loss(p,maximum(1,p.maxhp/16)*p.toxic_turns);}
    }else if(handlers[k].kind==2)p.protect=0;
    else if(handlers[k].kind==3){if(--p.stall_duration<=0){p.stall_duration=0;p.stall_counter=0;}}
    else if(handlers[k].kind==4&&!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/16));
    else if(handlers[k].kind==5)p.flinch=0;
    else if(handlers[k].kind==6)--p.taunt;
    else if(handlers[k].kind==7){--p.encore;for(int j=0;j<4;j++)if(p.move_ids[j]==p.encore_move&&p.pp[j]<=0)p.encore=0;}
    else if(handlers[k].kind==8)--p.disable;
    else if(handlers[k].kind==9){Mon& target=active(s,1-side);if(target.hp>0&&!target.fainted){int loss=indirect_loss(p,maximum(1,p.maxhp/8));if(loss)drain_heal(s,1-side,loss);}}
    else if(handlers[k].kind==13){if(p.status&&effective_weather(s)==RAIN){p.status=NONE;p.status_turns=0;p.toxic_turns=0;}}
    else if(handlers[k].kind==15){if(grounded(p)&&!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/16));}
    else if(handlers[k].kind==16)p.roost=0;
    else if(handlers[k].kind==20){if(--p.locked_duration==0)end_locked(s,side);else if(p.status==SLP)end_locked(s,side,false);else --p.locked_true;}
    else if(handlers[k].kind==18)--p.heal_block;
    else if(handlers[k].kind==19)--p.throat_chop;
    else if(handlers[k].kind==28){if(--p.charge_duration<=0){p.charge_duration=0;p.charge_move=0;p.charge_kind=0;p.charge_released=0;}}
    else if(handlers[k].kind==26||handlers[k].kind==27)indirect_loss(p,maximum(1,p.maxhp/(handlers[k].kind==26||has_type(p,2)||has_type(p,15)?4:8)));
    else if(handlers[k].kind==21)set_status(s,p,p.item==FLAME_ORB?BRN:TOX,side);
    else if(handlers[k].kind==22)use_white_herb(s,side);
    else if(handlers[k].kind==32)--p.magnet_rise;
    else if(handlers[k].kind==35){if(handlers[k].action==33)p.chilly_reception=0;else if(handlers[k].action==35)p.focus_punch=0;else if(handlers[k].action==37){p.mirror_coat=0;p.mirror_damage=0;}else p.beak_blast=0;}
    else if(handlers[k].kind==36&&p.cud_berry&&p.hp>0&&--p.cud_counter<=0){int berry=p.cud_berry;p.cud_berry=0;p.cud_counter=0;s.public_mons[side][s.public_active[side]].item=0;
      if(berry==SITRUS_BERRY&&!p.heal_block)p.hp=minimum(p.maxhp,p.hp+maximum(1,p.maxhp/4));
      if(berry==LUM_BERRY||(berry==CHESTO_BERRY&&p.status==SLP)){p.status=NONE;p.status_turns=0;p.toxic_turns=0;if(berry==LUM_BERRY)p.confusion=0;}
      if(berry==LEPPA_BERRY){int selected=-1;for(int i=0;i<4;i++)if(p.move_ids[i]>=0&&p.pp[i]==0){selected=i;break;}if(selected<0)for(int i=0;i<4;i++)if(p.move_ids[i]>=0&&p.pp[i]<p.maxpp[i]){selected=i;break;}if(selected>=0)p.pp[selected]=minimum(p.maxpp[selected],p.pp[selected]+10);}
    }
    else if(handlers[k].kind==37){if(p.ability==HUNGER_SWITCH&&p.form_family==2&&!p.terastallized)change_form(s,side,1-p.form_current);if(p.ability==SHIELDS_DOWN&&!p.transformed&&p.form_family==7&&p.hp>0)change_form(s,side,p.hp*2>p.maxhp?1:0);}
    else if(handlers[k].kind==33)--p.recharge;
    else if(handlers[k].kind==34){if(--p.partial_trap>0){const Mon& source=active(s,1-side);if(s.active[1-side]+1!=p.partial_source||source.hp<=0||!source.active_turns)p.partial_trap=0;else indirect_loss(p,maximum(1,p.maxhp/8));}}
    else if(handlers[k].kind==30){if(p.active_turns&&p.slow_start)--p.slow_start;}
    else if(handlers[k].kind==31){Mon& foe=active(s,1-side);if(p.hp>0&&foe.hp>0&&(foe.status==SLP||foe.ability==COMATOSE)){int loss=indirect_loss(foe,maximum(1,foe.maxhp/8));if(loss)publish_active(s,1-side,true);}}
    else if(handlers[k].kind==29){if((effective_weather(s)==SUN||random(s,2)==0)&&p.hp>0&&!p.item&&berry_item(p.last_item)){p.item=p.last_item;p.last_item=0;s.public_mons[side][s.public_active[side]].item=p.item;}}
    else if(handlers[k].kind==23){if(p.active_turns){int32_t delta[7]={0,0,0,0,1,0,0};if(apply_boosts(s,p,delta))publish_ability(s,side,s.public_active[side],p.ability);}}
    else if(handlers[k].kind==24){if(p.hp>0&&p.status&&random(s,100)<33){p.status=NONE;p.status_turns=0;p.toxic_turns=0;}}
    else if(handlers[k].kind==10){if(--p.yawn==0)set_status(s,p,SLP,p.yawn_source,true,true,false,true,true);}
    publish_active(s,side,p.hp!=before);resolve_faints(s);outcomes(s);if(s.terminated)return;
  }
  resolve_faints(s);outcomes(s);
}
PK_HD inline int compare_actions(const QueuedAction& a,const QueuedAction& b){
  if(a.order!=b.order)return a.order<b.order?-1:1;
  if(a.priority!=b.priority)return a.priority>b.priority?-1:1;
  if(a.fractional_priority!=b.fractional_priority)return a.fractional_priority>b.fractional_priority?-1:1;
  if(a.speed!=b.speed)return a.speed>b.speed?-1:1;
  if(a.sub_order!=b.sub_order)return a.sub_order<b.sub_order?-1:1;
  if(a.effect_order!=b.effect_order)return a.effect_order<b.effect_order?-1:1;
  return 0;
}
// Same selection order and forward Fisher-Yates tie shuffle as the oracle.
PK_HD inline void sort_action_list(State& s,QueuedAction* list,int length){
  for(int sorted=0;sorted+1<length;){
    int indices[QUEUE_CAPACITY],count=1;indices[0]=sorted;
    for(int i=sorted+1;i<length;i++){
      int delta=compare_actions(list[indices[0]],list[i]);
      if(delta>0){count=1;indices[0]=i;}else if(delta==0)indices[count++]=i;
    }
    for(int i=0;i<count;i++){int index=indices[i];if(index!=sorted+i){QueuedAction q=list[sorted+i];list[sorted+i]=list[index];list[index]=q;}}
    for(int i=sorted;i<sorted+count-1;i++){int j=i+random(s,sorted+count-i);QueuedAction q=list[i];list[i]=list[j];list[j]=q;}
    sorted+=count;
  }
}
PK_HD inline void sort_queue(State& s){sort_action_list(s,s.queue,s.queue_length);}
PK_HD inline void append_action(State& s,QueuedAction action){
  if(s.queue_length>=QUEUE_CAPACITY){s.fault=UNSUPPORTED;return;}
  s.queue[s.queue_length++]=action;
}
PK_HD inline void insert_action(State& s,QueuedAction action){
  if(s.queue_length>=QUEUE_CAPACITY){s.fault=UNSUPPORTED;return;}
  int first=-1,last=s.queue_length;
  for(int i=0;i<s.queue_length;i++){
    int delta=compare_actions(action,s.queue[i]);
    if(delta<=0&&first<0)first=i;
    if(delta<0){last=i;break;}
  }
  if(first<0){append_action(s,action);return;}
  int index=first==last?first:first+random(s,last-first+1);
  for(int i=s.queue_length;i>index;i--)s.queue[i]=s.queue[i-1];
  s.queue[index]=action;s.queue_length++;
}
PK_HD inline QueuedAction pop_action(State& s){
  QueuedAction action=s.queue[0];for(int i=1;i<s.queue_length;i++)s.queue[i-1]=s.queue[i];
  s.queue[--s.queue_length]={};return action;
}
PK_HD inline void check_replacements(State& s){
  if(s.terminated)return;
  for(int side=0;side<2;side++){
    s.forced[side]=active(s,side).fainted&&living(s,side)>0;
    if(s.forced[side])active(s,side).status=FNT;
  }
}
PK_HD inline int action_priority(const State& s,int side,int id,const Rules& rules){
  if(id<0||id>=rules.move_count)return 0;
  const Mon& p=s.mons[side][s.active[side]];const Move& m=rules.moves[id];int priority=m.priority;
  if(p.ability==PRANKSTER&&m.category==2)priority++;
  if(p.ability==TRIAGE&&(m.traits&262144))priority+=3;
  if(p.ability==GALE_WINGS&&m.type==9&&p.hp==p.maxhp)priority++;
  if(m.kind==GRASSY_GLIDE_MOVE&&s.terrain==GRASSY_TERRAIN&&grounded(p))priority++;
  return priority;
}
PK_HD inline QueuedAction make_action(State& s,int kind,int side,int action,int order,int priority=0){
  if(side>=0)boost_event_order(s);
  QueuedAction q{kind,side,side<0?-1:s.active[side],action,order,priority,side<0?1:speed(active(s,side)),0,0,-1};
  if(kind==MOVE)q.move_id=active(s,side).recharge?-4:struggling(active(s,side))?-2:active(s,side).charge_duration?active(s,side).charge_move:active(s,side).locked_duration?active(s,side).locked_move:active(s,side).move_ids[action%4];
  return q;
}
PK_HD inline void ability_start(State& s,int side,bool switch_in){
  Mon& p=active(s,side);
  if(p.hp<=0||p.fainted)return;
  if(p.ability>=EMBODY_TEAL&&p.ability<=EMBODY_CORNERSTONE&&p.terastallized&&p.form_family>=9&&p.form_family<=12&&p.form_current==1&&!p.embodied){p.embodied=1;int32_t delta[7]={};int stat=p.ability==EMBODY_TEAL?4:p.ability==EMBODY_WELLSPRING?3:p.ability==EMBODY_HEARTHFLAME?0:1;delta[stat]=1;if(apply_boosts(s,p,delta))publish_ability(s,side,s.public_active[side],p.ability);}
  if(switch_in&&p.ability==IMPOSTER){transform_into(s,side);return;}
  if(p.ability==TRACE){p.trace_seek=active(s,1-side).ability!=NO_ABILITY;
    if(p.trace_seek&&active(s,1-side).hp>0&&!active(s,1-side).fainted&&traceable_ability(active(s,1-side).ability)){random(s,1);if(!unassignable_ability(active(s,1-side).ability)){p.ability=active(s,1-side).ability;p.trace_seek=0;publish_ability(s,side,s.public_active[side],p.ability);s.requests[side].abilities[s.active[side]]=p.ability;ability_start(s,side);}}return;
  }
  if((p.ability==UNNERVE||p.ability==AS_ONE_GLASTRIER||p.ability==AS_ONE_SPECTRIER)&&!p.unnerved){p.unnerved=1;publish_ability(s,side,s.public_active[side],UNNERVE);}
        if(p.hp>0&&(p.ability==AIR_LOCK||p.ability==CLOUD_NINE)){if(switch_in)publish_ability(s,side,s.public_active[side],p.ability);each_event(s,false,WEATHER_CHANGE);}
        if(p.hp>0&&(p.ability==BEADS_OF_RUIN||p.ability==TABLETS_OF_RUIN||p.ability==VESSEL_OF_RUIN||p.ability==SWORD_OF_RUIN)&&(!s.ignore_ability_side||s.ignore_ability_side==side+1))publish_ability(s,side,s.public_active[side],p.ability);
        if(p.hp>0&&p.ability==SLOW_START)p.slow_start=5;
        if(p.hp>0&&p.ability==SUPREME_OVERLORD)p.supreme_fallen=minimum(5,s.total_fainted[side]);
        if(p.hp>0&&p.ability==FRISK){const Mon& foe=active(s,1-side);if(foe.hp>0&&!foe.fainted&&foe.item)s.public_mons[1-side][s.public_active[1-side]].item=foe.item;}
        if(p.hp>0){int terrain=(p.ability==ELECTRIC_SURGE||p.ability==HADRON_ENGINE)?ELECTRIC_TERRAIN:p.ability==GRASSY_SURGE?GRASSY_TERRAIN:p.ability==MISTY_SURGE?MISTY_TERRAIN:p.ability==PSYCHIC_SURGE?PSYCHIC_TERRAIN:0;if(terrain)set_terrain(s,terrain,p);
          int32_t delta[7]={0,0,0,0,0,0,0};
          if(p.ability==DOWNLOAD&&active(s,1-side).hp>0&&!active(s,1-side).fainted){const Mon& foe=active(s,1-side);boost_event_order(s);int def=boost_stat(foe.stats[1],foe.boosts[1]);boost_event_order(s);int spd=boost_stat(foe.stats[3],foe.boosts[3]);delta[def>=spd?2:0]=1;}
          if(p.ability==WIND_RIDER&&s.side_conditions[side][TAILWIND])delta[0]=1;
          if(p.ability==INTREPID_SWORD&&!(p.once_boost&1)){p.once_boost|=1;delta[0]=1;}
          if(p.ability==DAUNTLESS_SHIELD&&!(p.once_boost&2)){p.once_boost|=2;delta[1]=1;}
          if(apply_boosts(s,p,delta))publish_ability(s,side,s.public_active[side],p.ability);
        }
        if(p.hp>0&&p.ability==ORICHALCUM_PULSE)set_weather(s,SUN,p);
        if(p.hp>0&&p.ability>=DRIZZLE&&p.ability<=SNOW_WARNING)set_weather(s,p.ability-DRIZZLE+1,p);
        if(p.hp>0&&(p.ability==PRESSURE||p.ability==COMATOSE||p.ability==MOLD_BREAKER||p.ability==TERAVOLT||p.ability==TURBOBLAZE||(p.ability==INTIMIDATE&&active(s,1-side).hp>0&&!active(s,1-side).fainted)))publish_ability(s,side,s.public_active[side],p.ability);
        if(p.hp>0&&p.ability==INTIMIDATE){Mon& foe=active(s,1-side);if(foe.hp>0&&!foe.substitute&&foe.ability!=OWN_TEMPO&&foe.ability!=INNER_FOCUS&&foe.ability!=OBLIVIOUS&&foe.ability!=SCRAPPY){const int32_t delta[7]={-1,0,0,0,0,0,0};bool dropped=apply_boosts(s,foe,delta,side);if(dropped&&foe.ability==RATTLED){int32_t raised[7]={0,0,0,0,1,0,0};if(apply_boosts(s,foe,raised))publish_ability(s,1-side,s.public_active[1-side],foe.ability);}}}
}
PK_HD inline bool transform_into(State& s,int side){
  Mon& p=active(s,side);const Mon& q=active(s,1-side);
  if((p.illusion_slot||q.illusion_slot)||(q.form_family>=8&&(p.terastallized||q.terastallized))||p.transformed||q.transformed||q.fainted||q.hp<=0||q.substitute||(p.terastallized&&p.tera_type==STELLAR))return false;
  if(p.form_count){s.fault=UNSUPPORTED;return false;}
  p.original_species=p.species;p.original_species_flags=p.species_flags;p.original_trapping_flags=p.trapping_flags;p.original_weight=p.weight_hg;
  for(int i=0;i<2;i++)p.original_types[i]=p.base_types[i];for(int i=0;i<5;i++)p.original_stats[i]=p.stats[i];
  for(int i=0;i<4;i++){p.original_moves[i]=p.move_ids[i];p.original_pp[i]=p.pp[i];p.original_maxpp[i]=p.maxpp[i];p.original_categories[i]=p.move_categories[i];p.original_traits[i]=p.move_traits[i];
    p.move_ids[i]=q.move_ids[i];p.pp[i]=p.maxpp[i]=q.move_ids[i]<0?0:q.maxpp[i]==1?1:5;p.move_categories[i]=q.move_categories[i];p.move_traits[i]=q.move_traits[i];p.disabled[i]=0;}
  p.transformed=1;p.known_type=p.terastallized?1:0;p.trapping_flags=q.trapping_flags;p.species=q.species;p.species_flags=(p.species_flags&~0xF00)|(q.species_flags&0xF00);/* bits 8-11: Paldea Tauros form (Raging Bull) and Hoopa-Unbound (Hyperspace Fury) follow pokemon.species in Showdown; the item/ability bits read the base species */p.weight_hg=q.weight_hg;p.form_family=q.form_family;p.form_current=q.form_current;
  for(int i=0;i<2;i++)p.type[i]=p.base_types[i]=q.type[i];for(int i=0;i<5;i++)p.stats[i]=q.stats[i];for(int i=0;i<7;i++)p.boosts[i]=q.boosts[i];p.times_attacked=q.times_attacked;
  int old_ability=p.ability;p.ability=q.suppressed_ability?q.suppressed_ability:q.ability;p.suppressed_ability=0;p.trace_seek=0;p.form_busted=0;p.cud_berry=0;p.cud_counter=0;p.unnerved=0;p.slow_start=0;p.supreme_fallen=0;
  if(notransform_ability(p.ability)){p.suppressed_ability=p.ability;p.ability=NO_ABILITY;}
  p.speed_cache=(2*q.base_speed+p.iv_spe+p.ev_spe/4)*p.level/100+5;
  if(p.ability!=old_ability)ability_start(s,side);
  return true;
}
PK_HD inline void run_switches(State& s,int first_side,const Rules& rules){
      suppress_opponent(s);
      int switchers[2]={first_side,-1},count=1;
      while(s.queue_length&&s.queue[0].kind==RUN_SWITCH){QueuedAction next=pop_action(s);if(count<2)switchers[count++]=next.side;}
      int switch_order=each_event(s,true,SPEED_ORDER_ONLY); // Establish speed order, including fainted holders.
      if(count==2){switchers[0]=switch_order;switchers[1]=1-switch_order;}
      if(count==2&&active(s,switchers[0]).status==TOX&&active(s,switchers[1]).status==TOX&&speed(active(s,0))==speed(active(s,1)))random(s,2);
      // Tera Shift precedes Unnerve and every hazard or ordinary Start hook.
      for(int i=0;i<count;i++){int side=switchers[i];Mon& p=active(s,side);if(p.hp>0&&p.ability==TERA_SHIFT&&p.form_family==8&&p.form_current!=1){change_form(s,side,1,true);p.ability=p.base_ability=TERA_SHELL;s.requests[side].abilities[s.active[side]]=p.ability;}}
      // Unnerve has SwitchIn priority 1 and is active before hazards/berries.
      for(int i=0;i<count;i++){int side=switchers[i];Mon& p=active(s,side);if(p.hp>0&&(p.ability==UNNERVE||p.ability==AS_ONE_GLASTRIER||p.ability==AS_ONE_SPECTRIER)&&!p.unnerved){p.unnerved=1;publish_ability(s,side,s.public_active[side],UNNERVE);}}
      // These Start callbacks have no random effects in the admitted family.
      for(int i=0;i<count;i++){
        if(active(s,switchers[i]).unsupported){s.fault=UNSUPPORTED;return;}
        {int side=switchers[i];Mon& p=active(s,side);if(s.healing_wish[side]&&!p.fainted&&p.hp>0&&(p.hp<p.maxhp||p.status)){if(!p.heal_block)p.hp=p.maxhp;p.status=NONE;p.status_turns=0;p.toxic_turns=0;s.healing_wish[side]=0;publish_active(s,side,true);}}
        entry_hazards(s,switchers[i],rules);if(s.terminated)return;
        ability_start(s,switchers[i],true);
      }
      for(int i=0;i<count;i++){
        int side=switchers[i];Mon& p=active(s,side);if(p.hp<=0||p.fainted)continue;
        update_paradox_field(s,side,p.ability==PROTOSYNTHESIS?WEATHER_CHANGE:TERRAIN_CHANGE);
        if(p.item==AIR_BALLOON)s.public_mons[side][s.public_active[side]].item=AIR_BALLOON;
        if(p.item==BOOSTER_ENERGY){p.booster_started=1;update_booster(s,side);}
      }
      for(int i=0;i<count;i++){int side=switchers[i];Mon& p=active(s,side);if(p.hp<=0||p.fainted)continue;if(p.ability==SHIELDS_DOWN&&!p.transformed&&p.form_family==7)change_form(s,side,p.hp*2>p.maxhp?1:0);if(p.ability==ICE_FACE&&p.form_family==5&&p.form_current==1&&(effective_weather(s)==SNOW||effective_weather(s)==HAIL)){p.form_busted=0;change_form(s,side,0,true);}}
      // White Herb's AnySwitchIn callback is priority -2, after all Start
      // effects. Switch-in speedOrder gives equal-speed holders unique ranks.
      for(int i=0;i<2;i++)use_white_herb(s,i?1-switch_order:switch_order);
      publish_active(s,0);publish_active(s,1);

}
PK_HD inline void step(State& s,int a0,int a1,const Rules& rules,int level=100) {
  if(s.terminated||s.truncated||s.fault)return;
  int actions[2]={a0,a1};uint8_t legal[14];
  for(int side=0;side<2;side++) {
    mask(s,side,legal);bool ready=false;for(int i=0;i<14;i++)ready|=legal[i];
    if(!ready){if(actions[side]!=-1){s.fault=WAITING;return;}continue;}
    if(actions[side]<0||actions[side]>=14||!legal[actions[side]]){s.fault=ILLEGAL;return;}
    if(actions[side]<8&&!active(s,side).recharge&&!struggling(active(s,side))){
      int mid=active(s,side).charge_duration?active(s,side).charge_move:active(s,side).locked_duration?active(s,side).locked_move:active(s,side).move_ids[actions[side]%4];
      if(mid<0||mid>=rules.move_count||rules.moves[mid].unsupported){s.fault=UNSUPPORTED;return;}
    }
    if(active(s,side).unsupported){s.fault=UNSUPPORTED;return;}
  }
  s.phases++;
  if(s.public_count[0]==0||s.public_count[1]==0)initialize_public(s);
  bool rejected=false;
  for(int side=0;side<2;side++)if(actions[side]>=0){
    RequestCache& r=s.requests[side];
    if(actions[side]>=8&&!s.forced[side]&&!s.reviving[side]&&s.trapped[side]==2){
      s.trapped[side]=1;r.trapped=1;r.maybe_trapped=0;r.waiting=0;++s.unavailable_choices[side];rejected=true;
    }else{s.pending_actions[side]=actions[side];s.pending_mask|=1<<side;r.waiting=1;}
  }
  if(rejected){if(s.phases>=s.limit)s.truncated=1;return;}
  for(int side=0;side<2;side++)actions[side]=(s.pending_mask&(1<<side))?s.pending_actions[side]:-1;
  s.pending_mask=0;
  update_speeds(s);
  // New decisions are sorted independently of any suspended mid-turn queue.
  QueuedAction suspended[QUEUE_CAPACITY];int suspended_count=s.queue_length;
  for(int i=0;i<suspended_count;i++)suspended[i]=s.queue[i];
  s.queue_length=0;
  for(int side=0;side<2;side++){
    int action=actions[side];if(action<0)continue;
    if(action>=8){
      int kind=s.reviving[side]?REVIVE:SWITCH,order=s.reviving[side]?6:s.forced[side]?3:103;
      append_action(s,make_action(s,kind,side,action,order));
    }else{
      if(action>=4)append_action(s,make_action(s,TERA,side,action,106));
      int selected=active(s,side).charge_duration?active(s,side).charge_move:active(s,side).locked_duration?active(s,side).locked_move:active(s,side).move_ids[action%4];
      int priority=action_priority(s,side,struggling(active(s,side))?-2:selected,rules);
      QueuedAction moving=make_action(s,MOVE,side,action,200,priority);
      if(moving.move_id>=0&&active(s,side).ability==MYCELIUM_MIGHT&&rules.moves[moving.move_id].category==2)moving.fractional_priority=-1;
      if(moving.fractional_priority==0&&active(s,side).item==CUSTAP_BERRY&&active(s,side).hp*4<=active(s,side).maxhp&&can_eat(s,side)){use_item(s,side,true);moving.fractional_priority=1;}
      if(moving.move_id>=0&&(rules.moves[moving.move_id].kind==CHILLY_RECEPTION_MOVE||rules.moves[moving.move_id].kind==FOCUS_PUNCH_MOVE||rules.moves[moving.move_id].kind==BEAK_BLAST_MOVE)){QueuedAction prepare=moving;prepare.kind=PREPARE_MOVE;prepare.order=107;prepare.priority=0;prepare.fractional_priority=0;append_action(s,prepare);}
      if(moving.move_id>=0&&rules.moves[moving.move_id].kind==MIRROR_COAT_MOVE){QueuedAction prepare=moving;prepare.kind=PREPARE_MOVE;prepare.order=5;prepare.priority=0;prepare.fractional_priority=0;append_action(s,prepare);}
      append_action(s,moving);
    }
  }
  sort_queue(s);for(int i=0;i<suspended_count;i++)append_action(s,suspended[i]);
  if(!s.mid_turn){insert_action(s,make_action(s,BEFORE_TURN,-1,-1,4));append_action(s,make_action(s,RESIDUAL,-1,-1,300));s.mid_turn=1;}
  for(int side=0;side<2;side++)s.forced[side]=0;
  while(s.queue_length&&!s.fault&&!s.terminated){
    QueuedAction action=pop_action(s);
    if(action.kind==BEFORE_TURN){s.last_successful_move_kind=0;each_event(s,false,BEFORE_TURN_EVENT);}
    else if(action.kind==PREPARE_MOVE){Mon& p=active(s,action.side);if(s.active[action.side]==action.actor&&!p.fainted&&p.hp>0){int kind=rules.moves[action.move_id].kind;mark_volatile(p,kind==MIRROR_COAT_MOVE?37:kind==CHILLY_RECEPTION_MOVE?33:kind==FOCUS_PUNCH_MOVE?35:36);if(kind==MIRROR_COAT_MOVE){p.mirror_coat=1;p.mirror_damage=0;}else if(kind==CHILLY_RECEPTION_MOVE)p.chilly_reception=1;else if(kind==FOCUS_PUNCH_MOVE)p.focus_punch=1;else p.beak_blast=1;}}
    else if(action.kind==TERA){if(active(s,action.side).illusion_slot&&s.mons[action.side][active(s,action.side).illusion_slot-1].form_family>=8)break_illusion(s,action.side);if(active(s,action.side).form_family==2&&!active(s,action.side).transformed&&active(s,action.side).form_current){active(s,action.side).form_permanent=active(s,action.side).form_current;active(s,action.side).form_regression=1;}active(s,action.side).known_type=1;active(s,action.side).terastallized=1;s.tera_used[action.side]=1;s.public_mons[action.side][s.public_active[action.side]].tera=1;s.public_mons[action.side][s.public_active[action.side]].tera_type=active(s,action.side).tera_type;if(s.public_active[action.side]==s.active[action.side])s.requests[action.side].tera[s.active[action.side]]=1;publish_active(s,action.side);
      Mon& p=active(s,action.side);if(!p.transformed&&(p.form_family>=9||p.form_family==8)){
        change_form(s,action.side,p.form_family==8?2:1,true);p.form_regression=1;p.embodied=0;
        p.ability=p.base_ability=p.form_family==8?TERAFORM_ZERO:EMBODY_TEAL+p.form_family-9;s.requests[action.side].abilities[s.active[action.side]]=p.ability;
        if(p.ability==TERAFORM_ZERO&&(s.weather||s.terrain)){publish_ability(s,action.side,s.public_active[action.side],TERAFORM_ZERO);set_weather(s,0,p);set_terrain(s,0,p);}else ability_start(s,action.side);
      }}
    else if(action.kind==SWITCH){
      if(action.order==103&&active(s,action.side).hp>0)each_event(s); // BeforeSwitchOut/Pursuit activation update.
      switch_to(s,action.side,action.action-8);
      insert_action(s,make_action(s,RUN_SWITCH,action.side,-1,101));
    }else if(action.kind==REVIVE)switch_to(s,action.side,action.action-8);
    else if(action.kind==RUN_SWITCH){run_switches(s,action.side,rules);if(s.terminated||s.fault)return;    }else if(action.kind==MOVE){
      if(s.active[action.side]!=action.actor||s.mons[action.side][action.actor].fainted)continue;
      s.pending_recoil[action.side]=0;s.after_move[action.side]=0;
      int executed_id=action.move_id;s.used_move[action.side]=executed_id;s.used_move_hit[action.side]=0;
      int hit=execute(s,action.side,action.action,rules,level,false,executed_id,false,-1,action.priority);
      if(s.fault){restore_abilities(s);return;}
      if(hit){
        s.last_successful_move_kind=executed_id>=0&&executed_id<rules.move_count?rules.moves[executed_id].kind:0;
        executed_id=s.used_move[action.side];
        bool effective_hit=executed_id==active(s,action.side).last_move||s.used_move_hit[action.side];
        resolve_faints(s);
        if(s.pending_recoil[action.side]){Mon& p=active(s,action.side);p.hp=maximum(0,p.hp-s.pending_recoil[action.side]);s.pending_recoil[action.side]=0;publish_active(s,action.side,true);}
        if(hit!=3)each_event(s);
        if(effective_hit&&executed_id>=0&&executed_id<rules.move_count&&living(s,1-action.side)>0)apply_boosts(s,active(s,action.side),rules.moves[executed_id].after_stat_delta);
        Mon& p=active(s,action.side);
        if(effective_hit&&executed_id>=0&&rules.moves[executed_id].kind==RELIC_SONG_MOVE&&p.form_family==1&&!p.transformed&&p.hp>0&&!(rules.moves[executed_id].traits&32768))change_form(s,action.side,1-p.form_current);
        if(effective_hit&&p.hp>0&&p.item==THROAT_SPRAY&&executed_id>=0&&(rules.moves[executed_id].traits&16)&&!(p.ability==SHEER_FORCE&&(rules.moves[executed_id].traits&16384))){int32_t delta[7]={0,0,1,0,0,0,0};apply_boosts(s,p,delta);use_item(s,action.side);}
        if(effective_hit&&(s.after_move[action.side]&2)&&p.hp>0&&p.item==LIFE_ORB&&p.ability!=MAGIC_GUARD&&!(p.ability==SHEER_FORCE&&executed_id>=0&&(rules.moves[executed_id].traits&16384))&&(executed_id==-2||(executed_id>=0&&rules.moves[executed_id].category!=2&&rules.moves[executed_id].kind!=FUTURE_SIGHT_MOVE))){p.hp=maximum(0,p.hp-maximum(1,p.maxhp/10));publish_active(s,action.side,true);}
      }
      if(s.after_move[action.side]&&active(s,action.side).locked_duration==1)end_locked(s,action.side);
      if(!hit&&s.after_move[action.side])s.last_successful_move_kind=0;
      if(hit&&executed_id>=0&&(rules.moves[executed_id].traits&1073741824)&&active(s,1-action.side).ability==DANCER&&active(s,1-action.side).hp>0){
        restore_abilities(s);int dancer=1-action.side;Mon& p=active(s,dancer);bool no_lock=!p.locked_duration;
        s.pending_recoil[dancer]=0;s.after_move[dancer]=0;
        int copied=execute(s,dancer,0,rules,level,false,executed_id,false,-1,rules.moves[executed_id].priority,(-2147483647-1));
        if(s.fault)return;
        if(copied){resolve_faints(s);if(s.pending_recoil[dancer]){p.hp=maximum(0,p.hp-s.pending_recoil[dancer]);s.pending_recoil[dancer]=0;publish_active(s,dancer,true);}if(copied!=3)each_event(s);
          if(living(s,1-dancer)>0)apply_boosts(s,p,rules.moves[executed_id].after_stat_delta);
          if((s.after_move[dancer]&2)&&p.hp>0&&p.item==LIFE_ORB&&rules.moves[executed_id].category!=2){p.hp=maximum(0,p.hp-maximum(1,p.maxhp/10));publish_active(s,dancer,true);}
          s.last_successful_move_kind=rules.moves[executed_id].kind;
        }else if(s.after_move[dancer])s.last_successful_move_kind=0;
        if(p.locked_duration==1)end_locked(s,dancer);
        if(no_lock){p.locked_duration=0;p.locked_true=0;}
        if(s.after_move[dancer])after_move_items(s);
      }
      if(s.after_move[action.side])after_move_items(s);
      publish_active(s,0);publish_active(s,1);
    }else if(action.kind==RESIDUAL){residual(s,rules);if(s.terminated)return;}
    for(int side=0;side<2;side++)if(s.drag[side]){
      s.drag[side]=0;Mon& p=active(s,side);if(p.hp<=0)continue;
      int options[5],count=0;for(int i=1;i<6;i++){const Mon& q=s.mons[side][s.order[side][i]];if(q.hp>0&&!q.fainted)options[count++]=i;}
      if(count){int chosen=options[random(s,count)];switch_to(s,side,chosen,true);run_switches(s,side,rules);if(s.terminated||s.fault){restore_abilities(s);return;}}
    }
    restore_abilities(s);
    resolve_faints(s);outcomes(s);if(s.terminated)return;
    if(!s.queue_length)check_replacements(s);
    if(!s.queue_length||s.queue[0].order!=3)each_event(s);
    for(int side=0;side<2;side++)if(s.pivot[side]){
      if(active(s,side).hp>0&&(s.reviving[side]||living(s,side)>1)){if(!s.reviving[side]&&!s.shed_tail[side])each_event(s);s.forced[side]=1;}
      else s.pivot[side]=0;
    }
    if(s.forced[0]||s.forced[1])break;
    if(s.queue_length&&s.queue[0].kind==MOVE){
      update_speeds(s);
      for(int i=0;i<s.queue_length;i++)if(s.queue[i].side>=0){boost_event_order(s);s.queue[i].speed=speed(s.mons[s.queue[i].side][s.queue[i].actor]);if(s.queue[i].kind==MOVE&&s.queue[i].actor==s.active[s.queue[i].side])s.queue[i].priority=action_priority(s,s.queue[i].side,s.queue[i].move_id,rules);}
      sort_queue(s);
    }
  }
  if(!s.terminated&&!s.fault&&!s.forced[0]&&!s.forced[1]&&!s.queue_length){
    s.turn++;s.mid_turn=0;
    for(int side=0;side<2;side++)if(!active(s,side).fainted){
      Mon& p=active(s,side);if(p.got_attacked)p.known_type=1;p.active_turns++;p.last_result=p.current_result;p.current_result=0;p.stats_raised=0;p.stats_lowered=0;p.hit_by_foe=0;p.newly_switched=0;
      int handlers=int(p.taunt>0)+int(p.encore>0)+int(p.disable>0)+int(p.heal_block>0)+int(p.throat_chop>0)+int(p.choice_lock>=0);
      for(int left=handlers;left>1;left--)random(s,left);
      // Then TrapPokemon: the 'trapped' and 'partiallytrapped' volatiles (holder's speed) and a foe's Arena Trap,
      // Magnet Pull or Shadow Tag (foe's speed) are priority-0 handlers; Showdown's sort spends k-1 draws on k ties.
      {int speeds[3],count=0;if(p.trapped_volatile)speeds[count++]=speed(p);if(p.partial_trap)speeds[count++]=speed(p);
       const Mon& f=active(s,1-side);int fa=f.suppressed_ability?f.suppressed_ability:f.ability;
       if(f.hp>0&&!f.fainted&&(fa==ARENA_TRAP||fa==MAGNET_PULL||fa==SHADOW_TAG))speeds[count++]=speed(f);
       for(int i=0;i<count;i++)for(int j=i+1;j<count;j++)if(speeds[j]>speeds[i]){int x=speeds[i];speeds[i]=speeds[j];speeds[j]=x;}
       for(int i=0;i<count;){int k=i;while(k<count&&speeds[k]==speeds[i])k++;for(int r=i;r<k-1;r++)random(s,k-r);i=k;}}
      if(!choice_item(p)||!valid_choice_lock(p))p.choice_lock=-1;
    }
  }
  if(s.turn>1000){s.terminated=1;s.winner=-1;return;}
  if(!s.terminated&&!s.fault&&!s.mid_turn)update_trapping(s);
  if(!s.terminated&&!s.fault)refresh_requests(s);
  if(!s.terminated&&s.phases>=s.limit)s.truncated=1;
}
}
