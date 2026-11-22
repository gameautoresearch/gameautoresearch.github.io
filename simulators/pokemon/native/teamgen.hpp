// SPDX-License-Identifier: MIT
// Native frozen-format team generation. Battle-mechanics admission is separate
// and stays closed until the complete task qualifies.
#pragma once
#include "sim.hpp"
namespace pkn {
constexpr int GEN_SPECIES_CAPACITY=1024,GEN_GROUP_CAPACITY=1024;
enum GenCompatibility:uint32_t { GEN_BLISSEY=1,GEN_CHANSEY=2,GEN_ILLUMISE=4,GEN_VOLBEAT=8,
  GEN_WEB_SETTER=16,GEN_SCREEN_SETTER=32,GEN_TOXICROAK=64,GEN_SUN_SETTER=128 };
struct GenSpecies {
  int32_t base,types[2],battle_types[2],level,form_begin,form_count,sample_form;
  uint32_t compatibility;
  int32_t priority_species,raging_bull_type,ivy_cudgel_type,mega_sun;
  int32_t ability_species;
  uint32_t id_tag;
  int32_t required_move,stats[6];
  uint32_t base_tag;int32_t nfe,required_item_begin,required_item_count,effectiveness[19];
  int32_t set_begin,set_count,gender,fire_weak_abilities;
};
enum GenMoveFlag:uint32_t { GM_DAMAGE=1,GM_POWER=2,GM_TECHNICIAN=4,GM_SKILL_LINK=8,GM_RECOIL=16,GM_DRAIN=32,
  GM_STRONG_JAW=64,GM_IRON_FIST=128,GM_SOUND=256,GM_SHEER_FORCE=512,GM_INACCURATE=1024,GM_RECOVERY=2048,
  GM_CONTRARY=4096,GM_PHYSICAL_SETUP=8192,GM_SPECIAL_SETUP=16384,GM_MIXED_SETUP=32768,GM_SPEED_SETUP=65536,
  GM_SPEED_CONTROL=131072,GM_SETUP=262144,GM_HAZARDS=524288,GM_NO_STAB=1048576,GM_GRASSY_GLIDE=2097152 };
enum GenAbilityFlag:uint64_t { GA_GRASSY_SURGE=1ull,GA_SEED_SOWER=2ull,GA_PSYCHIC_SURGE=4ull,GA_ELECTRIC_SURGE=8ull,
  GA_CHLOROPHYLL=16ull,GA_DROUGHT=32ull,GA_MEGA_SOL=64ull,GA_ORICHALCUM_PULSE=128ull,GA_DRIZZLE=256ull,
  GA_AERILATE=512ull,GA_GALVANIZE=1024ull,GA_PIXILATE=2048ull,GA_REFRIGERATE=4096ull,GA_PRANKSTER=8192ull,GA_GUTS=16384ull };
enum GenCounterKey:int32_t { GC_PHYSICAL=19,GC_SPECIAL=20,GC_STATUS=21,GC_DAMAGE=22,GC_TECHNICIAN=23,
  GC_SKILL_LINK=24,GC_RECOIL=25,GC_DRAIN=26,GC_STRONG_JAW=27,GC_IRON_FIST=28,GC_SOUND=29,GC_PRIORITY=30,
  GC_SHEER_FORCE=31,GC_INACCURATE=32,GC_RECOVERY=33,GC_CONTRARY=34,GC_PHYSICAL_SETUP=35,GC_SPECIAL_SETUP=36,
  GC_MIXED_SETUP=37,GC_SPEED_SETUP=38,GC_SPEED_CONTROL=39,GC_SETUP=40,GC_HAZARDS=41,GC_STAB=42,GC_STAB_TERA=43 };
struct GenMove {int32_t type,category,priority,kind;uint32_t flags,id_tag;};
struct GenCounter {int32_t counts[44],damaging_moves[32],damaging_count,base_power_moves[32],base_power_count;};
struct GenMoveList {int32_t values[64],count,fault;};
struct GenRange {int32_t begin,count;};
struct GenSetSpec {int32_t role;GenRange moves,abilities,teras;int32_t sample_tera;uint64_t ability_flags;};
struct GenSet {int32_t species,forme,ability,item,level,tera,role,evs[6],ivs[6],gender,shiny;GenMoveList moves;};
struct GenTeam {GenSet pokemon[6];int32_t count,fault;};
struct GenRules {
  const GenSpecies* species;int32_t species_count,group_count;
  const int32_t* formes;
  const uint8_t* squawkabilly_groups;
  const GenMove* moves;int32_t move_count;
  const int32_t* ability_codes;
  const int32_t* selector_moves;const GenRange* selector_ranges;
  const uint32_t *ability_tags,*item_tags;int32_t item_count;const int32_t* required_items;
  const GenSetSpec* sets;const int32_t *set_moves,*set_abilities,*set_teras;
};
struct GenPool {
  int32_t groups[GEN_GROUP_CAPACITY],group_count;
  int32_t species[GEN_SPECIES_CAPACITY],species_count;
};
PK_HD inline bool gen_has_type(const int32_t types[2],int type){return types[0]==type||types[1]==type;}
PK_HD inline bool gen_initialize_pool(const GenRules& rules,GenPool& pool,int monotype=-1,const uint8_t* excluded=nullptr){
  if(rules.species_count<0||rules.species_count>GEN_SPECIES_CAPACITY||rules.group_count>GEN_GROUP_CAPACITY)return false;
  pool.group_count=0;pool.species_count=0;
  int counts[GEN_GROUP_CAPACITY]={},base_order[GEN_GROUP_CAPACITY],base_count=0;
  for(int i=0;i<rules.species_count;i++){
    const GenSpecies& species=rules.species[i];
    if(species.base<0||species.base>=rules.group_count)return false;
    if(excluded&&excluded[i])continue;
    if(monotype>=0&&(!gen_has_type(species.types,monotype)||!gen_has_type(species.battle_types,monotype)))continue;
    pool.species[pool.species_count++]=i;if(!counts[species.base]++)base_order[base_count++]=species.base;
  }
  for(int index=0;index<base_count;index++){
    int base=base_order[index];
    int weight=rules.squawkabilly_groups[base]?int(counts[base]>0):minimum(3,(counts[base]+2)/3);
    if(pool.group_count+weight>GEN_GROUP_CAPACITY)return false;
    for(int i=0;i<weight;i++)pool.groups[pool.group_count++]=base;
  }
  return true;
}
PK_HD inline uint32_t gen_random(uint64_t& rng,uint32_t count){rng=rng*0x5D588B656C078965ULL+0x269EC3ULL;return uint32_t((uint64_t(uint32_t(rng>>32))*count)>>32);}
PK_HD inline int gen_sample_no_replace(uint64_t& rng,int32_t* values,int& count){
  if(count<=0)return -1;int index=gen_random(rng,count),value=values[index];values[index]=values[--count];return value;
}
PK_HD inline int gen_forme(const GenRules& rules,int species_id,uint64_t& rng){
  if(species_id<0||species_id>=rules.species_count)return -1;
  const GenSpecies& species=rules.species[species_id];if(species.form_count<=0)return -1;
  return rules.formes[species.form_begin+(species.sample_form?gen_random(rng,species.form_count):0)];
}
PK_HD inline bool gen_compatible(const GenRules& rules,int candidate,const int32_t* team,int length){
  if(candidate<0||candidate>=rules.species_count||length<0||length>6)return false;
  uint32_t a=rules.species[candidate].compatibility;
  for(int i=0;i<length;i++){
    if(team[i]<0||team[i]>=rules.species_count)return false;
    uint32_t b=rules.species[team[i]].compatibility;
    if(((a&GEN_BLISSEY)&&(b&GEN_CHANSEY))||((a&GEN_CHANSEY)&&(b&GEN_BLISSEY))||
       ((a&GEN_ILLUMISE)&&(b&GEN_VOLBEAT))||((a&GEN_VOLBEAT)&&(b&GEN_ILLUMISE))||
       ((a&GEN_WEB_SETTER)&&(b&GEN_WEB_SETTER))||((a&GEN_SCREEN_SETTER)&&(b&GEN_SCREEN_SETTER))||
       ((a&GEN_TOXICROAK)&&(b&GEN_SUN_SETTER))||((a&GEN_SUN_SETTER)&&(b&GEN_TOXICROAK)))return false;
  }
  return true;
}
PK_HD inline int gen_move_type(const GenMove& move,const GenSpecies& species,uint64_t abilities,int tera){
  if(move.kind==1)return tera;
  if(move.kind==2)return species.types[0];
  if(move.kind==3&&species.raging_bull_type>=0)return species.raging_bull_type;
  if(move.kind==4&&species.ivy_cudgel_type>=0)return species.ivy_cudgel_type;
  if(move.kind==5){if(abilities&(GA_GRASSY_SURGE|GA_SEED_SOWER))return 3;if(abilities&GA_PSYCHIC_SURGE)return 10;if(abilities&GA_ELECTRIC_SURGE)return 4;}
  if(move.kind==6){if(species.mega_sun||(abilities&(GA_CHLOROPHYLL|GA_DROUGHT|GA_MEGA_SOL|GA_ORICHALCUM_PULSE)))return 1;if(abilities&GA_DRIZZLE)return 2;}
  if(move.type==0){if(abilities&GA_AERILATE)return 9;if(abilities&GA_GALVANIZE)return 4;if(abilities&GA_PIXILATE)return 17;if(abilities&GA_REFRIGERATE)return 5;}
  return move.type;
}
PK_HD inline bool gen_query_moves(const GenRules& rules,int species_id,const int32_t* moves,int count,uint64_t abilities,int tera,GenCounter& result){
  if(species_id<0||species_id>=rules.species_count||count<0||count>32||tera<0||tera>18)return false;
  result={};const GenSpecies& species=rules.species[species_id];
  const uint32_t flags[18]={GM_TECHNICIAN,GM_SKILL_LINK,GM_RECOIL,GM_DRAIN,GM_STRONG_JAW,GM_IRON_FIST,GM_SOUND,GM_SHEER_FORCE,GM_INACCURATE,GM_RECOVERY,GM_CONTRARY,GM_PHYSICAL_SETUP,GM_SPECIAL_SETUP,GM_MIXED_SETUP,GM_SPEED_SETUP,GM_SPEED_CONTROL,GM_SETUP,GM_HAZARDS};
  const int keys[18]={GC_TECHNICIAN,GC_SKILL_LINK,GC_RECOIL,GC_DRAIN,GC_STRONG_JAW,GC_IRON_FIST,GC_SOUND,GC_SHEER_FORCE,GC_INACCURATE,GC_RECOVERY,GC_CONTRARY,GC_PHYSICAL_SETUP,GC_SPECIAL_SETUP,GC_MIXED_SETUP,GC_SPEED_SETUP,GC_SPEED_CONTROL,GC_SETUP,GC_HAZARDS};
  for(int i=0;i<count;i++){
    int id=moves[i];if(id<0||id>=rules.move_count)return false;const GenMove& move=rules.moves[id];
    int type=gen_move_type(move,species,abilities,tera);bool damaging=move.flags&GM_DAMAGE;
    if(damaging)result.counts[GC_DAMAGE]++;else result.counts[GC_PHYSICAL+move.category]++;
    for(int j=0;j<18;j++)if(move.flags&flags[j])result.counts[keys[j]]++;
    // Bite, punch and sound are counted only for moves with ordinary power.
    if(!(move.flags&GM_POWER)){for(int j=4;j<=6;j++)if(move.flags&flags[j])result.counts[keys[j]]--;}
    if(move.flags&GM_POWER){
      result.base_power_moves[result.base_power_count++]=id;
      if(!(move.flags&GM_NO_STAB)||(species.priority_species&&move.priority>0)){
        result.counts[type]++;if(gen_has_type(species.types,type))result.counts[GC_STAB]++;
        if(type==tera)result.counts[GC_STAB_TERA]++;damaging=true;
      }
      if(move.priority>0||((move.flags&GM_GRASSY_GLIDE)&&(abilities&GA_GRASSY_SURGE)))result.counts[GC_PRIORITY]++;
    }
    if(damaging)result.damaging_moves[result.damaging_count++]=id;
  }
  return true;
}
enum GenAbilityCode:int32_t { GAC_CHLOROPHYLL=0,GAC_SOLAR_POWER=1,GAC_DEFIANT=2,GAC_HYDRATION=3,GAC_SWIFT_SWIM=4,
  GAC_IRON_FIST=5,GAC_SKILL_LINK=6,GAC_OVERGROW=7,GAC_PRANKSTER=8,GAC_SAND_FORCE=9,GAC_SAND_RUSH=10,
  GAC_SLUSH_RUSH=11,GAC_SWARM=12,GAC_TORRENT=13,GAC_FLASH_FIRE=14,GAC_AFTERMATH=15,GAC_UNBURDEN=16 };
enum GenTeamFlag:uint32_t { GT_RAIN=1,GT_SUN=2,GT_SAND=4,GT_SNOW=8,GT_SCREENS=16,GT_STICKY_WEB=32,
  GT_STEALTH_ROCK=64,GT_DEFOG=128,GT_RAPID_SPIN=256,GT_TOXIC_SPIKES=512,GT_TWO_SPIKES=1024,GT_STATUS_CURE=2048,GT_TERA_BLAST=4096 };
enum GenChosenMoveFlag:uint32_t { GCM_DEFOG=1,GCM_FLIP_TURN=2,GCM_SNOWSCAPE=4 };
PK_HD inline bool gen_cull_ability(const GenRules& rules,int ability,const GenSpecies& species,const GenCounter& counter,uint32_t team,uint32_t moves){
  const int32_t* a=rules.ability_codes;
  if(ability==a[GAC_CHLOROPHYLL]||ability==a[GAC_SOLAR_POWER])return !(team&GT_SUN);
  if(ability==a[GAC_DEFIANT])return species.ability_species==2&&counter.counts[GC_STATUS]>0;
  if(ability==a[GAC_HYDRATION]||ability==a[GAC_SWIFT_SWIM])return !(team&GT_RAIN);
  if(ability==a[GAC_IRON_FIST])return !counter.counts[GC_IRON_FIST];
  if(ability==a[GAC_SKILL_LINK])return !counter.counts[GC_SKILL_LINK];
  if(ability==a[GAC_OVERGROW])return !counter.counts[3];
  if(ability==a[GAC_PRANKSTER])return !counter.counts[GC_STATUS];
  if(ability==a[GAC_SAND_FORCE]||ability==a[GAC_SAND_RUSH])return !(team&GT_SAND);
  if(ability==a[GAC_SLUSH_RUSH])return !(team&GT_SNOW);
  if(ability==a[GAC_SWARM])return !counter.counts[11];
  if(ability==a[GAC_TORRENT])return !counter.counts[2]&&!(moves&GCM_FLIP_TURN);
  return false;
}
PK_HD inline int gen_ability(const GenRules& rules,int species_id,const int32_t* abilities,int count,const GenCounter& counter,uint32_t team,uint32_t moves,int tera,uint64_t& rng){
  if(species_id<0||species_id>=rules.species_count||count<=0||count>32||tera<0||tera>18)return -1;
  if(count==1)return abilities[0];
  const GenSpecies& species=rules.species[species_id];const int32_t* a=rules.ability_codes;
  if(species.ability_species==1)return a[moves&GCM_DEFOG?GAC_AFTERMATH:GAC_UNBURDEN];
  bool flash=false,slush=false;for(int i=0;i<count;i++){flash|=abilities[i]==a[GAC_FLASH_FIRE];slush|=abilities[i]==a[GAC_SLUSH_RUSH];}
  if(flash&&(tera==3||tera==5||tera==11||tera==15))return a[GAC_FLASH_FIRE];
  if((species.ability_species==2||species.ability_species==3)&&!counter.counts[GC_PHYSICAL])return a[GAC_PRANKSTER];
  if(species.ability_species==4&&(counter.counts[2]||(moves&GCM_FLIP_TURN)))return a[GAC_TORRENT];
  if(species.ability_species==5&&counter.counts[GC_SKILL_LINK])return a[GAC_SKILL_LINK];
  if(slush&&(moves&GCM_SNOWSCAPE))return a[GAC_SLUSH_RUSH];
  if(species.ability_species==6&&(team&GT_RAIN))return a[GAC_SWIFT_SWIM];
  int allowed[32],length=0;
  for(int i=0;i<count;i++)if(!gen_cull_ability(rules,abilities[i],species,counter,team,moves))allowed[length++]=abilities[i];
  if(length)return allowed[gen_random(rng,length)];
  for(int i=0;i<count;i++){
    int id=abilities[i];if(id==a[GAC_CHLOROPHYLL]||id==a[GAC_HYDRATION]||id==a[GAC_SAND_FORCE]||id==a[GAC_SAND_RUSH]||id==a[GAC_SLUSH_RUSH]||id==a[GAC_SOLAR_POWER]||id==a[GAC_SWIFT_SWIM])allowed[length++]=id;
  }
  return length?allowed[gen_random(rng,length)]:abilities[gen_random(rng,count)];
}
PK_HD constexpr uint32_t gen_tag(const char* name){uint32_t value=2166136261u;for(int i=0;name[i];i++)value=(value^uint8_t(name[i]))*16777619u;return value;}
PK_HD inline int gen_find_move(const GenRules& rules,uint32_t tag){for(int i=0;i<rules.move_count;i++)if(rules.moves[i].id_tag==tag)return i;return -1;}
PK_HD inline int gen_list_index(const GenMoveList& list,int id){for(int i=0;i<list.count;i++)if(list.values[i]==id)return i;return -1;}
PK_HD inline bool gen_list_has(const GenMoveList& list,int id){return id>=0&&gen_list_index(list,id)>=0;}
PK_HD inline bool gen_list_remove(GenMoveList& list,int id){int i=gen_list_index(list,id);if(i<0)return false;list.values[i]=list.values[--list.count];return true;}
PK_HD inline bool gen_list_add(GenMoveList& list,int id){if(id<0)return false;if(gen_list_has(list,id))return true;if(list.count>=64)return false;list.values[list.count++]=id;return true;}
enum GenSelectorGroup:int32_t { GS_STATUS=0,GS_SETUP=1,GS_PHYSICAL_SETUP=2,GS_PIVOT=3,GS_HAZARDS=4,GS_SPEED_SETUP=5,GS_PROTECT=6,GS_RECOVERY=7,GS_SPEED_CONTROL=8 };
struct GenSelector {int group;uint32_t names[8];int count;};
PK_HD inline GenSelector gen_group(int group){return {group,{},0};}
PK_HD inline GenSelector gen_names(uint32_t a,uint32_t b=0,uint32_t c=0,uint32_t d=0,uint32_t e=0){return {-1,{a,b,c,d,e},1+int(b!=0)+int(c!=0)+int(d!=0)+int(e!=0)};}
PK_HD inline int gen_selector_size(const GenRules& rules,const GenSelector& selector){return selector.group>=0?rules.selector_ranges[selector.group].count:selector.count;}
PK_HD inline int gen_selector_move(const GenRules& rules,const GenSelector& selector,int index){return selector.group>=0?rules.selector_moves[rules.selector_ranges[selector.group].begin+index]:gen_find_move(rules,selector.names[index]);}
PK_HD inline bool gen_selector_has(const GenRules& rules,const GenSelector& selector,int id){for(int i=0;i<gen_selector_size(rules,selector);i++)if(gen_selector_move(rules,selector,i)==id)return true;return false;}
PK_HD inline void gen_incompatible(const GenRules& rules,const GenMoveList& chosen,GenMoveList& pool,GenSelector a,GenSelector b){
  if(chosen.count+pool.count<=4)return;
  for(int i=0;i<chosen.count;i++){
    int id=chosen.values[i];
    for(int direction=0;direction<2;direction++){
      const GenSelector& test=direction?a:b;const GenSelector& remove=direction?b:a;
      if(!gen_selector_has(rules,test,id))continue;
      for(int j=0;j<gen_selector_size(rules,remove);j++){
        int other=gen_selector_move(rules,remove,j);if(id!=other&&gen_list_remove(pool,other)&&chosen.count+pool.count<=4)return;
      }
    }
  }
}
PK_HD inline void gen_cull_move_pool(const GenRules& rules,int species_id,const GenMoveList& chosen,GenMoveList& pool,uint64_t abilities,uint32_t team,int tera){
  if(chosen.count+pool.count<=4)return;
  const GenSpecies& species=rules.species[species_id];
  const uint32_t pairs[5][2]={{gen_tag("lightscreen"),gen_tag("reflect")},{gen_tag("sleeptalk"),gen_tag("rest")},{gen_tag("protect"),gen_tag("wish")},{gen_tag("leechseed"),gen_tag("protect")},{gen_tag("leechseed"),gen_tag("substitute")}};
  if(chosen.count==2){
    GenMoveList unpaired=pool;
    for(int i=0;i<5;i++){int a=gen_find_move(rules,pairs[i][0]),b=gen_find_move(rules,pairs[i][1]);if(gen_list_has(pool,a)&&gen_list_has(pool,b)){if(!gen_list_remove(unpaired,a)||!gen_list_remove(unpaired,b)){pool.fault=1;return;}}}
    if(unpaired.count==1)gen_list_remove(pool,unpaired.values[0]);
  }
  if(chosen.count==3)for(int i=0;i<5;i++){int a=gen_find_move(rules,pairs[i][0]),b=gen_find_move(rules,pairs[i][1]);if(gen_list_has(pool,a)&&gen_list_has(pool,b)){gen_list_remove(pool,a);gen_list_remove(pool,b);}}
  if(team&GT_SCREENS){gen_list_remove(pool,gen_find_move(rules,gen_tag("auroraveil")));if(pool.count>=6){gen_list_remove(pool,gen_find_move(rules,gen_tag("reflect")));gen_list_remove(pool,gen_find_move(rules,gen_tag("lightscreen")));}}
  const uint32_t team_flags[6]={GT_STICKY_WEB,GT_STEALTH_ROCK,GT_DEFOG|GT_RAPID_SPIN,GT_TOXIC_SPIKES,GT_TWO_SPIKES,GT_STATUS_CURE};
  const uint32_t removal[6]={gen_tag("stickyweb"),gen_tag("stealthrock"),gen_tag("defog"),gen_tag("toxicspikes"),gen_tag("spikes"),gen_tag("healbell")};
  for(int i=0;i<6;i++)if(team&team_flags[i]){gen_list_remove(pool,gen_find_move(rules,removal[i]));if(i==2)gen_list_remove(pool,gen_find_move(rules,gen_tag("rapidspin")));if(chosen.count+pool.count<=4)return;}
  gen_incompatible(rules,chosen,pool,gen_group(GS_STATUS),gen_names(gen_tag("healingwish"),gen_tag("switcheroo"),gen_tag("trick")));
  gen_incompatible(rules,chosen,pool,gen_group(GS_SETUP),gen_group(GS_PIVOT));
  gen_incompatible(rules,chosen,pool,gen_group(GS_SETUP),gen_group(GS_HAZARDS));
  gen_incompatible(rules,chosen,pool,gen_group(GS_SETUP),gen_names(gen_tag("defog"),gen_tag("nuzzle"),gen_tag("toxic"),gen_tag("yawn"),gen_tag("haze")));
  gen_incompatible(rules,chosen,pool,gen_group(GS_PHYSICAL_SETUP),gen_group(GS_PHYSICAL_SETUP));
  gen_incompatible(rules,chosen,pool,gen_names(gen_tag("substitute")),gen_group(GS_PIVOT));
  gen_incompatible(rules,chosen,pool,gen_group(GS_SPEED_SETUP),gen_names(gen_tag("aquajet"),gen_tag("rest"),gen_tag("trickroom")));
  const GenSelector rest[][2]={
    {gen_names(gen_tag("curse")),gen_names(gen_tag("irondefense"),gen_tag("rapidspin"))},
    {gen_names(gen_tag("dragondance")),gen_names(gen_tag("dracometeor"))},
    {gen_names(gen_tag("yawn")),gen_names(gen_tag("roar"))},
    {gen_names(gen_tag("trick")),gen_names(gen_tag("uturn"))},
    {gen_names(gen_tag("psychic"),gen_tag("psychicnoise")),gen_names(gen_tag("psyshock"),gen_tag("psychicnoise"))},
    {gen_names(gen_tag("surf")),gen_names(gen_tag("hydropump"))},
    {gen_names(gen_tag("liquidation")),gen_names(gen_tag("wavecrash"))},
    {gen_names(gen_tag("aquajet")),gen_names(gen_tag("flipturn"))},
    {gen_names(gen_tag("gigadrain")),gen_names(gen_tag("leafstorm"))},
    {gen_names(gen_tag("powerwhip")),gen_names(gen_tag("hornleech"))},
    {gen_names(gen_tag("airslash")),gen_names(gen_tag("hurricane"))},
    {gen_names(gen_tag("knockoff")),gen_names(gen_tag("foulplay"))},
    {gen_names(gen_tag("throatchop")),gen_names(gen_tag("crunch"),gen_tag("lashout"))},
    {gen_names(gen_tag("doubleedge")),gen_names(gen_tag("bodyslam"),gen_tag("headbutt"))},
    {gen_names(gen_tag("fireblast"),gen_tag("magmastorm")),gen_names(gen_tag("fierydance"),gen_tag("flamethrower"),gen_tag("lavaplume"))},
    {gen_names(gen_tag("thunderpunch")),gen_names(gen_tag("wildcharge"))},
    {gen_names(gen_tag("thunderbolt")),gen_names(gen_tag("discharge"))},
    {gen_names(gen_tag("gunkshot")),gen_names(gen_tag("direclaw"),gen_tag("poisonjab"),gen_tag("sludgebomb"))},
    {gen_names(gen_tag("aurasphere")),gen_names(gen_tag("focusblast"))},
    {gen_names(gen_tag("closecombat")),gen_names(gen_tag("drainpunch"))},
    {gen_names(gen_tag("dragonpulse"),gen_tag("spacialrend")),gen_names(gen_tag("dracometeor"))},
    {gen_names(gen_tag("dragonclaw")),gen_names(gen_tag("outrage"))},
    {gen_names(gen_tag("heavyslam")),gen_names(gen_tag("flashcannon"))},
    {gen_names(gen_tag("alluringvoice")),gen_names(gen_tag("dazzlinggleam"))},
    {gen_names(gen_tag("taunt")),gen_names(gen_tag("disable"))},
    {gen_names(gen_tag("thunderwave"),gen_tag("toxic")),gen_names(gen_tag("thunderwave"),gen_tag("willowisp"))},
    {gen_names(gen_tag("thunderwave"),gen_tag("toxic"),gen_tag("willowisp")),gen_names(gen_tag("toxicspikes"))},
    {gen_names(gen_tag("nastyplot")),gen_names(gen_tag("rockslide"),gen_tag("knockoff"))},
    {gen_names(gen_tag("switcheroo")),gen_names(gen_tag("fakeout"))},
    {gen_names(gen_tag("toxic")),gen_names(gen_tag("clearsmog"))},
    {gen_names(gen_tag("healbell")),gen_names(gen_tag("stealthrock"))},
    {gen_names(gen_tag("mirrorcoat")),gen_names(gen_tag("hydropump"),gen_tag("bodypress"))},
  };
  for(const auto& pair:rest)gen_incompatible(rules,chosen,pool,pair[0],pair[1]);
  if(!gen_has_type(species.types,5))gen_incompatible(rules,chosen,pool,gen_names(gen_tag("icebeam")),gen_names(gen_tag("icywind")));
  gen_incompatible(rules,chosen,pool,gen_names(gen_tag("taunt")),gen_names(gen_tag("encore")));
  if(!gen_has_type(species.types,16)&&tera!=16)gen_incompatible(rules,chosen,pool,gen_names(gen_tag("knockoff")),gen_names(gen_tag("suckerpunch")));
  if(!(abilities&GA_PRANKSTER))gen_incompatible(rules,chosen,pool,gen_names(gen_tag("thunderwave")),gen_names(gen_tag("yawn")));
  if(species.id_tag==gen_tag("barraskewda"))gen_incompatible(rules,chosen,pool,gen_names(gen_tag("psychicfangs"),gen_tag("throatchop")),gen_names(gen_tag("poisonjab"),gen_tag("throatchop")));
  if(species.id_tag==gen_tag("quagsire"))gen_incompatible(rules,chosen,pool,gen_names(gen_tag("spikes")),gen_names(gen_tag("icebeam")));
  if(species.id_tag==gen_tag("cyclizar"))gen_incompatible(rules,chosen,pool,gen_names(gen_tag("taunt")),gen_names(gen_tag("knockoff")));
  if(species.id_tag==gen_tag("coalossal"))gen_incompatible(rules,chosen,pool,gen_names(gen_tag("flamethrower")),gen_names(gen_tag("overheat")));
}
enum GenRole:int32_t { GR_BULKY_ATTACKER=0,GR_BULKY_SETUP=1,GR_BULKY_SUPPORT=2,GR_FAST_ATTACKER=3,GR_FAST_BULKY_SETUP=4,
  GR_FAST_SUPPORT=5,GR_SETUP_SWEEPER=6,GR_TERA_BLAST=7,GR_WALLBREAKER=8,GR_AV_PIVOT=9,GR_BULKY_PROTECT=10,GR_OFFENSIVE_PROTECT=11,GR_CHOICE_ITEM=12 };
PK_HD inline bool gen_pool_named(const GenRules& rules,const GenMoveList& pool,const char* name){return gen_list_has(pool,gen_find_move(rules,gen_tag(name)));}
PK_HD inline bool gen_add_chosen(const GenRules& rules,int species,GenMoveList& chosen,GenMoveList& pool,int move,uint64_t abilities,uint32_t team,int tera,GenCounter& counter){
  if(!gen_list_add(chosen,move)||!gen_list_remove(pool,move)){chosen.fault=1;return false;}
  if(!gen_query_moves(rules,species,chosen.values,chosen.count,abilities,tera,counter)){chosen.fault=1;return false;}
  gen_cull_move_pool(rules,species,chosen,pool,abilities,team,tera);if(pool.fault){chosen.fault=1;return false;}return true;
}
PK_HD inline bool gen_stab_needed(const GenRules& rules,const GenSpecies& species,const GenMoveList& pool,const GenCounter& counter,uint64_t abilities,int role,int type){
  switch(type){
    case 11:return gen_pool_named(rules,pool,"megahorn")||gen_pool_named(rules,pool,"xscissor")||(!counter.counts[11]&&(gen_has_type(species.types,4)||gen_has_type(species.types,10)));
    case 16:return (counter.counts[16]<2&&species.priority_species&&role==GR_WALLBREAKER)||!counter.counts[16];
    case 3:return !counter.counts[3]&&(gen_pool_named(rules,pool,"leafstorm")||species.stats[1]>=100||gen_has_type(species.types,4)||(abilities&GA_SEED_SOWER));
    case 5:return gen_pool_named(rules,pool,"freezedry")||gen_pool_named(rules,pool,"blizzard")||!counter.counts[5];
    case 0:return gen_pool_named(rules,pool,"boomburst")||gen_pool_named(rules,pool,"hypervoice");
    case 7:return !gen_has_type(species.types,8)&&!counter.counts[7];
    case 10:
      if(species.id_tag==gen_tag("bruxish")&&gen_pool_named(rules,pool,"psychicfangs"))return true;
      if(species.id_tag==gen_tag("hoopaunbound")&&gen_pool_named(rules,pool,"psychic"))return true;
      return !gen_has_type(species.types,16)&&!gen_has_type(species.types,15)&&!gen_has_type(species.types,2)&&!counter.counts[10];
    case 12:return !counter.counts[12]&&species.stats[1]>=80;
    case 15:return !counter.counts[15]&&(species.stats[1]>=90||gen_pool_named(rules,pool,"gigatonhammer")||gen_pool_named(rules,pool,"makeitrain"));
    case 2:return !counter.counts[2]&&!gen_has_type(species.types,8);
    case 1:case 4:case 6:case 8:case 9:case 13:case 14:case 17:return !counter.counts[type];
    default:return false;
  }
}
PK_HD inline bool gen_random_moveset(const GenRules& rules,int species_id,GenMoveList& pool,uint64_t abilities,uint32_t team,int tera,int role,uint64_t& rng,GenMoveList& chosen){
  chosen={};if(species_id<0||species_id>=rules.species_count||pool.count<0||pool.count>64||role<0||role>12)return false;
  const GenSpecies& species=rules.species[species_id];GenCounter counter{};
  gen_cull_move_pool(rules,species_id,chosen,pool,abilities,team,tera);if(pool.fault)return false;
  if(pool.count<=4){chosen=pool;return true;}
  if(role==GR_TERA_BLAST&&!gen_add_chosen(rules,species_id,chosen,pool,gen_find_move(rules,gen_tag("terablast")),abilities,team,tera,counter))return false;
  if(species.required_move>=0&&!gen_add_chosen(rules,species_id,chosen,pool,species.required_move,abilities,team,tera,counter))return false;
  if((abilities&GA_GUTS)&&gen_pool_named(rules,pool,"facade"))if(!gen_add_chosen(rules,species_id,chosen,pool,gen_find_move(rules,gen_tag("facade")),abilities,team,tera,counter))return false;
  const uint32_t enforced[4]={gen_tag("nightshade"),gen_tag("revelationdance"),gen_tag("revivalblessing"),gen_tag("stickyweb")};
  for(int i=0;i<4;i++){int move=gen_find_move(rules,enforced[i]);if(gen_list_has(pool,move)&&!gen_add_chosen(rules,species_id,chosen,pool,move,abilities,team,tera,counter))return false;}
  if(role==GR_BULKY_SUPPORT&&!(team&(GT_DEFOG|GT_RAPID_SPIN))){
    const uint32_t removal[2]={gen_tag("rapidspin"),gen_tag("defog")};for(int i=0;i<2;i++){int move=gen_find_move(rules,removal[i]);if(gen_list_has(pool,move)&&!gen_add_chosen(rules,species_id,chosen,pool,move,abilities,team,tera,counter))return false;}
  }
  if(species.stats[5]<=60&&(gen_pool_named(rules,pool,"dragontail")||gen_pool_named(rules,pool,"roar")||gen_pool_named(rules,pool,"whirlwind"))&&!(team&GT_STEALTH_ROCK)&&gen_pool_named(rules,pool,"stealthrock"))if(!gen_add_chosen(rules,species_id,chosen,pool,gen_find_move(rules,gen_tag("stealthrock")),abilities,team,tera,counter))return false;
  if(!(team&GT_SCREENS)&&gen_pool_named(rules,pool,"auroraveil"))if(!gen_add_chosen(rules,species_id,chosen,pool,gen_find_move(rules,gen_tag("auroraveil")),abilities,team,tera,counter))return false;
  if(species.types[1]<0&&(species.types[0]==0||species.types[0]==6)&&gen_pool_named(rules,pool,"knockoff"))if(!gen_add_chosen(rules,species_id,chosen,pool,gen_find_move(rules,gen_tag("knockoff")),abilities,team,tera,counter))return false;
  if(species.id_tag==gen_tag("smeargle")&&gen_pool_named(rules,pool,"spore"))if(!gen_add_chosen(rules,species_id,chosen,pool,gen_find_move(rules,gen_tag("spore")),abilities,team,tera,counter))return false;
  GenMoveList candidates{};
  if(role==GR_BULKY_ATTACKER||role==GR_BULKY_SETUP||role==GR_WALLBREAKER||species.priority_species){
    for(int i=0;i<pool.count;i++){int id=pool.values[i];const GenMove& move=rules.moves[id];int type=gen_move_type(move,species,abilities,tera);if(gen_has_type(species.types,type)&&(move.priority>0||((move.flags&GM_GRASSY_GLIDE)&&(abilities&GA_GRASSY_SURGE)))&&(move.flags&GM_POWER))candidates.values[candidates.count++]=id;}
    if(candidates.count&&!gen_add_chosen(rules,species_id,chosen,pool,candidates.values[gen_random(rng,candidates.count)],abilities,team,tera,counter))return false;
  }
  for(int t=0;t<2;t++){
    int type=species.types[t];if(type<0||(t&&type==species.types[0]))continue;candidates={};
    for(int i=0;i<pool.count;i++){int id=pool.values[i];const GenMove& move=rules.moves[id];if(!(move.flags&GM_NO_STAB)&&(move.flags&GM_POWER)&&type==gen_move_type(move,species,abilities,tera))candidates.values[candidates.count++]=id;}
    while(gen_stab_needed(rules,species,pool,counter,abilities,role,type)&&candidates.count){int move=gen_sample_no_replace(rng,candidates.values,candidates.count);if(!gen_add_chosen(rules,species_id,chosen,pool,move,abilities,team,tera,counter))return false;}
  }
  for(int stage=0;stage<9;stage++){
    candidates={};bool active=false;
    if(stage==0)active=!counter.counts[GC_STAB_TERA]&&role!=GR_BULKY_SUPPORT;
    if(stage==1)active=!counter.counts[GC_STAB];
    if(stage==2)active=role==GR_BULKY_SUPPORT||role==GR_BULKY_ATTACKER||role==GR_BULKY_SETUP;
    if(stage==3)active=role==GR_AV_PIVOT;
    if(stage==4)active=role==GR_BULKY_SETUP||role==GR_FAST_BULKY_SETUP||role==GR_SETUP_SWEEPER||role==GR_TERA_BLAST;
    if(stage==5)active=role==GR_BULKY_PROTECT||role==GR_OFFENSIVE_PROTECT;
    if(stage==6)active=counter.damaging_count==0;
    if(stage==7)active=counter.damaging_count==1&&role!=GR_AV_PIVOT&&role!=GR_FAST_SUPPORT&&role!=GR_BULKY_SUPPORT&&role!=GR_BULKY_PROTECT;
    if(!active)continue;
    for(int i=0;i<pool.count;i++){
      int id=pool.values[i];const GenMove& move=rules.moves[id];int type=gen_move_type(move,species,abilities,tera);bool include=false;
      if(stage<=1)include=!(move.flags&GM_NO_STAB)&&(move.flags&GM_POWER)&&(stage?gen_has_type(species.types,type):type==tera);
      if(stage==2)include=move.flags&GM_RECOVERY;
      if(stage==3)include=move.id_tag==gen_tag("uturn")||move.id_tag==gen_tag("voltswitch");
      if(stage==4)include=(move.flags&GM_SETUP)&&!(move.flags&GM_SPEED_SETUP);
      if(stage==5)include=gen_selector_has(rules,gen_group(GS_PROTECT),id);
      if(stage==6)include=!(move.flags&GM_NO_STAB)&&move.category!=2;
      if(stage==7)include=!(move.flags&GM_NO_STAB)&&(move.flags&GM_POWER)&&rules.moves[counter.damaging_moves[0]].type!=type;
      if(include)candidates.values[candidates.count++]=id;
    }
    if(stage==4&&!candidates.count)for(int i=0;i<pool.count;i++)if(rules.moves[pool.values[i]].flags&GM_SETUP)candidates.values[candidates.count++]=pool.values[i];
    if(candidates.count&&!gen_add_chosen(rules,species_id,chosen,pool,candidates.values[gen_random(rng,candidates.count)],abilities,team,tera,counter))return false;
  }
  const uint32_t pairs[5][2]={{gen_tag("lightscreen"),gen_tag("reflect")},{gen_tag("sleeptalk"),gen_tag("rest")},{gen_tag("protect"),gen_tag("wish")},{gen_tag("leechseed"),gen_tag("protect")},{gen_tag("leechseed"),gen_tag("substitute")}};
  while(chosen.count<4&&pool.count){
    if(chosen.count+pool.count<=4){for(int i=0;i<pool.count;i++)if(!gen_list_add(chosen,pool.values[i]))return false;break;}
    int move=pool.values[gen_random(rng,pool.count)];if(!gen_add_chosen(rules,species_id,chosen,pool,move,abilities,team,tera,counter))return false;
    for(int i=0;i<5;i++)for(int direction=0;direction<2;direction++){
      int selected=gen_find_move(rules,pairs[i][direction]),other=gen_find_move(rules,pairs[i][1-direction]);
      if(move==selected&&gen_list_has(pool,other)&&!gen_add_chosen(rules,species_id,chosen,pool,other,abilities,team,tera,counter))return false;
    }
  }
  return !chosen.fault;
}
PK_HD inline int gen_item(const GenRules& rules,const char* name){uint32_t tag=gen_tag(name);for(int i=0;i<rules.item_count;i++)if(rules.item_tags[i]==tag)return i;return -2;}
PK_HD inline bool gen_is_ability(const GenRules& rules,int ability,const char* name){return ability>=0&&rules.ability_tags[ability]==gen_tag(name);}
PK_HD inline bool gen_named_chosen(const GenRules& rules,const GenMoveList& chosen,const char* name){return gen_pool_named(rules,chosen,name);}
PK_HD inline bool gen_has_any(const GenRules& rules,const GenMoveList& chosen,GenSelector selector){for(int i=0;i<chosen.count;i++)if(gen_selector_has(rules,selector,chosen.values[i]))return true;return false;}
PK_HD inline int gen_priority_item(const GenRules& rules,int species_id,int ability,const GenMoveList& moves,const GenCounter& counter,uint32_t team,bool lead,int role,uint64_t& rng){
  const GenSpecies& s=rules.species[species_id];const int32_t* c=counter.counts;
  if(role==GR_FAST_BULKY_SETUP&&(gen_is_ability(rules,ability,"quarkdrive")||gen_is_ability(rules,ability,"protosynthesis")))return gen_item(rules,"boosterenergy");
  if(s.id_tag==gen_tag("lokix"))return gen_item(rules,role==GR_FAST_ATTACKER?"silverpowder":"lifeorb");
  if(s.required_item_count){int index=s.base_tag==gen_tag("arceus")?0:gen_random(rng,s.required_item_count);return rules.required_items[s.required_item_begin+index];}
  if(s.id_tag==gen_tag("pikachu"))return gen_item(rules,"lightball");
  if(role==GR_AV_PIVOT)return gen_item(rules,"assaultvest");
  if(s.id_tag==gen_tag("regieleki"))return gen_item(rules,"magnet");
  if(gen_has_type(s.types,0)&&gen_named_chosen(rules,moves,"doubleedge")&&gen_named_chosen(rules,moves,"fakeout"))return gen_item(rules,"silkscarf");
  if(s.id_tag==gen_tag("froslass")||gen_named_chosen(rules,moves,"populationbomb")||(gen_is_ability(rules,ability,"hustle")&&c[GC_SETUP]&&gen_random(rng,2)==0))return gen_item(rules,"widelens");
  if(s.id_tag==gen_tag("smeargle"))return gen_item(rules,"focussash");
  if(gen_named_chosen(rules,moves,"clangoroussoul")||(s.id_tag==gen_tag("toxtricity")&&gen_named_chosen(rules,moves,"shiftgear")))return gen_item(rules,"throatspray");
  if(s.base_tag==gen_tag("magearna")&&role==GR_TERA_BLAST)return gen_item(rules,"weaknesspolicy");
  if(gen_has_any(rules,moves,gen_names(gen_tag("dragonenergy"),gen_tag("lastrespects"),gen_tag("waterspout"))))return gen_item(rules,"choicescarf");
  if(gen_is_ability(rules,ability,"imposter")||(s.id_tag==gen_tag("magnezone")&&role==GR_FAST_ATTACKER))return gen_item(rules,"choicescarf");
  if(s.id_tag==gen_tag("rampardos")&&role==GR_FAST_ATTACKER)return gen_item(rules,"choicescarf");
  if(s.id_tag==gen_tag("palkia")&&c[GC_STATUS])return gen_item(rules,"lustrousorb");
  if(gen_named_chosen(rules,moves,"courtchange")||s.id_tag==gen_tag("luvdisc")||(s.id_tag==gen_tag("terapagos")&&!gen_named_chosen(rules,moves,"rest")))return gen_item(rules,"heavydutyboots");
  if(gen_is_ability(rules,ability,"cheekpouch")||gen_is_ability(rules,ability,"cudchew")||gen_is_ability(rules,ability,"harvest")||gen_is_ability(rules,ability,"ripen")||gen_named_chosen(rules,moves,"bellydrum")||gen_named_chosen(rules,moves,"filletaway"))return gen_item(rules,"sitrusberry");
  if(gen_has_any(rules,moves,gen_names(gen_tag("healingwish"),gen_tag("switcheroo"),gen_tag("trick")))){
    if(s.stats[5]>=60&&s.stats[5]<=108&&role!=GR_WALLBREAKER&&!c[GC_PRIORITY])return gen_item(rules,"choicescarf");
    return gen_item(rules,c[GC_PHYSICAL]>c[GC_SPECIAL]?"choiceband":"choicespecs");
  }
  if(c[GC_STATUS]&&(s.id_tag==gen_tag("latias")||s.id_tag==gen_tag("latios")))return gen_item(rules,"souldew");
  if(s.id_tag==gen_tag("scyther"))return gen_item(rules,lead&&!gen_named_chosen(rules,moves,"uturn")?"eviolite":"heavydutyboots");
  if(gen_is_ability(rules,ability,"poisonheal")||gen_is_ability(rules,ability,"quickfeet"))return gen_item(rules,"toxicorb");
  if(s.nfe)return gen_item(rules,"eviolite");
  if((gen_is_ability(rules,ability,"guts")||gen_named_chosen(rules,moves,"facade"))&&!gen_named_chosen(rules,moves,"sleeptalk"))return gen_item(rules,gen_has_type(s.types,1)||gen_is_ability(rules,ability,"toxicboost")?"toxicorb":"flameorb");
  if(gen_is_ability(rules,ability,"magicguard")||(gen_is_ability(rules,ability,"sheerforce")&&c[GC_SHEER_FORCE]))return gen_item(rules,"lifeorb");
  if(gen_is_ability(rules,ability,"angershell")){const char* choices[4]={"expertbelt","lumberry","scopelens","sitrusberry"};return gen_item(rules,choices[gen_random(rng,4)]);}
  if(c[GC_SKILL_LINK]&&!gen_is_ability(rules,ability,"skilllink")&&s.id_tag!=gen_tag("breloom"))return gen_item(rules,"loadeddice");
  if(gen_is_ability(rules,ability,"unburden"))return gen_item(rules,gen_named_chosen(rules,moves,"closecombat")||gen_named_chosen(rules,moves,"leafstorm")?"whiteherb":"sitrusberry");
  if(gen_named_chosen(rules,moves,"shellsmash")&&!gen_is_ability(rules,ability,"weakarmor"))return gen_item(rules,"whiteherb");
  if(gen_named_chosen(rules,moves,"meteorbeam")||(gen_named_chosen(rules,moves,"electroshot")&&!(team&GT_RAIN)))return gen_item(rules,"powerherb");
  if(gen_named_chosen(rules,moves,"acrobatics")&&!gen_is_ability(rules,ability,"protosynthesis"))return gen_item(rules,"");
  if(gen_named_chosen(rules,moves,"auroraveil")||(gen_named_chosen(rules,moves,"lightscreen")&&gen_named_chosen(rules,moves,"reflect")))return gen_item(rules,"lightclay");
  if(gen_is_ability(rules,ability,"gluttony")){const char* choices[5]={"aguavberry","figyberry","iapapaberry","magoberry","wikiberry"};return gen_item(rules,choices[gen_random(rng,5)]);}
  if(s.id_tag==gen_tag("giratina")&&gen_named_chosen(rules,moves,"rest")&&!gen_named_chosen(rules,moves,"sleeptalk"))return gen_item(rules,"leftovers");
  if(gen_named_chosen(rules,moves,"rest")&&!gen_named_chosen(rules,moves,"sleeptalk")&&!gen_is_ability(rules,ability,"naturalcure")&&!gen_is_ability(rules,ability,"shedskin"))return gen_item(rules,"chestoberry");
  if(s.id_tag!=gen_tag("yanmega")&&s.effectiveness[12]>=2)return gen_item(rules,"heavydutyboots");
  return -1;
}
PK_HD inline int gen_regular_item(const GenRules& rules,int species_id,int ability,const GenMoveList& moves,const GenCounter& counter,uint32_t team,bool lead,int role,uint64_t& rng){
  const GenSpecies& s=rules.species[species_id];const int32_t* c=counter.counts;
  bool life_orb=!gen_has_any(rules,moves,gen_names(gen_tag("flamecharge"),gen_tag("nuzzle"),gen_tag("rapidspin")));
  if(s.id_tag!=gen_tag("jirachi")&&c[GC_PHYSICAL]>=moves.count&&!gen_has_any(rules,moves,gen_names(gen_tag("dragontail"),gen_tag("fakeout"),gen_tag("firstimpression"),gen_tag("flamecharge"),gen_tag("rapidspin")))&&!gen_named_chosen(rules,moves,"trailblaze")){
    bool scarf=role!=GR_WALLBREAKER&&(s.stats[1]>=100||gen_is_ability(rules,ability,"hugepower")||gen_is_ability(rules,ability,"purepower"))&&s.stats[5]>=60&&s.stats[5]<=108&&!gen_is_ability(rules,ability,"speedboost")&&!c[GC_PRIORITY];
    return gen_item(rules,scarf&&gen_random(rng,2)==0?"choicescarf":"choiceband");
  }
  if(c[GC_SPECIAL]>=moves.count||(c[GC_SPECIAL]>=moves.count-1&&gen_has_any(rules,moves,gen_names(gen_tag("flipturn"),gen_tag("uturn"))))){
    bool scarf=role!=GR_WALLBREAKER&&s.stats[3]>=100&&s.stats[5]>=60&&s.stats[5]<=108&&!gen_is_ability(rules,ability,"speedboost")&&!gen_is_ability(rules,ability,"tintedlens")&&!gen_named_chosen(rules,moves,"uturn")&&!c[GC_PRIORITY];
    return gen_item(rules,scarf&&gen_random(rng,2)==0?"choicescarf":"choicespecs");
  }
  if(c[GC_SPEED_SETUP]&&!c[GC_PHYSICAL_SETUP]&&role==GR_BULKY_SETUP)return gen_item(rules,"weaknesspolicy");
  if(!c[GC_STATUS]&&role!=GR_FAST_ATTACKER&&role!=GR_WALLBREAKER&&role!=GR_TERA_BLAST)return gen_item(rules,"assaultvest");
  if(s.id_tag==gen_tag("golem"))return gen_item(rules,c[GC_SPEED_SETUP]?"weaknesspolicy":"custapberry");
  if(gen_named_chosen(rules,moves,"substitute"))return gen_item(rules,"leftovers");
  if(gen_named_chosen(rules,moves,"stickyweb")&&lead&&s.stats[0]+s.stats[2]+s.stats[4]<=235)return gen_item(rules,"focussash");
  if(s.effectiveness[12]>=1)return gen_item(rules,"heavydutyboots");
  if(gen_named_chosen(rules,moves,"chillyreception")||(role==GR_FAST_SUPPORT&&!gen_has_type(s.types,9)&&!gen_is_ability(rules,ability,"levitate")&&(gen_has_any(rules,moves,gen_group(GS_PIVOT))||gen_has_any(rules,moves,gen_names(gen_tag("defog"),gen_tag("mortalspin"),gen_tag("rapidspin"))))))return gen_item(rules,"heavydutyboots");
  if(gen_named_chosen(rules,moves,"dragondance")&&role==GR_BULKY_SETUP)return gen_item(rules,"weaknesspolicy");
  if(gen_is_ability(rules,ability,"roughskin")||(gen_is_ability(rules,ability,"regenerator")&&(role==GR_BULKY_SUPPORT||role==GR_BULKY_ATTACKER)&&s.stats[0]+s.stats[2]>=180&&gen_random(rng,2)==0)||(!gen_is_ability(rules,ability,"regenerator")&&!c[GC_SETUP]&&c[GC_RECOVERY]&&s.effectiveness[6]<1&&s.stats[0]+s.stats[2]>200&&gen_random(rng,2)==0))return gen_item(rules,"rockyhelmet");
  if(gen_named_chosen(rules,moves,"outrage")&&c[GC_SETUP])return gen_item(rules,"lumberry");
  if(gen_named_chosen(rules,moves,"protect")&&!gen_is_ability(rules,ability,"speedboost"))return gen_item(rules,"leftovers");
  if(role==GR_FAST_SUPPORT&&lead&&!c[GC_RECOVERY]&&!c[GC_RECOIL]&&(c[GC_HAZARDS]||c[GC_SETUP])&&s.stats[0]+s.stats[2]+s.stats[4]<258)return gen_item(rules,"focussash");
  if(!c[GC_SETUP]&&!gen_is_ability(rules,ability,"levitate")&&s.effectiveness[8]>=2)return gen_item(rules,"airballoon");
  if(role==GR_BULKY_ATTACKER||role==GR_BULKY_SUPPORT||role==GR_BULKY_SETUP)return gen_item(rules,"leftovers");
  if(s.id_tag==gen_tag("pawmot")&&gen_named_chosen(rules,moves,"nuzzle"))return gen_item(rules,"leppaberry");
  if(role==GR_FAST_SUPPORT||role==GR_FAST_BULKY_SETUP)return gen_item(rules,c[GC_PHYSICAL]+c[GC_SPECIAL]>c[GC_STATUS]&&life_orb?"lifeorb":"leftovers");
  if(role==GR_TERA_BLAST&&(s.id_tag==gen_tag("alcremie")||s.id_tag==gen_tag("bellossom")||s.id_tag==gen_tag("comfey")||s.id_tag==gen_tag("fezandipiti")||s.id_tag==gen_tag("florges")))return gen_item(rules,"leftovers");
  if(life_orb&&(role==GR_FAST_ATTACKER||role==GR_SETUP_SWEEPER||role==GR_TERA_BLAST||role==GR_WALLBREAKER))return gen_item(rules,"lifeorb");
  return gen_item(rules,"leftovers");
}
PK_HD inline int gen_choose_item(const GenRules& rules,int species,int ability,const GenMoveList& moves,const GenCounter& counter,uint32_t team,bool lead,int role,uint64_t& rng){
  int priority=gen_priority_item(rules,species,ability,moves,counter,team,lead,role,rng);
  return priority==-1?gen_regular_item(rules,species,ability,moves,counter,team,lead,role,rng):priority;
}
PK_HD inline bool gen_is_item(const GenRules& rules,int item,const char* name){return item>=0&&item<rules.item_count&&rules.item_tags[item]==gen_tag(name);}
PK_HD inline bool gen_random_set(const GenRules& rules,int species_id,uint32_t team,bool lead,uint64_t& rng,GenSet& result){
  result={};if(species_id<0||species_id>=rules.species_count)return false;
  const GenSpecies& s=rules.species[species_id];result.species=species_id;result.forme=gen_forme(rules,species_id,rng);if(result.forme<0)return false;
  int possible[64],count=0;
  for(int i=0;i<s.set_count;i++){
    int index=s.set_begin+i;const GenSetSpec& set=rules.sets[index];
    if(lead&&set.role==GR_FAST_BULKY_SETUP){bool paradox=false;for(int j=0;j<set.abilities.count;j++){int a=rules.set_abilities[set.abilities.begin+j];paradox|=gen_is_ability(rules,a,"protosynthesis")||gen_is_ability(rules,a,"quarkdrive");}if(paradox)continue;}
    if((team&GT_TERA_BLAST)&&set.role==GR_TERA_BLAST)continue;
    if(count>=64)return false;possible[count++]=index;
  }
  if(!count)return false;const GenSetSpec& set=rules.sets[possible[gen_random(rng,count)]];
  if(set.teras.count<=0||set.moves.count>64||set.abilities.count<=0)return false;
  result.role=set.role;result.tera=rules.set_teras[set.teras.begin+(set.sample_tera?gen_random(rng,set.teras.count):0)];
  GenMoveList pool{};pool.count=set.moves.count;for(int i=0;i<pool.count;i++)pool.values[i]=rules.set_moves[set.moves.begin+i];
  if(!gen_random_moveset(rules,species_id,pool,set.ability_flags,team,result.tera,result.role,rng,result.moves))return false;
  GenCounter counter{};if(!gen_query_moves(rules,species_id,result.moves.values,result.moves.count,set.ability_flags,result.tera,counter))return false;
  uint32_t move_flags=(gen_named_chosen(rules,result.moves,"defog")?GCM_DEFOG:0)|(gen_named_chosen(rules,result.moves,"flipturn")?GCM_FLIP_TURN:0)|(gen_named_chosen(rules,result.moves,"snowscape")?GCM_SNOWSCAPE:0);
  result.ability=gen_ability(rules,species_id,rules.set_abilities+set.abilities.begin,set.abilities.count,counter,team,move_flags,result.tera,rng);
  if(result.ability<0)return false;result.item=gen_choose_item(rules,species_id,result.ability,result.moves,counter,team,lead,result.role,rng);if(result.item<0)return false;
  result.level=s.level;for(int i=0;i<6;i++){result.evs[i]=85;result.ivs[i]=31;}
  bool immune=gen_is_ability(rules,result.ability,"magicguard")||gen_is_item(rules,result.item,"heavydutyboots");int weakness=immune?0:s.effectiveness[12];
  if(gen_has_any(rules,result.moves,gen_names(gen_tag("axekick"),gen_tag("highjumpkick"),gen_tag("jumpkick"),gen_tag("supercellslam"))))weakness=2;
  while(result.evs[0]>1){
    int hp=(2*s.stats[0]+result.ivs[0]+result.evs[0]/4+100)*result.level/100+10;
    if((gen_named_chosen(rules,result.moves,"substitute")&&gen_is_item(rules,result.item,"sitrusberry"))||s.id_tag==gen_tag("minior")){if(hp%4==0)break;}
    else if(gen_has_any(rules,result.moves,gen_names(gen_tag("bellydrum"),gen_tag("filletaway"),gen_tag("shedtail")))&&(gen_is_item(rules,result.item,"sitrusberry")||gen_is_ability(rules,result.ability,"gluttony"))){if(hp%2==0)break;}
    else if(gen_named_chosen(rules,result.moves,"substitute")&&gen_named_chosen(rules,result.moves,"endeavor")){if(hp%4>0)break;}
    else{
      if(weakness<=0||gen_is_ability(rules,result.ability,"regenerator")||gen_is_item(rules,result.item,"leftovers")||gen_is_item(rules,result.item,"lifeorb"))break;
      int divisor=4/weakness;if(!gen_is_item(rules,result.item,"sitrusberry")&&hp%divisor>0)break;
      if(gen_is_item(rules,result.item,"sitrusberry")&&hp%divisor==0)break;
    }
    result.evs[0]-=4;
  }
  bool no_attack=true;
  for(int i=0;i<result.moves.count;i++){
    const GenMove& move=rules.moves[result.moves.values[i]];if(move.flags&GM_DAMAGE)continue;
    if(move.id_tag==gen_tag("shellsidearm")){no_attack=false;break;}
    if(move.id_tag==gen_tag("terablast")&&(s.id_tag==gen_tag("porygon2")||gen_is_ability(rules,result.ability,"contrary")||gen_is_ability(rules,result.ability,"defiant")||gen_named_chosen(rules,result.moves,"shiftgear")||s.stats[1]>s.stats[3])){no_attack=false;break;}
    if(move.category==0&&move.id_tag!=gen_tag("bodypress")&&move.id_tag!=gen_tag("foulplay")){no_attack=false;break;}
  }
  if(no_attack&&!gen_named_chosen(rules,result.moves,"transform")){result.evs[1]=0;result.ivs[1]=0;}
  if(gen_named_chosen(rules,result.moves,"gyroball")||gen_named_chosen(rules,result.moves,"trickroom")){result.evs[5]=0;result.ivs[5]=0;}
  for(int i=0;i<result.moves.count-1;i++){int j=i+gen_random(rng,result.moves.count-i),move=result.moves.values[i];result.moves.values[i]=result.moves.values[j];result.moves.values[j]=move;}
  result.gender=s.base_tag==gen_tag("greninja")?1:s.gender?s.gender:gen_random(rng,2)?2:1;
  result.shiny=gen_random(rng,1024)==0;return true;
}
// team_size is the format's Max Team Size (1..6); 6 is unmodified gen9randombattle. The frozen
// generator scales its type/weakness limits by max(1,round(team_size/6)), which is 1 for every size here.
PK_HD inline bool gen_random_team(const GenRules& rules,uint64_t& rng,GenTeam& result,int team_size=6){
  if(team_size<1||team_size>6){result={};result.fault=1;return false;}
  result={};gen_random(rng,18); // The frozen source samples an unused monotype type.
  GenPool pool{};if(!gen_initialize_pool(rules,pool))return false;
  uint8_t bases[GEN_GROUP_CAPACITY]={};int type_count[19]={},weaknesses[19]={},double_weaknesses[19]={},freeze_dry=0,max_level=0,spikes=0;
  uint32_t team=0;bool needs_lead=true;
  while(pool.group_count&&result.count<team_size){
    int base=gen_sample_no_replace(rng,pool.groups,pool.group_count);GenMoveList formes{};
    for(int i=0;i<pool.species_count;i++){int id=pool.species[i];if(rules.species[id].base==base){if(formes.count>=64){result.fault=1;return false;}formes.values[formes.count++]=id;}}
    if(!formes.count){result.fault=1;return false;}
    int id=formes.values[gen_random(rng,formes.count)];const GenSpecies& s=rules.species[id];
    if(bases[s.base])continue;
    if((team&GT_TERA_BLAST)&&(s.id_tag==gen_tag("ogerpon")||s.id_tag==gen_tag("ogerponhearthflame")||s.id_tag==gen_tag("terapagos")))continue;
    if(s.base_tag==gen_tag("zoroark")&&result.count>=team_size-1)continue;
    bool icy=s.effectiveness[5]>0||(s.effectiveness[5]>-2&&gen_has_type(s.types,2)),skip=false;
    for(int i=0;i<2;i++)if(s.types[i]>=0&&type_count[s.types[i]]>=2)skip=true;
    if(skip)continue;
    for(int i=0;i<19;i++)if((s.effectiveness[i]>0&&weaknesses[i]>=3)||(s.effectiveness[i]>1&&double_weaknesses[i]>=1))skip=true;
    if(skip)continue;
    if(s.effectiveness[1]==0&&s.fire_weak_abilities&&weaknesses[1]>=3)continue;
    if(icy&&freeze_dry>=4)continue;
    if(s.level==100&&max_level>=1)continue;
    int32_t selected_species[6];for(int i=0;i<result.count;i++)selected_species[i]=result.pokemon[i].species;
    if(!gen_compatible(rules,id,selected_species,result.count))continue;
    bool no_lead=s.id_tag==gen_tag("dugtrio")||s.id_tag==gen_tag("gothitelle")||s.id_tag==gen_tag("ironthorns")||s.id_tag==gen_tag("kingambit")||s.id_tag==gen_tag("zacian")||s.id_tag==gen_tag("zaciancrowned")||s.id_tag==gen_tag("zamazenta")||s.id_tag==gen_tag("zamazentacrowned");
    if(needs_lead&&no_lead&&result.count+1==team_size)continue;
    GenSet set{};bool lead=needs_lead&&!no_lead;if(!gen_random_set(rules,id,team,lead,rng,set)){result.fault=1;return false;}
    if(lead){for(int i=result.count;i>0;i--)result.pokemon[i]=result.pokemon[i-1];result.pokemon[0]=set;needs_lead=false;}
    else result.pokemon[result.count]=set;
    if(++result.count==team_size)break;
    bases[s.base]=1;for(int i=0;i<2;i++)if(s.types[i]>=0)type_count[s.types[i]]++;
    for(int i=0;i<19;i++){if(s.effectiveness[i]>0)weaknesses[i]++;if(s.effectiveness[i]>1)double_weaknesses[i]++;}
    if(s.effectiveness[1]==0&&(gen_is_ability(rules,set.ability,"dryskin")||gen_is_ability(rules,set.ability,"fluffy")))weaknesses[1]++;
    if(icy)freeze_dry++;if(set.level==100)max_level++;
    if(gen_is_ability(rules,set.ability,"drizzle")||gen_named_chosen(rules,set.moves,"raindance"))team|=GT_RAIN;
    if(gen_is_ability(rules,set.ability,"drought")||gen_is_ability(rules,set.ability,"orichalcumpulse")||gen_named_chosen(rules,set.moves,"sunnyday"))team|=GT_SUN;
    if(gen_is_ability(rules,set.ability,"sandstream"))team|=GT_SAND;
    if(gen_is_ability(rules,set.ability,"snowwarning")||gen_named_chosen(rules,set.moves,"snowscape")||gen_named_chosen(rules,set.moves,"chillyreception"))team|=GT_SNOW;
    if(gen_named_chosen(rules,set.moves,"healbell"))team|=GT_STATUS_CURE;
    if(gen_named_chosen(rules,set.moves,"spikes")||gen_named_chosen(rules,set.moves,"ceaselessedge")){if(++spikes>=2)team|=GT_TWO_SPIKES;}
    if(gen_named_chosen(rules,set.moves,"toxicspikes")||gen_is_ability(rules,set.ability,"toxicdebris"))team|=GT_TOXIC_SPIKES;
    if(gen_named_chosen(rules,set.moves,"stealthrock")||gen_named_chosen(rules,set.moves,"stoneaxe"))team|=GT_STEALTH_ROCK;
    if(gen_named_chosen(rules,set.moves,"stickyweb"))team|=GT_STICKY_WEB;
    if(gen_named_chosen(rules,set.moves,"defog"))team|=GT_DEFOG;
    if(gen_named_chosen(rules,set.moves,"rapidspin")||gen_named_chosen(rules,set.moves,"mortalspin"))team|=GT_RAPID_SPIN;
    if(gen_named_chosen(rules,set.moves,"auroraveil")||(gen_named_chosen(rules,set.moves,"reflect")&&gen_named_chosen(rules,set.moves,"lightscreen")))team|=GT_SCREENS;
    if(set.role==GR_TERA_BLAST||s.id_tag==gen_tag("ogerpon")||s.id_tag==gen_tag("ogerponhearthflame")||s.id_tag==gen_tag("terapagos"))team|=GT_TERA_BLAST;
  }
  result.fault=result.count!=team_size;return !result.fault;
}
}
