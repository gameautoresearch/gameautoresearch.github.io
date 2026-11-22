// SPDX-License-Identifier: MIT
#pragma once
#include "teamgen.hpp"
namespace pkn {
// Frozen generator sets have neutral nature. These are public Dex tables, not
// pre-generated teams or simulator states. Unknown battle effects stay flagged.
struct ResetSpecies {int32_t id,types[2],stats[6],fixed_hp,nfe,weight_hg,paradox_species,species_flags,battle_form,signature_move,form_family,form_count,forms[3];};
struct ResetMove {int32_t category,maxpp;uint32_t traits,unsupported;};
struct ResetRules {
  const ResetSpecies* formes;int32_t forme_count;
  const ResetMove* moves;int32_t move_count;
  const int32_t *abilities,*items;const uint8_t *ability_supported,*item_supported;
  int32_t ability_count,item_count,iron_head_id;
};
PK_HD inline bool materialize_gen_set(const ResetRules& rules,const GenSet& source,Mon& target){
  if(source.moves.fault||source.forme<0||source.forme>=rules.forme_count||source.ability<0||source.ability>=rules.ability_count||source.item<0||source.item>=rules.item_count||source.level<1||source.level>9999||source.tera<0||source.tera>18||source.moves.count<1||source.moves.count>4)return false;
  for(uint32_t i=0;i<sizeof(Mon);i++)reinterpret_cast<uint8_t*>(&target)[i]=0;
  const ResetSpecies& species=rules.formes[source.forme];
  target.gender=source.gender;target.species=species.id;target.type[0]=species.types[0];target.type[1]=species.types[1];target.base_types[0]=species.types[0];target.base_types[1]=species.types[1];
  target.ability=rules.abilities[source.ability];target.base_ability=target.ability;target.original_ability=target.ability;target.item=rules.items[source.item];target.level=source.level;target.tera_type=source.tera;
  target.base_speed=species.stats[5];target.iv_spe=source.ivs[5];target.ev_spe=source.evs[5];target.base_attack=species.stats[1];target.known_type=1;target.trapping_flags=species.species_flags>>16;target.species_flags=species.species_flags;target.nfe=species.nfe;target.weight_hg=species.weight_hg;target.paradox_species=species.paradox_species;
  target.unsupported=(!rules.ability_supported[source.ability]?1u:0u)|(!rules.item_supported[source.item]?2u:0u);
  for(int i=0;i<6;i++){
    if(source.ivs[i]<0||source.ivs[i]>31||source.evs[i]<0||source.evs[i]>255)return false;
    int value=(2*species.stats[i]+source.ivs[i]+source.evs[i]/4+(i?0:100))*source.level/100+(i?5:10);
    if(i)target.stats[i-1]=value;else target.hp=target.maxhp=species.fixed_hp?species.fixed_hp:value;
  }
  target.form_family=species.form_family;target.form_count=species.form_count;
  for(int form=0;form<species.form_count;form++){
    int index=species.forms[form];if(index<0||index>=rules.forme_count)return false;const ResetSpecies& base=rules.formes[index];MonForm& f=target.forms[form];
    f.species=base.id;f.type[0]=base.types[0];f.type[1]=base.types[1];f.base_speed=base.stats[5];f.base_attack=base.stats[1];f.weight_hg=base.weight_hg;
    for(int i=0;i<6;i++){int value=(2*base.stats[i]+source.ivs[i]+source.evs[i]/4+(i?0:100))*source.level/100+(i?5:10);if(i)f.stats[i-1]=value;else f.maxhp=base.fixed_hp?base.fixed_hp:value;}
  }
  target.newly_switched=1;target.speed_cache=target.stats[4];target.choice_lock=-1;target.sleep_source=-1;
  target.last_move=-1;target.encore_move=-1;target.disable_move=-1;target.yawn_source=-1;target.locked_move=-1;
  for(int i=0;i<4;i++){
    target.move_ids[i]=-1;
    if(i>=source.moves.count)continue;
    int id=source.moves.values[i];if(id<0||id>=rules.move_count)return false;
    const ResetMove& move=rules.moves[id];target.move_ids[i]=id;target.move_categories[i]=move.category;
    target.pp[i]=target.maxpp[i]=move.maxpp;target.move_traits[i]=move.traits;
    target.unsupported|=move.unsupported?4u:0u;
    // Unknown effects preserve their actual IDs and explicitly close execution.
  }
  return true;
}
// Slots at and beyond team_size are left all-zero: an empty slot (maxhp 0) that every loop skips.
PK_HD inline bool materialize_gen_team(const ResetRules& rules,const GenTeam& source,Mon* pokemon,int team_size=6){
  if(team_size<1||team_size>6||source.fault||source.count!=team_size)return false;
  for(int i=0;i<team_size;i++)if(!materialize_gen_set(rules,source.pokemon[i],pokemon[i]))return false;
  for(int i=team_size;i<6;i++)for(uint32_t k=0;k<sizeof(Mon);k++)reinterpret_cast<uint8_t*>(&pokemon[i])[k]=0;
  return true;
}
// BattleStart changes occur after construction and before either side enters.
PK_HD inline bool prepare_gen_start(const ResetRules& rules,const GenSet& source,Mon& target){
  const ResetSpecies& species=rules.formes[source.forme];
  bool crowned=((species.species_flags&64)&&target.item==RUSTED_SWORD)||((species.species_flags&128)&&target.item==RUSTED_SHIELD);
  if(!crowned||species.battle_form<0)return true;
  GenSet changed=source;changed.forme=species.battle_form;
  for(int i=0;i<changed.moves.count;i++)if(changed.moves.values[i]==rules.iron_head_id){
    if(species.signature_move<0)return false;
    changed.moves.values[i]=species.signature_move;
  }
  if(!materialize_gen_set(rules,changed,target))return false;
  target.ability=(species.species_flags&64)?INTREPID_SWORD:DAUNTLESS_SHIELD;target.base_ability=target.ability;target.original_ability=target.ability;
  return true;
}

}
