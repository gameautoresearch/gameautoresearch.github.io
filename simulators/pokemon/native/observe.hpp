// SPDX-License-Identifier: MIT
#pragma once
#include "sim.hpp"
namespace pkn {
struct SpeciesInfo {int32_t token,types[2],type_tokens[2],stats[5];};
struct MoveInfo {int32_t token,type_token,category_token,type,category,power,accuracy,priority;};
struct PublicRules {
  const SpeciesInfo* species;int32_t species_count;
  const MoveInfo* moves;int32_t move_count,struggle_index,recharge_index;
  const int32_t *ability_tokens,*item_tokens,*type_tokens,*status_tokens;
  const int8_t* chart;
};
PK_HD inline int visible_token(const char* value){uint32_t hash=2166136261u;for(int i=0;value[i];i++)hash=(hash^uint8_t(value[i]))*16777619u;return 1+hash%16383u;}
PK_HD inline const MoveInfo* visible_move(int id,const PublicRules& rules){
  if(id==-2)id=rules.struggle_index;
  if(id==-4)id=rules.recharge_index;
  return id>=0&&id<rules.move_count?rules.moves+id:nullptr;
}
PK_HD inline int potential_dead(const State& s,int side){int dead=0;for(int k=0;k<6;k++)dead+=s.mons[side][k].maxhp>0&&s.requests[side].hp[k]==0;return dead;}
PK_HD inline int public_dead(const State& s,int side){int dead=0;for(int k=0;k<s.public_count[side];k++)dead+=s.public_mons[side][s.public_order[side][k]].fainted;return dead;}
template<class Scalar> PK_HD inline void observe(const State& s,int player,Scalar* out,const PublicRules& rules){
  for(int i=0;i<700;i++)out[i]=0;
  const int foe=1-player;const RequestCache& request=s.requests[player];
  for(int row=0;row<12;row++){
    bool own=row<6;int side=own?player:foe,index=own?row:row-6;
    if(!own&&index>=s.public_count[side])continue;
    int slot=own?request.order[index]:s.public_order[side][index];
    const Mon& p=s.mons[side][slot];const PublicMon& v=s.public_mons[side][slot];
    if(own&&p.maxhp==0)continue;
    int species=own?request.species[slot]:v.details_known?v.species:0;
    if(species<0||species>=rules.species_count)continue;
    const SpeciesInfo& d=rules.species[species];
    bool tera=own?bool(request.tera[slot]):bool(v.tera);
    int ability=own?rules.ability_tokens[request.abilities[slot]]:v.ability?rules.ability_tokens[v.ability]:0;
    int item=own?rules.item_tokens[request.unannounced_item_loss[slot]?request.unannounced_item_loss[slot]:p.item]:v.item?rules.item_tokens[v.item]:0;
    int status=own?request.status[slot]:v.status;
    Scalar* tokens=out+row*12;Scalar* numbers=out+144+row*24;
    tokens[0]=d.token;tokens[1]=ability;tokens[2]=item;
    tokens[3]=tera?rules.type_tokens[own?p.tera_type:v.tera_type]:d.type_tokens[0];tokens[4]=tera?0:d.type_tokens[1];
    tokens[5]=own||tera?rules.type_tokens[own?p.tera_type:v.tera_type]:0;tokens[6]=rules.status_tokens[status];
    int first_empty=4;for(int k=0;k<4;k++)if(request.mon_moves[slot][k]<0){first_empty=k;break;}
    for(int k=0;k<4;k++){
      int id=own?(k==first_empty?request.appended_move[slot]:request.mon_moves[slot][k]):(k<v.move_count?v.moves[k]:-1);
      const MoveInfo* m=visible_move(id,rules);tokens[7+k]=m?m->token:0;
    }
    const MoveInfo* last=visible_move(v.last_move,rules);tokens[11]=last?last->token:0;
    numbers[0]=1;numbers[1]=slot==s.public_active[side];
    numbers[2]=own?request.hp_ratio[slot]:double(v.hp)/100;
    numbers[3]=own?1:v.known_hp;numbers[4]=own?request.hp[slot]==0:v.fainted;
    numbers[5]=double(own?request.levels[slot]:v.details_known?v.level:100)/100;
    for(int k=0;k<5;k++)numbers[6+k]=double(own?request.stats[slot][k]:d.stats[k])/300;
    for(int k=0;k<7;k++)numbers[11+k]=double(v.boosts[k])/6;
    numbers[18]=tera;numbers[19]=v.substitute;numbers[20]=v.confusion;numbers[22]=v.taunt;numbers[23]=own;
  }
  Scalar* g=out+432;
  g[0]=double(minimum(s.turn,1000))/1000;g[1]=request.forced;
  g[2]=request.active?request.trapped:0;g[3]=request.active?request.maybe_trapped:0;g[4]=request.can_tera;g[5]=s.tera_used[foe];g[6]=double(s.public_count[foe])/6;
  g[7]=double(public_dead(s,foe)-potential_dead(s,player))/6;
  for(int side=0;side<2;side++)for(int j=0;j<10;j++)g[8+side*10+j]=double(s.visible_side_conditions[side?foe:player][j])/3;
  g[32]=request.waiting;
  g[33]=request.tera[request.actor];
  if(s.weather){const char* names[6]={"","weather:raindance","weather:sunnyday","weather:sandstorm","weather:snowscape","weather:hail"};out[472]=visible_token(names[s.weather]);}
  if(s.visible_terrain){const char* names[5]={"","terrain:moveelectricterrain","terrain:movegrassyterrain","terrain:movemistyterrain","terrain:movepsychicterrain"};out[473]=visible_token(names[s.visible_terrain]);}
  const PublicMon& target=s.public_mons[foe][s.public_active[foe]];
  const Mon& true_target=active(s,foe);
  const SpeciesInfo& target_species=rules.species[target.details_known?target.species:0];
  const SpeciesInfo& actor=rules.species[request.species[request.actor]];
  for(int k=0;k<4;k++){
    const MoveInfo* m=visible_move(request.moves[k],rules);if(!m||!request.active)continue;
    int effectiveness=0;bool immune=false;
    for(int t=0;t<2;t++){
      int type=target.tera?(t?-1:target.tera_type):target_species.types[t];if(type<0||type>=18||m->type<0||m->type>=18)continue;
      int exponent=rules.chart[m->type*18+type];if(exponent==-127)immune=true;else effectiveness+=exponent;
    }
    double factor=immune?0:effectiveness>=0?double(1<<effectiveness):1.0/double(1<<(-effectiveness));
    for(int tera=0;tera<2;tera++){
      int action=k+tera*4;Scalar* tokens=out+476+action*3;Scalar* numbers=out+518+action*12;
      tokens[0]=m->token;tokens[1]=m->type_token;tokens[2]=m->category_token;
      numbers[0]=double(m->power)/150;numbers[1]=double(m->accuracy)/100;numbers[2]=double(m->priority)/5;
      numbers[3]=double(request.pp[k])/64;numbers[4]=double(request.maxpp[k])/64;
      numbers[5]=m->category==0;numbers[6]=m->category==1;numbers[7]=m->category==2;
      numbers[8]=factor/4;numbers[9]=m->type>=0&&(actor.types[0]==m->type||actor.types[1]==m->type);numbers[10]=tera;
      out[686+action]=!request.waiting&&!request.disabled[k]&&(!tera||request.can_tera);
    }
  }
  for(int i=0;i<6;i++){
    int slot=request.order[i];const Mon& p=s.mons[player][slot];if(p.maxhp==0)continue;
    const SpeciesInfo& d=rules.species[request.species[slot]];Scalar* tokens=out+476+(8+i)*3;Scalar* numbers=out+518+(8+i)*12;
    tokens[0]=d.token;tokens[1]=d.type_tokens[0];tokens[2]=d.type_tokens[1];
    numbers[0]=request.hp_ratio[slot];numbers[1]=request.tera[slot];numbers[11]=1;
    out[686+8+i]=!request.waiting&&slot!=request.actor&&(s.reviving[player]?p.hp==0:p.hp>0)&&(request.forced||!request.trapped);
  }
}
}
