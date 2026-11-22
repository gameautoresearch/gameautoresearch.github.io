// SPDX-License-Identifier: Apache-2.0
// Ported from crforge env/src/main/java/org/crforge/env/LadderEnv.java:
// header(), rememberTowers(), writeWorld(), writeOther(), readPlays();
// env/src/main/java/org/crforge/env/SeatView.java and
// core/src/main/java/org/crforge/core/battle/unit/{CharacterEntity,TowerEntity}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// Extracted POD fields are outputs only; no policy observation feeds engine state.
#pragma once
#include "../world.hpp"
namespace clash_native {
CN_HD inline int observation_packed_level(int packed){return add32(add32(sar32(packed,8),packed_steps(packed)),1);}
CN_HD inline void NativeWorld::extract_observation(){
    observationCount=0;
    // EntityHolder's live list is already sorted by kind-band ID. The observer
    // writes character/tower rows first, then other kinds newest first itself.
    for(int at=0;at<holder.live_count;++at){
        int i=holder.live[at];if(observationCount>=WORLD_CAPACITY){fault(321);break;}
        ObservationEntity out{};out.id=holder.ids[i];
        if(holder.kinds[i]==5){
            const PfEntity&e=entities[i];const NativeUnitConfig&u=spec->units[rows[i]];const PfTargeting&t=targets[i];const CombatHealth&hp=health[i];
            bool tower=u.king||u.princess;
            out.kind=tower?obs_schema::KIND_TOWER:obs_schema::KIND_CHARACTER;
            out.side=e.side;out.row=u.rowVocab;out.x=e.x;out.y=e.y;
            out.hp=hp.hp;out.max_hp=hp.maximum;out.shield=hp.shield;out.has_hp=e.hpPresent;
            out.state=e.state;out.level=observation_packed_level(u.packedLevel);
            out.alive=e.alive;out.air=e.air;out.building=u.building;out.king=u.king;
            out.no_move=(e.flags&e.flagBits.noMove)!=0;out.no_attack=(e.flags&e.flagBits.noAttack)!=0;
            // No fixed8 card charges, cloaks, tunnels or clones. Waiting Archer
            // units are hidden to combat and carry F_HIDDEN on their owner's row.
            out.charged=false;out.clone=false;out.invisible=false;out.underground=false;
            out.waiting=!tower&&e.state==WaitingToDeploy;
            out.hidden=!tower&&(out.waiting||(e.flags&spec->tags.hidden)!=0);
            out.awake=tower&&(e.flags&spec->tags.keepsTargetingOff)==0;
            out.deploy_ms=tower?0:e.deployCountdown;
            out.attack_ms=t.attackTimerMs;out.load_ms=t.loadTimerMs;
            out.speed_percent=speed_percent(i,false);out.hit_speed_percent=speed_percent(i,true);
            out.target_id=-1;
            if(t.reference>=0&&t.reference<WORLD_CAPACITY&&holder.allocated[t.reference]&&is_unit(t.reference))out.target_id=holder.ids[t.reference];
            out.bar_always=true;out.bar_damaged=hp.hp!=hp.maximum;out.bar_rolling=false;out.ground_trail=false;
            out.ability_ran_tick=-1;out.buff_count=buffs[i].count;
            if(out.buff_count>8){fault(322);out.buff_count=8;}
            for(int b=0;b<out.buff_count;++b){const CombatBuff&buff=buffs[i].items[b];if(buff.id!=spec->stun.id){fault(323);continue;}
                out.buffs[b]={buff.id,buff.remaining,spec->stun.speed_percent,spec->stun.hit_speed_percent};}
        }else if(holder.kinds[i]==4){
            const ProjectileState&p=projectiles[i];int row=projectileRows[i];
            if(row<0||row>=6){fault(324);continue;}
            out.kind=obs_schema::KIND_PROJECTILE;out.side=p.side;out.row=spec->projectileVocab[row];out.x=p.x;out.y=p.y;
            out.level=observation_packed_level(spec->projectiles[row].packed_level);
            out.aim_x=p.aim_x;out.aim_y=p.aim_y;out.target_id=p.target_id>0?p.target_id:-1;out.deploy_index=p.deploy_index;
            out.alive=true;out.has_hp=false;
        }else if(holder.kinds[i]==3){
            const CombatAreaEffect&a=areas[i];out.kind=obs_schema::KIND_AREA;out.side=a.side;out.row=spec->zapVocab;out.x=a.x;out.y=a.y;
            out.level=observation_packed_level(spec->zap.packed_level);out.time_left_ms=a.countdown;out.radius=combat_area_radius(a,spec->zap);
            out.target_id=-1;out.alive=true;out.has_hp=false;out.clone=false;
        }else{fault(325);continue;}
        observationEntities[observationCount++]=out;
    }
    for(int seat=0;seat<2;++seat){
        ObservationOwn own{};const MatchSide&s=match.sides[seat];const Hand&hand=s.hand;
        own.elixir=s.elixir;own.reserved=mul32(commands.promised(seat,match.tick),10000);own.crowns=match.crowns(*this,seat);own.hand_refill_ms=hand.cooldown_ms;
        for(int k=0;k<4;++k){int card=hand.slots[k];if(card>=0&&card<8){own.hand_card[k]=spec->cardVocab[card];own.hand_cost[k]=spec->cards[card].cost;own.hand_pending[k]=commands.pending(seat,card,match.tick)?1:0;}else own.hand_card[k]=-1;
            own.hand_form[k]=0;own.queue[k]=k<hand.count&&hand.queue[k]>=0&&hand.queue[k]<8?spec->cardVocab[hand.queue[k]]:-1;}
        own.next_card=hand.count>0&&hand.queue[0]>=0&&hand.queue[0]<8?spec->cardVocab[hand.queue[0]]:-1;
        for(int k=0;k<8;++k){own.deck[k]=spec->cardVocab[k];own.deck_slots[k]=0;own.deck_evolution[k]=0;own.deck_level[k]=11;}
        // Towers are remembered at reset in original order: each side's king,
        // low-x princess, high-x princess. Their original IDs remain known after
        // removal; looking them up by ID avoids reading a recycled payload slot.
        for(int owner=0;owner<2;++owner){int base=owner*3,at=owner==seat?0:3;
            int a=base+1,b=base+2;bool aLeft=observation_x(seat,spec->towerX[a])<=observation_x(seat,spec->towerX[b]);
            int original[3]={base,aLeft?a:b,aLeft?b:a};
            for(int k=0;k<3;++k){int tower=original[k],slot=holder.find_live(5000000+tower);
                own.tower_hp[at+k]=slot>=0&&entities[slot].alive?health[slot].hp:0;
                own.tower_max_hp[at+k]=spec->units[spec->towerRow[tower]].maximumHp;}
        }
        own.last_refusal=lastRefusal[seat];own.last_run_refusal=lastRunRefusal[seat];own.last_ability_run_refusal=0;own.illegal_total=illegalTotal[seat];
        // Waiting rows end when a command runs, even if its refused promise lasts.
        // Their order is the original command-giving order, never hand-slot order.
        own.pending_count=0;
        for(int k=0;k<commands.count&&own.pending_count<4;++k){const QueuedPlayerCommand&c=commands.queued[k];if(c.side!=seat||c.ran)continue;
            if(c.card<0||c.card>=8){fault(326);continue;}own.pending[own.pending_count++]={spec->cardVocab[c.card],c.x,c.y,c.runTick};}
        observationOwn[seat]=own;
    }
    ObservationClock clock{};clock.tick=match.tick;clock.episode_step=episodeStep;clock.time_ms=match.timeline.timeMs;
    clock.section=match.timeline.section;clock.section_type=spec->rules.sectionTypes[match.timeline.section];clock.section_left_ms=match.timeline.section_left_ms(spec->rules);clock.full_bar_ms=spec->rules.fullBarMs[match.timeline.rate];
    clock.start_cooldown_ticks=spec->rules.startCooldownTicks;clock.decision_ticks=DECISION_TICKS;clock.play_delay_ticks=20;
    clock.winner=match.winner;clock.first_side=firstSide;clock.crowns[0]=match.crowns(*this,0);clock.crowns[1]=match.crowns(*this,1);
    clock.illegal_now[0]=illegalNow[0];clock.illegal_now[1]=illegalNow[1];clock.fault=faultCode;clock.not_modelled=0;
    clock.over=match.is_over(*this);clock.overtime=match.timeline.overtime(spec->rules);clock.truncated=match.truncated;clock.reset=false; // LadderEnv.observe writes I_RESET=0, including an explicit reset.
    observationClock=clock;
}
} // namespace clash_native
