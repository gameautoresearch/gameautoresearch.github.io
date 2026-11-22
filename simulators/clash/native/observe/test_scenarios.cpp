// SPDX-License-Identifier: Apache-2.0
// Source scenarios for the observer ports attributed in observe.hpp.
#include "observe.hpp"
#include <cassert>
#include <cstdio>
using namespace clash_native;
struct W {
 ObservationEntity e[220];int n=0,error=0,own_reads[2]={};ObservationClock c;ObservationOwn players[2];
 int entity_count(){return n;}const ObservationEntity& entity_at(int i){return e[i];}
 const ObservationEntity* find_entity(int id){for(int i=0;i<n;++i)if(e[i].id==id)return &e[i];return nullptr;}
 const ObservationOwn& own(int seat){++own_reads[seat];return players[seat];}const ObservationClock& clock(){return c;}void fault(int code){error=code;}
};
int main(){
 static ObserverState<> s;s.reset(60000);W w;w.n=3;w.e[0].id=100;w.e[0].kind=1;w.e[0].side=1;w.e[0].waiting=true;w.e[0].x=12000;w.e[0].y=15000;w.e[0].row=81;
 w.e[1].id=120;w.e[1].kind=1;w.e[1].side=0;w.e[1].target_id=100;w.e[1].x=3000;w.e[1].y=4000;
 w.e[2].id=200002;w.e[2].kind=3;w.e[2].side=1;w.e[2].target_id=100;w.e[2].x=10000;w.e[2].y=18000;
 int o[7024];++s.serial;observe_seat(s,w,0,o);assert(w.own_reads[0]==1&&w.own_reads[1]==0);assert(o[48]==2);int* row=o+224;assert(row[0]==0&&row[4]==15000&&row[5]==4000&&row[14]==-1);assert((row+34)[0]==1&&(row+34)[14]==-1);assert(s.given[0]==2);assert(s.battle_id(0,0)==120);
 // Waiting enemy first becomes visible now; it receives the next first-sight identity,
 // independent of its smaller raw battle ID. No movement is fabricated across first sight.
 w.e[0].waiting=false;w.e[1].x+=100;++s.serial;observe_seat(s,w,0,o);row=o+224;assert(row[0]==2&&row[12]==0);assert((row+34)[0]==0&&(row+34)[12]==-100);assert((row+34)[14]==2);
 // Enemy trail discloses position and state, while every private card/health/timer field is unknown.
 w.e[0].underground=true;w.e[0].ground_trail=true;w.e[0].hp=1000;w.e[0].level=11;w.e[0].attack_ms=500;++s.serial;observe_seat(s,w,0,o);row=o+224;assert(row[3]==-1&&row[6]==-1&&row[7]==-1&&row[8]==-1&&row[11]==-1&&row[18]==0&&row[14]==-1);assert(row[10]==(1|2|8));
 // Launched enemy spell exposes its card and public cost at run, but neither aim nor placed point
 // until the first projectile releases; remaining arrows then expose all their aims together.
 PublicPlay play;play.side=1;play.card=9;play.cast=9;play.kind=3;play.tick=50;play.x=1500;play.y=26000;play.cost=3;play.projectile_count=30;play.deploy_index=0;assert(s.on_play(play));w.e[2].deploy_index=0;w.e[2].aim_x=1500;w.e[2].aim_y=26000;w.c.tick=55;++s.serial;observe_seat(s,w,0,o);assert(o[115]==30000&&o[28]==9&&o[120]==-1&&o[121]==-1&&o[123]==1);row=o+224+2*34;assert(row[29]==-1&&row[30]==-1);
 s.projectile_released(1,0,54);++s.serial;observe_seat(s,w,0,o);assert(o[120]==16500&&o[121]==26000&&o[124]==1&&o[123]==3);row=o+224+2*34;assert(row[29]==16500&&row[30]==26000);
 // Public-elixir regeneration never reads private elixir, and obeys stop/clamp/suppressed ticks.
 s.public_elixir[1].amount=99999;s.tick(28000,10);assert(s.public_elixir[1].amount==100000);s.public_elixir[1].play(1,100);s.tick(28000,10);assert(s.public_elixir[1].amount==90000&&s.public_elixir[1].stop_ms==50);s.tick(0,10);assert(s.public_elixir[1].stop_ms==50);s.tick(28000,10);assert(s.public_elixir[1].stop_ms==0);s.tick(28000,10);assert(s.public_elixir[1].amount==90178);
 // Entity overflow prioritizes all world entities and newest non-worlds, and counts every omitted
 // visible entity. Hidden waiting characters never consume a seat row or cause an ID gap.
 s.reset(60000);w.n=205;for(int i=0;i<w.n;++i){w.e[i]=ObservationEntity{};w.e[i].id=i;w.e[i].kind=i<3?2:3;w.e[i].row=i;w.e[i].side=0;}++s.serial;observe_seat(s,w,0,o);assert(o[48]==200&&o[49]==5);assert(o[224+3*34+3]==204);assert(o[224+199*34+3]==8);
 // Fixed8 empty champion controllers expose -1 charges, champion and entity (not a nonexistent 0).
 assert(o[101]==-1&&o[102]==-1&&o[105]==-1&&o[107]==-1&&o[108]==-1&&o[111]==-1);
 w.c.over=true;w.c.winner=1;w.c.tick=6000;int both[14048],info[12];float rewards[2];observe(s,w,both,info,rewards);assert(both[1]==2&&both[7025]==1&&info[2]==1&&rewards[0]==-1&&rewards[1]==1);
 std::puts("observer source scenarios passed: first-sight IDs, public-only own queries, waiting/trail visibility, aim secrecy, public elixir, overflow, terminal");
}
