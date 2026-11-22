// SPDX-License-Identifier: Apache-2.0; provenance is in rules.hpp includes.
#include "rules.hpp"
using namespace clash_native;
static RulesConfig config(const int* p){RulesConfig c{};int k=0;c.startingElixir=p[k++];c.maxMana=p[k++];c.startCooldownTicks=p[k++];c.endDelayMs=p[k++];c.sectionCount=p[k++];c.rateCount=p[k++];c.cooldownCount=p[k++];c.eventCount=p[k++];for(int i=0;i<8;++i)c.sectionLengths[i]=p[k++];for(int i=0;i<8;++i)c.sectionTypes[i]=p[k++];for(int i=0;i<8;++i)c.rateLengths[i]=p[k++];for(int i=0;i<8;++i)c.fullBarMs[i]=p[k++];for(int i=0;i<8;++i)c.cooldownLengths[i]=p[k++];for(int i=0;i<8;++i)c.cooldownMs[i]=p[k++];for(int i=0;i<16;++i)c.eventTimes[i]=p[k++];return c;}
static CardRule card(const int*p){CardRule c{};int k=0;c.cost=p[k++];c.productionStopMs=p[k++];c.unitType=p[k++];c.count=p[k++];c.summonRadius=p[k++];c.summonWidth=p[k++];c.summonDelayMs=p[k++];c.deployTimeMs=p[k++];c.collisionRadius=p[k++];c.spawnRadius=p[k++];c.spawnAngleShift=p[k++];c.flyingHeight=p[k++];c.spawnPathfindSpeed=p[k++];c.tileSizeOverride=p[k++];c.margin=p[k++];c.startY=p[k++];c.endY=p[k++];c.enemyOk=p[k++];c.onBuildings=p[k++];c.onWater=p[k++];c.fullLane=p[k++];c.building=p[k++];c.spellAsDeploy=p[k++];return c;}
static MaskEntity entity(const int*p){return {bool(p[0]),p[1],p[2],p[3],bool(p[4]),bool(p[5]),bool(p[6]),p[7],p[8],p[9]};}
extern "C" {
void rules_timeline(const int*p,const int*ticks,const int*equal,int n,int freezeAt,int*out){auto c=config(p);Timeline t{};for(int i=0;i<n;++i){t.crownsEqual=equal[i];t.advance(c,ticks[i]);if(i==freezeAt)t.freeze();int row[]={t.timeMs,t.section,t.rate,c.fullBarMs[t.rate],c.cooldownMs[t.cooldown],t.section_left_ms(c),t.overtime(c),t.time_up(c),t.firstAdvance,t.frozen};for(int j=0;j<10;++j)out[10*i+j]=row[j];}}
void rules_placement(const int*cp,const int*cells,int width,int height,const int*ep,int n,int side,int x,int y,int symmetry,int laneSequence,int*out,uint8_t*mask,uint8_t*landing){auto c=card(cp);ArenaView a{width,height,cells};MaskEntity es[32];for(int i=0;i<n;++i)es[i]=entity(ep+10*i);deploy_mask(a,c,side,es,n,mask);landing_tiles(a,c,mask,landing);auto p=place_card(a,c,x,y,side,es,n,symmetry,laneSequence);int top[]={p.code,p.x,p.y,p.hasInterval,p.intervalLo,p.intervalHi,p.originLane,p.count};for(int i=0;i<8;++i)out[i]=top[i];for(int i=0;i<p.count;++i){auto&u=p.units[i];int row[]={u.index,u.dx,u.dy,u.x,u.y,u.lane,u.state,u.waitMs};for(int j=0;j<8;++j)out[8+i*8+j]=row[j];}}
void rules_side(const int*cp,const int*order,const int*plays,int n,int*out){auto c=config(cp);Timeline t{};MatchSide s{};s.elixir=c.startingElixir*10000;s.hand.deal(order);for(int i=0;i<n;++i){t.advance(c,i);if(plays[3*i]>=0)s.play(plays[3*i],plays[3*i+1],plays[3*i+2]);int filled=s.visit(c,t);int*row=out+18*i;row[0]=filled;row[1]=s.elixir;row[2]=s.spent;row[3]=s.wasted;row[4]=s.productionStopMs;row[5]=s.deployCounter;row[6]=s.lastPlayed;row[7]=s.hand.cooldown_ms;row[8]=s.hand.count;for(int j=0;j<4;++j)row[9+j]=s.hand.slots[j];for(int j=0;j<s.hand.count;++j)row[13+j]=s.hand.queue[j];}}
}
// Minimal tower-only host for differential matching of LadderMatch's own logic.
// It deliberately has no troop combat: combat is qualified by the whole-world suite.
struct RuleTestWorld {
    LadderMatch* match=nullptr;int hp[6]={},listed[6]={1,1,1,1,1,1},x[6]={},y[6]={};
    bool hitsHeld=false,matchEnded=false;bool circlePending[6]={};
    bool king_alive(int side)const{return hp[3*side]>0;}
    int princess_count(int side)const{return listed[3*side+1]+listed[3*side+2];}
    int lowest_tower_hp(int side)const{int low=hp[3*side];for(int i=1;i<3;++i)if(listed[3*side+i]&&hp[3*side+i]<low)low=hp[3*side+i];return low;}
    void end_match(){matchEnded=true;}
    void hold_hits(bool h){hitsHeld=h;}
    int clear_field(){return 0;}
    int arena_length()const{return 32000;}
    void circle(int side,int radius){for(int i=3*side;i<3*side+3;++i)if(hp[i]>0&&squared_distance(x[i],y[i],x[3*side],y[3*side])<=radius*radius)circlePending[i]=true;}
    void tick_entities(int){for(int i=0;i<6;++i)if(circlePending[i]){hp[i]=0;circlePending[i]=false;}for(int s=0;s<2;++s)match->sides[s].visit(match->config,match->timeline);cleanup();}
    void cleanup(){for(int s=0;s<2;++s)for(int i=1;i<3;++i)if(hp[3*s+i]<=0)listed[3*s+i]=0;}
    void tiebreak_step_end(){cleanup();}
    void drain_towers(int side,int step){for(int i=0;i<3;++i)if(listed[3*side+i]){int&v=hp[3*side+i];v=v>step?v-step:0;}}
    void execute_due_commands(int){}
};
extern "C" void rules_match(const int*cp,const int*initial,const int*events,int eventCount,int steps,int*out){
    LadderMatch m{};m.config=config(cp);RuleTestWorld w{};w.match=&m;
    for(int i=0;i<6;++i){w.hp[i]=initial[3*i];w.x[i]=initial[3*i+1];w.y[i]=initial[3*i+2];}
    for(int s=0;s<2;++s)m.sides[s].elixir=m.config.startingElixir*10000;
    for(int t=0;t<steps;++t){for(int e=0;e<eventCount;++e)if(events[3*e]==t)w.hp[events[3*e+1]]=events[3*e+2];m.step(w);
        int row[]={m.tick,m.clockMs,m.timeline.timeMs,m.timeline.section,m.timeline.rate,m.timeline.section_left_ms(m.config),m.timeline.frozen,m.endTimerMs,m.winner,m.tiebreakMs,m.ended,m.lastTicked,m.is_over(w),m.crowns(w,0),m.crowns(w,1),w.hitsHeld,w.matchEnded,w.hp[0],w.hp[1],w.hp[2],w.hp[3],w.hp[4],w.hp[5]};
        for(int i=0;i<23;++i)out[t*23+i]=row[i];
    }
}
struct CommandTestWorld {
    bool alive[2]={true,true};int princess[2]={2,2};
    bool king_alive(int s)const{return alive[s];}int princess_count(int s)const{return princess[s];}
};
extern "C" void rules_giving(const int*cp,const int*cardsPacked,const int*cells,const int*ep,int entityCount,const int*snapshot,const int*queue,int queueCount,int side,int slot,int x,int y,int*out){
    LadderMatch m{};m.config=config(cp);m.tick=snapshot[0];m.endTimerMs=snapshot[1];m.timeline.section=snapshot[2];m.timeline.sectionElapsedMs=snapshot[3];m.timeline.crownsEqual=snapshot[4];
    CommandTestWorld w{};for(int s=0;s<2;++s){w.alive[s]=snapshot[5+s];w.princess[s]=snapshot[7+s];m.sides[s].elixir=snapshot[9+s];for(int k=0;k<4;++k)m.sides[s].hand.slots[k]=snapshot[11+4*s+k];}
    CardRule cards[8];for(int i=0;i<8;++i)cards[i]=card(cardsPacked+23*i);PlayerCommands commands{};commands.count=queueCount;
    for(int i=0;i<queueCount;++i){const int*q=queue+11*i;commands.queued[i]={q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],bool(q[9]),bool(q[10])};}
    MaskEntity es[32];for(int i=0;i<entityCount;++i)es[i]=entity(ep+10*i);
    Giving g=commands.give(m,w,cards,side,slot,x,y,ArenaView{36,64,cells},es,entityCount);
    int row[]={int(g.refusal),g.code,g.givenTick,g.runTick,g.cost,g.x,g.y,commands.promised(0,m.tick),commands.promised(1,m.tick)};for(int i=0;i<9;++i)out[i]=row[i];
}
struct ExecTestWorld:RuleTestWorld {
    PlayerCommands commands{};CardRule cards[8]{};MaskEntity entities[6]{};ArenaView arena{};int deployments=0;
    void deploy_card(int,int,int,const Placement&){++deployments;}
    void execute_due_commands(int){commands.execute_due(*match,*this,cards,arena,entities,6);}
};
extern "C" void rules_command_run(const int*cp,const int*cardp,const int*cells,const int*ep,int seed,const int*requests,int requestCount,int steps,int*giveout,int*stateout){
    LadderMatch m{};ExecTestWorld w{};w.match=&m;w.arena={36,64,cells};BattleRandom random{seed};random.next(1);random.next(1);int words[2]={0,0};m.init(config(cp),random,words);
    for(int i=0;i<8;++i)w.cards[i]=card(cardp+23*i);
    for(int i=0;i<6;++i){w.entities[i]=entity(ep+10*i);w.hp[i]=w.entities[i].summoner?4824:3052;w.x[i]=w.entities[i].x;w.y[i]=w.entities[i].y;}
    for(int t=0;t<steps;++t){for(int j=0;j<requestCount;++j){const int*r=requests+5*j;if(r[0]!=t)continue;auto g=w.commands.give(m,w,w.cards,r[1],r[2],r[3],r[4],w.arena,w.entities,6);int row[]={int(g.refusal),g.code,g.givenTick,g.runTick,g.cost,g.x,g.y};for(int k=0;k<7;++k)giveout[7*j+k]=row[k];}
        m.step(w);int*row=stateout+26*t;row[0]=m.tick;row[1]=w.commands.promised(0,m.tick);row[2]=w.commands.promised(1,m.tick);row[3]=w.deployments;
        for(int s=0;s<2;++s){auto&side=m.sides[s];int*o=row+4+11*s;o[0]=side.elixir;o[1]=side.spent;o[2]=side.deployCounter;o[3]=side.hand.cooldown_ms;o[4]=side.hand.count;for(int k=0;k<4;++k)o[5+k]=side.hand.slots[k];o[9]=side.hand.queue[0];o[10]=side.hand.queue[1];}
    }
}
extern "C" void rules_legal(const int*cp,const int*cardsPacked,const int*cells,const int*ep,int entityCount,const int*snapshot,const int*queue,int queueCount,int side,uint8_t*out){
    LadderMatch m{};m.config=config(cp);m.tick=snapshot[0];m.endTimerMs=snapshot[1];m.timeline.section=snapshot[2];m.timeline.sectionElapsedMs=snapshot[3];m.timeline.crownsEqual=snapshot[4];
    CommandTestWorld w{};for(int s=0;s<2;++s){w.alive[s]=snapshot[5+s];w.princess[s]=snapshot[7+s];m.sides[s].elixir=snapshot[9+s];for(int k=0;k<4;++k)m.sides[s].hand.slots[k]=snapshot[11+4*s+k];}
    CardRule cards[8];for(int i=0;i<8;++i)cards[i]=card(cardsPacked+23*i);PlayerCommands commands{};commands.count=queueCount;
    for(int i=0;i<queueCount;++i){const int*q=queue+11*i;commands.queued[i]={q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],bool(q[9]),bool(q[10])};}
    MaskEntity es[32];for(int i=0;i<entityCount;++i)es[i]=entity(ep+10*i);
    legal_actions(m,w,commands,cards,ArenaView{36,64,cells},es,entityCount,side,false,out);
}
