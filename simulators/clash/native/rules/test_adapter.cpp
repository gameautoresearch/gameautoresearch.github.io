// SPDX-License-Identifier: Apache-2.0; provenance in world_state_adapter.hpp.
#include "world_state_adapter.hpp"
#include <cassert>
using namespace clash_native;
struct World {
    PfEntity e{};PfMovement m{};PfTargeting t{};StateTimers timers{};StateVisitConfig c{1000,250,false,false};bool active=true;int scaled=50,routes=0,clears=0,faultCode=0;
    PfEntity& entity(int){return e;}PfMovement* movement(int){return e.movementComponent?&m:nullptr;}PfTargeting& targeting(int){return t;}
    const StateVisitConfig& state_config(int){return c;}StateTimers& state_timers(int){return timers;}
    bool targeting_active(int){return active;}void set_targeting_active(int,bool on){active=on;}
    int scaled_hit_step(int){return scaled;}int scaled_deploy_step(int){return 50;}uint64_t keeps_targeting_off_mask(){return 1;}
    void prepare_route(int){++routes;}void clear_reference(int){t.reference=-1;++clears;}
    void fault(int code){faultCode=code;}
};
int main(){
    World w{};w.e.state=WaitingToDeploy;w.e.delay=100;w.e.movementComponent=true;w.e.movementActive=false;
    assert(fixed8_post_state(w,0)==0&&w.e.delay==50);assert(fixed8_post_state(w,0)==0&&w.e.state==Deploying&&w.e.deployCountdown==1000&&w.e.movementActive&&!w.active);
    for(int i=0;i<19;++i)assert(fixed8_post_state(w,0)==0);
    assert(fixed8_post_state(w,0)==1&&w.e.state==Moving&&w.e.deployCountdown==0&&w.active&&w.routes==1);
    w.e.state=Attacking;w.t.reference=4;w.scaled=0;assert(fixed8_post_state(w,0)==1&&!w.active&&w.t.reference<0&&w.clears==1);
    w.t.reference=4;w.active=false;assert(fixed8_post_state(w,0)==1&&w.t.reference<0&&w.clears==2);
    w.scaled=50;w.e.flags=1;w.t.reference=4;assert(fixed8_post_state(w,0)==1&&!w.active&&w.t.reference==4&&w.clears==2);
    w.e.flags=0;assert(fixed8_post_state(w,0)==1&&w.active);w.e.alive=false;assert(fixed8_post_state(w,0)==1&&!w.active&&w.clears==3);
    w.e.state=Moving;w.e.deployCountdown=50;assert(fixed8_set_state(w,0,Standing)==0&&w.e.state==Moving);
    w.e.deployCountdown=0;assert(fixed8_set_state(w,0,Standing)==1&&w.e.state==Standing);assert(w.faultCode==0);
    assert(fixed8_set_state(w,0,Casting)==-1&&w.faultCode==231);
}
