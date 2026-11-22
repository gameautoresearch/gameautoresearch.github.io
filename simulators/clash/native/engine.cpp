// SPDX-License-Identifier: Apache-2.0
// CPU ABI for the independent fixed8 C++ world. Same world headers are CUDA code.
#include "world_impl.hpp"
#include "rules/observation_world.hpp"
#include "task_spec.hpp"
#include <cstring>
#include <new>
using namespace clash_native;
static const NativeTaskSpec task=make_task_spec();
extern "C" {
void* cn_world_create(int seed){auto*w=new(std::nothrow) NativeWorld();if(!w)return nullptr;w->init(&task,seed);return w;}
void cn_world_free(void* p){delete static_cast<NativeWorld*>(p);}
int cn_world_reset(void*p,int seed){auto&w=*static_cast<NativeWorld*>(p);w.init(&task,seed);return w.faultCode;}
int cn_world_step(void*p,int a0,int a1,int limit){auto&w=*static_cast<NativeWorld*>(p);return w.advance(a0,a1,limit);}
int cn_world_read(void*p,int*obs,uint8_t*masks,int*info,float*reward){auto&w=*static_cast<NativeWorld*>(p);w.render(obs,masks,info,reward);return w.faultCode;}
int cn_world_tick(void*p){auto&w=*static_cast<NativeWorld*>(p);w.rendered=false;if(!w.match.step(w))return 0;w.finish_public_tick();return 1;}
size_t cn_world_bytes(){return sizeof(NativeWorld);}
// The cadence and timeline this binary runs; native/match_forms.py checks it against its form.
int cn_match_form(int*out,int cap){return report_match_form(task.rules,out,cap);}
int cn_world_save(void*p,void*out,size_t n){if(n!=sizeof(NativeWorld))return -1;std::memcpy(out,p,n);return 0;}
int cn_world_load(void*p,const void*src,size_t n){if(n!=sizeof(NativeWorld))return -1;auto&w=*static_cast<NativeWorld*>(p);std::memcpy(p,src,n);w.bind(&task);return w.faultCode;}
int cn_world_debug(void*p,int*out,int cap){auto&w=*static_cast<NativeWorld*>(p);int n=0;auto put=[&](int v){if(n<cap)out[n]=v;++n;};put(w.match.tick);put(w.match.clockMs);put(w.random.state);put(w.faultCode);put(w.holder.live_count);put(w.match.timeline.timeMs);put(w.match.endTimerMs);put(w.match.tiebreakMs);put(w.kings[0].waitListed);put(w.kings[0].activationListed);for(int k=0;k<w.holder.live_count;++k){int i=w.holder.live[k];put(w.holder.ids[i]);put(w.holder.kinds[i]);if(w.is_unit(i)){auto&e=w.entities[i];auto&t=w.targets[i];put(w.rows[i]);put(e.side);put(e.x);put(e.y);put(e.state);put(e.deployCountdown);put(e.delay);put(e.hp);put(t.reference<0?-1:w.holder.ids[t.reference]);put(t.attackTimerMs);put(t.loadTimerMs);put(e.dirX);put(e.dirY);put(int(e.flags));put(int(e.pendingFlags));put(e.pendingDamageAmount);put(e.pendingDamageDurationMs);put(w.holder.active[i]);}else if(w.holder.kinds[i]==4){auto&q=w.projectiles[i];put(w.projectileRows[i]);put(q.side);put(q.x);put(q.y);put(q.z);put(q.target_id);put(q.aim_x);put(q.aim_y);put(q.delay_ms);}else{put(w.areas[i].countdown);}}return n;}
}
