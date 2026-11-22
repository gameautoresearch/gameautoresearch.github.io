// SPDX-License-Identifier: Apache-2.0
// Persistent CPU scheduling around the independent native fixed8 world. Game
// rules remain in world_impl.hpp (crforge full-path/commit attribution there).
// One World per lane; no Java, simulation substitutions or shared per-game state.
#include "world_impl.hpp"
#include "rules/observation_world.hpp"
#include "task_spec.hpp"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
namespace {
using namespace clash_native;
const NativeTaskSpec batch_task=make_task_spec();
enum class Job {Reset,Step,Read,Save,Load,StepRead,ResetRead};
struct Work {
    Job kind=Job::Read;
    const int32_t* seeds=nullptr;const int32_t* selected=nullptr;const int32_t* actions=nullptr;int tick_limit=0;
    int32_t* observations=nullptr;uint8_t* masks=nullptr;int32_t* infos=nullptr;float* rewards=nullptr;
    uint8_t* output_state=nullptr;const uint8_t* input_state=nullptr;
};
class CpuBatch {
public:
    const int count,requested_threads;
    std::unique_ptr<NativeWorld[]> worlds;
    CpuBatch(int n,int threads):count(n),requested_threads(threads),worlds(new NativeWorld[n]()){
        if(threads>1){
            try {for(int i=0;i<threads;++i)workers.emplace_back([this]{worker_loop();});}
            catch(...){shutdown();throw;}
        }
    }
    ~CpuBatch(){shutdown();}
    int dispatch(const Work& value){
        // A single owner may call the C ABI from different Python threads; calls
        // are serialized, and every buffer remains borrowed only until return.
        std::lock_guard<std::mutex> call_lock(call_mutex);
        internal_error.store(0,std::memory_order_relaxed);
        if(workers.empty()){
            for(int i=0;i<count;++i)run_lane(value,i);
        }else{
            std::unique_lock<std::mutex> lock(mutex);
            work=value;next.store(0,std::memory_order_relaxed);remaining=int(workers.size());++generation;
            available.notify_all();finished.wait(lock,[this]{return remaining==0;});
        }
        if(internal_error.load(std::memory_order_relaxed))return -2;
        if(value.kind==Job::Save)return 0; // Fault snapshots remain available for diagnostics.
        for(int i=0;i<count;++i)if(worlds[i].faultCode)return worlds[i].faultCode;
        return 0;
    }
private:
    std::vector<std::thread> workers;
    std::mutex mutex,call_mutex;
    std::condition_variable available,finished;
    uint64_t generation=0;int remaining=0;bool stopping=false;
    std::atomic<int> next{0},internal_error{0};Work work{};
    void render(const Work& w,int i){worlds[i].render(w.observations+size_t(i)*14048,w.masks+size_t(i)*4614,w.infos+size_t(i)*12,w.rewards+size_t(i)*2);}
    void run_lane(const Work& w,int i){
        NativeWorld& world=worlds[i];
        switch(w.kind){
        case Job::Reset:case Job::ResetRead:
            if(!w.selected||w.selected[i])world.init(&batch_task,w.seeds[i]);
            if(w.kind==Job::ResetRead)render(w,i);
            break;
        case Job::Step:case Job::StepRead:
            world.advance(w.actions[size_t(i)*2],w.actions[size_t(i)*2+1],w.tick_limit);
            if(w.kind==Job::StepRead)render(w,i);
            break;
        case Job::Read:render(w,i);break;
        case Job::Save:std::memcpy(w.output_state+size_t(i)*sizeof(NativeWorld),&world,sizeof(NativeWorld));break;
        case Job::Load:std::memcpy(&world,w.input_state+size_t(i)*sizeof(NativeWorld),sizeof(NativeWorld));world.bind(&batch_task);break;
        }
    }
    void worker_loop(){
        uint64_t seen=0;
        for(;;){
            std::unique_lock<std::mutex> lock(mutex);
            available.wait(lock,[this,seen]{return stopping||generation!=seen;});
            if(stopping)return;seen=generation;Work current=work;lock.unlock();
            for(int i=next.fetch_add(1,std::memory_order_relaxed);i<count;i=next.fetch_add(1,std::memory_order_relaxed)){
                try{run_lane(current,i);}catch(...){internal_error.store(1,std::memory_order_relaxed);}
            }
            lock.lock();if(--remaining==0)finished.notify_one();
        }
    }
    void shutdown(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        available.notify_all();for(auto& thread:workers)if(thread.joinable())thread.join();workers.clear();
    }
};
CpuBatch* batch(void* handle){return static_cast<CpuBatch*>(handle);}
bool outputs(int32_t*o,uint8_t*m,int32_t*i,float*r){return o&&m&&i&&r;}
}
extern "C" {
void* cn_batch_create(int count,int threads,const int32_t* seeds){
    if(count<1||threads<0||threads>256||!seeds)return nullptr;
    try{auto value=std::make_unique<CpuBatch>(count,threads);Work w{};w.kind=Job::Reset;w.seeds=seeds;if(value->dispatch(w))return nullptr;return value.release();}catch(...){return nullptr;}
}
void cn_batch_free(void* handle){delete batch(handle);}
int cn_batch_count(void* handle){return handle?batch(handle)->count:0;}
int cn_batch_threads(void* handle){return handle?batch(handle)->requested_threads:0;}
size_t cn_batch_world_bytes(){return sizeof(clash_native::NativeWorld);}
int cn_batch_handles(void* handle,void** out,int count){if(!handle||!out||count!=batch(handle)->count)return -1;for(int i=0;i<count;++i)out[i]=&batch(handle)->worlds[i];return 0;}
int cn_batch_reset(void* handle,const int32_t* seeds,const int32_t* selected){if(!handle||!seeds)return -1;Work w{};w.kind=Job::Reset;w.seeds=seeds;w.selected=selected;return batch(handle)->dispatch(w);}
int cn_batch_step(void* handle,const int32_t* actions,int tick_limit){if(!handle||!actions||tick_limit<0)return -1;Work w{};w.kind=Job::Step;w.actions=actions;w.tick_limit=tick_limit;return batch(handle)->dispatch(w);}
int cn_batch_read(void* handle,int32_t* obs,uint8_t* masks,int32_t* infos,float* rewards){if(!handle||!outputs(obs,masks,infos,rewards))return -1;Work w{};w.kind=Job::Read;w.observations=obs;w.masks=masks;w.infos=infos;w.rewards=rewards;return batch(handle)->dispatch(w);}
int cn_batch_step_read(void* handle,const int32_t* actions,int tick_limit,int32_t* obs,uint8_t* masks,int32_t* infos,float* rewards){if(!handle||!actions||tick_limit<0||!outputs(obs,masks,infos,rewards))return -1;Work w{};w.kind=Job::StepRead;w.actions=actions;w.tick_limit=tick_limit;w.observations=obs;w.masks=masks;w.infos=infos;w.rewards=rewards;return batch(handle)->dispatch(w);}
int cn_batch_reset_read(void* handle,const int32_t* seeds,const int32_t* selected,int32_t* obs,uint8_t* masks,int32_t* infos,float* rewards){if(!handle||!seeds||!outputs(obs,masks,infos,rewards))return -1;Work w{};w.kind=Job::ResetRead;w.seeds=seeds;w.selected=selected;w.observations=obs;w.masks=masks;w.infos=infos;w.rewards=rewards;return batch(handle)->dispatch(w);}
int cn_batch_save(void* handle,void* out,size_t bytes){if(!handle||!out||bytes!=size_t(batch(handle)->count)*sizeof(clash_native::NativeWorld))return -1;Work w{};w.kind=Job::Save;w.output_state=static_cast<uint8_t*>(out);return batch(handle)->dispatch(w);}
int cn_batch_load(void* handle,const void* src,size_t bytes){if(!handle||!src||bytes!=size_t(batch(handle)->count)*sizeof(clash_native::NativeWorld))return -1;Work w{};w.kind=Job::Load;w.input_state=static_cast<const uint8_t*>(src);return batch(handle)->dispatch(w);}
}
