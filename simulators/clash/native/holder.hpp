// SPDX-License-Identifier: Apache-2.0
// Ported from crforge core/src/main/java/org/crforge/core/battle/{EntityHolder,BattleEntity}.java
// @ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
// Flat slot/index storage replaces Java objects; hooks are statically dispatched to World.
#pragma once
#include "arith.hpp"
namespace clash_native {
template<int CAP> struct NativeHolder {
    int ids[CAP]{},kinds[CAP]{},required[CAP]{},birth[6]{},live[CAP]{},pending[CAP]{},snapshot[CAP]{};
    uint8_t components[CAP]{},active[CAP]{},allocated[CAP]{};
    int live_count=0,pending_count=0,snapshot_count=0,free_cursor=0,run_tick=-1;
    bool ticking=false,in_pending=false,pending_done=false;
    template<class World> CN_HD int reserve(World& w,int kind,int component_bits,int active_bits,int required_id=0){
        if(kind<0||kind>5||pending_count+live_count>=CAP){w.fault(201);return -1;}
        int slot=-1;for(int n=0;n<CAP;++n){int k=(free_cursor+n)%CAP;if(!allocated[k]){slot=k;free_cursor=(k+1)%CAP;break;}}
        if(slot<0){w.fault(202);return -1;}
        allocated[slot]=1;kinds[slot]=kind;ids[slot]=kind*1000000+(birth[kind]++%1000000);required[slot]=required_id;
        components[slot]=uint8_t(component_bits);active[slot]=uint8_t(active_bits);pending[pending_count++]=slot;
        // Payload must be initialized before hand_over() folds its tags.
        return slot;
    }
    template<class World> CN_HD void hand_over(World& w,int slot,bool registered=false){
        w.add_tag_fold(slot);
        if(registered){w.before_registration(slot);for(int k=0;k<4;++k)if((components[slot]&active[slot])&(1<<k))w.component_visit(slot,k);}
    }
    CN_HD int find_live(int id)const{for(int k=0;k<live_count;++k)if(ids[live[k]]==id)return live[k];return -1;}
    CN_HD int find(int id)const{int found=find_live(id);if(found>=0)return found;for(int k=0;k<pending_count;++k)if(ids[pending[k]]==id)return pending[k];return -1;}
    CN_HD void free_slot(int slot){allocated[slot]=0;components[slot]=active[slot]=0;required[slot]=0;}
    template<class World> CN_HD void notices(World& w,int gone,bool notify_holder){
        // A notice's list is fixed: a child made by it does not hear this notice.
        int listed[CAP],n=0;for(int k=0;k<pending_count;++k)listed[n++]=pending[k];for(int k=0;k<live_count;++k)listed[n++]=live[k];
        for(int k=0;k<n;++k){int s=listed[k];w.entity_removed(s,gone);w.actions_object_left(s,ids[gone]);w.actions_instigator_left(s,gone);w.parent_removed(s,gone);}
        if(notify_holder)w.holder_entity_removed(gone);
    }
    template<class World> CN_HD bool removal_round(World& w){
        bool any=false;
        for(int k=0;k<pending_count;++k){int gone=pending[k];if(!w.removable(gone))continue;any=true;notices(w,gone,true);
            pending[k]=pending[--pending_count];--k;w.actions_released(gone);free_slot(gone);}
        int removed[CAP],count=0,kept=0;
        for(int k=0;k<live_count;++k){int slot=live[k];if(w.removable(slot))removed[count++]=slot;else live[kept++]=slot;}live_count=kept;
        for(int k=0;k<count;++k){int gone=removed[k];w.leaving(gone);notices(w,gone,false);w.actions_released(gone);w.holder_entity_removed(gone);free_slot(gone);}
        return any||count;
    }
    template<class World> CN_HD void cleanup(World& w){
        while(removal_round(w)){}
        if(!pending_count)return;
        int admitted[CAP],count=0;
        for(int k=0;k<pending_count;++k){int slot=pending[k],need=required[slot],owner=need?find_live(need):-1;
            if(need&&(owner<0||w.removable(owner))){w.released_at_fold(slot);free_slot(slot);continue;}
            w.add_tag_fold(slot);admitted[count++]=slot;required[slot]=0;
        }
        pending_count=0;
        // Kind bands and per-kind IDs define order, not allocation/reuse slots.
        for(int k=1;k<count;++k){int value=admitted[k],j=k;while(j>0&&ids[admitted[j-1]]>ids[value]){admitted[j]=admitted[j-1];--j;}admitted[j]=value;}
        for(int k=0;k<count;++k){int slot=admitted[k],j=live_count;while(j>0&&ids[live[j-1]]>ids[slot]){live[j]=live[j-1];--j;}live[j]=slot;++live_count;}
        for(int k=0;k<count;++k)w.on_registered(admitted[k]);
    }
    template<class World> CN_HD void pending_pass(World& w,int phase){in_pending=true;for(int k=0;k<snapshot_count;++k)w.pending_actions(snapshot[k],phase);in_pending=false;}
    template<class World> CN_HD void tick(World& w,int tick_value){
        if(ticking){w.fault(203);return;}ticking=true;pending_done=false;cleanup(w);
        snapshot_count=live_count;for(int k=0;k<live_count;++k)snapshot[k]=live[k];
        w.pre_pass(tick_value,snapshot,snapshot_count);
        for(int k=0;k<snapshot_count;++k)w.pre_hook(snapshot[k]);
        pending_pass(w,1);
        for(int component=0;component<4;++component)for(int k=0;k<snapshot_count;++k){int slot=snapshot[k];if(!(components[slot]&(1<<component)))continue;w.component_refresh(slot,component);if(active[slot]&(1<<component))w.component_visit(slot,component);}
        run_tick=tick_value;for(int k=0;k<snapshot_count;++k)w.running_actions(snapshot[k],tick_value);run_tick=-1;
        pending_pass(w,2);for(int k=0;k<snapshot_count;++k)w.post_hook(snapshot[k]);
        w.after_post_hooks();pending_pass(w,3);pending_done=true;w.post_pass(tick_value);cleanup(w);
        for(int k=0;k<live_count;++k)w.action_countdown(live[k]);
        ticking=false;run_tick=-1;
    }
};
}
