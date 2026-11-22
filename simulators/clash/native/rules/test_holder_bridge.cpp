// SPDX-License-Identifier: Apache-2.0; pinned source provenance in ../holder.hpp.
#include "../holder.hpp"
#include <cstdio>
#include <cstdlib>
using namespace clash_native;
struct HolderTestWorld {
    NativeHolder<64> holder{};int label[64]{},slots[32]{},scenario=0,tick=-1;bool dead[64]{};
    void emit(int code,int l,int value){std::printf("%d,%d,%d\n",code,l,value);}
    void fault(int code){std::fprintf(stderr,"fault%d\n",code);std::abort();}
    int make(int l,int kind,int components,int active,bool registered=false,int required=0){int s=holder.reserve(*this,kind,components,active,required);label[s]=l;dead[s]=false;slots[l]=s;holder.hand_over(*this,s,registered);return s;}
    void add_tag_fold(int s){emit(1,label[s],0);}void before_registration(int s){emit(2,label[s],0);}void on_registered(int s){emit(3,label[s],0);}void pre_hook(int s){emit(4,label[s],0);}
    void pending_actions(int s,int phase){emit(5,label[s],phase);if(tick==0&&phase==1&&label[s]==1){if(scenario==0)make(4,5,3,1,true);if(scenario==1)make(4,5,1,1,true,holder.ids[s]);}}
    void component_refresh(int s,int k){emit(6,label[s],k);}void component_visit(int s,int k){emit(7,label[s],k);if(tick==0&&k==0){if(scenario==0&&label[s]==2)make(5,4,2,2);if(scenario==1&&label[s]==1)dead[s]=true;}}
    void running_actions(int s,int t){emit(8,label[s],t);}void post_hook(int s){emit(10,label[s],0);if(tick==0&&scenario==0&&label[s]==3)make(6,3,1,1);}
    void after_post_hooks(){emit(11,0,0);}void post_pass(int t){emit(12,0,t);}void action_countdown(int s){emit(13,label[s],0);}
    bool removable(int s){return dead[s];}void leaving(int s){emit(14,label[s],0);}void entity_removed(int s,int gone){emit(15,label[s],label[gone]);if(scenario==1&&label[s]==2&&label[gone]==1)dead[s]=true;}
    void actions_object_left(int s,int id){int gone=0;for(int i=0;i<64;++i)if(holder.ids[i]==id)gone=label[i];emit(16,label[s],gone);}void actions_instigator_left(int s,int gone){emit(17,label[s],label[gone]);}
    void parent_removed(int s,int gone){emit(18,label[s],label[gone]);}void holder_entity_removed(int s){emit(19,label[s],0);}void actions_released(int s){emit(20,label[s],0);}void released_at_fold(int s){emit(21,label[s],0);}
    void pre_pass(int t,const int*,int){emit(22,0,t);}
    void run(int which){scenario=which;if(scenario==0){make(1,5,5,1);make(2,3,3,3);make(3,4,4,4);}else if(scenario==1){make(1,5,1,1);make(2,5,1,1);holder.cleanup(*this);dead[make(3,3,1,1)]=true;dead[make(5,4,1,1)]=true;}else{int a=make(1,5,1,1);make(2,4,1,1,true,holder.ids[a]);holder.cleanup(*this);make(3,4,1,1,true,holder.ids[a]);}
        holder.cleanup(*this);for(tick=0;tick<3;++tick)holder.tick(*this,tick);
    }
};
int main(int argc,char**argv){if(argc!=2)return 2;HolderTestWorld w{};w.run(std::atoi(argv[1]));}
