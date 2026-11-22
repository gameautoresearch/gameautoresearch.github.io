#include "rules.h"
#include <algorithm>
#include <cstring>
extern "C" {
int dg_state_bytes(){return sizeof(dd::State);}int dg_action_bytes(){return sizeof(dd::Action);}
void dg_cpu_reset(void*states,const int8_t*hands,const int8_t*bottom,int n){auto*s=(dd::State*)states;std::memset(states,0,n*sizeof(dd::State));for(int g=0;g<n;++g){for(int r=0;r<3;++r)for(int k=0;k<16;++k)s[g].hand[r][k]=hands[(g*3+r)*16+k];for(int k=0;k<16;++k)s[g].bottom[k]=bottom[g*16+k];}}
int dg_cpu_generate(void*state,void*actions){auto&s=*(dd::State*)state;auto*a=(dd::Action*)actions;dd::Generator gen(s,a);gen.generate();if(gen.overflow)return-1;std::sort(a,a+gen.n,dd::less);return std::unique(a,a+gen.n,dd::equal)-a;}
void dg_cpu_encode(void*state,int8_t*out){auto&s=*(dd::State*)state;dd::encode(s,s.role,out);}
void dg_cpu_encode_warp(void*state,int8_t*out){auto&s=*(dd::State*)state;int row=s.steps-1;if(row>0&&!dd::total(s.history[row]))--row;const uint8_t*p=row>=0?s.history[row]:nullptr;
 for(int i=0;i<1240;++i)out[i]=dd::encode_byte(s,s.role,p,dd::remaining(s,0),dd::remaining(s,1),dd::remaining(s,2),i);}
int dg_cpu_step(void*state,void*action){return dd::step(*(dd::State*)state,*(dd::Action*)action);}
int dg_cpu_choose(void*state,void*actions,int count,int greedy,int kind,uint64_t seed,int turn,uint64_t epsilon){return dd::choose(*(dd::State*)state,(dd::Action*)actions,count,greedy,kind,seed,turn,epsilon);}
}
