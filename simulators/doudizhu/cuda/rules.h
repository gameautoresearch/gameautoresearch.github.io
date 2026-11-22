// Apache-2.0. Complete fixed-role DouZero rules port; see ../NOTICE.md.
// Shared by the compiled CPU reference and CUDA kernels. No Python transitions.
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define DD_HD __host__ __device__ __forceinline__
#else
#define DD_HD inline
#endif
namespace dd {
constexpr int MAX_ACTIONS=4096;
// A conservative raw-generation bound for a 20-card hand is <2496: basic55,
// triple-wings138, straights87, airplane singles1327/pairs148, quad wings740.
// 4096 is scratch capacity, not legal truncation. Overflow is an unscored fault.
struct alignas(16) Action{uint8_t c[16];int64_t hi;uint64_t lo;};
struct alignas(16) State{uint8_t hand[3][16],played[3][16],bottom[16],last[3][16],history[162][16];
 int role,last_pid,bombs,steps,done,won;};
DD_HD int rank(int i){return i<12?i+3:i==12?17:i==13?20:30;}
DD_HD int total(const uint8_t*c){int n=0;for(int i=0;i<15;++i)n+=c[i];return n;}
DD_HD void copy(uint8_t*to,const uint8_t*from){for(int i=0;i<16;++i)to[i]=from[i];}
DD_HD int nth(const uint8_t*c,int pos){for(int i=0;i<15;++i){if(pos<c[i])return rank(i);pos-=c[i];}return 0;}
DD_HD int maximum_count(const uint8_t*c,int count){int x=0;for(int i=0;i<15;++i)if(c[i]==count)x=rank(i);return x;}
DD_HD void key(Action&a){uint64_t hi=0,lo=0;int p=0;for(int i=0;i<15;++i)for(int j=0;j<a.c[i];++j){
 if(p<16)hi|=uint64_t(i+1)<<(60-4*p);else lo|=uint64_t(i+1)<<(12-4*(p-16));++p;}
 a.hi=int64_t(hi^uint64_t(0x8000000000000000ULL));a.lo=lo;}
DD_HD bool less(const Action&a,const Action&b){return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}
DD_HD bool equal(const Action&a,const Action&b){return a.hi==b.hi&&a.lo==b.lo;}
struct Type{int type,rank,len;};
DD_HD Type detect(const uint8_t*c){
 int n=total(c),keys[15],nk=0,f[5]={0,0,0,0,0};
 for(int i=0;i<15;++i)if(c[i]){keys[nk++]=rank(i);++f[c[i]];}
 if(!n)return{0,0,1};if(n==1)return{1,nth(c,0),1};
 if(n==2){if(nk==1)return{2,nth(c,0),1};if(c[13]&&c[14])return{5,0,1};return{15,0,1};}
 if(n==3)return nk==1?Type{3,nth(c,0),1}:Type{15,0,1};
 if(n==4){if(nk==1)return{4,nth(c,0),1};if(nk==2&&(nth(c,0)==nth(c,2)||nth(c,1)==nth(c,3)))return{6,nth(c,1),1};return{15,0,1};}
 bool continuous=n==nk;for(int i=1;i<nk;++i)if(keys[i]!=keys[i-1]+1)continuous=false;
 if(continuous)return{8,keys[0],n};
 if(n==5)return nk==2?Type{7,nth(c,2),1}:Type{15,0,1};
 if(n==6&&(nk==2||nk==3)&&f[4]==1&&(f[2]==1||f[1]==2))return{13,nth(c,2),1};
 if(n==8&&(((nk==3||nk==2)&&f[4]==1&&f[2]==2)||f[4]==2))return{14,maximum_count(c,4),1};
 bool seq=true;for(int i=1;i<nk;++i)if(keys[i]!=keys[i-1]+1)seq=false;
 if(nk==f[2]&&seq)return{9,keys[0],nk};if(nk==f[3]&&seq)return{10,keys[0],nk};
 if(f[3]>=2){int t[6],nt=0;for(int i=0;i<15;++i)if(c[i]){if(c[i]==3)t[nt++]=rank(i);else if(c[i]!=1&&c[i]!=2)return{15,0,1};}
  bool s=true;for(int i=1;i<nt;++i)if(t[i]!=t[i-1]+1)s=false;
  if(s){if(nt==f[1]+2*f[2])return{11,t[0],nt};if(nt==f[2]&&nk==2*nt)return{12,t[0],nt};}
  if(nt==4){if(t[2]==t[1]+1&&t[3]==t[2]+1)return{11,t[1],3};if(t[1]==t[0]+1&&t[2]==t[1]+1)return{11,t[0],3};}}
 return{15,0,1};
}
DD_HD void rival(const State&s,uint8_t*out){for(int i=0;i<16;++i)out[i]=0;if(s.steps){int row=s.steps-1;
 if(!total(s.history[row])&&row>0)--row;copy(out,s.history[row]);}}
struct Generator{
 const State&s;Action*out;uint8_t previous[16];Type target;int n=0,overflow=0;
 DD_HD Generator(const State&state,Action*actions):s(state),out(actions){rival(s,previous);target=detect(previous);}
 DD_HD bool want(int family)const{if(target.type==0)return true;if(target.type==5)return false;
  if(family==4||family==5)return true;return family==target.type;}
 DD_HD void emit(Action a,int family){
  if(overflow)return;
  bool accept=target.type==0||family==0||((family==4||family==5)&&target.type!=4);
  if(!accept){switch(target.type){case 6:accept=nth(a.c,1)>nth(previous,1);break;
    case 7:case 13:accept=nth(a.c,2)>nth(previous,2);break;
    case 11:case 12:accept=maximum_count(a.c,3)>maximum_count(previous,3);break;
    case 14:accept=maximum_count(a.c,4)>maximum_count(previous,4);break;
    default:accept=nth(a.c,0)>nth(previous,0);}}
  if(!accept)return;if(n>=MAX_ACTIONS){overflow=1;return;}key(a);out[n++]=a;
 }
 DD_HD void wings(Action base,int family,int k,bool pairs){
  int values[20],nv=0,indices[6];const uint8_t*h=s.hand[s.role];
  for(int r=0;r<15;++r)if(!base.c[r]){int copies=pairs?(h[r]>=2?1:0):h[r];for(int j=0;j<copies;++j)values[nv++]=r;}
  if(nv<k)return;for(int j=0;j<k;++j)indices[j]=j;
  while(true){Action a=base;for(int j=0;j<k;++j)a.c[values[indices[j]]]+=pairs?2:1;emit(a,family);
   int j=k-1;while(j>=0&&indices[j]==nv-k+j)--j;if(j<0)break;++indices[j];for(int q=j+1;q<k;++q)indices[q]=indices[q-1]+1;}
 }
 DD_HD void generate(){
  if(s.done)return;const uint8_t*h=s.hand[s.role];
  for(int family=1;family<=4;++family)if(want(family))for(int r=0;r<15;++r)if(h[r]>=family){Action a{};a.c[r]=family;emit(a,family);}
  if(want(5)&&h[13]&&h[14]){Action a{};a.c[13]=a.c[14]=1;emit(a,5);}
  for(int family=6;family<=7;++family)if(want(family))for(int t=0;t<15;++t)if(h[t]>=3)for(int r=0;r<15;++r)if(r!=t&&h[r]>=(family==6?1:2)){
   Action a{};a.c[t]=3;a.c[r]=family==6?1:2;emit(a,family);}
  for(int family=8;family<=12;++family)if(want(family)){
   int repeat=family==8?1:family==9?2:3,minimum=family==8?5:family==9?3:2;
   for(int start=0;start<12;++start){int longest=0;while(start+longest<12&&h[start+longest]>=repeat)++longest;
    for(int length=minimum;length<=longest;++length){if(target.type==family&&target.len>=minimum&&length!=target.len)continue;
     Action a{};for(int r=start;r<start+length;++r)a.c[r]=repeat;
     if(family<=10)emit(a,family);else wings(a,family,length,family==12);}}}
  for(int family=13;family<=14;++family)if(want(family))for(int r=0;r<15;++r)if(h[r]==4){Action a{};a.c[r]=4;wings(a,family,2,family==14);}
  if(total(previous)){Action a{};emit(a,0);}
 }
};
DD_HD void bits(const uint8_t*c,int8_t*out){for(int i=0;i<13;++i)for(int j=0;j<4;++j)out[i*4+j]=j<c[i];out[52]=c[13]>0;out[53]=c[14]>0;}
DD_HD int remaining(const State&s,int role){return total(s.hand[role]);}
DD_HD void encode(const State&s,int role,int8_t*out){for(int i=0;i<1240;++i)out[i]=0;if(s.done)return;
 uint8_t unseen[16]={},previous[16]={};for(int i=0;i<15;++i){int v=i<13?4:1;v-=s.hand[role][i];for(int r=0;r<3;++r)v-=s.played[r][i];unseen[i]=v;}
 rival(s,previous);int cursor=0;bits(s.hand[role],out+cursor);cursor+=54;bits(unseen,out+cursor);cursor+=54;
 if(role==0){bits(previous,out+cursor);cursor+=54;bits(s.played[2],out+cursor);cursor+=54;bits(s.played[1],out+cursor);cursor+=54;
  out[cursor+remaining(s,2)-1]=1;cursor+=17;out[cursor+remaining(s,1)-1]=1;cursor+=17;}
 else{int mate=3-role;bits(s.played[0],out+cursor);cursor+=54;bits(s.played[mate],out+cursor);cursor+=54;bits(previous,out+cursor);cursor+=54;
  bits(s.last[0],out+cursor);cursor+=54;bits(s.last[mate],out+cursor);cursor+=54;
  out[cursor+remaining(s,0)-1]=1;cursor+=20;out[cursor+remaining(s,mate)-1]=1;cursor+=17;}
 out[cursor+s.bombs]=1;int width=role==0?319:430,begin=s.steps>15?s.steps-15:0,padding=15-(s.steps-begin);
 for(int row=begin;row<s.steps;++row)bits(s.history[row],out+width+(padding+row-begin)*54);
}
DD_HD int bit(const uint8_t*c,int p){if(!c)return 0;return p<52?(c[p/4]>p%4):(c[p-39]>0);}
DD_HD int encode_byte(const State&s,int role,const uint8_t*previous,int r0,int r1,int r2,int i){
 if(s.done)return 0;int width=role==0?319:430;
 if(i>=width){int p=i-width;if(p>=810)return 0;int row=s.steps-15+p/54;if(row<0)return 0;return bit(s.history[row],p%54);}
 int fields=role==0?5:7,offset=fields*54;
 if(i<offset){int field=i/54,p=i%54;int mate=3-role;
  if(field==0)return bit(s.hand[role],p);
  if(field==1){int r=p<52?p/4:p-39;int c=(r<13?4:1)-s.hand[role][r];for(int k=0;k<3;++k)c-=s.played[k][r];return p<52?(c>p%4):(c>0);}
  if(role==0){if(field==2)return bit(previous,p);return bit(s.played[field==3?2:1],p);}
  if(field==2)return bit(s.played[0],p);if(field==3)return bit(s.played[mate],p);
  if(field==4)return bit(previous,p);return bit(s.last[field==5?0:mate],p);
 }
 int p=i-offset;if(role==0){if(p<17)return p==r2-1;if(p<34)return p-17==r1-1;return p-34==s.bombs;}
 int mate=role==1?r2:r1;if(p<20)return p==r0-1;if(p<37)return p-20==mate-1;return p-37==s.bombs;
}
DD_HD int step(State&s,const Action&a){if(s.done||s.steps>=162)return 1;int n=total(a.c);if(n)s.last_pid=s.role;
 if((n==4&&maximum_count(a.c,4))||(n==2&&a.c[13]&&a.c[14]))++s.bombs;
 for(int i=0;i<16;++i){if(a.c[i]>s.hand[s.role][i])return 1;s.last[s.role][i]=a.c[i];s.history[s.steps][i]=a.c[i];
  s.hand[s.role][i]-=a.c[i];s.played[s.role][i]+=a.c[i];if(s.role==0){int remove=a.c[i]<s.bottom[i]?a.c[i]:s.bottom[i];s.bottom[i]-=remove;}}
 ++s.steps;if(!remaining(s,s.role)){s.done=1;s.won=s.role==0;}else s.role=(s.role+1)%3;return 0;
}
DD_HD uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
DD_HD uint32_t word(uint64_t seed,int turn,int domain,int nonce=0){return uint32_t(mix(seed^uint64_t(turn)*0xd2b74407b1ce6e93ULL^uint64_t(domain)*0xca5a826395121157ULL^uint64_t(nonce))>>32);}
DD_HD int uniform(uint64_t seed,int turn,int bound){uint32_t threshold=uint32_t(-uint32_t(bound))%uint32_t(bound);int nonce=0;
 while(true){uint64_t product=uint64_t(word(seed,turn,2,nonce++))*uint32_t(bound);if(uint32_t(product)>=threshold)return int(product>>32);}}
DD_HD int choose(const State&s,const Action*actions,int count,int greedy,int kind,uint64_t seed,int turn,uint64_t epsilon){
 if(kind==2||(kind==1&&uint64_t(word(seed,turn,1))<epsilon))return uniform(seed,turn,count);
 if(kind!=3)return greedy;
 bool cooperate=s.role!=0&&s.last_pid!=0&&total(actions[0].c)==0;
 if(cooperate){for(int i=0;i<count;++i)if(total(actions[i].c)==remaining(s,s.role))return i;return 0;}
 int best=0,best_n=-1,best_sum=0;for(int i=0;i<count;++i){int n=total(actions[i].c),sum=0;for(int r=0;r<15;++r)sum+=rank(r)*actions[i].c[r];
  if(n>best_n||(n==best_n&&sum<best_sum)){best=i;best_n=n;best_sum=sum;}}return best;
}
}
