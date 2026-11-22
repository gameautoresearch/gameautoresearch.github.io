// Fixed-role DouZero rules port. Upstream is the differential rules oracle.
// No RNG or hidden-hand feature access: deals are supplied by the pinned Python codec.
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using Cards=std::vector<uint8_t>;
using Moves=std::vector<Cards>;
using Counts=std::array<int,31>;
static constexpr int ranks[15]={3,4,5,6,7,8,9,10,11,12,13,14,17,20,30};
static Counts counts(const Cards &a){Counts c{};for(auto x:a)++c[x];return c;}
static bool consecutive(const Cards&a){for(size_t i=1;i<a.size();++i)if(a[i]!=a[i-1]+1)return false;return true;}
struct Type{int type=15,rank=0,len=1;};
static Type detect(const Cards&a){
 int n=a.size();auto c=counts(a);Cards keys;std::array<int,21> freq{};
 for(int r:ranks)if(c[r]){keys.push_back(r);++freq[c[r]];}
 if(n==0)return{0};if(n==1)return{1,a[0]};
 if(n==2){if(a[0]==a[1])return{2,a[0]};if(a==Cards{20,30})return{5};return{15};}
 if(n==3)return keys.size()==1?Type{3,a[0]}:Type{15};
 if(n==4){if(keys.size()==1)return{4,a[0]};if(keys.size()==2&&(a[0]==a[2]||a[1]==a[3]))return{6,a[1]};return{15};}
 if(consecutive(a))return{8,a[0],n};
 if(n==5)return keys.size()==2?Type{7,a[2]}:Type{15};
 if(n==6&&(keys.size()==2||keys.size()==3)&&freq[4]==1&&(freq[2]==1||freq[1]==2))return{13,a[2]};
 if(n==8&&(((keys.size()==3||keys.size()==2)&&freq[4]==1&&freq[2]==2)||freq[4]==2)){
  int rank=0;for(int r:ranks)if(c[r]==4)rank=r;return{14,rank};}
 if(int(keys.size())==freq[2]&&consecutive(keys))return{9,keys[0],int(keys.size())};
 if(int(keys.size())==freq[3]&&consecutive(keys))return{10,keys[0],int(keys.size())};
 if(freq[3]>=2){Cards triples;int singles=0,pairs=0;
  for(auto r:keys){if(c[r]==3)triples.push_back(r);else if(c[r]==1)++singles;else if(c[r]==2)++pairs;else return{15};}
  if(consecutive(triples)){
   if(int(triples.size())==singles+pairs*2)return{11,triples[0],int(triples.size())};
   if(int(triples.size())==pairs&&keys.size()==triples.size()*2)return{12,triples[0],int(triples.size())};}
  if(triples.size()==4){Cards tail(triples.begin()+1,triples.end()),head(triples.begin(),triples.end()-1);
   if(consecutive(tail))return{11,tail[0],3};if(consecutive(head))return{11,head[0],3};}}
 return{15};
}
static void choose(const Cards &a,int k,int start,Cards &prefix,const std::function<void(const Cards&)> &fn){
 if(k==0){fn(prefix);return;}for(int i=start;i<=int(a.size())-k;++i){
  if(i>start&&a[i]==a[i-1])continue;prefix.push_back(a[i]);choose(a,k-1,i+1,prefix,fn);prefix.pop_back();}}
static Moves serial(const Counts&c,int repeat,int minimum,int length){
 Moves out;for(int start=3;start<=14;++start){int longest=0;while(start+longest<=14&&c[start+longest]>=repeat)++longest;
  for(int n=(length>=minimum?length:minimum);n<=longest;++n){Cards a;for(int r=start;r<start+n;++r)for(int j=0;j<repeat;++j)a.push_back(r);out.push_back(a);if(length>=minimum)break;}}
 return out;
}
static Moves generate(const Cards&hand,int type,int length){
 auto c=counts(hand);Moves out;auto add=[&](Cards a){std::sort(a.begin(),a.end());out.push_back(std::move(a));};
 if(type>=1&&type<=4){for(int r:ranks)if(c[r]>=(type==4?4:type)){if(type!=4||c[r]==4)add(Cards(type,r));}return out;}
 if(type==5){if(c[20]&&c[30])add({20,30});return out;}
 if(type==6||type==7){for(int t:ranks)if(c[t]>=3)for(int r:ranks)if(r!=t&&c[r]>=(type==6?1:2)){Cards a(3,t);a.insert(a.end(),type==6?1:2,r);add(a);}return out;}
 if(type>=8&&type<=10)return serial(c,type-7,type==8?5:type==9?3:2,length);
 if(type==11||type==12){for(auto triple:serial(c,3,2,length)){
  auto tc=counts(triple);Cards available;int n=triple.size()/3;
  if(type==11){for(auto r:hand)if(!tc[r])available.push_back(r);}else for(int r:ranks)if(c[r]>=2&&!tc[r])available.push_back(r);
  Cards prefix;choose(available,n,0,prefix,[&](const Cards &wing){auto a=triple;for(auto r:wing){a.push_back(r);if(type==12)a.push_back(r);}add(a);});}return out;}
 if(type==13||type==14){for(int t:ranks)if(c[t]==4){Cards available;
  if(type==13){for(auto r:hand)if(r!=t)available.push_back(r);}else for(int r:ranks)if(r!=t&&c[r]>=2)available.push_back(r);
  Cards prefix;choose(available,2,0,prefix,[&](const Cards &wing){Cards a(4,t);for(auto r:wing){a.push_back(r);if(type==14)a.push_back(r);}add(a);});}return out;}
 return out;
}
static int triple_rank(const Cards&a){auto c=counts(a);int rank=0;for(int r:ranks)if(c[r]==3)rank=r;return rank;}
static int quad_rank(const Cards&a){auto c=counts(a);int rank=0;for(int r:ranks)if(c[r]==4)rank=r;return rank;}
static Moves legal_moves(const Cards&hand,const Cards&rival){
 auto rt=detect(rival);Moves out;
 auto append=[&](Moves a){out.insert(out.end(),std::make_move_iterator(a.begin()),std::make_move_iterator(a.end()));};
 if(rt.type==0){for(int type=1;type<=14;++type)append(generate(hand,type,0));}
 else if(rt.type!=5&&rt.type!=15){auto options=generate(hand,rt.type,rt.len);if(rt.type==4){auto rocket=generate(hand,5,0);options.insert(options.end(),rocket.begin(),rocket.end());}
  for(auto&a:options){bool wins=false;
   switch(rt.type){case 6:wins=a[1]>rival[1];break;case 7:case 13:wins=a[2]>rival[2];break;
    case 11:case 12:wins=triple_rank(a)>triple_rank(rival);break;case 14:wins=quad_rank(a)>quad_rank(rival);break;
    default:wins=a[0]>rival[0];}
   if(wins)out.push_back(std::move(a));}}
 if(rt.type!=0&&rt.type!=4&&rt.type!=5){append(generate(hand,4,0));append(generate(hand,5,0));}
 if(!rival.empty())out.push_back({});
 std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());
 if(out.empty())throw std::runtime_error("empty native legal set");return out;
}
static void bits(const Cards&a,int8_t*out){auto c=counts(a);std::fill(out,out+54,0);for(int i=0;i<13;++i)for(int n=0;n<c[ranks[i]];++n)out[i*4+n]=1;out[52]=c[20]>0;out[53]=c[30]>0;}
struct Game{
 std::array<Cards,3> hand,played,last;Cards bottom;std::vector<Cards> history;Moves legal;
 int role=0,last_pid=0,bombs=0,decisions=0;bool done=false,won=false;
 Cards rival()const{if(history.empty())return{};if(!history.back().empty())return history.back();return history.size()>1?history[history.size()-2]:Cards{};}
 void refresh(){legal=legal_moves(hand[role],rival());}
 void step(int action){if(done||action<0||action>=int(legal.size()))throw std::runtime_error("illegal native action index");
  Cards a=legal[action];auto t=detect(a);if(!a.empty())last_pid=role;if(t.type==4||t.type==5)++bombs;
  last[role]=a;history.push_back(a);for(auto c:a){auto it=std::find(hand[role].begin(),hand[role].end(),c);if(it==hand[role].end())throw std::runtime_error("native card conservation");hand[role].erase(it);played[role].push_back(c);
   if(role==0){auto b=std::find(bottom.begin(),bottom.end(),c);if(b!=bottom.end())bottom.erase(b);}}
  if(++decisions>162)throw std::runtime_error("native finite game guard");
  done=hand[role].empty();if(done){won=role==0;legal.clear();}else{role=(role+1)%3;refresh();}}
 int heuristic()const{bool cooperate=role!=0&&last_pid!=0&&!legal[0].size();if(cooperate){for(int i=0;i<int(legal.size());++i)if(legal[i].size()==hand[role].size())return i;return 0;}
  int best=0;for(int i=1;i<int(legal.size());++i){auto&a=legal[i];auto&b=legal[best];if(a.size()>b.size()||(a.size()==b.size()&&std::accumulate(a.begin(),a.end(),0)<std::accumulate(b.begin(),b.end(),0)))best=i;}return best;}
 void features(int8_t*out)const{
  int cursor=0;auto put=[&](const Cards&a){bits(a,out+cursor);cursor+=54;};Cards unseen;Counts left{};for(int i=0;i<13;++i)left[ranks[i]]=4;left[20]=left[30]=1;
  for(auto r:hand[role])--left[r];for(auto&p:played)for(auto r:p)--left[r];for(int r:ranks){if(left[r]<0)throw std::runtime_error("native public accounting");for(int k=0;k<left[r];++k)unseen.push_back(r);}
  int width=role==0?319:430;std::fill(out,out+1240,0);put(hand[role]);put(unseen);
  if(role==0){put(rival());put(played[2]);put(played[1]);out[cursor+int(hand[2].size())-1]=1;cursor+=17;out[cursor+int(hand[1].size())-1]=1;cursor+=17;}
  else{int mate=3-role;put(played[0]);put(played[mate]);put(rival());put(last[0]);put(last[mate]);out[cursor+int(hand[0].size())-1]=1;cursor+=20;out[cursor+int(hand[mate].size())-1]=1;cursor+=17;}
  out[cursor+bombs]=1;int begin=std::max(0,int(history.size())-15),padding=15-(int(history.size())-begin);
  for(int i=begin;i<int(history.size());++i)bits(history[i],out+width+(padding+i-begin)*54);
 }
};
// Persistent native threads, bounded to the caller's fixed CPU allocation.
class Pool{
 std::vector<std::thread> threads;std::mutex mutex;std::condition_variable ready,finished;
 std::function<void(int)> fn;int count=0,left=0,generation=0;bool stop=false;std::exception_ptr error;
 public:explicit Pool(int n){for(int k=0;k<n;++k)threads.emplace_back([&,k,n]{int seen=0;std::unique_lock<std::mutex>lock(mutex);while(true){ready.wait(lock,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;auto job=fn;int total=count;lock.unlock();try{for(int i=k;i<total;i+=n)job(i);}catch(...){std::lock_guard<std::mutex>g(mutex);if(!error)error=std::current_exception();}lock.lock();if(--left==0)finished.notify_one();}});}
 void run(int n,const std::function<void(int)>&job){if(threads.empty()){for(int i=0;i<n;++i)job(i);return;}std::unique_lock<std::mutex>lock(mutex);fn=job;count=n;left=threads.size();error=nullptr;++generation;ready.notify_all();finished.wait(lock,[&]{return left==0;});if(error)std::rethrow_exception(error);}
 ~Pool(){{std::lock_guard<std::mutex>g(mutex);stop=true;ready.notify_all();}for(auto&t:threads)t.join();}
};
struct Batch{std::vector<Game>games;Pool pool;std::vector<int32_t>meta;std::vector<int8_t>features,hand,cards,lens,actions;
 explicit Batch(int threads):pool(threads>1?threads:0){}
 void frame(const int32_t*ids,int n){meta.assign(n*9,0);features.resize(n*1240);hand.assign(n*20,0);std::vector<int>offset(n+1);for(int j=0;j<n;++j)offset[j+1]=offset[j]+games.at(ids[j]).legal.size();
  cards.assign(offset[n]*20,0);lens.resize(offset[n]);actions.resize(offset[n]*54);
  pool.run(n,[&](int j){auto&g=games.at(ids[j]);auto*m=meta.data()+j*9;m[0]=g.role;m[1]=g.hand[g.role].size();m[2]=offset[j];m[3]=g.legal.size();m[4]=g.bombs;m[5]=g.decisions;m[6]=g.done;m[7]=g.won;m[8]=g.last_pid;
   if(!g.done){g.features(features.data()+j*1240);std::copy(g.hand[g.role].begin(),g.hand[g.role].end(),hand.data()+j*20);
    for(int k=0;k<int(g.legal.size());++k){int row=offset[j]+k;auto&a=g.legal[k];lens[row]=a.size();std::copy(a.begin(),a.end(),cards.data()+row*20);bits(a,actions.data()+row*54);}}});}
};
static thread_local std::string error;
extern "C" {
const char* db_error(){return error.c_str();}
void* db_new(int threads){try{return new Batch(threads);}catch(const std::exception&e){error=e.what();return nullptr;}}
void db_delete(void*p){delete static_cast<Batch*>(p);}
int db_reset(void*p,const int8_t*hands,const int8_t*bottom,int n){try{auto&b=*static_cast<Batch*>(p);b.games.assign(n,Game{});b.pool.run(n,[&](int i){auto&g=b.games[i];for(int r=0;r<3;++r)for(int k=0;k<15;++k)for(int c=0;c<hands[(i*3+r)*15+k];++c)g.hand[r].push_back(ranks[k]);for(int k=0;k<15;++k)for(int c=0;c<bottom[i*15+k];++c)g.bottom.push_back(ranks[k]);g.refresh();});return 0;}catch(const std::exception&e){error=e.what();return-1;}}
int db_step(void*p,const int32_t*ids,const int32_t*actions,int n){try{auto&b=*static_cast<Batch*>(p);b.pool.run(n,[&](int i){b.games.at(ids[i]).step(actions[i]);});return 0;}catch(const std::exception&e){error=e.what();return-1;}}
int db_frame(void*p,const int32_t*ids,int n){try{static_cast<Batch*>(p)->frame(ids,n);return 0;}catch(const std::exception&e){error=e.what();return-1;}}
const int32_t* db_meta(void*p){return static_cast<Batch*>(p)->meta.data();}
const int8_t* db_features(void*p){return static_cast<Batch*>(p)->features.data();}
const int8_t* db_hand(void*p){return static_cast<Batch*>(p)->hand.data();}
const int8_t* db_cards(void*p){return static_cast<Batch*>(p)->cards.data();}
const int8_t* db_lens(void*p){return static_cast<Batch*>(p)->lens.data();}
const int8_t* db_actions(void*p){return static_cast<Batch*>(p)->actions.data();}
int db_heuristic(void*p){try{auto&b=*static_cast<Batch*>(p);b.pool.run(b.games.size(),[&](int i){auto&g=b.games[i];while(!g.done)g.step(g.heuristic());});return 0;}catch(const std::exception&e){error=e.what();return-1;}}
}
