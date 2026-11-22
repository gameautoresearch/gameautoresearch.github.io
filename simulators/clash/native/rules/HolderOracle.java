// SPDX-License-Identifier: Apache-2.0
// Differential oracle harness only. EntityHolder/BattleEntity/EntityActions/BattleComponent
// from crforge core/src/main/java/org/crforge/core/battle/ @
// f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de; pinned source remains unmodified.
package org.crforge.core.battle;
import java.util.*;
public final class HolderOracle {
  static final StringBuilder log=new StringBuilder();
  static EntityHolder holder;static E[] all=new E[32];static int scenario,tick=-1;
  static void emit(int code,int label,int value){log.append(code).append(',').append(label).append(',').append(value).append('\n');}
  static class E extends BattleEntity {
    final int label;boolean dead=false;final EntityActions actions;
    E(int label,int kind,int components,int active){super(kind);this.label=label;all[label]=this;
      for(int k=0;k<4;k++)if((components&(1<<k))!=0){final int slot=k;attach(new BattleComponent(){public int index(){return slot;}public void refresh(){emit(6,label,slot);}public void visit(){emit(7,label,slot);if(tick==0&&slot==0){if(scenario==0&&label==2)make(5,4,2,2,false,0);if(scenario==1&&label==1)dead=true;}}});setActive(k,(active&(1<<k))!=0);}
      actions=new EntityActions(){
        public void pendingPass(int phase){emit(5,label,phase);if(tick==0&&phase==1&&label==1){if(scenario==0)make(4,5,3,1,true,0);if(scenario==1)make(4,5,1,1,true,getId());}}
        public void runPass(int t){emit(8,label,t);}
        public void endOfTick(){emit(13,label,0);}
        public void objectLeft(int id){emit(16,label,labelOf(id));}
        public void instigatorLeft(EntityActions left){int gone=0;for(E e:all)if(e!=null&&e.actions==left)gone=e.label;emit(17,label,gone);}
        public void released(){emit(20,label,0);}
      };
    }
    public EntityActions actions(){return actions;}
    public boolean isRemovable(){return dead;}
    protected void addTagFold(){emit(1,label,0);}
    protected void beforeRegistrationVisit(){emit(2,label,0);}
    protected void onRegistered(){emit(3,label,0);}
    protected void preHook(){emit(4,label,0);}
    protected void postHook(){emit(10,label,0);if(tick==0&&scenario==0&&label==3)make(6,3,1,1,false,0);}
    protected void leaving(){emit(14,label,0);}
    protected void entityRemoved(BattleEntity e){int gone=((E)e).label;emit(15,label,gone);if(scenario==1&&label==2&&gone==1)dead=true;}
    protected void parentRemoved(BattleEntity e){emit(18,label,((E)e).label);}
    protected void releasedAtFold(){emit(21,label,0);}
  }
  static int labelOf(int id){for(E e:all)if(e!=null&&e.getId()==id)return e.label;return -1;}
  static E make(int label,int kind,int components,int active,boolean registered,int required){E e=new E(label,kind,components,active);if(registered)holder.addRegistered(e,required);else{holder.add(e);if(required!=0)holder.requireListed(e,required);}return e;}
  public static void main(String[] args){scenario=Integer.parseInt(args[0]);holder=new EntityHolder(new HolderPasses(){public void prePass(int t,List<BattleEntity> snapshot){emit(22,0,t);}public void afterPostHooks(){emit(11,0,0);}public void postPass(int t){emit(12,0,t);}public void entityRemoved(BattleEntity e){emit(19,((E)e).label,0);}});
    if(scenario==0){make(1,5,5,1,false,0);make(2,3,3,3,false,0);make(3,4,4,4,false,0);}
    else if(scenario==1){make(1,5,1,1,false,0);make(2,5,1,1,false,0);holder.cleanup();make(3,3,1,1,false,0).dead=true;make(5,4,1,1,false,0).dead=true;}
    else if(scenario==2){E a=make(1,5,1,1,false,0);make(2,4,1,1,true,a.getId());holder.cleanup();make(3,4,1,1,true,a.getId());}
    holder.cleanup();for(tick=0;tick<3;tick++)holder.tick(tick);System.out.print(log);
  }
}
