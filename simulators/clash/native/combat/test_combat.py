"""Pinned-Java differential checks. Runtime combat never calls Java; this is a test oracle."""
from pathlib import Path
import ctypes,json,sys,subprocess,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from game import make_vector,resolve
import jpype

def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    lib=output/'combat-tests.dylib'
    subprocess.run(['clang++','-std=c++20','-O2','-fPIC','-shared',str(HERE/'test_bridge.cpp'),'-o',str(lib)],check=True)
    native=ctypes.CDLL(str(lib));ptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    native.cn_combat_scaling.argtypes=[ptr,ctypes.c_int,ptr,ctypes.c_int,ptr]
    native.cn_combat_trig.argtypes=[ptr,ctypes.c_int,ptr]
    native.cn_combat_damage.argtypes=[ptr,ctypes.c_int,ptr]
    native.cn_combat_launch_flight.argtypes=[ptr,ptr]
    native.cn_combat_homing.argtypes=[ptr,ptr,ptr]
    native.cn_combat_arrows.argtypes=[ctypes.c_int]*7+[ptr]
    config=json.loads((HERE.parents[1]/'configs/profile-trimmed.json').read_text());config.update(envs=1,threads=1)
    checks={}
    with make_vector(config) as v:
        J=jpype.JClass;Math=J('org.crforge.core.pathfinding.math.FixedMath');Scaling=J('org.crforge.core.pathfinding.combat.LevelScaling');Mode=J('org.crforge.core.pathfinding.combat.ScalingMode');Globals=J('org.crforge.core.pathfinding.combat.ScalingGlobals');Rarity=J('org.crforge.core.pathfinding.combat.RarityTable');Packed=J('org.crforge.core.pathfinding.combat.PackedLevel');Direct=J('org.crforge.core.pathfinding.target.DirectHit')
        rng=np.random.default_rng(1141)
        xy=rng.integers(-30000,30000,(2000,3),dtype=np.int32);xy[:,2]=rng.integers(-10000,10000,2000,dtype=np.int32);xy[:5,:2]=[[0,0],[1,0],[0,1],[-1,0],[0,-1]]
        trig=np.empty((len(xy),4),np.int32);native.cn_combat_trig(xy,len(xy),trig)
        for (x,y,a),row in zip(xy,trig):
            vec=jpype.JArray(jpype.JInt)([int(x),int(y)]);Math.rotate1024(vec,int(a));expected=[Math.sine1024(int(a)),Math.angleOfVector(int(x),int(y)),*list(vec)];np.testing.assert_array_equal(row,expected)
        checks['integer_trigonometry']=len(xy)
        modes=[Mode.NONE,Mode.CARD_DAMAGE,Mode.CARD_HITPOINTS,Mode.KING_DAMAGE,Mode.KING_HITPOINTS,Mode.TOWER_DAMAGE,Mode.TOWER_HITPOINTS]
        n=0
        for rarity in [Rarity.COMMON,Rarity.RARE,Rarity.EPIC,Rarity.LEGENDARY,Rarity.CHAMPION]:
            multipliers=np.array(list(rarity.powerLevelMultiplier()),np.int32)
            rows=np.array([[int(base),int(Packed.fromLevel(level,rarity)),mode,int(rarity.relativeLevel()),1] for base in [0,1,17,55,1200,2147483647] for level in [1,3,8,11,16,23] for mode in range(7)],np.int32)
            result=np.empty((len(rows),3),np.int32);native.cn_combat_scaling(rows,len(rows),multipliers,len(multipliers),result)
            for row,actual in zip(rows,result):
                base,packed,mode,relative,_=map(int,row);expected=[Scaling.scale(Globals.standard(),base,packed,modes[mode],rarity),Direct.crownTowerDamage(relative,base),Packed.pack(packed,rarity)];np.testing.assert_array_equal(actual,expected)
            n+=len(rows)
        checks['level_and_crown_scaling']=n
        HP=J('org.crforge.core.pathfinding.combat.HitPoints');Damage=J('org.crforge.core.pathfinding.combat.DamageApplication')
        rows=np.array([[int(hp),int(shield),int(amount),dedupe,flags,reduction,attack,crown] for hp in [1,17,1000] for shield in [0,13,100] for amount in [0,1,3,17,1250] for dedupe in [0,911] for reduction in [0,33,100,-33] for attack in [30,100,155] for crown in [0,1] for flags in [0,1,2,4,8,16,32,64,128,256]],np.int32)
        result=np.empty((len(rows),8),np.int32);native.cn_combat_damage(rows,len(rows),result)
        for row,actual in zip(rows,result):
            hp,shield,amount,dedupe,flags,reduction,attack,crown=map(int,row);health=HP(hp);health.setShield(shield)
            q=jpype.JProxy('org.crforge.core.pathfinding.combat.DamageQueries',dict={'damageForbidden':lambda flags=flags:bool(flags&1),'immune':lambda flags=flags:bool(flags&2),'hidden':lambda flags=flags:bool(flags&4),'untouchable':lambda flags=flags:bool(flags&8),'damageHeld':lambda flags=flags:bool(flags&16),'battleEnded':lambda flags=flags:bool(flags&32),'unkillable':lambda flags=flags:bool(flags&64),'targetHasBuffComponent':lambda flags=flags:not bool(flags&128),'attackerHasBuffComponent':lambda flags=flags:not bool(flags&256),'heard':lambda amount:amount,'hitPointsTaken':lambda amount:None,'modifyDamage':lambda d,reduction=reduction:((100-reduction)*d)//100,'attackerDamagePercent':lambda attack=attack:attack,'attackerCrownTowerPercent':lambda:30,'crownTowerTarget':lambda crown=crown:bool(crown)})
            r=Damage.damage(health,amount,dedupe,17,-90,q)
            expected=[health.getHitPoints(),health.getShield(),r.applied(),r.landed(),r.died(),r.accepted(),health.getLastHitHeading()]
            again=Damage.damage(health,amount,dedupe,0,0,q);expected.append(again.applied());np.testing.assert_array_equal(actual,expected)
        checks['damage_shield_dedupe']=len(rows)
        Vector=J('org.crforge.env.VectorEnv');PathJ=J('java.nio.file.Path');tables=Vector.tables(PathJ.of(str(resolve(config['tables']))));records=J('org.crforge.core.battle.data.BattleRecords')(tables)
        Battle=J('org.crforge.core.battle.unit.Standard1v1Battle');battle=Battle(tables,11,False);world=battle.getWorld();battle.getBattle().step();king=world.present().get(0)
        Projectile=J('org.crforge.core.battle.projectile.ProjectileEntity');post=Projectile.class_.getDeclaredMethod('postHook');post.setAccessible(True)
        count=0
        for name in ['Fireball','Arrows']:
            card=records.card(name);data=records.projectile(str(card.projectile()))
            for delay in [0,1,50,77,199]:
                p=Projectile(world,data,0);p.cast(king,10,3500,1000,2250,14500,18000,delay,0)
                # The public table rows give exact speed, gravity and height.
                row=np.array([data.speed(),data.gravity(),data.constantHeight(),3500,1000,2250,14500,18000,delay,40],np.int32);out=np.empty((40,5),np.int32);native.cn_combat_launch_flight(row,out)
                for actual in out:
                    if not p.isReleased():post.invoke(p)
                    expected=[p.getX(),p.getY(),p.getZ(),p.getDelayMs(),p.isReleased()];np.testing.assert_array_equal(actual,expected)
                count+=40
        checks['real_table_projectile_flight_ticks']=count
        target=world.present().get(3);target_field=Projectile.class_.getDeclaredField('target');target_field.setAccessible(True)
        homing_checks=0
        for name in ['ArcherArrow','MusketeerProjectile','KingProjectile','TowerPrincessProjectile']:
            data=records.projectile(name)
            for trajectory in [0,1,2]:
                p=Projectile(world,data,0);p.cast(king,10,3500,1000,2250,14500,18000,0,0);target_field.set(p,target)
                path=np.array([[14500+(k*30 if trajectory else 0),18000-(k*25 if trajectory==2 else 0),(k*33)%500] for k in range(60)],np.int32)
                row=np.array([data.speed(),data.gravity(),3500,1000,2250,14500,18000,len(path)],np.int32);actual=np.empty((len(path),7),np.int32);native.cn_combat_homing(row,path,actual)
                for point,out in zip(path,actual):
                    target.getView().setX(int(point[0]));target.getView().setY(int(point[1]));target.getView().setZ(int(point[2]))
                    if not p.isReleased():post.invoke(p)
                    expected=[p.getX(),p.getY(),p.getZ(),p.getAimX(),p.getAimY(),p.getAimZ(),p.isReleased()];np.testing.assert_array_equal(out,expected);homing_checks+=1
        checks['real_table_moving_homing_ticks']=homing_checks
        # Source scenario for all 30 independently flying Arrows; JVM FixedMath and BattleRandom
        # derive reference positions directly from the pinned castProjectiles implementation.
        card=records.card('Arrows');data=records.projectile(str(card.projectile()));Random=J('org.crforge.core.battle.BattleRandom')
        count=int(card.multipleProjectiles());waves=int(card.projectileWaves());radius=int(card.radius());pr=int(data.radius());interval=int(card.projectileIntervalMs());wave_interval=int(card.projectileWaveIntervalMs())
        for seed in [0,1,-1,991,2147483647]:
            out=np.empty((count*waves,8),np.int32);native.cn_combat_arrows(seed,radius,pr,count,waves,interval,wave_interval,out);random=Random(seed);expected=[]
            for wave in range(waves):
                for k in range(count):
                    rx,ry=14500,18000
                    if k:
                        vec=jpype.JArray(jpype.JInt)([radius-pr*60//100,0]);Math.rotate1024(vec,(360//(count-1))*(k-1));rx+=int(vec[0]);ry+=int(vec[1])
                    vec=jpype.JArray(jpype.JInt)([0,pr*60//100]);Math.rotate1024(vec,random.next(359));vec[0]+=rx-14500;vec[1]+=ry-18000
                    if int(vec[0])**2+int(vec[1])**2>(radius*90//100)**2:Math.normalize(vec,radius*90//100)
                    ox,oy=map(int,vec);expected.append([9000+(ox>>2),1000+oy,2250,14500+ox,18000+oy,rx,ry,wave*wave_interval+k*interval])
            np.testing.assert_array_equal(out,np.array(expected,np.int32))
        checks['arrows_independent_launches']=5*count*waves
    executable=output/'source-scenarios'
    subprocess.run(['clang++','-std=c++20','-O2',str(HERE/'test_scenarios.cpp'),'-o',str(executable)],check=True)
    scenario=subprocess.run([str(executable)],check=True,capture_output=True,text=True)
    (output/'source-scenarios.log').write_text(scenario.stdout)
    checks['native_source_scenarios']='passed'
    summary={'status':'passed','engine_commit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','checks':checks,'full_task_parity':False,'header_sha256':hashlib.sha256((HERE/'combat.hpp').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
