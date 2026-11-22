"""Fixed8 tactical opponents using only one seat's public arrays.

Inspired by the utility/teacher curriculum in itzik123/ClashRoyaleAi, not a port
of its incompatible engine or policy. No state fork, hidden hand, RNG or search.
Coordinates are already in the observing seat's canonical arena frame.
"""
import json
import numpy as np
from .common import DATA,setup_native

VERSION='fixed8-visible-utility-v2'
COST={'Knight':3,'Archer':3,'Musketeer':4,'Giant':5,'MiniPekka':4,'Fireball':4,'Arrows':3,'Zap':2}

class PublicTeacher:
 def __init__(self,kind='teacher',data=None):
  if kind not in ('teacher','intermediate'):raise ValueError('unknown visible teacher')
  setup_native()
  from evaluate import StaticSchema,frozen_facts,scripted_bot
  self.data=json.loads(DATA.read_text()) if data is None else data
  self.schema=StaticSchema(self.data);self.kind=kind
  facts=frozen_facts(self.data);self.types,self.spells,self.units,_=facts
  self.baseline=scripted_bot(self.data,facts)
  self.names={i:n for i,n in enumerate(self.data['vocabulary'])};self.ids={n:i for i,n in self.names.items()}
  self.unit_cost={f.unit:COST[self.names[c]]/max(1,self.data['cards'][self.names[c]]['count']) for c,f in self.units.items()}
  self.last_reason='initial'

 def _tile(self,mask,slot,x,y):
  legal=np.flatnonzero(mask[1+576*slot:1+576*(slot+1)])
  if not len(legal):return None
  cx=legal%18+.5;cy=legal//18+.5
  k=int(np.argmin((cx-x)**2+(cy-y)**2))
  return int(1+576*slot+legal[k])

 def __call__(self,observation,mask):
  o=np.asarray(observation);mask=np.asarray(mask)
  if o.shape!=(7024,) or mask.shape!=(2307,):raise ValueError('fixed8 public arrays required')
  if not mask[0] or mask[2305:].any():raise ValueError('fixed8 action mask differs')
  h=self.schema.header;e=self.schema.entity;s=self.schema
  def field(n):return int(o[h[n][0]])
  self.last_reason='wait'
  if np.count_nonzero(mask)==1:return 0
  # An intermediate configuration has the same resources/rules but a one-second
  # decision cadence and a smaller tactical repertoire. No elixir handicap.
  if self.kind=='intermediate' and field('tick')%20!=0:return 0
  if field('pending_count'):return 0
  rows=s.entities(o);flags=rows[:,e['flags']]
  visible=((flags&s.flags['alive'])!=0)&((flags&(s.flags['hidden']|s.flags['underground']|s.flags['invisible']))==0)
  troops=rows[visible&(rows[:,e['kind']]==s.kinds['character'])]
  enemy=troops[troops[:,e['side']]==1];own=troops[troops[:,e['side']]==0]
  towers=rows[visible&(rows[:,e['kind']]==s.kinds['tower'])&(rows[:,e['side']]==1)]
  elixir=(field('elixir')-field('elixir_reserved'))/10000
  slots=[i for i in range(4) if mask[1+i*576:1+(i+1)*576].any()]
  hand={i:self.names.get(int(o[h['hand_card'][i]]),'') for i in slots}
  candidates=[]
  def offer(score,slot,x,y,reason):
   a=self._tile(mask,slot,x,y)
   if a is not None:candidates.append((float(score),-a,a,reason))
  if self.kind=='teacher':
   # Keep the qualified standard economy/placement foundation. Tactical utility
   # overrides it only for an explicit visible opportunity. A fresh fixed RNG
   # makes tie-breaking a pure function of this observation, not batch order.
   base=int(self.baseline(o,mask,np.random.default_rng(0)))
   if base:candidates.append((20.,-base,base,'standard-foundation'))
  # Finish exposed towers before other plays. Only visible tower HP is used.
  for slot,name in hand.items():
   fact=self.spells.get(self.ids.get(name))
   if fact is None:continue
   for tower in towers:
    hp=int(tower[e['hp']]);shield=max(0,int(tower[e['shield']]))
    if hp>0 and hp+shield<=fact.tower_damage:
     offer(1000-COST[name],slot,tower[e['x']]/1000,tower[e['y']]/1000,'tower-finish')
  # Spell value: clipped damage as a share of visible unit value, plus kills.
  if self.kind=='teacher' and len(enemy):
   xy=enemy[:,[e['x'],e['y']]].astype(float)/1000
   health=np.where(enemy[:,e['hp']]>=0,enemy[:,e['hp']],enemy[:,e['max_hp']])+np.maximum(enemy[:,e['shield']],0)
   maximum=np.maximum(enemy[:,e['max_hp']],1)
   worth=np.array([self.unit_cost.get(int(v),2.) for v in enemy[:,e['row']]])
   centers=np.unique(np.floor(xy),axis=0)+.5
   # Only public entity positions, with a bounded set of pair midpoints.
   near=xy[np.argsort(xy[:,1],kind='stable')[:12]]
   mid=[(a+b)/2 for i,a in enumerate(near) for b in near[i+1:] if np.linalg.norm(a-b)<=6]
   if mid:centers=np.concatenate((centers,mid))
   for slot,name in hand.items():
    fact=self.spells.get(self.ids.get(name))
    if fact is None:continue
    covered=((centers[:,None,:]-xy[None,:,:])**2).sum(-1)<=(fact.radius/1000)**2
    damage=np.minimum(health,fact.damage)/maximum*worth+.35*(health<=fact.damage)
    values=covered@damage
    k=int(np.argmax(values));threshold=COST[name]*(.9 if elixir>=9 else 1.1)
    if values[k]>=threshold:offer(80+values[k]-COST[name],slot,*centers[k],'spell-value')
  threats=enemy[enemy[:,e['y']]<22000]
  if len(threats):
   lead=threats[np.argmin(threats[:,e['y']])];x=lead[e['x']]/1000;y=lead[e['y']]/1000
   lane=threats[(threats[:,e['x']]<9000)==(x<9)]
   nearby=own[((own[:,e['x']]/1000-x)**2+(own[:,e['y']]/1000-y)**2)<49]
   air=bool(np.any((lane[:,e['flags']]&s.flags['air'])!=0));tank=bool(np.any(lane[:,e['max_hp']]>2500))
   # Do not repeatedly spend the whole hand on an already covered push.
   need=y<12 and (len(nearby)<min(2,len(lane)) or (y<7 and len(nearby)<3))
   if need:
    for slot,name in hand.items():
     if name not in ('Knight','Archer','Musketeer','MiniPekka'):continue
     if air and name not in ('Archer','Musketeer'):continue
     rank=(8 if name=='MiniPekka' and tank else 7 if name=='Musketeer' and air else 5 if name=='Knight' else 4)
     ranged=name in ('Archer','Musketeer');px=np.clip(x+(2.5 if x<9 else -2.5),1,16)
     py=np.clip(y-(4 if ranged else 1.5),3,13.5)
     offer(65+rank-COST[name]-.2*y,slot,px,py,'defense')
  giant_id=self.units[self.ids['Giant']].unit
  giants=own[own[:,e['row']]==giant_id]
  if len(giants) and self.kind=='teacher':
   lead=giants[np.argmax(giants[:,e['y']])];x=lead[e['x']]/1000;y=lead[e['y']]/1000
   support=own[(own[:,e['row']]!=giant_id)&((own[:,e['x']]/1000-x)**2+(own[:,e['y']]/1000-y)**2<36)]
   if not len(support) and elixir>=4:
    for slot,name in hand.items():
     if name in ('Musketeer','Archer','Knight'):offer(35+(3 if name=='Musketeer' else 0),slot,x,np.clip(y-2.5,2,14),'tank-support')
  # Reserve defensive elixir; build a tank push, then cycle when near the cap.
  left,right=field('enemy_left_hp'),field('enemy_right_hp')
  lane=0 if left>0 and (right<=0 or left<=right) else 1;x=3.5 if lane==0 else 14.5
  late=bool(field('overtime'));threshold=9 if self.kind=='teacher' else 9.5
  for slot,name in hand.items():
   if name=='Giant' and elixir>=threshold and not len(giants):offer(25,slot,x,13.5 if late else 2.5,'tank-pressure')
   elif name in ('Knight','Archer','MiniPekka','Musketeer') and elixir>=9.5:
    offer(15-COST[name],slot,x,13.5,'elixir-cycle')
  if not candidates:return 0
  _,_,action,reason=max(candidates);self.last_reason=reason
  if not mask[action]:raise ValueError('teacher generated illegal action')
  return action
