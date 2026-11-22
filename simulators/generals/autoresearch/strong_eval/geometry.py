"""Read-only map-family diagnostics; no hidden coordinates enter policy frames."""
from collections import deque,Counter,defaultdict
import numpy as np

def diagnose(bank,dimensions):
 with np.load(bank,allow_pickle=False) as data:
  mountains=data['mountains'];generals=data['generals'];ownership=data['ownership']
 maps=[]
 if len(dimensions)!=len(mountains):raise ValueError('Map dimension count differs')
 for i,(h,w) in enumerate(dimensions):
  starts=[tuple(np.argwhere(generals[i]&ownership[i,p])[0]) for p in range(2)]
  queue=deque([(starts[0],0)]);seen={starts[0]};distance=None
  while queue:
   (r,c),d=queue.popleft()
   if (r,c)==starts[1]:distance=d;break
   for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
    if 0<=nr<h and 0<=nc<w and not mountains[i,nr,nc] and (nr,nc) not in seen:
     seen.add((nr,nc));queue.append(((nr,nc),d+1))
  if distance is None or distance<17:raise ValueError('Map violates full-task walking separation')
  maps.append(dict(map_index=i,height=h,width=w,area=h*w,walking_separation=distance,distance_family='17-28' if distance<=28 else '29+',area_family='289-360' if h*w<=360 else '361-440' if h*w<=440 else '441-529',shape_family='square' if h==w else 'tall' if h>w else 'wide'))
 return dict(maps=maps,walking_distance_definition='Four-neighbour shortest path over non-mountain cells, including castles, matching the pinned generator.',distance_counts=dict(Counter(x['distance_family'] for x in maps)),area_counts=dict(Counter(x['area_family'] for x in maps)),shape_counts=dict(Counter(x['shape_family'] for x in maps)),coordinates_included=False)

def aggregate(rows,geometry):
 lookup={x['map_index']:x for x in geometry['maps']};groups=defaultdict(list)
 for row in rows:
  for field in ('distance_family','area_family','shape_family'):
   groups[(field,lookup[row['map_index']][field])].append(row)
 return {field:{name:dict(games=len(values),points=100*sum(x['score'] for x in values)/len(values),wins=sum(x['capture_win'] for x in values),losses=sum(x['capture_loss'] for x in values),caps=sum(x['truncated'] for x in values)) for (key,name),values in groups.items() if key==field} for field in ('distance_family','area_family','shape_family')}
