"""Complete paired-map scoring, cluster uncertainty and calibrated optional anchors."""
import numpy as np
from .contract import digest

def by_map(rows):
 if not rows:raise ValueError('No games')
 from integrations.generals.environment import RULESET
 cap=RULESET['truncation']
 opponents={r['opponent'] for r in rows};maps={r['map_index'] for r in rows};seen=set();values={}
 for r in rows:
  k=(r['opponent'],r['map_index'],r['seat'])
  if k in seen or r['seat'] not in (0,1) or r['score'] not in (0.,.5,1.):raise ValueError('Invalid/duplicate game')
  if bool(r['natural'])==bool(r['truncated']):raise ValueError('Natural outcomes and caps must be distinct')
  if r['truncated'] and r['score']!=.5:raise ValueError('Cap cannot be land-adjudicated')
  if not 1<=r['length']<=cap or (r['truncated'] and r['length']!=cap):raise ValueError('Wrong game boundary')
  if bool(r['capture_win'])!=(r['score']==1) or bool(r['capture_loss'])!=(r['score']==0):raise ValueError('Capture diagnostics differ from outcome')
  seen.add(k);values.setdefault(r['map_index'],[]).append(r['score'])
 if len(seen)!=len(opponents)*len(maps)*2:raise ValueError('Incomplete paired-seat/opponent schedule')
 return {i:float(np.mean(v)) for i,v in values.items()}

def summary(rows,replicates=2000):
 values=by_map(rows);v=np.asarray([values[i] for i in sorted(values)]);rng=np.random.default_rng(83)
 boots=v[rng.integers(len(v),size=(replicates,len(v)))].mean(1)
 return dict(quality=float(v.mean()),quality_ci95=np.quantile(boots,[.025,.975]).tolist(),games=len(rows),unique_maps=len(v),
  captures_won=sum(r['capture_win'] for r in rows),captures_lost=sum(r['capture_loss'] for r in rows),
  cap_draws=sum(r['truncated'] for r in rows),
  by_opponent={p:float(np.mean([r['score'] for r in rows if r['opponent']==p])) for p in sorted({r['opponent'] for r in rows})},
  by_seat={str(s):float(np.mean([r['score'] for r in rows if r['seat']==s])) for s in (0,1)},
  aggregation='Equal opponent and seat weights; bootstrap clusters entire maps, preserving seat/opponent pairing')

def calibration_gate(starter_rows,reference_rows,minimum_gap=.05,replicates=2000):
 if {(r['opponent'],r['map_index'],r['seat']) for r in starter_rows}!={(r['opponent'],r['map_index'],r['seat']) for r in reference_rows}:raise ValueError('Anchor schedules differ')
 a=by_map(starter_rows);b=by_map(reference_rows);delta=np.array([b[i]-a[i] for i in sorted(a)])
 rng=np.random.default_rng(73);boots=delta[rng.integers(len(delta),size=(replicates,len(delta)))].mean(1);ci=np.quantile(boots,[.025,.975]).tolist()
 return dict(qualified=float(delta.mean())>=minimum_gap and ci[0]>0,gap=float(delta.mean()),gap_ci95=ci,minimum_gap=minimum_gap,
  starter_quality=float(np.mean(list(a.values()))),reference_quality=float(np.mean(list(b.values()))))

def multi_seed_gate(starter_parts,reference_parts,minimum_gap=.05,replicates=2000):
 if len(starter_parts)<3 or len(starter_parts)!=len(reference_parts):raise ValueError('At least three paired training seeds required')
 keyed=[by_map(r) for r in starter_parts+reference_parts];indices=sorted(keyed[0]);n=len(starter_parts)
 schedules=[{(r['opponent'],r['map_index'],r['seat']) for r in rows} for rows in starter_parts+reference_parts]
 if any(s!=schedules[0] for s in schedules):raise ValueError('Anchor schedules differ')
 a=np.array([[keyed[s][i] for i in indices] for s in range(n)]);b=np.array([[keyed[n+s][i] for i in indices] for s in range(n)])
 rng=np.random.default_rng(731);boots=[]
 for _ in range(replicates):
  seeds=rng.integers(n,size=n);maps=rng.integers(len(indices),size=len(indices));boots.append(float((b[seeds][:,maps]-a[seeds][:,maps]).mean()))
 ci=np.quantile(boots,[.025,.975]).tolist();gaps=(b-a).mean(1)
 return dict(qualified=float(gaps.mean())>=minimum_gap and ci[0]>0 and bool(np.all(gaps>0)),starter_quality=float(a.mean()),
  reference_quality=float(b.mean()),gap=float(gaps.mean()),gap_ci95=ci,minimum_gap=minimum_gap,
  paired_training_seeds=n,seed_gaps=gaps.tolist())

ELO_LOW,ELO_HIGH=-3000.,3000.

def _fit_elo(points,games,ratings,iterations=60):
 """Maximum-likelihood Elo against opponents of known rating, vectorized over rows of (points, games)."""
 c=np.log(10)/400;x=np.full(points.shape[0],float(np.average(ratings)))
 for _ in range(iterations):
  p=1/(1+np.exp(-c*(x[:,None]-ratings[None,:])))
  gradient=c*(points-games*p).sum(1);information=c*c*(games*p*(1-p)).sum(1)
  x=np.clip(x+np.clip(gradient/np.maximum(information,1e-12),-400,400),ELO_LOW,ELO_HIGH)
 return x,information

def panel_elo(rows,ratings,replicates=2000):
 """The candidate's Elo on the panel's ladder: one rating fitted to its games against the rated opponents.

 `ratings` is {opponent: Elo} from the panel's calibration (ladder.py); unrated opponents are left out.
 A cap draw counts as half a win. The interval resamples whole maps, as the score's does. At either
 bound the candidate won or lost nearly everything, and the value is a bound rather than an estimate.
 """
 names=sorted({r['opponent'] for r in rows}&set(ratings or {}))
 if not names:return None
 maps=sorted({r['map_index'] for r in rows});column={n:k for k,n in enumerate(names)};row={m:k for k,m in enumerate(maps)}
 points=np.zeros((len(maps),len(names)));games=np.zeros_like(points)
 for r in rows:
  if r['opponent'] in column:
   games[row[r['map_index']],column[r['opponent']]]+=1;points[row[r['map_index']],column[r['opponent']]]+=r['score']
 known=np.asarray([ratings[n] for n in names],float)
 estimate,information=_fit_elo(points.sum(0,keepdims=True),games.sum(0,keepdims=True),known)
 picks=np.random.default_rng(83).integers(len(maps),size=(replicates,len(maps)))
 boots,_=_fit_elo(points[picks].sum(1),games[picks].sum(1),known);value=float(estimate[0])
 return dict(elo=value,ci95=np.quantile(boots,[.025,.975]).tolist(),se=float(1/np.sqrt(max(information[0],1e-12))),
  at_bound=value in (ELO_LOW,ELO_HIGH),opponents=names,
  note='On the panel ladder (its anchor is 0). The panel ratings carry their own calibration uncertainty.')

def grade(rows,anchors=None,*,development=False,ratings=None):
 scores=summary(rows);anchors=anchors or {};quality=scores['quality']
 result=dict(status='smoke_only' if development else 'complete',reward=quality,panel_points=100*quality,**scores,
  normalized_quality=None,anchor_status='not_qualified')
 rated=panel_elo(rows,ratings)
 if rated:result['panel_elo']=rated
 if anchors.get('qualified'):
  gap=anchors['reference_quality']-anchors['starter_quality']
  if gap<anchors.get('minimum_gap',.05):raise ValueError('Invalid anchor gap')
  result.update(normalized_quality=(quality-anchors['starter_quality'])/gap,anchor_status='qualified',anchor_commitment=digest(anchors))
 return result
