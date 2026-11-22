"""Observer-only request/run refusals and actual publicly placed card counts.

Never supplies the other seat's hand or RNG to an actor. The supervisor passes
only its public enemy_play_count, which counts this actor's visibly placed plays.
Counters are decision samples/lower bounds where the ABI exposes only the last
refusal in a decision, not an invented count of all internal command events.
"""
from collections import Counter
import numpy as np

VERSION='clash-visible-execution-observer-v1'

class DecisionCounters:
 def __init__(self):
  self.counts=Counter();self.request_codes=Counter();self.run_codes=Counter();self.attempted_cards=Counter();self.placed_cards=Counter();self.pending_at_end=0
 def observe(self,before,mask,action,after,*,own_placed_before,own_placed_after,own_placed_cards=()):
  if int(after[54])<=int(before[54]):self.counts['nonadvancing_observation_pairs_skipped']+=1;return
  self.counts['decisions']+=1
  if not 0<=action<2307 or not mask[action]:raise ValueError('observer saw illegal chosen action')
  if action==0:
   self.counts['wait_actions']+=1
   self.counts['forced_wait_actions' if np.count_nonzero(mask)==1 else 'voluntary_wait_actions']+=1
  else:
   self.counts['nonwait_action_attempts']+=1
   if action<2305:self.attempted_cards[str(int(before[7+(action-1)//576]))]+=1
  request=int(after[62]);run=int(after[63]);advanced=int(after[0])-int(before[0])
  if request:self.counts['request_refusal_decisions']+=1;self.request_codes[str(request)]+=1
  if advanced>0:
   self.counts['advancing_decisions']+=1;self.counts['advanced_ticks']+=advanced
   if run:self.counts['run_refusal_advancing_decisions']+=1;self.run_codes[str(run)]+=1
   placed=int(own_placed_after)-int(own_placed_before)
   if placed<0:self.counts['public_play_count_reset_boundaries']+=1
   else:
    self.counts['actual_publicly_placed_cards']+=placed
    for card in own_placed_cards[:placed]:self.placed_cards[str(int(card))]+=1
  elif run:self.counts['stale_run_refusal_nonadvancing_samples']+=1
  self.pending_at_end=int(after[53])
 def summary(self):
  keys=['decisions','wait_actions','forced_wait_actions','voluntary_wait_actions','nonwait_action_attempts','request_refusal_decisions','run_refusal_advancing_decisions','advancing_decisions','advanced_ticks','actual_publicly_placed_cards','public_play_count_reset_boundaries','nonadvancing_observation_pairs_skipped','stale_run_refusal_nonadvancing_samples']
  return {'version':VERSION,**{k:self.counts[k] for k in keys},'request_refusal_codes':dict(self.request_codes),'run_refusal_codes':dict(self.run_codes),'attempted_card_ids':dict(self.attempted_cards),'placed_card_ids':dict(self.placed_cards),'pending_commands_at_end':self.pending_at_end,'interpretation':'last_run_refusal counts advancing decision samples, a lower bound on internal failed executions. Actual placed cards use the other seat public enemy_play_count only. Requests may remain pending at natural end.'}

def visible_own_plays(before_other,after_other):
 """Whitelist public history fields only; the actor never sees other-seat arrays."""
 old,new=int(before_other[114]),int(after_other[114]);delta=max(0,new-old)
 return old,new,[int(after_other[116+9*i]) for i in range(min(delta,8))]
