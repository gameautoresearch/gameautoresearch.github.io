"""Independent public-information tempo, economy and scaling policies.

Only supplied numeric/card observations, legal masks and the current static card
catalog are consumed. No Game, state_json, opponent hidden boards or RNG access.
Unrepresented Choose-One option descriptions remain an explicit encoder gap.
"""
from functools import lru_cache
import json
import numpy as np
import taverngym as tg

L = dict(tg.layout())
MF, IF = L['MINION_F'], L['ITEM_F']
BF = L['GLOBAL_F'] + L['N_QUESTS'] * L['QUEST_F'] + L['PENDING_KINDS']
HF, SF, CF = BF + 7 * MF, BF + 7 * MF + 10 * IF, BF + 7 * MF + 18 * IF
BC = 1 + L['N_POWERS'] + 2 + 3 * L['N_QUESTS'] + L['N_SECRETS']
HC, SC, CC = BC + 7, BC + 17, BC + 25
STYLES = ('tempo', 'economy', 'synergy')

@lru_cache(maxsize=1)
def catalog():
    # Observation IDs are CardId+1; raw cid is never used as an observation index.
    return {int(row['cid']) + 1: row for row in json.loads(tg.catalog_json())}

def meta(card):
    return catalog().get(int(card), {})

@lru_cache(maxsize=None)
def text(card):
    m=meta(card); value=str(m.get('text',''))
    if m.get('hero_power_id'):
        power=next((row for row in catalog().values() if row['id']==m['hero_power_id']),{})
        value += ' '+str(power.get('text',''))
    return value.lower()

def minion_value(row):
    if not row[0]: return 0.
    attack, health = max(0., row[1]*10), max(0., row[2]*10)
    value = .9*attack + health + .25*np.sqrt(attack*health) + row[4]*6 + row[5]*4
    value += 5*row[9] + 4*row[11] + 4*row[12] + 6*row[13] + 3*row[10] + 6*row[17]
    return float(value)

def position_priority(row):
    # Attack order: cleave/venomous/shielded attackers first, fragile support later.
    return float(row[1]*10 + 18*row[17] + 12*row[12] + 7*row[9] + 4*row[10] - 2*row[8])

def act_visible(features, cards, action_mask, style='tempo'):
    if style not in STYLES: raise ValueError('Unknown public script style')
    f, c, mask = np.asarray(features), np.asarray(cards), np.asarray(action_mask, dtype=bool)
    if f.shape != (3227,) or c.shape != (124,) or mask.shape != (245,): raise ValueError('Pinned visible interface required')
    legal = np.flatnonzero(mask)
    if not len(legal): raise ValueError('No action requested for a waiting seat')
    board = f[BF:HF].reshape(7, MF)
    hand = f[HF:SF].reshape(10, IF)
    shop = f[SF:CF].reshape(8, IF)
    choices = f[CF:CF+16*IF].reshape(16, IF)
    board_values = np.array([minion_value(m) for m in board])
    n_board = int(np.count_nonzero(board[:,0])); n_hand = int(np.count_nonzero(hand[:,0]))
    turn, tier, gold, health = f[0]*20, f[10]*6, f[7]*10, f[5]*40 + f[6]*20
    owned = np.concatenate((c[BC:BC+7], c[HC:HC+10])); races = board[:,20:31].sum(0)
    scores = np.full(245, -1000., dtype=np.float64); scores[0] = 0.
    def item_value(row, card):
        m = meta(card); t = text(card)
        if row[1] > .5:
            val = minion_value(row[9:])
            duplicates = int(np.count_nonzero(owned == card)) if card else 0
            val += 5*min(duplicates, 1) + 15*(duplicates >= 2)
            if style == 'synergy':
                val += 3*float(np.dot(row[29:40], races))
                val += 5*sum(word in t for word in ('end of your turn', 'whenever', 'battlecry', 'rally', 'deathrattle'))
            if style == 'economy': val += 5*sum(word in t for word in ('gold', 'refresh', 'tavern spell', 'sell'))
            return float(val)
        val = 3 + float(m.get('tier',0))
        val += min(n_board, 4)*sum(word in t for word in ('give', 'gain', 'blood gem'))
        if style == 'synergy': val += 3*sum(word in t for word in ('permanent', 'each', 'end of'))
        if style == 'economy': val += 4*sum(word in t for word in ('gold', 'discount', 'refresh'))
        return float(val)
    pending_kind = int(np.argmax(f[L['GLOBAL_F']+6:BF]))
    choice_ids = legal[(legal >= 30) & (legal < 46)]
    if len(choice_ids):
        harmful = any(word in text(c[4]) for word in ('destroy', 'remove', 'consume'))
        for a in choice_ids:
            i = int(a-30); card = c[CC+i]
            value = item_value(choices[i], card)
            if pending_kind == 4 and harmful: value = -value
            if pending_kind in (1,5):
                t=text(card); value += 3*sum(word in t for word in (('attack','health','combat') if style=='tempo' else ('gold','refresh','tavern') if style=='economy' else ('battlecry','friendly','each')))
            if pending_kind == 3: value += int(i == (1 if style=='economy' else 0))*.1
            scores[a] = value - i*1e-6
        return int(legal[np.argmax(scores[legal])])
    scores[7] = -5.  # Cancel is supported, but never silently rescues a failed action.
    shop_values = np.array([item_value(x, card) if x[0] else -1000 for x,card in zip(shop,c[SC:SC+8])])
    hand_values = np.array([item_value(x, card) if x[0] else -1000 for x,card in zip(hand,c[HC:HC+10])])
    weakest = int(np.argmin(np.where(board[:,0]>.5, board_values, np.inf))) if n_board else 0
    best_hand = max([hand_values[i] for i in range(10) if hand[i,0] and hand[i,1]>.5] or [-1000])
    legal_buys = [int(a-8) for a in legal if 8 <= a < 16]
    best_shop = max([shop_values[i] for i in legal_buys if shop[i,1]>.5] or [-1000])
    for a in legal:
        if 8 <= a < 16:
            i=int(a-8); value=shop_values[i]
            minion=shop[i,1]>.5; dup=int(np.count_nonzero(owned==c[SC+i]))
            if minion and n_board + np.count_nonzero(hand[:,1]>.5) >= 7 and dup<2 and value<board_values[weakest]+4: continue
            scores[a]=4 + value*.15 - shop[i,2]*.3
        elif 16 <= a < 23:
            i=int(a-16)
            if n_board == 7 and i==weakest and max(best_hand,best_shop)>board_values[i]+4:
                scores[a]=9 + (max(best_hand,best_shop)-board_values[i])*.05
        elif 23 <= a < 30: scores[a]=3.5 + (1. if style=='synergy' else 0.)
        elif 46 <= a < 126:
            h,pos=divmod(int(a-46),8)
            desired=int(np.count_nonzero([position_priority(m)>position_priority(hand[h,9:]) for m in board[:n_board]]))
            scores[a]=15 + hand_values[h]*.12 - abs(pos-desired)*.1
        elif 126 <= a < 196:
            h,b=divmod(int(a-126),7);scores[a]=16 + hand_values[h]*.12 + .02*board_values[b]
        elif 196 <= a < 245 and n_board>1:
            fr,to=divmod(int(a-196),7);order=sorted(range(n_board),key=lambda j:(-position_priority(board[j]),int(c[BC+j]),j))
            mismatch=next((j for j in range(n_board) if order[j]!=j),None)
            if mismatch is not None and fr==order[mismatch] and to==mismatch:scores[a]=.3
    upgrade_cost=f[11]*10
    min_turn=(2*tier+1 if style=='tempo' else 2*tier-1 if style=='economy' else 2*tier)
    if turn>=min_turn and (n_board>=min(5,int(turn)+1) or upgrade_cost<=2) and health>8:
        scores[3]=10 if style=='economy' else 6.5
    for slot in range(2):
        cost=f[17+3*slot]*5
        scores[4+slot]=2.5 + (1 if style=='economy' else 0) - .1*cost
    if n_board<7 and health>12:scores[6]=3.
    promising=bool(np.max(shop_values)>max(8.,board_values[weakest]+6 if n_board==7 else 8.))
    if not legal_buys and promising and f[15]<.5:scores[2]=.2
    elif not promising and f[15]>.5:scores[2]=.1
    if gold>=max(1.,f[12]*2) and health>8 and n_hand<8:scores[1]=.5
    # Every supported action can be selected when uniquely legal, including unusual pending/cancel states.
    return int(legal[np.argmax(scores[legal])])

class VisibleScript:
    def __init__(self, style='tempo'):
        if style not in STYLES: raise ValueError(style)
        self.style=style
    def remote_act(self, obs, indices):
        indices=np.asarray(indices,dtype=np.int64)
        actions=np.array([act_visible(obs['features'][i],obs['cards'][i],obs['action_mask'][i],self.style) for i in indices],np.int64)
        return actions,np.zeros(len(indices),np.float32),np.zeros(len(indices),np.float32)

class ScriptMixture:
    """Collect-compatible opponent router for all8 controlled seats.

    `indices` addresses rows in obs. Compact observations must supply original
    flattened slot IDs in obs['slots']; it never infers seats from packed row order.
    """
    def __init__(self, styles_by_seat=('tempo','economy','synergy','tempo','economy','synergy','tempo','economy'), rotate_by_lobby=False):
        if len(styles_by_seat)!=8 or any(s not in STYLES for s in styles_by_seat):raise ValueError('Eight valid script styles required')
        self.styles=tuple(styles_by_seat);self.rotate_by_lobby=bool(rotate_by_lobby)
    def remote_act(self,obs,indices):
        indices=np.asarray(indices,np.int64);slots=np.asarray(obs.get('slots',np.arange(len(obs['features']))))[indices]
        actions=[]
        for row,slot in zip(indices,slots):
            seat=int(slot)%8
            if self.rotate_by_lobby:seat=(seat+int(slot)//8)%8
            actions.append(act_visible(obs['features'][row],obs['cards'][row],obs['action_mask'][row],self.styles[seat]))
        return np.asarray(actions,np.int64),np.zeros(len(indices),np.float32),np.zeros(len(indices),np.float32)
