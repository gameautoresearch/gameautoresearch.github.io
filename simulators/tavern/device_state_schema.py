"""Project device checkpoint metadata onto the unchanged native Episode schema.

The native Episode serializes game/index/finished. Device checkpoints additionally
retain their batch seed and controlled seats so they can reset and resume alone.
The complete Game object (including all RNG state) is never normalized or changed.
"""


def native_episode(state,*,seed,controlled):
    expected={'game','index','finished','seed','controlled','k'}
    if set(state)!=expected:raise ValueError('unexpected device checkpoint schema')
    seats=list(controlled);k=len(seats)
    if not k or k>8 or state['seed']!=seed or state['k']!=k:
        raise ValueError('device checkpoint batch metadata mismatch')
    if state['controlled']!=seats+[0]*(8-k):
        raise ValueError('device checkpoint controlled-seat metadata mismatch')
    if len(state['finished'])!=8 or any(state['finished'][k:]):
        raise ValueError('device checkpoint unused finished seats changed')
    return {'game':state['game'],'index':state['index'],'finished':state['finished'][:k]}
