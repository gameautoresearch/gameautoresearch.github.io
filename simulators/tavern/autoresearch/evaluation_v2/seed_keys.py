"""Pure native seed arithmetic; safe in the no-Torch operator environment."""
MASK=(1<<64)-1
STRIDE=1000003
INV_STRIDE=pow(STRIDE,-1,1<<64)

def sm(x):
    x=(x+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)

def undo_xor(value,shift):
    result=value
    for offset in range(shift,64,shift):result^=value>>offset
    return result&MASK

def inverse_sm(value):
    value=undo_xor(value,31)*pow(0x94d049bb133111eb,-1,1<<64)&MASK
    value=undo_xor(value,27)*pow(0xbf58476d1ce4e5b9,-1,1<<64)&MASK
    return (undo_xor(value,30)-0x9e3779b97f4a7c15)&MASK

def lobby_key(seed,game=0):return int(seed)^sm(game)

def direct_game_key(seed):return inverse_sm(inverse_sm(int(seed)))

def training_overlap(seed,config):
    key=lobby_key(seed)
    return any((((key^sm(g))-config['seed'])*INV_STRIDE)&MASK < config['updates'] for g in range(config['n_games']))
