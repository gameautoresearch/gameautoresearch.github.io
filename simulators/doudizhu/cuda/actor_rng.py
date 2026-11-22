"""Host reference for the exact device counter RNG; never a policy input."""
import numpy as np


def actor_seeds(rng,n):
    high=np.iinfo(np.uint64).max
    if isinstance(rng,dict):
        return np.array([rng[i].integers(0,high,dtype=np.uint64) for i in range(n)],dtype=np.uint64)
    return rng.integers(0,high,size=n,dtype=np.uint64)


def words(seeds,turn,domain,nonce=0):
    with np.errstate(over='ignore'):
        x=np.asarray(seeds,dtype=np.uint64)^np.uint64(turn)*np.uint64(0xd2b74407b1ce6e93)^np.uint64(domain)*np.uint64(0xca5a826395121157)^np.asarray(nonce,dtype=np.uint64)
        x=x+np.uint64(0x9e3779b97f4a7c15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        x=x^(x>>np.uint64(31))
        return x>>np.uint64(32)


def uniform(seeds,turn,counts):
    seeds=np.asarray(seeds,dtype=np.uint64)
    bounds=np.asarray(counts,dtype=np.uint32)
    if np.any(bounds==0):raise ValueError('Positive legal counts required')
    with np.errstate(over='ignore'):
        threshold=(np.uint32(0)-bounds)%bounds
        product=words(seeds,turn,2)*bounds.astype(np.uint64)
        nonce=np.zeros(len(bounds),dtype=np.uint64)
        reject=product.astype(np.uint32)<threshold
        while np.any(reject):
            nonce[reject]+=1
            product[reject]=words(seeds[reject],turn,2,nonce[reject])*bounds[reject].astype(np.uint64)
            reject=product.astype(np.uint32)<threshold
    return (product>>np.uint64(32)).astype(np.int32)


def explore(seeds,turn,counts,greedy,epsilon):
    choices=greedy.copy()
    mask=words(seeds,turn,1)<min(1<<32,int(epsilon*(1<<32)))
    if np.any(mask):choices[mask]=uniform(np.asarray(seeds)[mask],turn,np.asarray(counts)[mask])
    return choices


class CounterDraw:
    def __init__(self,value):self.value=int(value)
    def __lt__(self,epsilon):return self.value<min(1<<32,int(epsilon*(1<<32)))


class CounterGameRNG:
    def __init__(self,seed,turn):self.seed=int(seed);self.turn=turn
    def word(self,domain,nonce=0):
        # Scalar bridge for the original object-based actor. NumPy one-element
        # arrays make an unfairly slow baseline and are unnecessary here.
        mask=(1<<64)-1
        x=(self.seed^(self.turn*0xd2b74407b1ce6e93&mask)^(domain*0xca5a826395121157&mask)^nonce)
        x=(x+0x9e3779b97f4a7c15)&mask
        x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
        x=((x^(x>>27))*0x94d049bb133111eb)&mask
        return (x^(x>>31))>>32
    def random(self):return CounterDraw(self.word(1))
    def integers(self,high):
        high=int(high)
        if not 0<high<1<<32:raise ValueError('Counter bound must fit positive uint32')
        threshold=((1<<32)-high)%high
        nonce=0
        while True:
            product=self.word(2,nonce)*high
            if product&0xffffffff>=threshold:return product>>32
            nonce+=1
