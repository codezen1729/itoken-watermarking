"""Two frozen keyed statistics over open, reader-generated 6-bit step IDs."""
from collections import Counter
from functools import lru_cache
import hashlib
import hmac
from itertools import combinations
import math

import numpy as np

MODES=('exact','tolerant')
BLOCKS=tuple(tuple((a+d)%6 for d in offsets) for offsets in ((0,1,2),(0,2,3)) for a in range(6))

def features(ids, mode):
    f=Counter()
    for token in ids:
        if len(token)!=6 or set(token)-{'0','1'}: raise ValueError('Expected a signed, 6-bit ID')
        if mode=='exact': f['full/'+token]+=1;continue
        if mode!='tolerant':raise ValueError('Unknown watermark rule')
        for n,indices in enumerate(BLOCKS):
            value=int(''.join(token[j] for j in indices),2)
            for i in range(8): f[f'{n}/{i}']+=(7 if i==value else -1)
    return f

@lru_cache(maxsize=800000)
def coefficient(key,problem,mode,name):
    data='itoken-key-v1\x1f'+problem+'\x1f'+mode+'\x1f'+name
    raw=hmac.digest(bytes.fromhex(key),data.encode(),'sha256')
    u=((int.from_bytes(raw[:8],'big')>>11)+0.5)/2**53
    v=((int.from_bytes(raw[8:16],'big')>>11)+0.5)/2**53
    return math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)

def score_features(f,key,problem,mode):
    norm=math.sqrt(sum(x*x for x in f.values()))
    if not norm:return None
    return math.fsum(v*coefficient(key,problem,mode,n) for n,v in sorted(f.items()) if v)/norm

def test(ids,keys,problem,mode):
    f=features(ids,mode);target=score_features(f,keys[0],problem,mode)
    if target is None:return {'p':None,'positive':None,'score':None}
    p=(1+sum(score_features(f,k,problem,mode)>=target for k in keys[1:]))/len(keys)
    return {'p':p,'positive':p<=.01,'score':target}

def select(pool,utilities,target,problem,arm):
    valid=[j for j,row in enumerate(pool) if row]
    if len(valid)<6: raise ValueError('Insufficient supported candidates')
    sets=np.array(list(combinations(valid,6)))
    objective=np.array(utilities)[sets].mean(axis=1)
    if arm!='control':
        fs=[features(ids,arm) for ids in pool]
        names=sorted({key for f in fs for key in f})
        x=np.asarray([[f[k] for k in names] for f in fs],dtype=float)
        g=np.array([coefficient(target,problem,arm,k) for k in names])
        gram=x@x.T
        denom=np.sqrt(np.maximum(gram[sets[:,:,None],sets[:,None,:]].sum(axis=(1,2)),0))
        z=np.divide((x@g)[sets].sum(axis=1),denom,out=np.zeros(len(sets)),where=denom>0)
        objective=objective+12*z
    return sets[np.argmax(objective)].tolist()
