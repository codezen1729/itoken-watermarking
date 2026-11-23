"""Run the frozen experiment with external blind semantic readings."""
import argparse
import hashlib
import math
import random
import secrets
import sys
from pathlib import Path

from packets import ARMS, ROOT, answers, batches, read, save, sources, validated
from watermark import select, test

def identify(enc,quant,steps):
    from quantize import embed_steps
    from canonical import canonical_step
    return [('1' if canonical_step(s)['sign']=='avoid' else '0')+c[:5] for s,c in zip(steps,quant.codes(embed_steps(enc,steps)),strict=True)]

def engines():
    from encoder import SentenceEncoder
    from quantize import Hash
    return SentenceEncoder(),Hash()

def freeze():
    names=['PROTOCOL.md','sources_a.json','sources_b.json','foils.json','packets.py','encoder.py','quantize.py','canonical.py','prefix_calibration.json','prefix_calibration.py','calibration.json','weights.json','public_axes.npz','watermark.py','trial.py','endpoints.py','../open_study/study.py','../real_model.py']
    for i in range(8):
        answers(f'pool{i}')
        names.extend(f'calls/pool{i}/{f}.json' for f in ('in','out'))
    data={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in names}
    dest=ROOT/'freeze.json'
    if dest.exists() and read(dest)!=data:raise ValueError('Frozen files changed')
    save(dest,data)

def model_weights():
    sys.path.insert(0,str(ROOT.parent/'open_study'))
    from study import Encoder
    checkpoint=ROOT.parents[2]/'work/models/qwen2.5-0.5b-instruct-4bit'
    lm=Encoder(checkpoint)
    save(ROOT/'weights.json',{'model':lm.lm.metadata,'weights':[[lm.weight(s['problem'],t) for t in s['passages']] for s in sources()]})

def generate():
    freeze()
    dest=ROOT/'generation.json'
    if dest.exists(): raise ValueError('Draw already completed')
    units=sources(); keypath=ROOT/'keys.json'
    if not keypath.exists():save(keypath,[[secrets.token_hex(32) for _ in range(200)] for _ in units])
    keys=read(keypath); weights=read(ROOT/'weights.json')['weights'];enc,q=engines();rows=[]
    for i,s in enumerate(units):
        readings=answers(f'pool{i}');p=s['problem'];rng=random.Random(67482+i)
        noise=[-math.log(-math.log(rng.random())) for _ in s['passages']]
        base=[w['log_weight']+n for w,n in zip(weights[i],noise,strict=True)]
        planned=[identify(enc,q,validated(readings[f's{j}'],t)) for j,t in enumerate(s['passages'])]
        variants={}
        for arm in ARMS:
            chosen=select(planned,base,keys[i][0],p,arm);segments=[s['passages'][j] for j in chosen]
            variants[arm]={'selected':chosen,'id':secrets.token_hex(8),'text':' '.join(segments[:3])+'\n\n'+' '.join(segments[3:])}
        rows.append(dict(unit=i,problem=p,pool_ids=planned,noise=noise,variants=variants))
    save(dest,{'units':rows})

def evaluate():
    enc,q=engines();bank=read(ROOT/'keys.json');results=[]
    for batch,pairs in batches():
        for unit,arm in pairs:
            v=unit['variants'][arm];uid=v['id'];p=unit['problem'];i=unit['unit']
            planned=sum([unit['pool_ids'][j] for j in v['selected']],[])
            judges=[answers(f'judge{j}{batch}')[uid] for j in 'ab']
            row=dict(unit=i,arm=arm,id=uid,planned_ids=planned,judges=judges,faithful=all(j['status']=='preserved' for j in judges),readings={})
            modes=('exact','tolerant') if arm=='control' else (arm,)
            row['plan']={m:test(planned,bank[i],p,m) for m in modes}
            for stage in ('orig','new'):
                for reader in 'ab':
                    label=f'{stage}{reader}{batch}'
                    source=next(d['text'] for d in read(ROOT/'calls'/label/'in.json')['documents'] if d['id']==uid)
                    steps=validated(answers(label)[uid],source);ids=identify(enc,q,steps)
                    row['readings'][stage+reader]={'ids':ids,'quotes':[s['quote'] for s in steps],'test':{m:test(ids,bank[i],p,m) for m in modes}}
            results.append(row)
    save(ROOT/'results.json',sorted(results,key=lambda r:(r['unit'],ARMS.index(r['arm']))))
    foils=read(ROOT/'foils.json');foilout=[]
    for i,f in enumerate(foils):
        ids={}
        for n,typ in enumerate(('original','paraphrase','foil')):
            b=(n-i)%3;uid=hashlib.sha256(f'foil-ids-v1/{i}/{typ}'.encode()).hexdigest()[:12]
            ids[typ]=identify(enc,q,validated(answers(f'contrast{b}')[uid],f[typ]))
        foilout.append({'type':f['type'],**ids})
    save(ROOT/'foil_results.json',foilout)
if __name__=='__main__':
    tasks={'weights':model_weights,'freeze':freeze,'generate':generate,'evaluate':evaluate}
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=tasks)
    tasks[parser.parse_args().phase]()
