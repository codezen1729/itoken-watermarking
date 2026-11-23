"""Frozen evaluation endpoints; gold alignment is isolated from both detectors."""
from itertools import permutations
from packets import ROOT, ARMS, answers, read, save, sources

def interval(text,quote):
    if not quote:return None
    a=text.find(quote)
    if a<0:raise ValueError('Alignment quote missing')
    if text.find(quote,a+1)>=0:return None
    return (a,a+len(quote))

def assign(text,quotes,anchors):
    regions=[interval(text,a) for a in anchors];result=[]
    for q in quotes:
        loc=interval(text,q);hits=[]
        for j,r in enumerate(regions):
            if loc is None or r is None:continue
            shared=min(loc[1],r[1])-max(loc[0],r[0])
            if shared>0 and shared>=0.5*min(r[1]-r[0],loc[1]-loc[0]):hits.append(j)
        result.append(hits[0] if len(hits)==1 else None)
    return result

def align(r,u,corpus):
    arm=r['arm'];v=u['variants'][arm];batch=(ARMS.index(arm)-u['unit'])%3;uid=v['id']
    original=v['text'];new=answers(f'word{batch}')[uid]['text']
    orig=[corpus[k] for k in v['selected']]
    gold=[]
    for judge in r['judges']:
        if judge['status'] not in ('changed','preserved','uncertain'):raise ValueError('Fidelity status')
        mapping={a['index']:a['quote'] for a in judge['links']}
        if set(mapping)!=set(range(6)) or len(judge['links'])!=6:raise ValueError('Alignment length')
        if any(q is not None and (not isinstance(q,str) or not q or q not in new) for q in mapping.values()):raise ValueError('Alignment quotation')
        gold.append([mapping[i] for i in range(6)])
    result={'plan':[x[0] if len(x)==1 else None for x in (u['pool_ids'][k] for k in v['selected'])]}
    for name,rd in r['readings'].items():
        if name.startswith('orig'):label=assign(original,rd['quotes'],orig)
        else:
            votes=[assign(new,rd['quotes'],g) for g in gold]
            label=[a if a==b else None for a,b in zip(*votes)]
        result[name]=[rd['ids'][label.index(i)] if label.count(i)==1 else None for i in range(6)]
    return result

def pair_rates(pairs):
    # Each planned source contributes once; ambiguous/missing matches remain in denominator.
    a=b=m=wrong=total=0
    for left,right in pairs:
        for x,y in zip(left,right,strict=True):
            a+=1
            if x is not None and y is not None:
                b+=1;m+=x==y
        for i,j in permutations(range(len(left)),2):
            x,y=left[i],right[j]
            if x is not None and y is not None:
                total+=1;wrong+=x==y
    return dict(exact=m,aligned=b,attempted=a,wrong_step_collisions=wrong,wrong_step_comparisons=total)

def both(row,stage,mode):
    return all(row['readings'][stage+j]['test'][mode]['positive'] is True for j in 'ab')

def summarize():
    rows=read(ROOT/'results.json');units=read(ROOT/'generation.json')['units'];corpus=sources()
    mapped=[(r,align(r,units[r['unit']],corpus[r['unit']]['passages'])) for r in rows]
    out={'detection':{},'agreement':{},'challenge':{}}
    for arm,mode in [('control','exact'),('control','tolerant'),('exact','exact'),('tolerant','tolerant')]:
        ss=[r for r in rows if r['arm']==arm];f=[r for r in ss if r['faithful']]
        el=[r for r in f if both(r,'orig',mode)]
        out['detection'][arm+'_'+mode]=dict(n=len(ss),plan=sum(r['plan'][mode]['positive'] is True for r in ss),
              original_both=sum(both(r,'orig',mode) for r in ss),rewrite_both=sum(both(r,'new',mode) for r in ss),
              faithful=len(f),faithful_original=sum(both(r,'orig',mode) for r in f),faithful_rewrite=sum(both(r,'new',mode) for r in f),
              survival=[sum(both(r,'new',mode) for r in el),len(el)],
              individual={n:sum(r['readings'][n]['test'][mode]['positive'] is True for r in ss) for n in ('origa','origb','newa','newb')})
    for scope in ['all','faithful']:
        mm=[m for r,m in mapped if scope=='all' or r['faithful']]
        out['agreement'][scope]={x+'_'+y:pair_rates([(m[x],m[y]) for m in mm]) for x,y in [('origa','origb'),('newa','newb'),('origa','newb'),('origb','newa'),('plan','origa'),('plan','origb'),('plan','newa'),('plan','newb')]}
    fs=read(ROOT/'foil_results.json')
    for t in ['all']+sorted({f['type'] for f in fs}):
        selected=[f for f in fs if t=='all' or f['type']==t]
        eq=lambda x,y:len(x)==len(y)==1 and x==y
        out['challenge'][t]=dict(n=len(selected),all_single=sum(all(len(f[z])==1 for z in ('original','paraphrase','foil')) for f in selected),
           paraphrase_exact=sum(eq(f['original'],f['paraphrase']) for f in selected),foil_collision=sum(eq(f['original'],f['foil']) for f in selected))
    return out
if __name__=='__main__':save(ROOT/'summary.json',summarize())
