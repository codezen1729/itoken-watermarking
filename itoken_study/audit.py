"""Replay the catalog-free iToken trial from probabilities, discrete IDs, keys and raw calls.

The default numerical replay only requires NumPy. Add --reencode to execute the
frozen encoder and quantizer on every raw source reading, including the foils.
"""
import argparse
import ast
from collections import Counter
from functools import cache, lru_cache
import hashlib
import hmac
from itertools import combinations
import json
import math
from pathlib import Path
import random
import re
import struct

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ('control', 'exact', 'tolerant')
MODES = ('exact', 'tolerant')
NAMES = tuple(f'{b}/{j}' for b in range(12) for j in range(8))


@cache
def data(rel):
    return json.loads((ROOT / rel).read_text())


def require(result, msg):
    if not result:
        raise ValueError(msg)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_freeze():
    entries = data('freeze.json')
    minimum = {'canonical.py','prefix_calibration.json','prefix_calibration.py','PROTOCOL.md','sources_a.json','sources_b.json','foils.json',
               'packets.py','encoder.py','quantize.py','weights.json','calibration.json','public_axes.npz','watermark.py',
               'trial.py','endpoints.py','../open_study/study.py','../real_model.py'}
    for u in range(8):
        minimum.update(f'calls/pool{u}/{name}.json' for name in ('in','out'))
    require(minimum <= set(entries), 'Freeze manifest lacks required inputs')
    for rel, expected in entries.items():
        require(digest(ROOT/rel) == expected, f'Freeze hash differs: {rel}')
    calibration = data('calibration.json')
    require(digest(ROOT/'public_axes.npz') == calibration['quantizer_sha256'], 'Wrong fitted axes')
    for rel, expected in calibration['source_sha256'].items():
        require(digest(ROOT.parent/'bias_study'/rel) == expected, f'Historical input changed: {rel}')
    return digest(ROOT/'freeze.json')


@cache
def prompts():
    return {e.targets[0].id: e.value.value for e in ast.parse((ROOT/'packets.py').read_text()).body
            if isinstance(e,ast.Assign) and isinstance(e.targets[0],ast.Name)
            and isinstance(e.value,ast.Constant) and isinstance(e.value.value,str)}


@cache
def packet(name, instr):
    given = data(f'calls/{name}/in.json')
    require(given['instruction'] == prompts()[instr], f'Prompt drift in {name}')
    def indexed(records):
        result = {r['id']: r for r in records}
        require(len(result) == len(records), f'Duplicate ID in {name}')
        return result
    inputs = indexed(given['documents'])
    outputs = indexed(data(f'calls/{name}/out.json')['documents'])
    require(inputs.keys() == outputs.keys(), f'Packet ID mismatch: {name}')
    return inputs, outputs


def accepted(answer, source):
    seen, result = set(), []
    for s in answer['steps']:
        require(s['sign'] in ('do','avoid') and all(isinstance(s[k],str) and s[k].strip()
                for k in ('premise','action','consequence','quote')) and s['quote'] in source,
                'Invalid step fields or unsupported quotation')
        sig = tuple(' '.join(s[k].lower().split()) for k in ('sign','premise','action','consequence'))
        if sig not in seen:
            result.append(s)
            seen.add(sig)
    return result


class Readings:
    def __init__(self, reencode):
        self.reencode = reencode
        self.steps = 0
        if reencode:
            from encoder import SentenceEncoder
            from quantize import Hash
            self.encoder, self.quantizer = SentenceEncoder(), Hash()

    def check(self, ids, raw, source):
        steps = accepted(raw, source)
        require(len(steps) == len(ids) and all(re.fullmatch('[01]{6}', v) for v in ids),
                'Step count or stored ID format differs')
        if self.reencode and steps:
            from canonical import canonical_step
            from quantize import embed_steps
            bits = self.quantizer.codes(embed_steps(self.encoder, steps))
            recovered = [('1' if canonical_step(s)['sign']=='avoid' else '0')+b[:5]
                         for s,b in zip(steps,bits,strict=True)]
            require(recovered == ids, f'Neural re-encoding differs at passage {raw["id"]}')
        self.steps += len(steps)
        return [s['quote'] for s in steps]


@lru_cache(maxsize=550000)
def prf(key, problem, mode, feature):
    message = b'\x1f'.join(part.encode() for part in ('itoken-key-v1',problem,mode,feature))
    h = hmac.new(bytes.fromhex(key), message, hashlib.sha256).digest()
    a,b = struct.unpack('>QQ', h[:16])
    uniform = lambda n: ((n // 2048) + 0.5) / (1 << 53)
    return math.sqrt(-2*math.log(uniform(a))) * math.cos(math.tau*uniform(b))


def histogram(ids, mode):
    if mode=='exact':
        return Counter('full/'+code for code in ids)
    values = np.full((12,8), -len(ids), dtype=np.int64)
    for code in ids:
        for family, offsets in enumerate(((0,1,2),(0,2,3))):
            for shift in range(6):
                digit = sum(int(code[(shift+d)%6])*(4>>j) for j,d in enumerate(offsets))
                values[family*6+shift,digit] += 8
    return dict(zip(NAMES,values.ravel().tolist(),strict=True))


def score(ids, keys, p, mode):
    f = histogram(ids,mode)
    norm = math.sqrt(sum(x*x for x in f.values()))
    if not norm:
        return dict(score=None,p=None,positive=None)
    names = sorted(k for k,v in f.items() if v)
    weights = [f[k] for k in names]
    scores = [math.fsum(x*prf(key,p,mode,k) for x,k in zip(weights,names,strict=True))/norm for key in keys]
    prob = (1+sum(v>=scores[0] for v in scores[1:]))/len(keys)
    return dict(score=scores[0],p=prob,positive=prob<=.01)


def check_score(saved,replayed):
    require(saved['positive'] is replayed['positive'] and saved['p']==replayed['p'],
            'Different detector verdict or Monte Carlo rank')
    if replayed['score'] is None:
        require(saved['score'] is None, 'Different abstention')
    else:
        require(math.isclose(saved['score'],replayed['score'],rel_tol=1e-11,abs_tol=1e-11),
                'Different target score')


def selected(pool,utility,key,problem,arm):
    sets = np.asarray(list(combinations([i for i,ids in enumerate(pool) if ids],6)))
    criteria = np.asarray(utility)[sets].mean(axis=1)
    if arm!='control':
        rows = [histogram(ids,arm) for ids in pool]
        names = sorted(set().union(*(r.keys() for r in rows)))
        counts = np.asarray([[r.get(n,0) for n in names] for r in rows])
        salt = np.asarray([prf(key,problem,arm,n) for n in names])
        # Independent, direct calculation on each combined feature vector.
        for i in range(0,len(sets),128):
            batch = sets[i:i+128]
            vectors = counts[batch].sum(axis=1)
            norms = np.linalg.norm(vectors,axis=1)
            z = np.divide(np.sum(vectors*salt,axis=1),norms,out=np.zeros(len(batch)),where=norms>0)
            criteria[i:i+len(batch)] += 12*z
    return sets[np.argmax(criteria)].tolist()


def generation(readings):
    corpus = data('sources_a.json')+data('sources_b.json')
    keys = data('keys.json')
    require(len(corpus)==len(keys)==8 and all(len(k)==200 for k in keys), 'Incomplete cohort or key bank')
    require(len({k for ks in keys for k in ks})==1600 and all(re.fullmatch('[a-f0-9]{64}',k) for ks in keys for k in ks),
            'Duplicate or malformed keys')
    units = data('generation.json')['units']
    ws = data('weights.json')['weights']
    require(len(units)==len(ws)==8,'Incomplete generation')
    cohort = []
    for i,(source,unit,weights) in enumerate(zip(corpus,units,ws,strict=True)):
        p, passages = source['problem'],source['passages']
        pool = unit['pool_ids']
        require(unit['unit']==i and unit['problem']==p and len(passages)==len(set(passages))==len(pool)==len(weights)==20,
                'Source cohort differs from generation')
        pin,pout = packet(f'pool{i}','EXTRACT')
        require(len(pin)==20,'Missing pool reading')
        for j,ids in enumerate(pool):
            tag=f's{j}'
            require(pin[tag]==dict(id=tag,problem=p,text=passages[j]),'Pool input binding differs')
            readings.check(ids,pout[tag],passages[j])
        rng = random.Random(67482+i)
        noise = [-math.log(-math.log(rng.random())) for _ in range(20)]
        require(np.allclose(noise,unit['noise'],atol=1e-14,rtol=1e-14),'Gumbel coupling differs')
        utility = []
        for j,w in enumerate(weights):
            prompt = f'Problem: {p}\nProposed step: {passages[j]}\nWould this step help achieve the objective? Answer Yes or No.'
            require(w['prompt']==prompt and w['token_ids']==[9454,2753], 'Qwen prompt or tokens differ')
            yes,no = w['logprobs']
            derived = -float(np.logaddexp(0.,no-yes))
            require(math.isclose(derived,w['log_weight'],abs_tol=1e-12), 'Inclusion probability differs')
            utility.append(derived+noise[j])
        require(set(unit['variants'])==set(ARMS),'Invalid arms')
        for arm in ARMS:
            v=unit['variants'][arm]
            expected=selected(pool,utility,keys[i][0],p,arm)
            require(expected==v['selected'], f'Selection differs: {i}/{arm}')
            chunks = [passages[k] for k in expected]
            require(v['text']==' '.join(chunks[:3])+'\n\n'+' '.join(chunks[3:]), 'Original passage differs')
            cohort.append(dict(unit=i,arm=arm,problem=p,v=v,segments=chunks,plan=[x for k in expected for x in pool[k]]))
    require(len({c['v']['id'] for c in cohort})==24,'Duplicate proposal IDs')
    return cohort,keys


def main_results(cohort,keys,readings):
    saved=data('results.json')
    by_id = {r['id']:r for r in saved}
    require(len(saved)==len(by_id)==len(cohort),'Results cardinality differs')
    totals={}
    for c in cohort:
        i,arm,p,v = (c[k] for k in ('unit','arm','problem','v'))
        uid=v['id']; row=by_id[uid]; b=(ARMS.index(arm)-i)%3
        require(row['unit']==i and row['arm']==arm and row['planned_ids']==c['plan'],'Wrong proposal binding')
        old=v['text']
        word_in,word_out=packet(f'word{b}','REWORD')
        require(word_in[uid]==dict(id=uid,problem=p,text=old),'Rewriter saw different source')
        new=word_out[uid]['text']
        require(isinstance(new,str) and new.strip() and new.strip()!=old.strip(),'Invalid rewrite')
        votes=[]
        for rd in 'ab':
            ji,jo=packet(f'judge{rd}{b}','JUDGE')
            require(ji[uid]==dict(id=uid,original=old,rewrite=new,sources=[dict(index=j,text=s) for j,s in enumerate(c['segments'])]),
                    'Fidelity input binding differs')
            require(jo[uid]['status'] in ('preserved','uncertain','changed'),'Invalid fidelity verdict')
            votes.append(jo[uid])
        faithful=all(e['status']=='preserved' for e in votes)
        require(row['judges']==votes and row['faithful'] is faithful,'Fidelity output binding differs')
        modes=MODES if arm=='control' else (arm,)
        flags = {m:{} for m in modes}
        for mode in modes:
            s=score(c['plan'],keys[i],p,mode)
            check_score(row['plan'][mode],s)
            flags[mode]['plan']=s['positive']
        for stage,source in (('orig',old),('new',new)):
            for rd in 'ab':
                ri,ro=packet(f'{stage}{rd}{b}','EXTRACT')
                require(ri[uid]==dict(id=uid,problem=p,text=source),'Reader saw different input')
                obs=row['readings'][stage+rd]
                quotes=readings.check(obs['ids'],ro[uid],source)
                require(obs['quotes']==quotes,'Source citations differ from reader response')
                for mode in modes:
                    s=score(obs['ids'],keys[i],p,mode)
                    check_score(obs['test'][mode],s)
                    flags[mode][stage+rd]=s['positive']
        for mode,results in flags.items():
            label=arm+'/'+mode; tally=totals.setdefault(label,Counter())
            orig=all(results['orig'+r] is True for r in 'ab')
            rewrite=all(results['new'+r] is True for r in 'ab')
            for name,flag in dict(n=True,plan=results['plan'],original_both=orig,rewrite_both=rewrite,
                                  **{k:results[k] for k in ('origa','origb','newa','newb')},
                                  faithful=faithful,faithful_rewrite_both=faithful and rewrite,
                                  faithful_initial_both=faithful and orig,
                                  faithful_initial_rewrite_both=faithful and orig and rewrite).items():
                tally[name]+=int(flag is True)
    return totals


def contrasts(readings):
    source=data('foils.json'); results=data('foil_results.json')
    require(len(source)==len(results)==24,'Missing foil cases')
    counts={}
    for i,(s,obs) in enumerate(zip(source,results,strict=True)):
        require(s['type']==obs['type'],'Foil category differs')
        for j,kind in enumerate(('original','paraphrase','foil')):
            b=(j-i)%3; qid=hashlib.sha256(f'foil-ids-v1/{i}/{kind}'.encode()).hexdigest()[:12]
            ins,outs=packet(f'contrast{b}','EXTRACT')
            require(ins[qid]['text']==s[kind] and len(ins)==24,'Foil packet binding differs')
            readings.check(obs[kind],outs[qid],s[kind])
        stats=counts.setdefault(s['type'],Counter())
        stats['groups']+=1
        stats['all_single']+=int(all(len(obs[k])==1 for k in ('original','paraphrase','foil')))
        for kind in ('paraphrase','foil'):
            single=len(obs['original'])==len(obs[kind])==1
            stats[kind+'_singleton']+=int(single)
            stats[kind+'_exact']+=int(single and obs['original']==obs[kind])
    return counts


def interval(text, quote):
    if quote is None:
        return None
    require(isinstance(quote,str) and quote and quote in text,'Alignment citation absent')
    positions=[m.start() for m in re.finditer('(?='+re.escape(quote)+')',text)]
    return (positions[0],positions[0]+len(quote)) if len(positions)==1 else None


def matched(text, readings, anchor_sets):
    # Resolve each reading to a unique source region under every judge's mapping.
    anchor_sets = [[interval(text,q) for q in claims] for claims in anchor_sets]
    assignments = [[] for _ in range(6)]
    for code,quote in zip(readings['ids'],readings['quotes'],strict=True):
        item=interval(text,quote)
        if item is None: continue
        votes=[]
        for anchors in anchor_sets:
            hits=[]
            for i,a in enumerate(anchors):
                if a is None: continue
                overlap=min(a[1],item[1])-max(a[0],item[0])
                if 2*overlap>=min(a[1]-a[0],item[1]-item[0]):
                    hits.append(i)
            votes.append(hits[0] if len(hits)==1 else None)
        if len(set(votes))==1 and votes[0] is not None:
            assignments[votes[0]].append(code)
    return [cs[0] if len(cs)==1 else None for cs in assignments]


def agreement(cohort):
    results={r['id']:r for r in data('results.json')}
    generation_rows=data('generation.json')['units']
    grouped={'all':[],'faithful':[]}
    for c in cohort:
        v=c['v']; unit=c['unit']; uid=v['id']; row=results[uid]
        batch=(ARMS.index(c['arm'])-unit)%3
        new=packet(f'word{batch}','REWORD')[1][uid]['text']
        anchors=[]
        for judge in row['judges']:
            links=judge['links']
            require(sorted(l['index'] for l in links)==list(range(6)),'Missing source-to-rewrite link')
            anchors.append([l['quote'] for l in sorted(links,key=lambda x:x['index'])])
        plan=[generation_rows[unit]['pool_ids'][k] for k in v['selected']]
        aligned={'plan':[p[0] if len(p)==1 else None for p in plan]}
        for stage in ('orig','new'):
            for rd in 'ab':
                label=stage+rd
                aligned[label]=matched(new if stage=='new' else v['text'],row['readings'][label],
                                       anchors if stage=='new' else [c['segments']])
        grouped['all'].append(aligned)
        if row['faithful']:grouped['faithful'].append(aligned)
    pairs=(('origa','origb'),('newa','newb'),('origa','newb'),('origb','newa'),
           ('plan','origa'),('plan','origb'),('plan','newa'),('plan','newb'))
    report={}
    for scope,blocks in grouped.items():
        report[scope]={}
        for a,b in pairs:
            t=dict(exact=0,aligned=0,attempted=len(blocks)*6,wrong_step_collisions=0,wrong_step_comparisons=0)
            for rows in blocks:
                for i,x in enumerate(rows[a]):
                    for j,y in enumerate(rows[b]):
                        if x is None or y is None: continue
                        if i==j:
                            t['aligned']+=1;t['exact']+=int(x==y)
                        else:
                            t['wrong_step_comparisons']+=1;t['wrong_step_collisions']+=int(x==y)
            report[scope][a+'_'+b]=t
    return report


def verify_summary(counts, foil, matches):
    saved=data('summary.json')
    require(saved['agreement']==matches,'Gold-aligned agreement endpoint differs')
    detection={}
    for name,t in counts.items():
        detection[name.replace('/','_')] = dict(n=t['n'],plan=t['plan'],original_both=t['original_both'],
            rewrite_both=t['rewrite_both'],faithful=t['faithful'],faithful_original=t['faithful_initial_both'],
            faithful_rewrite=t['faithful_rewrite_both'],survival=[t['faithful_initial_rewrite_both'],t['faithful_initial_both']],
            individual={k:t[k] for k in ('origa','origb','newa','newb')})
    require(saved['detection']==detection,'Aggregate detection or faithful survival endpoints differ')
    with_all={**foil,'all':sum(foil.values(),Counter())}
    challenge={k:dict(n=t['groups'],all_single=t['all_single'],paraphrase_exact=t['paraphrase_exact'],
                      foil_collision=t['foil_exact']) for k,t in with_all.items()}
    require(saved['challenge']==challenge,'Aggregate logical contrast endpoints differ')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reencode',action='store_true')
    parser.add_argument('--generation-only',action='store_true')
    args=parser.parse_args()
    manifest=check_freeze()
    r=Readings(args.reencode)
    cohort,keys=generation(r)
    out={'freeze_sha256':manifest,'keys_verified':sum(map(len,keys)),'selections_verified':len(cohort)}
    if not args.generation_only:
        out['detector_counts']=main_results(cohort,keys,r)
        out['foil_counts']=contrasts(r)
        out['agreement']=agreement(cohort)
        verify_summary(out['detector_counts'],out['foil_counts'],out['agreement'])
    out['step_readings_verified']=r.steps
    out['neural_reencoding']=args.reencode
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
