"""Open text recovery experiment. Run from this directory using the MLX environment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from real_model import MLXModel

FIELDS = ('premise','action','consequence')
EXTRACT = ('Read only the public problem and document. Recover the distinct reasoning steps actually advocated, in your own concise ordinary English. A step links a stated or clearly implied reason to a concrete mechanism/action and its intended consequence. Keep roles, ordering, conditions and scope. Do not fill in unstated reasoning. Separate distinct causal applications, merge duplicates, and abstain on unsupported or unresolved ones. Return any number of steps. For each give premise, action, consequence, and quote (one exact contiguous source passage supporting the step). Use [] if none. No fixed vocabulary or reference answers are supplied.')
REWRITE = ('Rewrite this proposal as an ordinary piece of advice in your own voice. Preserve every substantive argument, the motivation, action, outcome, actors, and qualifiers. Reorganize and change terminology freely; no phrase is required. Omit headings, lists, and bookkeeping labels. Return only the new text.')
FIDELITY = ('Compare the original and rewrite directly. Independently identify substantive reasoning relations in the original and check all against the rewrite, including omissions, additions, conditions, timing, actors, causal direction, and scope. Return status preserved only when all substantive reasoning survives. Use changed for a material change and uncertain for unresolved meaning. Give concrete evidence for your assessment. Wording and presentation order may differ.')
REVIEW = ('Independently read the source and assess the supplied open reasoning triples. Check every proposed triple for textually supported premise, action and consequence and inspect the source for omitted reasoning steps. Return status complete only if all substantive relations are covered and supported. Use partial for omissions, inaccurate relations or extra steps; uncertain if unresolved. Provide reasons with specific text evidence.')

def read(p): return json.loads(p.read_text())
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def problems(): return read(ROOT/'sources.json')['problems']+read(ROOT/'sources_b.json')['problems']
def packet(name,prompt,docs):
    path=ROOT/'calls'/name/'in.json'
    obj={'instruction':prompt,'documents':docs}
    if path.exists() and read(path)!=obj: raise ValueError(f'Packet drift: {name}')
    save(path,obj)
def outputs(name):
    base=ROOT/'calls'/name
    expected={d['id'] for d in read(base/'in.json')['documents']}
    rows=read(base/'out.json')['documents']; mapped={d['id']:d for d in rows}
    if set(mapped)!=expected or len(mapped)!=len(rows): raise ValueError(f'Incomplete responses: {name}')
    return mapped

def source_batches():
    for i,p in enumerate(problems()):
        if len(p['ideas'])!=24: raise ValueError('Expected 24 free proposals')
        packet('pool'+str(i+1),EXTRACT,[dict(id=f's{j}',problem=p['prompt'],text=d['text']) for j,d in enumerate(p['ideas'])])

def admitted(row,source):
    seen=set(); result=[]
    for s in row.get('steps',[]):
        if not isinstance(s,dict) or any(not isinstance(s.get(k),str) or not s[k].strip() for k in (*FIELDS,'quote')) or s['quote'] not in source: continue
        sig=tuple(' '.join(s[k].lower().split()) for k in FIELDS)
        if sig not in seen: result.append(s); seen.add(sig)
    return result

class Encoder:
    def __init__(self,path):
        self.lm=MLXModel(path,device='cpu',execution_mode='dense-fp32',source_repo='mlx-community/Qwen2.5-0.5B-Instruct-4bit',revision='a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3')
        self.cache={}
    def encode(self,text):
        import numpy as np
        from mlx_lm.models.base import create_attention_mask
        if text not in self.cache:
            mx=self.lm.mx;model=self.lm.model.model
            ids=self.lm.tokenizer.encode(text,add_special_tokens=False)
            hidden=model.embed_tokens(mx.array([ids])); mask=create_attention_mask(hidden,None)
            for layer in model.layers[:8]:hidden=layer(hidden,mask)
            value=np.asarray(mx.mean(hidden,axis=1)[0],dtype=float)
            self.cache[text]=value/max(float(np.linalg.norm(value)),1e-12)
        return self.cache[text]
    def vector(self,problem,steps):
        import numpy as np
        if not steps:return None
        parts=[]
        for s in steps:
            fields=[]
            for f in FIELDS:
                v=self.encode(s[f])-self.encode(problem)
                fields.append(v/max(np.linalg.norm(v),1e-12))
            parts.append(np.concatenate(fields))
        return np.mean(parts,axis=0)
    def weight(self,problem,text):
        import numpy as np
        prompt=f'Problem: {problem}\nProposed step: {text}\nWould this step help achieve the objective? Answer Yes or No.'
        _,ids,_=self.lm._prompt([{'role':'user','content':prompt}])
        lp,_=self.lm._prefill(ids)
        choices=[self.lm.tokenizer.encode(s,add_special_tokens=False) for s in ('Yes','No')]
        assert all(len(a)==1 for a in choices)
        values=[float(lp[a[0]].item()) for a in choices]
        return {'prompt':prompt,'token_ids':[a[0] for a in choices],'logprobs':values,'log_weight':values[0]-float(np.logaddexp(*values))}

def direction(key,problem,size):
    import numpy as np
    prefix=b'open-ideas-v1\0'+bytes.fromhex(key)+b'\0'+problem.encode()+b'\0'
    width=size//3
    return np.concatenate([np.unpackbits(np.frombuffer(hashlib.shake_256(prefix+f.encode()).digest((width+7)//8),dtype=np.uint8))[:width].astype(float)*2-1 for f in FIELDS])
def test(v,keys,problem):
    import numpy as np
    if v is None:return {'score':None,'p':None,'detected':None}
    scores=[float(np.dot(direction(k,problem,len(v)),v)/3) for k in keys]
    p=(1+sum(x>=scores[0] for x in scores[1:]))/len(scores)
    return {'score':scores[0],'p':p,'detected':p<=.01}

def generate(enc):
    import numpy as np
    dest=ROOT/'generation.json'
    if dest.exists():raise ValueError('Cohort already drawn')
    keypath=ROOT/'keys.json'
    if not keypath.exists(): save(keypath,[[secrets.token_hex(32) for _ in range(200)] for _ in range(8)])
    keyset=read(keypath); rows=[]; vectors={}; metadata=enc.lm.metadata
    for i,p in enumerate(problems()):
        raw=outputs('pool'+str(i+1)); vectors_p=[];weights=[];nsteps=[]
        for j,s in enumerate(p['ideas']):
            steps=admitted(raw[f's{j}'],s['text']);nsteps.append(len(steps));v=enc.vector(p['prompt'],steps)
            if v is None:v=np.zeros(3*len(enc.encode(p['prompt'])))
            vectors_p.append(v);weights.append(enc.weight(p['prompt'],s['text']))
        vectors['c'+str(i)]=np.array(vectors_p); rng=random.Random(5290+i)
        noise=[-math.log(-math.log(rng.random())) for _ in range(24)];keys=keyset[i]
        direction0=direction(keys[0],p['prompt'],len(vectors_p[0])); scores=[float(v@direction0/3) for v in vectors_p]
        variants={}
        for cond,strength in [('unmarked',0),('marked',4)]:
            ranked=sorted(range(24),key=lambda j:weights[j]['log_weight']+noise[j]+strength*scores[j],reverse=True)
            chosen=sorted(ranked[:12]);texts=[p['ideas'][j]['text'] for j in chosen]
            doc='\n\n'.join(' '.join(texts[k:k+4]) for k in range(0,12,4)); accepted=sum(nsteps[j] for j in chosen)
            planned=sum(vectors_p[j]*nsteps[j] for j in chosen)/accepted if accepted else None
            variants[cond]={'id':secrets.token_hex(8),'selected':chosen,'text':doc,'planned':test(planned,keys,p['prompt'])}
        rows.append({'unit':i,'problem':p['prompt'],'weights':weights,'pool_n_steps':nsteps,'noise':noise,'key_scores':scores,'variants':variants})
    np.savez_compressed(ROOT/'pool_vectors.npz',**vectors);save(dest,{'model':metadata,'units':rows})

def groups():
    units=read(ROOT/'generation.json')['units']
    for n,cond in enumerate(('unmarked','marked')):
        yield str(n+1),[(u,u['variants'][cond]) for u in units]

def originals():
    for group,pairs in groups():
        docs=[{'id':v['id'],'problem':u['problem'],'text':v['text']} for u,v in pairs]
        packet('rewrite'+group,REWRITE,docs);packet('original'+group,EXTRACT,docs)

def rewrites():
    for group,pairs in groups():
        first=outputs('rewrite'+group);docs=[];comp=[]
        for u,v in pairs:
            new=first[v['id']]['text'];docs.append({'id':v['id'],'problem':u['problem'],'text':new});comp.append({'id':v['id'],'original':v['text'],'rewrite':new})
        packet('rewritten'+group,EXTRACT,docs);packet('fidelity'+group,FIDELITY,comp)

def reviews():
    for group,_ in groups():
        for var in ['original','rewritten']:
            name=var+group
            if not (ROOT/'calls'/name/'out.json').exists():continue
            first=outputs(name);docs=[]
            for d in read(ROOT/'calls'/name/'in.json')['documents']:
                docs.append({**d,'steps':admitted(first[d['id']],d['text'])})
            packet('review_'+name,REVIEW,docs)

def evaluate(enc):
    import numpy as np
    rows=[];vecs={};keyset=read(ROOT/'keys.json')
    for group,pairs in groups():
        for u,v in pairs:
            ident=v['id'];row={'unit':u['unit'],'condition':'unmarked' if group=='1' else 'marked','id':ident,'planned':v['planned'],'fidelity':outputs('fidelity'+group)[ident]}
            for var in ('original','rewritten'):
                name=var+group;text=v['text'] if var=='original' else outputs('rewrite'+group)[ident]['text'];first=outputs(name)[ident];steps=admitted(first,text)
                vector=enc.vector(u['problem'],steps);det=test(vector,keyset[u['unit']],u['problem'])
                if vector is not None:vecs[name+ident]=vector
                row[var]={**det,'proposed':len(first['steps']),'retained':len(steps),'review':outputs('review_'+name)[ident]}
            rows.append(row)
    np.savez_compressed(ROOT/'document_vectors.npz',**vecs);save(ROOT/'results.json',rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pool','generate','originals','rewrites','reviews','evaluate']);parser.add_argument('--model',type=Path);args=parser.parse_args()
    if args.phase in ('generate','evaluate'):{'generate':generate,'evaluate':evaluate}[args.phase](Encoder(args.model))
    else:{'pool':source_batches,'originals':originals,'rewrites':rewrites,'reviews':reviews}[args.phase]()
