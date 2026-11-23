"""Blind inputs and output validation. IDs and packet names are administrative."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
ARMS = ('control','exact','tolerant')
EXTRACT = """Read only the public question and the passage. Identify each distinct advocated application of reasoning (or, for a factual statement, the asserted relation). Describe it in ordinary short English using premise, action and consequence. Give plain English subjects and objects, preserve the causal direction, conditions and timing, and leave an unstated field as 'unspecified'. Give one exact contiguous supporting quotation per step. Write sign as 'avoid' when the main intervention or assertion explicitly forbids, withholds or negates the primary action, otherwise 'do'. A negative motivation or adverse outcome does not by itself change the sign. Use your own words for the three fields. Recover any number of steps; no candidate list or target count is supplied. Do not add missing reasoning."""
REWORD = """Re-express the proposal as natural advice in your own voice. Preserve all the substantive reasons, concrete actions and effects, including actors, conditions, timing, scope and exceptions. Sentence and paragraph order may change. Check against the original, then submit one rewrite without a heading, list labels or references to the experiment."""
JUDGE = """Judge whether the rewrite preserves all substantive reasoning of the original, without omissions, unsupported additions or changes in actor, mechanism, condition, timing, scope, direction or effect. Use preserved only if all survive, changed for a material difference, uncertain if unresolved. Give specific evidence. Independently map each supplied original source passage index to the closest corresponding exact contiguous quotation in the rewrite; use null if absent. A quote can identify a changed relation; it is not a vote that the relation was preserved. Quotes are for analysis only."""

def read(path): return json.loads(path.read_text())
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def frozen_packet(name,prompt,docs):
    dest=ROOT/'calls'/name/'in.json'; obj={'instruction':prompt,'documents':docs}
    if dest.exists() and read(dest)!=obj: raise ValueError(f'Packet drift: {name}')
    save(dest,obj)
def answers(name):
    base=ROOT/'calls'/name
    requested=[r['id'] for r in read(base/'in.json')['documents']]
    rows=read(base/'out.json')['documents'];out={r['id']:r for r in rows}
    if len(set(requested))!=len(requested) or len(out)!=len(rows) or set(out)!=set(requested): raise ValueError(f'ID mismatch: {name}')
    return out
def sources(): return read(ROOT/'sources_a.json')+read(ROOT/'sources_b.json')
def validated(row,source):
    steps=[];seen=set()
    for r in row['steps']:
        if any(not isinstance(r.get(f),str) or not r[f].strip() for f in ('premise','action','consequence','quote')): raise ValueError('Incomplete field')
        if r['quote'] not in source or r.get('sign') not in ('do','avoid'): raise ValueError('Invalid quotation or sign')
        fingerprint=tuple(' '.join(r[f].lower().split()) for f in ('sign','premise','action','consequence'))
        if fingerprint not in seen: steps.append(r); seen.add(fingerprint)
    return steps
def pool():
    for u,s in enumerate(sources()):
        assert len(set(s['passages']))==20
        frozen_packet(f'pool{u}',EXTRACT,[dict(id=f's{j}',problem=s['problem'],text=t) for j,t in enumerate(s['passages'])])
def foil_packets():
    foils=read(ROOT/'foils.json'); variants=('original','paraphrase','foil')
    for b in range(3):
        docs=[]
        for i,f in enumerate(foils):
            typ=variants[(b+i)%3]
            uid=hashlib.sha256(f'foil-ids-v1/{i}/{typ}'.encode()).hexdigest()[:12]
            docs.append(dict(id=uid,problem='What reasoning or relation does this passage express?',text=f[typ]))
        random.Random(9283+b).shuffle(docs);frozen_packet(f'contrast{b}',EXTRACT,docs)
def batches():
    units=read(ROOT/'generation.json')['units']
    for b in range(3):
        yield b,[(u,ARMS[(b+u['unit'])%3]) for u in units]
def originals():
    for b,pairs in batches():
        docs=[dict(id=u['variants'][a]['id'],problem=u['problem'],text=u['variants'][a]['text']) for u,a in pairs]
        for j in ('a','b'): frozen_packet(f'orig{j}{b}',EXTRACT,docs)
        frozen_packet(f'word{b}',REWORD,docs)
def rewrites():
    corpus=sources()
    for b,pairs in batches():
        if not (ROOT/'calls'/f'word{b}'/'out.json').exists():continue
        out=answers(f'word{b}');docs=[];cmp=[]
        for u,a in pairs:
            v=u['variants'][a];uid=v['id'];w=out[uid]['text']
            assert w.strip() and w.strip()!=v['text'].strip()
            docs.append(dict(id=uid,problem=u['problem'],text=w))
            cmp.append(dict(id=uid,original=v['text'],rewrite=w,sources=[{'index':j,'text':corpus[u['unit']]['passages'][c]} for j,c in enumerate(v['selected'])]))
        for j in ('a','b'):
            frozen_packet(f'new{j}{b}',EXTRACT,docs);frozen_packet(f'judge{j}{b}',JUDGE,cmp)
if __name__=='__main__':
    import sys
    {'pool':pool,'foils':foil_packets,'originals':originals,'rewrites':rewrites}[sys.argv[1]]()
