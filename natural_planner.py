"""Actual learned candidate probabilities and key-biased natural idea choices.

The public catalog defines curated semantic alternatives, not a formal calculus.
IDs exist in computation only, never in authored or reworded prose. The model
policy is local to one decision and conditional on three listed expressions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import secrets

from real_model import MLXModel
from research_runner import save, digest
from watermark import bias_distribution, key_randomization_pvalue

ROOT=Path(__file__).resolve().parent
SOURCE='mlx-community/Qwen2.5-0.5B-Instruct-4bit'
REVISION='a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'
PROTOCOL='natural-language-choices-v1'
DEPENDENCIES=('natural_planner.py','real_model.py','watermark.py','research_runner.py')


def canonical(decision,option):
    return json.dumps({'decision':decision['question'],'mechanism':option['mechanism'],
                       'consequence':option['consequence']},sort_keys=True,ensure_ascii=False,separators=(',',':'))


def context(problem,decision):
    return json.dumps({'protocol':PROTOCOL,'problem':problem['id'],'decision':decision['id'],
                       'problem_sha256':hashlib.sha256(json.dumps(problem,sort_keys=True,ensure_ascii=False).encode()).hexdigest()},
                     sort_keys=True,separators=(',',':'))


def validate_catalog(catalog):
    if not isinstance(catalog,dict) or not isinstance(catalog.get('problems'),list) or not catalog['problems']:
        raise ValueError('Catalog needs problems')
    seen=set()
    for problem in catalog['problems']:
        if not isinstance(problem['id'],str) or problem['id'] in seen:raise ValueError('Duplicate/invalid problem ID')
        seen.add(problem['id'])
        if any(not isinstance(problem.get(k),str) or not problem[k].strip() for k in ('title','problem','goal')):raise ValueError('Missing problem text')
        decisions=problem['decisions']
        if not decisions or len({d['id'] for d in decisions})!=len(decisions):raise ValueError('Duplicate/empty decisions')
        for d in decisions:
            if len(d['options'])!=3 or len({o['id'] for o in d['options']})!=3:raise ValueError('Need three distinct options')
            for k in ('question','context'):
                if not isinstance(d[k],str) or not d[k].strip():raise ValueError('Missing decision text')
            for o in d['options']:
                if any(not isinstance(o.get(k),str) or not o[k].strip() for k in ('id','mechanism','consequence','application')):raise ValueError('Missing option text')
            if len({canonical(d,o) for o in d['options']})!=3 or len({o['application'] for o in d['options']})!=3:raise ValueError('Duplicate semantic alternatives')
    return catalog


def request(problem,decision):
    candidates=[{'text':o['application'],'itoken_id':canonical(decision,o)} for o in decision['options']]
    rng=random.Random(hashlib.sha256((problem['id']+'\0'+decision['id']).encode()).hexdigest())
    rng.shuffle(candidates)
    state={'problem':problem['problem'],'goal':problem['goal'],'decision':decision['question'],'context':decision['context']}
    prompt=('Choose one next planning step using the local decision below. Return exactly one listed statement and stop. '
            'No other planning choices have been supplied.\n'+json.dumps(state,ensure_ascii=False,sort_keys=True)+
            '\nCANDIDATE STATEMENTS:\n'+'\n'.join(c['text'] for c in candidates))
    return candidates,[{'role':'user','content':prompt}]


def build_bank(model,catalog_path,out):
    catalog=validate_catalog(json.loads(catalog_path.read_text()))
    freeze={'protocol':PROTOCOL,'catalog_sha256':digest(catalog_path),'model_metadata':model.metadata,
            'source_sha256':{n:digest(ROOT/n) for n in DEPENDENCIES},
            'context_policy':'public problem plus current local decision only; no prior choices or watermark key',
            'scope':'conditional on three listed EOS-terminated natural-language applications, one wording per semantic option',
            'numerical_tolerance_nats_per_token':.05}
    manifest=out/'bank_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=freeze:raise ValueError('Frozen bank configuration drift')
    save(manifest,freeze)
    entries=[]
    for pi,p in enumerate(catalog['problems']):
        for di,d in enumerate(p['decisions']):
            path=out/f'problem_{pi:02d}_decision_{di:02d}.json'
            candidates,messages=request(p,d)
            if path.exists():
                row=json.loads(path.read_text())
                if row['freeze']!=freeze or row['messages']!=messages:raise ValueError('Saved score drift')
            else:
                scores=model.score_candidates(messages,candidates,verify_first_uncached=True,verify_all_uncached=di==0)
                difference=max(c['max_absolute_token_logprob_difference'] for c in scores['uncached_crosschecks'])
                if not math.isfinite(difference) or difference>.05:
                    save(out/f'numerical_failure_{pi:02d}_{di:02d}.json',scores);raise ValueError('Numerical tolerance failed')
                row={'freeze':freeze,'problem_id':p['id'],'decision_id':d['id'],'messages':messages,'scores':scores,'max_checked_logprob_difference':difference}
                save(path,row)
            entries.append({'problem_id':p['id'],'decision_id':d['id'],'path':path.name,'itoken_probabilities':row['scores']['itoken_probabilities'],
                            'covered_probability_mass':row['scores']['covered_probability_mass'],'max_checked_logprob_difference':row['max_checked_logprob_difference']})
            print('scored',pi,di,'coverage',row['scores']['covered_probability_mass'],flush=True)
    save(out/'bank_summary.json',{'freeze':freeze,'entries':entries})


def detect(problem,choices,key,surrogates):
    pairs=[]
    if set(choices)-{d['id'] for d in problem['decisions']}:raise ValueError('Unknown decision')
    for d in problem['decisions']:
        if d['id'] not in choices:continue
        matches=[o for o in d['options'] if o['id']==choices[d['id']]]
        if len(matches)!=1:raise ValueError('Unknown option')
        pairs.append((context(problem,d),canonical(d,matches[0])))
    if not pairs:return {'n_choices':0,'p_value':None,'detected':None,'preferred':None}
    score,p,n=key_randomization_pvalue(pairs,key,surrogates)
    return {'n_choices':n,'p_value':p,'detected':p<=.01,'preferred':score}


def sample_cohort(catalog_path,bank_path,out,seed=1701,delta=6.0):
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh cohort; never reroll observed results')
    catalog=validate_catalog(json.loads(catalog_path.read_text()))
    if len(catalog['problems'])!=8 or any(len(p['decisions'])!=16 for p in catalog['problems']):raise ValueError('Locked cohort is eight problems, sixteen decisions each')
    if delta!=6.0 or seed!=1701:raise ValueError('Use the locked primary specification')
    bank=json.loads((bank_path/'bank_summary.json').read_text());freeze=bank['freeze']
    if freeze['catalog_sha256']!=digest(catalog_path) or freeze['source_sha256']!={n:digest(ROOT/n) for n in DEPENDENCIES}:raise ValueError('Scorer/catalog drift')
    if freeze!=json.loads((bank_path/'bank_manifest.json').read_text()):raise ValueError('Bank manifest mismatch')
    index={(e['problem_id'],e['decision_id']):e for e in bank['entries']}
    expected={(p['id'],d['id']) for p in catalog['problems'] for d in p['decisions']}
    if len(index)!=len(bank['entries']) or set(index)!=expected:raise ValueError('Incomplete or duplicated bank')
    for p in catalog['problems']:
        for d in p['decisions']:
            e=index[(p['id'],d['id'])];raw=json.loads((bank_path/e['path']).read_text())
            if raw['freeze']!=freeze or raw['messages']!=request(p,d)[1] or raw['scores']['itoken_probabilities']!=e['itoken_probabilities']:raise ValueError('Bank record drift')
            if set(e['itoken_probabilities'])!={canonical(d,o) for o in d['options']}:raise ValueError('Wrong support')
            bias_distribution(e['itoken_probabilities'],b'validation-only','validation',0)
    config={'protocol':PROTOCOL,'units':8,'decisions_per_unit':16,'delta':[0,6],'seed':seed,'alpha':.01,'surrogate_count':199,
            'catalog_sha256':digest(catalog_path),'bank_sha256':digest(bank_path/'bank_summary.json'),
            'key_sampling':'target and 199 surrogate keys independently drawn once with secrets.token_bytes(32)',
            'pairing':'common choice uniforms within unit, separate from all key randomness','source_sha256':freeze['source_sha256']}
    save(out/'config.json',config);documents=[]
    for unit,p in enumerate(catalog['problems']):
        key=secrets.token_bytes(32);surrogates=[secrets.token_bytes(32) for _ in range(199)]
        unit_spec={'unit_id':unit,'problem_id':p['id'],'public_demonstration_key_hex':key.hex(),'surrogate_keys_hex':[x.hex() for x in surrogates],'sampling_seed':seed+unit}
        save(out/f'unit_{unit:02d}_key.json',unit_spec)
        for condition,strength in [('unmarked',0),('marked',delta)]:
            rng=random.Random(seed+unit);choices={};policies=[];llr=0.0
            for d in p['decisions']:
                probabilities=index[(p['id'],d['id'])]['itoken_probabilities'];q=bias_distribution(probabilities,key,context(p,d),strength)
                alt=bias_distribution(probabilities,key,context(p,d),delta);u=rng.random();cumulative=0;selected=None
                for token,prob in q.items():
                    cumulative+=prob
                    if u<cumulative:selected=token;break
                if selected is None:selected=next(reversed(q))
                option=next(o for o in d['options'] if canonical(d,o)==selected)
                choices[d['id']]=option['id'];llr+=math.log(alt[selected]/probabilities[selected])
                policies.append({'decision_id':d['id'],'p':probabilities,'q':q,'uniform':u,'selected':selected})
            ident=secrets.token_hex(8)
            row={'document_id':ident,**unit_spec,'condition':condition,'choices':choices,'policy_rows':policies,'planned_log_likelihood_ratio':llr,
                 'planned_detection':detect(p,choices,key,surrogates)}
            save(out/'plans'/f'{ident}.json',row)
            documents.append({k:row[k] for k in ('document_id','unit_id','problem_id','condition','planned_detection')})
    save(out/'index.json',{'config':config,'documents':documents})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--catalog',type=Path,default=ROOT/'natural_ideas_catalog.json')
    ap.add_argument('--model',type=Path);ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--cohort',type=Path)
    args=ap.parse_args()
    if args.model:
        model=MLXModel(args.model,source_repo=SOURCE,revision=REVISION,device='cpu',execution_mode='dense-fp32',max_context_tokens=8192)
        build_bank(model,args.catalog,args.bank)
    if args.cohort:sample_cohort(args.catalog,args.bank,args.cohort)
