"""Replay natural-language recovery, lexical controls and paired detection.

No model is invoked. Fidelity records are separate agent judgments, not inferred
from agreement with the intended plan. All assigned documents remain included.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from natural_lexical_baseline import recover as lexical_recover
from natural_planner import detect
from natural_transport import replay
from natural_failures import replay_recovery
from natural_study import AUTHOR_INSTRUCTION, REWRITE_INSTRUCTION, AUTHOR_FIDELITY_INSTRUCTION, FIDELITY_INSTRUCTION
from research_runner import save,digest

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())

def fraction(a,b):return {'successes':a,'trials':b,'fraction':a/b if b else None}

def agreement(expected,actual):
    return {'exact':actual==expected,'correct':sum(actual.get(d)==o for d,o in expected.items()),'expected':len(expected),
            'predicted':len(actual),'missing':[d for d in expected if d not in actual],
            'wrong':{d:{'expected':expected[d],'actual':o} for d,o in actual.items() if d in expected and o!=expected[d]},
            'extra':{d:o for d,o in actual.items() if d not in expected}}

def _batch(path):
    result=replay_recovery(path) if (path/'assignment_manifest.json').exists() else replay(path)
    assert result==read(path/'results.json'),'Saved recovery differs from first responses'
    rows={r['id']:r for r in result['rows']};assert len(rows)==len(result['rows'])
    return rows

def challenge_summary(root):
    data=read(ROOT/'natural_challenge_holdout.json');p=data['public_problem'];cases=data['cases'];rows={}
    frozen=read(root/'challenge_freeze.json')
    for name,sha in frozen['source_sha256'].items():assert digest(ROOT/name)==sha
    for n in (1,2):
        for ident,row in _batch(root/'agent_artifacts'/f'challenge_{n:02d}').items():
            assert ident not in rows;rows[ident]=row
    assert set(rows)=={c['id'] for c in cases}
    detailed=[]
    for case in cases:
        r=rows[case['id']]['recovery'];actual=r['choices'];expected=case['expected_choices'];amb=case['expected_ambiguous_decisions']
        gold_status=case['expected_status']
        required_status={'complete':'complete_choices','partial':'partial_evidence','abstain':'abstained',
                         'ambiguous':'partial_evidence' if expected else 'ambiguous_choices'}[gold_status]
        status_correct=(r['status']==required_status and
                        (r['complete'] if gold_status=='complete' else
                         bool(actual) and not r['complete'] and not r['ambiguous_decisions'] if gold_status=='partial' else
                         not actual and not r['ambiguous_decisions'] if gold_status=='abstain' else bool(r['ambiguous_decisions'])))
        a=agreement(expected,actual);lex=lexical_recover(p,case['text']);la=agreement(expected,lex['choices'])
        detailed.append({'id':case['id'],'gold_status':gold_status,'agreement':a,'status_correct':status_correct,
                         'ambiguity_correct':sorted(r['ambiguous_decisions'])==sorted(amb),
                         'strict_success':a['exact'] and status_correct and sorted(r['ambiguous_decisions'])==sorted(amb),
                         'recovery':r,'lexical':lex,'lexical_agreement':la,'contrast_pair':case.get('contrast_pair')})
    correct=sum(x['agreement']['correct'] for x in detailed);gold=sum(x['agreement']['expected'] for x in detailed);pred=sum(x['agreement']['predicted'] for x in detailed)
    summary={'cases':len(cases),'strict_successes':sum(x['strict_success'] for x in detailed),'exact_choice_sets':sum(x['agreement']['exact'] for x in detailed),
             'correct_choices':correct,'gold_choices':gold,'predicted_choices':pred,'precision':correct/pred if pred else None,'recall':correct/gold if gold else None,
             'lexical_exact_choice_sets':sum(x['lexical_agreement']['exact'] for x in detailed),
             'lexical_correct_choices':sum(x['lexical_agreement']['correct'] for x in detailed),
             'lexical_predicted_choices':sum(x['lexical_agreement']['predicted'] for x in detailed),
             'semantic_false_acceptances':sum(not x['agreement']['expected'] and bool(x['agreement']['predicted']) for x in detailed)}
    for status in ('complete','partial','abstain','ambiguous'):
        group=[x for x in detailed if x['gold_status']==status]
        summary[status]=fraction(sum(x['strict_success'] for x in group),len(group))
    contrasts=[x for x in detailed if x['contrast_pair']]
    summary['same_word_contrast_cases']=fraction(sum(x['strict_success'] for x in contrasts),len(contrasts))
    summary['lexical_same_word_exact_sets']=fraction(sum(x['lexical_agreement']['exact'] for x in contrasts),len(contrasts))
    return summary,detailed

def _assessment(path,ids,problems_by_doc):
    response=read(path/'responses.json');assert set(response)=={'assessments'}
    rows=response['assessments'];found={}
    for row in rows:
        ident=row['document_id'];assert ident in ids and ident not in found
        assert row['status'] in {'preserved','changed','uncertain'}
        details=row['decisions'];p=problems_by_doc[ident]
        assert len(details)==len(p['decisions']) and {d['decision_id'] for d in details}=={d['id'] for d in p['decisions']}
        assert all(d['verdict'] in {'preserved','changed','omitted','uncertain'} for d in details)
        assert isinstance(row['added_conflicting_decisions'],list)
        if row['status']=='preserved':assert all(d['verdict']=='preserved' for d in details) and not row['added_conflicting_decisions']
        found[ident]=row
    assert set(found)==set(ids)
    return found

def audit(root):
    for name,sha in read(root/'primary_freeze.json')['source_sha256'].items():assert digest(ROOT/name)==sha
    challenge,challenge_rows=challenge_summary(root)
    cat=read(ROOT/'natural_ideas_catalog.json');problems={p['id']:p for p in cat['problems']};cohort=root/'cohort';index=read(cohort/'index.json');config=index['config']
    assert (config['units'],config['decisions_per_unit'],config['delta'],config['seed'],config['alpha'],config['surrogate_count'])==(8,16,[0,6],1701,.01,199)
    assert config['catalog_sha256']==digest(ROOT/'natural_ideas_catalog.json') and config['bank_sha256']==digest(root/'probability_bank/bank_summary.json')
    for name,sha in config['source_sha256'].items():assert digest(ROOT/name)==sha
    docs={d['document_id']:d for d in index['documents']};assert len(docs)==16
    plans={i:read(cohort/'plans'/f'{i}.json') for i in docs};bydoc={i:problems[p['problem_id']] for i,p in plans.items()}
    artifacts=root/'agent_artifacts';recoveries={};originals={};rewrites={};af={};rf={}
    for n in range(1,9):
        rows=_batch(artifacts/f'recovery_{n:02d}');assert len({docs[i]['condition'] for i in rows})==1
        variant='original' if n<=4 else 'reworded'
        for ident,row in rows.items():assert (ident,variant) not in recoveries;recoveries[(ident,variant)]=row
    for n in range(1,5):
        ar=read(artifacts/f'author_{n:02d}'/'requests.json');ids=[d['document_id'] for d in ar['documents']]
        assert len(ids)==4 and len({docs[i]['condition'] for i in ids})==1
        expected_authors=[]
        for ident in ids:
            p=bydoc[ident];plan=plans[ident];applications=[]
            for d in p['decisions']:
                option=next(o for o in d['options'] if o['id']==plan['choices'][d['id']])
                applications.append({'question':d['question'],'context':d['context'],'application':option['application']})
            expected_authors.append({'document_id':ident,'problem':p['problem'],'goal':p['goal'],'selected_applications':applications})
        assert ar=={'instruction':AUTHOR_INSTRUCTION,'documents':expected_authors}
        for family,target in [('author',originals),('rewrite',rewrites)]:
            responses=read(artifacts/f'{family}_{n:02d}'/'responses.json')['documents']
            assert len(responses)==4 and {r['document_id'] for r in responses}==set(ids)
            for r in responses:assert r['document_id'] not in target;target[r['document_id']]=r['text']
        rewrite_request=read(artifacts/f'rewrite_{n:02d}'/'requests.json')
        assert rewrite_request['instruction']==REWRITE_INSTRUCTION
        rewrite_inputs=rewrite_request['documents']
        assert {r['document_id']:r['text'] for r in rewrite_inputs}=={i:originals[i] for i in ids}
        authorship_inputs=[];fidelity_inputs=[]
        for ident in ids:
            p=bydoc[ident];plan=plans[ident];intended=[]
            for d in p['decisions']:
                option=next(o for o in d['options'] if o['id']==plan['choices'][d['id']])
                intended.append({'decision_id':d['id'],'application':option['application']})
            authorship_inputs.append({'document_id':ident,'intended_applications':intended,'text':originals[ident]})
            fidelity_inputs.append({'document_id':ident,'decision_topics':[{'decision_id':d['id'],'question':d['question']} for d in p['decisions']],
                                    'original':originals[ident],'rewrite':rewrites[ident]})
        assert read(artifacts/f'authorship_{n:02d}'/'requests.json')=={'instruction':AUTHOR_FIDELITY_INSTRUCTION,'documents':authorship_inputs}
        assert read(artifacts/f'fidelity_{n:02d}'/'requests.json')=={'instruction':FIDELITY_INSTRUCTION,'documents':fidelity_inputs}
        af.update(_assessment(artifacts/f'authorship_{n:02d}',ids,bydoc));rf.update(_assessment(artifacts/f'fidelity_{n:02d}',ids,bydoc))
    assert len(recoveries)==32 and len(originals)==len(rewrites)==len(af)==len(rf)==16
    rows=[]
    marker_pattern=re.compile(r'\b[pmde]\d+\b|\b(?:problem|decision|choice)_[0-9a-f]{6,}\b|[∧∨⇒]|->')
    for ident,plan in plans.items():
        p=bydoc[ident];key=bytes.fromhex(plan['public_demonstration_key_hex']);keys=[bytes.fromhex(k) for k in plan['surrogate_keys_hex']]
        planned=detect(p,plan['choices'],key,keys);assert planned==plan['planned_detection']==docs[ident]['planned_detection']
        for variant,texts in [('original',originals),('reworded',rewrites)]:
            rec=recoveries[(ident,variant)];text=texts[ident];assert rec['text']==text
            choices=rec['recovery']['choices'];lex=lexical_recover(p,text)
            rows.append({'document_id':ident,'unit_id':plan['unit_id'],'problem_id':p['id'],'condition':plan['condition'],'variant':variant,
                         'planned_detection':planned,'detection':detect(p,choices,key,keys),'recovery':rec['recovery'],
                         'agreement':agreement(plan['choices'],choices),'lexical':lex,'lexical_agreement':agreement(plan['choices'],lex['choices']),
                         'lexical_detection':detect(p,lex['choices'],key,keys),'author_fidelity':af[ident],'rewrite_fidelity':rf[ident],
                         'explicit_marker_matches':marker_pattern.findall(text),'text_changed':rewrites[ident]!=originals[ident]})
    summary={'units':8,'documents':32,'decisions_per_document':16,'alpha':.01,'surrogate_keys':199,'uncertainty_scope':'descriptive paired pilot, no population calibration claim',
             'explicit_marker_documents':sum(bool(r['explicit_marker_matches']) for r in rows)}
    for condition in ('unmarked','marked'):
        a=sorted([r for r in rows if r['condition']==condition and r['variant']=='original'],key=lambda r:r['unit_id'])
        b=sorted([r for r in rows if r['condition']==condition and r['variant']=='reworded'],key=lambda r:r['unit_id'])
        assert len(a)==len(b)==8
        positive=lambda r:r['detection']['detected'] is True
        planned_count=sum(r['planned_detection']['detected'] for r in a);source_count=sum(positive(r) for r in a)
        both_faithful=lambda r:r['author_fidelity']['status']=='preserved' and r['rewrite_fidelity']['status']=='preserved'
        faithful=[r for r in b if both_faithful(r)]
        group={'planned_detection':fraction(planned_count,8),'original_detection':fraction(source_count,8),'reworded_detection':fraction(sum(positive(r) for r in b),8),
               'survival_from_planned':fraction(sum(x['planned_detection']['detected'] and positive(y) for x,y in zip(a,b)),planned_count),
               'survival_from_original':fraction(sum(positive(x) and positive(y) for x,y in zip(a,b)),source_count),
               'faithful_reworded_detection_joint':fraction(sum(positive(r) for r in faithful),8),
               'detection_given_both_fidelities':fraction(sum(positive(r) for r in faithful),len(faithful)),
               'author_fidelity':dict(Counter(r['author_fidelity']['status'] for r in b)),
               'rewrite_fidelity':dict(Counter(r['rewrite_fidelity']['status'] for r in b))}
        for variant,part in [('original',a),('reworded',b)]:
            group[variant+'_exact_paths']=sum(r['agreement']['exact'] for r in part)
            group[variant+'_correct_decisions']=sum(r['agreement']['correct'] for r in part)
            group[variant+'_recovered_decisions']=sum(r['agreement']['predicted'] for r in part)
            group[variant+'_abstentions']=sum(r['detection']['detected'] is None for r in part)
            group[variant+'_lexical_exact_paths']=sum(r['lexical_agreement']['exact'] for r in part)
            group[variant+'_lexical_correct_decisions']=sum(r['lexical_agreement']['correct'] for r in part)
            group[variant+'_lexical_detection']=fraction(sum(r['lexical_detection']['detected'] is True for r in part),8)
        summary[condition]=group
    bank=read(root/'probability_bank/bank_summary.json');entries=bank['entries'];assert len(entries)==128
    cover=[e['covered_probability_mass'] for e in entries]
    probabilities={'contexts':128,'terminated_sequences':384,'coverage_min':min(cover),'coverage_mean':statistics.mean(cover),'coverage_max':max(cover),
                   'mean_entropy_nats':statistics.mean(-sum(p*math.log(p) for p in e['itoken_probabilities'].values() if p) for e in entries),
                   'max_checked_logprob_difference':max(e['max_checked_logprob_difference'] for e in entries)}
    return {'challenge':challenge,'paired':summary,'probabilities':probabilities},rows,challenge_rows

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT/'research_results/natural_language_v1');ap.add_argument('--write',action='store_true');ap.add_argument('--challenge-only',action='store_true')
    a=ap.parse_args()
    if a.challenge_only:
        summary,rows=challenge_summary(a.root)
        if a.write:save(a.root/'challenge_summary.json',summary);save(a.root/'challenge_rows.json',{'rows':rows})
    else:
        summary,rows,ch= audit(a.root)
        if a.write:save(a.root/'summary.json',summary);save(a.root/'paired_rows.json',{'rows':rows});save(a.root/'challenge_rows.json',{'rows':ch})
        else:assert summary==read(a.root/'summary.json') and {'rows':rows}==read(a.root/'paired_rows.json')
    print(json.dumps(summary,indent=2))
