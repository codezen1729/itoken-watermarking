"""Independently replay natural-language score, sampling and artifact arithmetic.

No model is invoked. Run with assertions enabled. Probability aggregation,
exponential tilting, route draws and HMAC tests are implemented independently
of natural_planner and watermark. Catalog semantics remain human/model
annotations, not machine-proved entailments.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import hmac
from itertools import accumulate
import json
import math
from pathlib import Path
import random
import re
import statistics

from natural_recovery import make_extraction_prompt, make_verification_prompt, parse_proposal

ROOT=Path(__file__).resolve().parent
DEFAULT_RUN=ROOT/'research_results/natural_language_v1'
PROTOCOL='natural-language-choices-v1'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(actual,expected):
    assert math.isclose(actual,expected,rel_tol=5e-12,abs_tol=1e-300),(actual,expected)


def canonical(decision,option):
    return json.dumps({'decision':decision['question'],'mechanism':option['mechanism'],
                       'consequence':option['consequence']},sort_keys=True,ensure_ascii=False,separators=(',',':'))


def context(problem,decision):
    fingerprint=hashlib.sha256(json.dumps(problem,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return json.dumps({'protocol':PROTOCOL,'problem':problem['id'],'decision':decision['id'],
                       'problem_sha256':fingerprint},sort_keys=True,separators=(',',':'))


def scoring_request(problem,decision):
    candidates=[{'text':o['application'],'itoken_id':canonical(decision,o)} for o in decision['options']]
    seed=hashlib.sha256((problem['id']+'\0'+decision['id']).encode()).hexdigest()
    random.Random(seed).shuffle(candidates)
    state={'problem':problem['problem'],'goal':problem['goal'],'decision':decision['question'],'context':decision['context']}
    prompt=('Choose one next planning step using the local decision below. Return exactly one listed statement and stop. '
            'No other planning choices have been supplied.\n'+json.dumps(state,ensure_ascii=False,sort_keys=True)+
            '\nCANDIDATE STATEMENTS:\n'+'\n'.join(c['text'] for c in candidates))
    return candidates,[{'role':'user','content':prompt}]


def replay_bank(run,catalog):
    bank=run/'probability_bank';summary=read(bank/'bank_summary.json');freeze=read(bank/'bank_manifest.json')
    assert summary['freeze']==freeze and freeze['protocol']==PROTOCOL
    assert freeze['catalog_sha256']==digest(ROOT/'natural_ideas_catalog.json')
    for filename,sha in freeze['source_sha256'].items():assert digest(ROOT/filename)==sha
    metadata=freeze['model_metadata']
    assert metadata['source_repo']=='mlx-community/Qwen2.5-0.5B-Instruct-4bit'
    assert metadata['revision']=='a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'
    assert metadata['execution_mode']=='dense-fp32'
    eos=metadata['scoring_eos_token_id'];assert eos==151645
    entries={(e['problem_id'],e['decision_id']):e for e in summary['entries']}
    assert len(entries)==len(summary['entries'])==128
    probabilities={};coverage=[];entropy=[];checks=[];sequence_count=0
    for pi,problem in enumerate(catalog['problems']):
        assert len(problem['decisions'])==16
        for di,decision in enumerate(problem['decisions']):
            identity=(problem['id'],decision['id']);entry=entries[identity]
            assert entry['path']==f'problem_{pi:02d}_decision_{di:02d}.json'
            raw=read(bank/entry['path']);scores=raw['scores'];expected,messages=scoring_request(problem,decision)
            assert raw['freeze']==freeze and raw['messages']==messages
            assert raw['problem_id']==problem['id'] and raw['decision_id']==decision['id']
            assert scores['metadata']==metadata
            candidates=scores['candidates'];assert len(candidates)==3
            assert [(c['text'],c['itoken_id']) for c in candidates]==[(c['text'],c['itoken_id']) for c in expected]
            assert len({tuple(c['token_ids']) for c in candidates})==3
            labels=[];logs=[]
            for candidate in candidates:
                ids,values=candidate['token_ids'],candidate['token_logprobs']
                assert len(ids)==len(values) and ids[-1]==eos and ids.count(eos)==1
                assert candidate['includes_termination'] and candidate['termination_token_id']==eos
                assert candidate['context_id']==scores['context_id']
                assert all(math.isfinite(value) and value<=1e-6 for value in values)
                total=math.fsum(values);close(total,candidate['sequence_log_probability'])
                labels.append(candidate['itoken_id']);logs.append(total);sequence_count+=1
            assert len(set(labels))==3
            shift=max(logs);scaled=[math.exp(x-shift) for x in logs];z=math.fsum(scaled)
            log_coverage=shift+math.log(z);covered=math.exp(log_coverage)
            close(log_coverage,scores['log_covered_probability_mass'])
            close(covered,scores['covered_probability_mass']);close(covered,entry['covered_probability_mass'])
            p={name:value/z for name,value in zip(labels,scaled)}
            for table in (scores['itoken_probabilities'],entry['itoken_probabilities']):
                assert table.keys()==p.keys()
                for name,value in p.items():close(value,table[name])
            for value,probability in zip(scaled,scores['candidate_probabilities'],strict=True):close(value/z,probability)
            crosschecks=scores['uncached_crosschecks'];expected_indices=set(range(3)) if di==0 else {0}
            assert {c['checked_candidate_index'] for c in crosschecks}==expected_indices
            differences=[]
            for check in crosschecks:
                cached=candidates[check['checked_candidate_index']]['token_logprobs'];uncached=check['uncached_token_logprobs']
                difference=max(abs(a-b) for a,b in zip(cached,uncached,strict=True))
                close(difference,check['max_absolute_token_logprob_difference'])
                assert difference<=freeze['numerical_tolerance_nats_per_token']==.05
                differences.append(difference)
            close(max(differences),entry['max_checked_logprob_difference'])
            close(max(differences),raw['max_checked_logprob_difference'])
            checks.extend(differences);coverage.append(covered)
            entropy.append(-math.fsum(v*math.log(v) for v in p.values() if v));probabilities[identity]=p
    assert len(probabilities)==128 and sequence_count==384 and len(checks)==144
    print(f'Bank: 128 contexts, 384 terminated sequences, {len(checks)} uncached checks; maximum discrepancy {max(checks):.8g} nats/token')
    print(f'Coverage min/mean/max {min(coverage):.8g}/{statistics.mean(coverage):.8g}/{max(coverage):.8g}; mean entropy {statistics.mean(entropy):.8g} nats')
    return probabilities


def preferred(key,ctx,token):
    value=json.dumps([ctx,token],ensure_ascii=False,separators=(',',':')).encode()
    return hmac.digest(key,b'itoken-watermark:v1\0'+value,'sha256')[0]<128


def detector(problem,choices,key,surrogates):
    assert set(choices)<={d['id'] for d in problem['decisions']}
    pairs=[]
    for decision in problem['decisions']:
        if decision['id'] not in choices:continue
        option=next(o for o in decision['options'] if o['id']==choices[decision['id']])
        pairs.append((context(problem,decision),canonical(decision,option)))
    if not pairs:return {'n_choices':0,'p_value':None,'detected':None,'preferred':None}
    assert len(pairs)==len(set(pairs))
    observed=sum(preferred(key,ctx,token) for ctx,token in pairs)
    null=[sum(preferred(k,ctx,token) for ctx,token in pairs) for k in surrogates]
    p=(1+sum(score>=observed for score in null))/(len(surrogates)+1)
    return {'n_choices':len(pairs),'p_value':p,'detected':p<=.01,'preferred':observed}


def replay_cohort(run,catalog,probabilities):
    cohort=run/'cohort';config=read(cohort/'config.json');index=read(cohort/'index.json')
    assert config==index['config']
    assert (config['units'],config['decisions_per_unit'],config['delta'],config['seed'],config['alpha'],config['surrogate_count'])==(8,16,[0,6],1701,.01,199)
    assert config['bank_sha256']==digest(run/'probability_bank/bank_summary.json')
    assert config['catalog_sha256']==digest(ROOT/'natural_ideas_catalog.json')
    for filename,sha in config['source_sha256'].items():assert digest(ROOT/filename)==sha
    assert len(index['documents'])==16 and len({d['document_id'] for d in index['documents']})==16
    all_keys=[];plans={};p_values=defaultdict(list);changes=[]
    for unit,problem in enumerate(catalog['problems']):
        record=read(cohort/f'unit_{unit:02d}_key.json')
        assert record['unit_id']==unit and record['problem_id']==problem['id'] and record['sampling_seed']==1701+unit
        key=bytes.fromhex(record['public_demonstration_key_hex']);surrogates=list(map(bytes.fromhex,record['surrogate_keys_hex']))
        assert len(surrogates)==199;all_keys.extend([key,*surrogates])
        entries=[e for e in index['documents'] if e['unit_id']==unit]
        assert sorted(e['condition'] for e in entries)==['marked','unmarked']
        choices_by_condition={}
        for entry in entries:
            ident,condition=entry['document_id'],entry['condition'];plan=read(cohort/'plans'/f'{ident}.json')
            assert ident not in plans and plan['document_id']==ident and plan['condition']==condition
            assert all(plan[k]==v for k,v in record.items());plans[ident]=plan
            rng=random.Random(record['sampling_seed']);expected_choices={};llr=0
            for decision,row in zip(problem['decisions'],plan['policy_rows'],strict=True):
                assert row['decision_id']==decision['id'];p=probabilities[(problem['id'],decision['id'])];ctx=context(problem,decision)
                weights={token:value*math.exp(6*preferred(key,ctx,token)) for token,value in p.items()}
                z=math.fsum(weights.values());alternative={token:value/z for token,value in weights.items()}
                q=alternative if condition=='marked' else p
                assert p.keys()==row['p'].keys()==row['q'].keys()
                for token in p:close(p[token],row['p'][token]);close(q[token],row['q'][token])
                u=rng.random();close(u,row['uniform'])
                selected=next(token for token,limit in zip(q,accumulate(q.values())) if u<limit)
                assert selected==row['selected']
                option=next(o for o in decision['options'] if canonical(decision,o)==selected)
                expected_choices[decision['id']]=option['id'];llr+=math.log(alternative[selected]/p[selected])
            assert expected_choices==plan['choices'];close(llr,plan['planned_log_likelihood_ratio'])
            score=detector(problem,expected_choices,key,surrogates)
            assert score==plan['planned_detection']==entry['planned_detection']
            p_values[condition].append(score['p_value']);choices_by_condition[condition]=expected_choices
        changes.append(sum(choices_by_condition['marked'][d]!=o for d,o in choices_by_condition['unmarked'].items()))
    assert len(all_keys)==len(set(all_keys))==1600 and all(len(k)==32 for k in all_keys)
    print(f'Cohort: 256 paired draws replayed, 16 likelihood ratios, 1,600 distinct 256-bit keys; changed choices per pair {changes}')
    for condition,values in p_values.items():print(f'{condition} planned p-values: {values}')
    return plans


def _unique(rows,key):
    result={r[key]:r for r in rows};assert len(result)==len(rows);return result


def check_artifacts(run,catalog,plans):
    artifacts=run/'agent_artifacts';problems={p['id']:p for p in catalog['problems']}
    originals={};rewrites={};seen=defaultdict(set)
    for number in range(1,5):
        author=artifacts/f'author_{number:02d}';rewrite=artifacts/f'rewrite_{number:02d}'
        requests=_unique(read(author/'requests.json')['documents'],'document_id');responses=_unique(read(author/'responses.json')['documents'],'document_id')
        assert set(requests)==set(responses) and len(requests)==4
        assert len({plans[i]['condition'] for i in requests})==1
        for ident,request in requests.items():
            plan=plans[ident];problem=problems[plan['problem_id']]
            expected=[]
            for decision in problem['decisions']:
                option=next(o for o in decision['options'] if o['id']==plan['choices'][decision['id']])
                expected.append({'question':decision['question'],'context':decision['context'],'application':option['application']})
            assert request=={'document_id':ident,'problem':problem['problem'],'goal':problem['goal'],'selected_applications':expected}
            originals[ident]=responses[ident]['text']
        rewritten_requests=_unique(read(rewrite/'requests.json')['documents'],'document_id')
        rewritten_responses=_unique(read(rewrite/'responses.json')['documents'],'document_id')
        assert set(rewritten_requests)==set(rewritten_responses)==set(requests)
        for ident,request in rewritten_requests.items():
            assert request=={'document_id':ident,'text':originals[ident]};rewrites[ident]=rewritten_responses[ident]['text']
        author_assess=read(artifacts/f'authorship_{number:02d}'/'requests.json')['documents']
        rewrite_assess=read(artifacts/f'fidelity_{number:02d}'/'requests.json')['documents']
        assert {r['document_id'] for r in author_assess}=={r['document_id'] for r in rewrite_assess}==set(requests)
        for row in author_assess:
            ident=row['document_id'];problem=problems[plans[ident]['problem_id']]
            intended=[]
            for decision in problem['decisions']:
                option=next(o for o in decision['options'] if o['id']==plans[ident]['choices'][decision['id']])
                intended.append({'decision_id':decision['id'],'application':option['application']})
            assert row=={'document_id':ident,'intended_applications':intended,'text':originals[ident]}
        for row in rewrite_assess:
            ident=row['document_id'];problem=problems[plans[ident]['problem_id']]
            assert row=={'document_id':ident,'decision_topics':[{'decision_id':d['id'],'question':d['question']} for d in problem['decisions']],
                         'original':originals[ident],'rewrite':rewrites[ident]}
    assert set(originals)==set(rewrites)==set(plans)
    marker_docs=[]
    for number in range(1,9):
        folder=artifacts/f'recovery_{number:02d}'
        assignment=read(folder/'assignment_manifest.json') if (folder/'assignment_manifest.json').exists() else None
        if assignment is not None:
            encoded=json.dumps(assignment['assignment'],sort_keys=True,ensure_ascii=False).encode()
            assert assignment['assignment_sha256']==hashlib.sha256(encoded).hexdigest()
            assert assignment['wrapper_source_sha256']==digest(ROOT/'natural_failures.py')
            assigned=assignment['assignment']['cases']
            blanks=[c for c in assigned if not c['text'].strip()]
            assert assignment['empty_case_ids']==[c['id'] for c in blanks]
        else:
            assigned=None;blanks=[]
        if assignment is not None and not assignment['transport_invoked']:
            assert all(not c['text'].strip() for c in assigned)
            bindings=[]
        else:
            manifest=read(folder/'private_manifest.json');bindings=manifest['bindings']
            if assigned is not None:assert [b['case'] for b in bindings]==[c for c in assigned if c['text'].strip()]
        proposals=_unique(read(folder/'proposals/requests.json')['requests'],'request_id')
        answers={r['request_id']:r['response'] for r in read(folder/'proposals/responses.json')['responses']} if proposals else {}
        verifiers=_unique(read(folder/'verification/requests.json')['requests'],'request_id') if (folder/'verification/requests.json').exists() else {}
        expected_verifiers=set()
        assert set(proposals)==set(answers)=={b['request_id'] for b in bindings}
        variant='original' if number<=4 else 'reworded';texts=originals if variant=='original' else rewrites
        batch_cases=assigned if assigned is not None else [b['case'] for b in bindings]
        assert len({plans[c['id']]['condition'] for c in batch_cases})==1
        for case in blanks:
            ident=case['id'];assert case['text']==texts[ident] and not case['text'].strip()
            assert ident not in seen[variant];seen[variant].add(ident)
        for binding in bindings:
            rid,case=binding['request_id'],binding['case'];ident=case['id'];text=case['text'];problem=problems[plans[ident]['problem_id']]
            assert text==texts[ident] and case['problem']==problem and ident not in seen[variant];seen[variant].add(ident)
            assert proposals[rid]['prompt']==make_extraction_prompt(problem,text)
            try:proposal=parse_proposal(problem,text,answers[rid])
            except (ValueError,TypeError):proposal={'choices':[]}
            if proposal['choices']:
                expected_verifiers.add(rid);assert verifiers[rid]['prompt']==make_verification_prompt(problem,text,proposal)
            forbidden=[problem['id'],*(d['id'] for d in problem['decisions']),*(o['id'] for d in problem['decisions'] for o in d['options'])]
            if any(name in text for name in forbidden) or re.search(r'\b[pmde]\d+\b|[∧∨⇒]|->',text):marker_docs.append((ident,variant))
        assert set(verifiers)==expected_verifiers
    assert seen['original']==seen['reworded']==set(plans)
    paired=read(run/'paired_rows.json')['rows'];assert len(paired)==32
    for row in paired:
        plan=plans[row['document_id']];problem=problems[plan['problem_id']]
        key=bytes.fromhex(plan['public_demonstration_key_hex']);keys=list(map(bytes.fromhex,plan['surrogate_keys_hex']))
        assert detector(problem,row['recovery']['choices'],key,keys)==row['detection']
        assert detector(problem,row['lexical']['choices'],key,keys)==row['lexical_detection']
    call_file=artifacts/'agent_calls.json'
    if call_file.exists():
        calls=read(call_file)['calls']
        assert len({c['agent'] for c in calls})==len(calls)
        for call in calls:
            assert digest(artifacts/call['input'])==call['input_sha256']
            assert digest(artifacts/call['output'])==call['output_sha256']
            assert call['attempt']==1 and call['fork_turns']=='none'
        print(f'Agent call hashes: {len(calls)} batches, {sum(c["items"] for c in calls)} assigned items')
    print(f'Artifacts: all16 author/rewrite pairs, all32 source-only recovery cases, independent fidelity inputs, 64 semantic/lexical detection scores verified; explicit marker documents {len(marker_docs)}')
    return marker_docs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--stage',choices=['bank','cohort','all'],default='all')
    args=parser.parse_args();catalog=read(ROOT/'natural_ideas_catalog.json');assert len(catalog['problems'])==8
    bank=replay_bank(args.root,catalog)
    if args.stage!='bank':
        plans=replay_cohort(args.root,catalog,bank)
        if args.stage=='all':check_artifacts(args.root,catalog,plans)
