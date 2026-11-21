"""Recompute the stronger-agent study from retained first-response artifacts.

This audits arithmetic, artifact consistency and protocol accounting. It cannot
verify a hosted model's identity or replace independent semantic annotation.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics

from blinded_recovery import _load, _response_map, _hash
from core import IToken, make_problem, parse_proof
from recovery_v2 import parse_extraction_output
from real_model_pilot import detect_structural
from research_runner import digest
from rewording_audit import summarize_pairs, summarize_recovery

ROOT=Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def replay_batch(path):
    manifest,requests=_load(path)
    proposals=_response_map(path/'proposals/responses.json',requests)
    vreq=read(path/'verification/requests.json')['requests']
    binding=read(path/'verification/private_binding.json')
    assert binding['requests_sha256']==_hash(vreq)
    assert binding['proposal_responses_sha256']==digest(path/'proposals/responses.json')
    verifications=_response_map(path/'verification/responses.json',vreq) if vreq else {}
    stored=read(path/'results.json')
    assert stored['freeze']==manifest['freeze']
    assert stored['proposal_response_artifact_sha256']==digest(path/'proposals/responses.json')
    assert stored['verification_response_artifact_sha256']==(digest(path/'verification/responses.json') if vreq else None)
    assert len(stored['rows'])==len(manifest['bindings'])
    rows={r['id']:r for r in stored['rows']}
    assert len(rows)==len(stored['rows'])
    for b in manifest['bindings']:
        c,rid=b['case'],b['request_id']; row=rows[c['id']]
        recovered=parse_extraction_output(make_problem(c['n_stages'],c['problem_id']),c['text'],proposals[rid],verifications.get(rid)).to_dict()
        assert recovered==row['recovery'],c['id']
        assert row['text']==c['text'] and row['request_id']==rid
        assert row['proposal_response']==proposals[rid]
        assert row['verification_response']==verifications.get(rid)
    return rows

def tokens(values):
    result=[]
    for value in values:
        if isinstance(value,str):value=json.loads(value)
        result.append(IToken(value['rule'],tuple(value['premises']),value['conclusion']))
    return result

def audit(root):
    artifacts=root/'agent_artifacts'
    held={}
    for batch in ('holdout_batch_00','holdout_batch_01'):
        for ident,row in replay_batch(artifacts/batch).items():
            assert ident not in held
            held[ident]=row['recovery']
    holdout=summarize_recovery(read(ROOT/'fresh_holdout.json')['cases'],held)
    config=read(root/'cohort/cohort_config.json')
    assert (config['units'],config['stages'],config['marked_delta'],config['unmarked_delta'],config['alpha'],config['surrogate_count'])==(8,32,6,0,.01,199)
    assert config['bank_sha256']==digest(root/'probability_bank/bank_summary.json')
    for name,sha in config['source_sha256'].items():assert digest(ROOT/name)==sha
    bank=read(root/'probability_bank/bank_summary.json')
    stages=bank['stages'];assert len(stages)==32
    coverage=[s['covered_probability_mass'] for s in stages]
    entropy=[-sum(p*math.log(p) for p in s['itoken_probabilities'].values()) for s in stages]
    index=read(root/'cohort/cohort_index.json')['documents']
    byid={d['document_id']:d for d in index}
    assert len(byid)==16
    recovered={};rewrites={};fidelity={};similarities=[]
    for n in range(1,9):
        rows=replay_batch(artifacts/f'recovery_{n:02d}')
        conditions={byid[ident]['condition'] for ident in rows}
        assert len(conditions)==1,'Mixed-condition model context'
        variant='original' if n<=4 else 'reworded'
        for ident,row in rows.items():
            identity=(ident,variant);assert identity not in recovered
            recovered[identity]=row
    for n in range(1,5):
        job=artifacts/f'language_{n:02d}'
        request=read(job/'requests.json');response=read(job/'responses.json')
        inputs={d['document_id']:d for d in request['documents']}
        output=response['rewrites'];assert len(output)==len(inputs)
        assert len({r['document_id'] for r in output})==len(output)
        assert {r['document_id'] for r in output}==set(inputs)
        assert len({byid[i]['condition'] for i in inputs})==1
        fjob=artifacts/f'comparison_{n:02d}'
        frequest=read(fjob/'requests.json');fresponse=read(fjob/'responses.json')
        comparisons={d['document_id']:d for d in frequest['documents']}
        assert set(comparisons)==set(inputs)
        assessments=fresponse['assessments'];assert len(assessments)==len(inputs)
        assert {a['document_id'] for a in assessments}==set(inputs)
        for a in assessments:
            assert a['status'] in {'preserved','changed','uncertain'}
            counts=a['original_applications']
            assert set(counts)=={'preserved','omitted','changed','uncertain'}
            assert all(type(v) is int and v>=0 for v in counts.values()) and sum(counts.values())==64
            assert isinstance(a['additional_applications'],list)
            if a['status']=='preserved':assert counts['preserved']==64 and not a['additional_applications']
            fidelity[a['document_id']]=a
        for r in output:
            ident,text=r['document_id'],r['text'];source=inputs[ident]['text']
            assert comparisons[ident]['original']==source and comparisons[ident]['reworded']==text
            assert recovered[(ident,'original')]['text']==source
            assert recovered[(ident,'reworded')]['text']==text
            try: parse_proof(text); old_parser_accepts=True
            except ValueError: old_parser_accepts=False
            similarities.append({'document_id':ident,'text_changed':text!=source,'old_template_parser_accepts':old_parser_accepts,
                                 'token_sequence_similarity':difflib.SequenceMatcher(None,source.split(),text.split(),autojunk=False).ratio()})
            rewrites[ident]=text
    assert len(recovered)==32 and len(rewrites)==16 and len(fidelity)==16
    paired=[]
    for ident,entry in byid.items():
        plan=read(root/'cohort/private_plans'/f'{ident}.json')
        problem=make_problem(plan['n_stages'],plan['problem_id'])
        key=bytes.fromhex(plan['public_demonstration_key_hex']);surrogates=[bytes.fromhex(k) for k in plan['surrogate_keys_hex']]
        planned=detect_structural(problem,tokens(plan['planned_canonical']),key,surrogates)
        assert planned==plan['planned_detection']==entry['planned_detection']
        expected=set(plan['planned_canonical'])
        for variant in ('original','reworded'):
            row=recovered[(ident,variant)];recovery=row['recovery']
            steps=tokens(recovery['steps']) if recovery['accepted'] else []
            detection=detect_structural(problem,steps,key,surrogates)
            observed={s.canonical() for s in steps}
            paired.append({'document_id':ident,'unit_id':plan['unit_id'],'condition':plan['condition'],'variant':variant,
                'planned_detected':planned['detected'],'planned_detection':planned,'detection':detection,'recovery':recovery,
                'fidelity':fidelity[ident] if variant=='reworded' else {'status':'not_assessed'},
                'path_agreement':{'exact_steps':observed==expected,'correct_steps':len(observed&expected),'gold_steps':len(expected),
                                  'missing_steps':sorted(expected-observed),'extra_steps':sorted(observed-expected)}})
    summary=summarize_pairs(paired)
    for condition in ('unmarked','marked'):
        for variant in ('original','reworded'):
            group=[r for r in paired if r['condition']==condition and r['variant']==variant]
            summary[condition][variant+'_exact_paths']=sum(r['path_agreement']['exact_steps'] for r in group)
            summary[condition][variant+'_correct_steps']=sum(r['path_agreement']['correct_steps'] for r in group)
    return {'holdout':holdout,'paired':summary,
            'probabilities':{'contexts':32,'terminated_sequences':288,'coverage_min':min(coverage),'coverage_max':max(coverage),
                            'coverage_mean':statistics.mean(coverage),'mean_itoken_entropy_nats':statistics.mean(entropy),
                            'max_checked_logprob_difference':max(s['max_checked_logprob_difference'] for s in stages)},
            'rewording_surface':similarities},paired

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,nargs='?',default=ROOT/'research_results/strong_recovery_v2')
    parser.add_argument('--write',action='store_true',help='write recomputed summary and paired rows; never invoke models')
    args=parser.parse_args();summary,rows=audit(args.root)
    if args.write:
        from research_runner import save
        save(args.root/'summary.json',summary);save(args.root/'paired_rows.json',{'rows':rows})
    else:
        assert read(args.root/'summary.json')==summary
        assert read(args.root/'paired_rows.json')=={'rows':rows}
    print(json.dumps(summary,indent=2))
