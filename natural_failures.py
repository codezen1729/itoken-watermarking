"""Preserve assigned empty texts without changing the frozen recovery protocol.

Nonempty cases use natural_transport unchanged. Empty or whitespace-only source
texts receive explicit failure records, with no invented placeholder and no LLM
call. Invalid response bundles still fail closed at their transport boundary.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import natural_transport as transport
from research_runner import save,digest

ROOT=Path(__file__).resolve().parent
PROTOCOL='natural-empty-source-accounting-v1'


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def _empty(case):
    text=case.get('text')
    if not isinstance(text,str):raise ValueError('Source text must be a string, including for an empty-source failure')
    return not text.strip()


def prepare_recovery(cases,out,*,label,batch_group):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh recovery directory required')
    if not cases or len({c['id'] for c in cases})!=len(cases):raise ValueError('Need nonempty unique assigned cases')
    nonempty=[c for c in cases if not _empty(c)]
    assignment={'cases':cases,'label':label,'batch_group':batch_group}
    manifest={'protocol':PROTOCOL,'assignment':assignment,'assignment_sha256':_hash(assignment),
              'empty_case_ids':[c['id'] for c in cases if _empty(c)],
              'wrapper_source_sha256':digest(Path(__file__)),
              'transport_invoked':bool(nonempty)}
    if nonempty:
        freeze=transport.prepare(nonempty,out,label=label,batch_group=batch_group)
    else:
        freeze={'protocol':PROTOCOL,'model_invoked':False,'reason':'all assigned source texts are empty'}
        save(out/'proposals/requests.json',{'requests':[]})
    save(out/'assignment_manifest.json',manifest)
    return freeze


def _load(out):
    out=Path(out);manifest=json.loads((out/'assignment_manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['wrapper_source_sha256']!=digest(Path(__file__)):
        raise ValueError('Empty-source wrapper changed')
    assignment=manifest['assignment']
    if manifest['assignment_sha256']!=_hash(assignment):raise ValueError('Full assignment binding changed')
    cases=assignment['cases'];nonempty=[c for c in cases if not _empty(c)]
    if manifest['empty_case_ids']!=[c['id'] for c in cases if _empty(c)] or manifest['transport_invoked']!=bool(nonempty):
        raise ValueError('Empty-source classification changed')
    if nonempty:
        bound,_=transport.load(out)
        if [b['case'] for b in bound['bindings']]!=nonempty:raise ValueError('Transport omits or substitutes a nonempty assignment')
        if bound['freeze']['label']!=assignment['label'] or bound['freeze']['batch_group']!=assignment['batch_group']:
            raise ValueError('Transport group differs from full assignment')
    elif json.loads((out/'proposals/requests.json').read_text())!={'requests':[]}:
        raise ValueError('An all-empty batch must not contain an LLM request')
    return manifest,nonempty


def prepare_verification(out):
    out=Path(out);_,nonempty=_load(out)
    if nonempty:return transport.prepare_verification(out)
    path=out/'verification/requests.json'
    if path.exists():raise ValueError('Verification preparation already recorded')
    save(path,{'requests':[]})
    return 0


def _failure(case):
    decisions=sorted(d['id'] for d in case['problem']['decisions'])
    return {'id':case['id'],'problem_id':case['problem']['id'],'text':case['text'],
            'proposal_response':None,'verification_response':None,
            'recovery':{'status':'empty_source','accepted':False,'complete':False,'choices':{},
                        'ambiguous_decisions':[],'unresolved_decisions':decisions,'evidence':{},'source_spans':{},
                        'model_verification_performed':False,'semantic_entailment_verified':False,
                        'reason':'The assigned raw source is empty or whitespace-only; no model call was made.'}}


def replay_recovery(out):
    out=Path(out);manifest,nonempty=_load(out)
    if nonempty:
        result=transport.replay(out)
        if not manifest['empty_case_ids']:return result
        admitted={r['id']:r for r in result['rows']}
    else:
        path=out/'verification/requests.json'
        if path.exists() and json.loads(path.read_text())!={'requests':[]}:
            raise ValueError('An all-empty batch must not contain verifier requests')
        result={'freeze':{'protocol':PROTOCOL,'model_invoked':False,'reason':'all assigned source texts are empty'},
                'proposal_responses_sha256':None,'verification_responses_sha256':None}
        admitted={}
    rows=[]
    for case in manifest['assignment']['cases']:
        rows.append(_failure(case) if _empty(case) else admitted[case['id']])
    result['rows']=rows
    result['failure_accounting']={'protocol':PROTOCOL,'assignment_sha256':manifest['assignment_sha256'],
                                 'wrapper_source_sha256':manifest['wrapper_source_sha256'],
                                 'assigned_cases':len(rows),'empty_case_ids':manifest['empty_case_ids']}
    return result


def finalize_recovery(out):
    out=Path(out)
    if (out/'results.json').exists():raise ValueError('Already finalized; first responses cannot be rerolled')
    result=replay_recovery(out);save(out/'results.json',result);return result['rows']
