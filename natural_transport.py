"""Frozen file transport for actual isolated natural-language recovery agents."""
from pathlib import Path
import hashlib
import json
import secrets

from blinded_recovery import MODEL, _response_map, _hash
from natural_recovery import make_extraction_prompt, parse_proposal, make_verification_prompt, parse_extraction_output
from research_runner import save,digest

ROOT=Path(__file__).resolve().parent
DEPENDENCIES=('natural_transport.py','natural_recovery.py','recovery_v2.py','blinded_recovery.py','semantic_recovery.py','research_runner.py')

def prepare(cases,out,*,label,batch_group):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh batch required')
    if not cases or len({c['id'] for c in cases})!=len(cases):raise ValueError('Need unique nonempty cases')
    requests=[];bindings=[]
    for case in cases:
        rid=secrets.token_hex(12);prompt=make_extraction_prompt(case['problem'],case['text'])
        requests.append({'request_id':rid,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
        bindings.append({'request_id':rid,'case':case})
    freeze={'protocol':'natural-agent-transport-v1','label':label,'batch_group':batch_group,'model':MODEL,
            'cases_sha256':_hash(cases),'bindings_sha256':_hash(bindings),'source_sha256':{n:digest(ROOT/n) for n in DEPENDENCIES},'proposal_requests_sha256':_hash(requests)}
    save(out/'private_manifest.json',{'freeze':freeze,'bindings':bindings})
    save(out/'proposals/requests.json',{'requests':requests})
    return freeze

def load(out):
    out=Path(out);manifest=json.loads((out/'private_manifest.json').read_text());requests=json.loads((out/'proposals/requests.json').read_text())['requests']
    if manifest['freeze']['source_sha256']!={n:digest(ROOT/n) for n in DEPENDENCIES}:raise ValueError('Recovery source drift')
    if manifest['freeze']['proposal_requests_sha256']!=_hash(requests):raise ValueError('Prompt drift')
    if manifest['freeze']['bindings_sha256']!=_hash(manifest['bindings']):raise ValueError('Request-to-case binding drift')
    if manifest['freeze']['cases_sha256']!=_hash([b['case'] for b in manifest['bindings']]):raise ValueError('Case drift')
    return manifest,requests

def prepare_verification(out):
    out=Path(out);manifest,requests=load(out);responses=_response_map(out/'proposals/responses.json',requests)
    if (out/'verification/requests.json').exists():raise ValueError('No verifier rerolls')
    prompts=[];early={}
    for b in manifest['bindings']:
        rid,c=b['request_id'],b['case']
        try:proposal=parse_proposal(c['problem'],c['text'],responses[rid])
        except (ValueError,TypeError):
            early[rid]=parse_extraction_output(c['problem'],c['text'],responses[rid]);continue
        if proposal['choices']:
            prompt=make_verification_prompt(c['problem'],c['text'],proposal)
            prompts.append({'request_id':rid,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
        else:early[rid]=parse_extraction_output(c['problem'],c['text'],responses[rid])
    save(out/'verification/requests.json',{'requests':prompts})
    save(out/'verification/private_binding.json',{'requests_sha256':_hash(prompts),'proposal_responses_sha256':digest(out/'proposals/responses.json'),'early_results':early})
    return len(prompts)

def replay(out):
    out=Path(out);manifest,requests=load(out);proposals=_response_map(out/'proposals/responses.json',requests)
    req=json.loads((out/'verification/requests.json').read_text())['requests'];binding=json.loads((out/'verification/private_binding.json').read_text())
    if binding['requests_sha256']!=_hash(req) or binding['proposal_responses_sha256']!=digest(out/'proposals/responses.json'):raise ValueError('Verifier packet drift')
    verifications=_response_map(out/'verification/responses.json',req) if req else {}
    rows=[]
    for b in manifest['bindings']:
        rid,c=b['request_id'],b['case']
        r=parse_extraction_output(c['problem'],c['text'],proposals[rid],verifications.get(rid))
        rows.append({'id':c['id'],'problem_id':c['problem']['id'],'text':c['text'],'recovery':r,
                     'proposal_response':proposals[rid],'verification_response':verifications.get(rid)})
    return {'freeze':manifest['freeze'],'rows':rows,'proposal_responses_sha256':digest(out/'proposals/responses.json'),
            'verification_responses_sha256':digest(out/'verification/responses.json') if req else None}

def finalize(out):
    out=Path(out)
    if (out/'results.json').exists():raise ValueError('Already finalized')
    result=replay(out);save(out/'results.json',result);return result['rows']
