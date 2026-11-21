"""Export blinded natural-language study packets; agents supply actual judgments.

This module does not call or simulate an LLM. Every phase preserves first raw
outputs and hashes. Model workers must be separate fresh contexts, grouped by
condition and variant as specified by the exported job manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from natural_failures import prepare_recovery as prepare
from research_runner import save,digest

ROOT=Path(__file__).resolve().parent
AUTHOR_INSTRUCTION='''Write a coherent natural-language proposal solving the supplied problem using every selected planning application. These are the intended decisions, not a menu to choose from. Express the mechanism and its consequence clearly for all sixteen decisions; you may integrate several decisions in a paragraph. Use ordinary prose, not numbered steps or option labels, and do not copy the supplied sentences verbatim. Do not put internal IDs, formal logic symbols, placeholder variables, or route tags in the text. References to alternatives are allowed only if their rejected or hypothetical status is clear. Do not silently replace, omit, or reverse a selected mechanism. A natural proposal of roughly 500–900 words is sufficient, but completeness takes priority over length. Return one document for every input, even if difficult.'''
REWRITE_INSTRUCTION='''Reword each proposal into natural prose that could plausibly be written by a different person. Preserve the decisions actually advocated, who does what, their order or timing where meaningful, and why the mechanisms lead to their stated consequences. Change sentence construction and organization where meaning allows; use synonyms and references freely. No particular words, phrases, labels, or atom names must be retained. Do not introduce formal symbols or identifiers. Do not replace the solution with a different mechanism that merely reaches the same goal. Keep distinctions between adopted, rejected, and hypothetical alternatives. Return a single first rewrite of every document; do not generate candidates and select among them.'''
AUTHOR_FIDELITY_INSTRUCTION='''Assess independently whether the authored text expresses each intended application. Check the action/mechanism and consequence, roles, timing, scope, and relevant premises. An endpoint match or shared words is insufficient. Do not repair missing or reversed reasoning. For every decision return preserved/changed/omitted/uncertain, explaining any non-preserved judgment with source evidence. Return an overall status of preserved/changed/uncertain and identify added conflicting decisions. The supplied applications are targets to compare with the text, not evidence that the text expressed them.'''
FIDELITY_INSTRUCTION='''Independently compare the original and rewrite. For every supplied decision topic, judge whether the rewrite preserves the mechanism the original actually advocates, its consequence, roles, timing, scope, and relevant premises. An endpoint match or shared words is insufficient. Do not repair missing or reversed reasoning. Return preserved/changed/omitted/uncertain for each decision and explain any non-preserved judgment with evidence. If a topic was absent from the original and remains absent, that may count as preserved; do not invent a missing original choice. Return an overall status of preserved/changed/uncertain and identify added conflicting decisions. No intended solution is supplied: compare only what these two texts actually say.'''


def read(path):return json.loads(Path(path).read_text())


def complete_map(path,key,ids):
    obj=read(path)
    if set(obj)!={key} or not isinstance(obj[key],list):raise ValueError('Wrong response bundle')
    rows=obj[key];found={}
    for row in rows:
        ident=row['document_id']
        if ident not in ids or ident in found:raise ValueError('Unknown/duplicate document')
        found[ident]=row
    if set(found)!=set(ids):raise ValueError('Missing assigned document')
    return found


def prepare_authors(catalog,cohort,jobs):
    jobs=Path(jobs)
    if jobs.exists() and any(jobs.iterdir()):raise ValueError('Fresh job root required')
    public=read(catalog);problems={p['id']:p for p in public['problems']};index=read(Path(cohort)/'index.json')
    if index['config']['catalog_sha256']!=digest(catalog):raise ValueError('Catalog drift')
    batches=[]
    for c,condition in enumerate(('unmarked','marked')):
        selected=[d for d in index['documents'] if d['condition']==condition]
        for b in range(2):
            number=2*c+b+1;docs=[]
            for entry in selected[b*4:b*4+4]:
                plan=read(Path(cohort)/'plans'/f"{entry['document_id']}.json");p=problems[plan['problem_id']]
                applications=[]
                for d in p['decisions']:
                    o=next(o for o in d['options'] if o['id']==plan['choices'][d['id']])
                    applications.append({'question':d['question'],'context':d['context'],'application':o['application']})
                docs.append({'document_id':entry['document_id'],'problem':p['problem'],'goal':p['goal'],'selected_applications':applications})
            if len(docs)!=4:raise ValueError('Locked batch needs four documents')
            save(jobs/f'author_{number:02d}'/'requests.json',{'instruction':AUTHOR_INSTRUCTION,'documents':docs})
            batches.append({'number':number,'condition':condition,'document_ids':[d['document_id'] for d in docs]})
    save(jobs/'private_jobs.json',{'catalog_sha256':digest(catalog),'cohort_index_sha256':digest(Path(cohort)/'index.json'),'batches':batches,
                                'author_instruction_sha256':hashlib.sha256(AUTHOR_INSTRUCTION.encode()).hexdigest(),
                                'author_fidelity_instruction_sha256':hashlib.sha256(AUTHOR_FIDELITY_INSTRUCTION.encode()).hexdigest(),
                                'rewrite_instruction_sha256':hashlib.sha256(REWRITE_INSTRUCTION.encode()).hexdigest(),
                                'fidelity_instruction_sha256':hashlib.sha256(FIDELITY_INSTRUCTION.encode()).hexdigest()})


def _setup(catalog,cohort,jobs):
    jobs=Path(jobs);manifest=read(jobs/'private_jobs.json')
    if manifest['catalog_sha256']!=digest(catalog) or manifest['cohort_index_sha256']!=digest(Path(cohort)/'index.json'):raise ValueError('Job context drift')
    for name,instruction in [('author',AUTHOR_INSTRUCTION),('author_fidelity',AUTHOR_FIDELITY_INSTRUCTION),('rewrite',REWRITE_INSTRUCTION),('fidelity',FIDELITY_INSTRUCTION)]:
        if manifest[name+'_instruction_sha256']!=hashlib.sha256(instruction.encode()).hexdigest():raise ValueError('Instructions changed')
    problems={p['id']:p for p in read(catalog)['problems']};plans={p['document_id']:p for p in [read(f) for f in (Path(cohort)/'plans').glob('*.json')]}
    return jobs,manifest,problems,plans


def import_authors(catalog,cohort,jobs):
    jobs,manifest,problems,plans=_setup(catalog,cohort,jobs)
    for batch in manifest['batches']:
        n=batch['number'];source=jobs/f'author_{n:02d}'/'responses.json';target=jobs/f'rewrite_{n:02d}'/'requests.json'
        if not source.exists() or target.exists():continue
        authored=complete_map(source,'documents',batch['document_ids']);cases=[];docs=[];comparisons=[]
        for ident in batch['document_ids']:
            text=authored[ident]['text']
            if not isinstance(text,str):raise ValueError('Text must be a string')
            p=problems[plans[ident]['problem_id']]
            cases.append({'id':ident,'problem':p,'text':text});docs.append({'document_id':ident,'text':text})
            intended=[]
            for d in p['decisions']:
                option=next(o for o in d['options'] if o['id']==plans[ident]['choices'][d['id']])
                intended.append({'decision_id':d['id'],'application':option['application']})
            comparisons.append({'document_id':ident,'intended_applications':intended,'text':text})
        prepare(cases,jobs/f'recovery_{n:02d}',label='natural-original',batch_group=f'c{batch["condition"]}v0b{n}')
        save(jobs/f'authorship_{n:02d}'/'requests.json',{'instruction':AUTHOR_FIDELITY_INSTRUCTION,'documents':comparisons})
        save(target,{'instruction':REWRITE_INSTRUCTION,'documents':docs})
        save(jobs/f'rewrite_{n:02d}'/'private_binding.json',{'source_response_sha256':digest(source)})
        print('Imported authors',n,flush=True)


def import_rewrites(catalog,cohort,jobs):
    jobs,manifest,problems,plans=_setup(catalog,cohort,jobs)
    for batch in manifest['batches']:
        n=batch['number'];source=jobs/f'rewrite_{n:02d}'/'responses.json';target=jobs/f'fidelity_{n:02d}'/'requests.json'
        if not source.exists() or target.exists():continue
        authored=complete_map(jobs/f'author_{n:02d}'/'responses.json','documents',batch['document_ids'])
        rewritten=complete_map(source,'documents',batch['document_ids']);cases=[];docs=[]
        for ident in batch['document_ids']:
            text=rewritten[ident]['text']
            if not isinstance(text,str):raise ValueError('Text must be a string')
            plan=plans[ident];p=problems[plan['problem_id']]
            cases.append({'id':ident,'problem':p,'text':text})
            topics=[{'decision_id':d['id'],'question':d['question']} for d in p['decisions']]
            docs.append({'document_id':ident,'decision_topics':topics,'original':authored[ident]['text'],'rewrite':text})
        prepare(cases,jobs/f'recovery_{n+4:02d}',label='natural-reworded',batch_group=f'c{batch["condition"]}v1b{n}')
        save(target,{'instruction':FIDELITY_INSTRUCTION,'documents':docs})
        save(jobs/f'fidelity_{n:02d}'/'private_binding.json',{'author_response_sha256':digest(jobs/f'author_{n:02d}'/'responses.json'),'rewrite_response_sha256':digest(source)})
        print('Imported rewrites',n,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['prepare','authors','rewrites'])
    ap.add_argument('--catalog',type=Path,default=ROOT/'natural_ideas_catalog.json');ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--jobs',type=Path,required=True)
    a=ap.parse_args();{'prepare':prepare_authors,'authors':import_authors,'rewrites':import_rewrites}[a.phase](a.catalog,a.cohort,a.jobs)
