"""Supporting packets and replay for the separately declared follow-up."""
import argparse
from pathlib import Path
from study import ROOT, EXTRACT, FIDELITY, Encoder, admitted, groups, outputs, packet, read, save, test


def prepare():
    for group,pairs in groups():
        if not (ROOT/'calls'/('edit'+group)/'out.json').exists():continue
        for unit,variant in pairs:
            ident=variant['id']
            if ident not in outputs('edit'+group):continue
            text=outputs('edit'+group)[ident]['text']
            packet('editread'+group,EXTRACT,[{'id':ident,'problem':unit['problem'],'text':text}])
            comparison=[{'id':ident,'original':variant['text'],'rewrite':text}]
            for role in ['editjudge','editcheck']:packet(role+group,FIDELITY,comparison)


def score(path):
    keys=read(ROOT/'keys.json');enc=Encoder(path);rows=[]
    for group,pairs in groups():
        for unit,variant in pairs:
            ident=variant['id'];edited=outputs('edit'+group)
            if ident not in edited:continue
            text=edited[ident]['text'];answers=outputs('editread'+group)[ident];steps=admitted(answers,text)
            result=test(enc.vector(unit['problem'],steps),keys[unit['unit']],unit['problem'])
            original=next(r for r in read(ROOT/'results.json') if r['id']==ident)['original']
            assert original['detected'] is True
            rows.append({'unit':unit['unit'],'id':ident,'condition':'unmarked' if group=='1' else 'marked',
                         'original_p':original['p'],'edited':result,'retained':len(steps),'proposed':len(answers['steps']),
                         'fidelity':[outputs(role+group)[ident] for role in ['editjudge','editcheck']]})
    save(ROOT/'copyedit_results.json',rows)
    for r in rows: print(r['condition'],r['original_p'],r['edited']['p'],[f['status'] for f in r['fidelity']])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','score']);parser.add_argument('--model',type=Path);args=parser.parse_args()
    if args.phase=='prepare':prepare()
    else:score(args.model)
