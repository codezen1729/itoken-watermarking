"""Catalog-free detector: supply a public question, prose, a blind reading and verifier keys."""
import argparse
import json
from pathlib import Path

from packets import EXTRACT, read, validated
from trial import engines, identify
from watermark import MODES, test

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prompt','score'))
    parser.add_argument('--problem',required=True)
    parser.add_argument('--text',type=Path,required=True)
    parser.add_argument('--steps',type=Path)
    parser.add_argument('--keys',type=Path)
    parser.add_argument('--mode',choices=MODES,default='exact')
    args=parser.parse_args();text=args.text.read_text()
    if args.command=='prompt':
        print(json.dumps({'instruction':EXTRACT,'problem':args.problem,'text':text,
              'format':{'steps':[dict(premise='...',action='...',consequence='...',sign='do',quote='...')]}},indent=2));return
    if args.steps is None or args.keys is None:parser.error('score needs --steps and --keys')
    keys=read(args.keys)
    if not isinstance(keys,list) or len(keys)!=200 or len(set(keys))!=200 or any(len(bytes.fromhex(k))!=32 for k in keys):parser.error('Expected 200 distinct 256-bit keys')
    steps=validated(read(args.steps),text);enc,quant=engines();ids=identify(enc,quant,steps)
    print(json.dumps({'itokens':ids,**test(ids,keys,args.problem,args.mode)},indent=2))
if __name__=='__main__':main()
