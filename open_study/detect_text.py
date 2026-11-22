"""Recover and score a new document with no source bank or reference plan.

First export the prompt and give it to a semantic reader. Supply that reader's
JSON output to `score` along with a model path and a verifier's key bank.
"""
import argparse
import json
from pathlib import Path

from study import EXTRACT, Encoder, admitted, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prompt', 'score'))
    parser.add_argument('--problem', required=True, help='Exact public problem text used as the key context')
    parser.add_argument('--text', required=True, type=Path)
    parser.add_argument('--steps', type=Path, help='Reader JSON object with a steps array')
    parser.add_argument('--keys', type=Path, help='JSON list of 200 hex keys, target followed by 199 surrogates')
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    source = args.text.read_text()
    if args.phase == 'prompt':
        print(json.dumps({'instruction': EXTRACT, 'problem': args.problem, 'text': source,
                          'output': {'steps': [{'premise': '...', 'action': '...', 'consequence': '...', 'quote': '...'}]}}, ensure_ascii=False,indent=2))
        return
    if not all((args.steps, args.keys, args.model)):
        parser.error('score requires --steps, --keys and --model')
    row = json.loads(args.steps.read_text())
    keys = json.loads(args.keys.read_text())
    if not isinstance(keys,list) or len(keys)!=200 or len(set(keys))!=200 or any(len(bytes.fromhex(k))!=32 for k in keys):
        parser.error('expected a target and 199 distinct 256-bit surrogate keys')
    if not isinstance(row.get('steps'),list):
        parser.error('expected a steps array')
    steps = admitted(row,source)
    encoder = Encoder(args.model)
    result = test(encoder.vector(args.problem,steps),keys,args.problem)
    print(json.dumps({**result,'proposed':len(row['steps']),'retained':len(steps)},indent=2))


if __name__ == '__main__':
    main()
