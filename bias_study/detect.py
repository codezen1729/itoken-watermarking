"""Recover open steps with a blind reader, then score a piece of prose."""
import argparse
import json
from pathlib import Path

from experiment import EXTRACT, MODEL, Encoder, admitted, rank_test, read, step_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prompt', 'score'))
    parser.add_argument('--problem', required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--steps', type=Path, help='Object containing a steps array from the reader')
    parser.add_argument('--keys', type=Path, help='Array of 200 hex keys, target first')
    parser.add_argument('--model', type=Path, default=MODEL)
    args = parser.parse_args()
    source = args.text.read_text()
    if args.command == 'prompt':
        print(json.dumps({'instruction': EXTRACT, 'problem': args.problem, 'text': source,
                          'format': {'steps': [dict(premise='...', action='...', consequence='...', quote='...')]}},
                         ensure_ascii=False, indent=2))
        return
    if args.steps is None or args.keys is None:
        parser.error('score requires --steps and --keys')
    keys = read(args.keys)
    if len(keys) != 200 or len(set(keys)) != 200 or any(len(bytes.fromhex(k)) != 32 for k in keys):
        parser.error('expected 200 distinct 256-bit keys')
    steps = admitted(read(args.steps), source)
    result = rank_test(step_vectors(Encoder(args.model), args.problem, steps), keys, args.problem)
    print(json.dumps({**result, 'retained': len(steps)}, indent=2))


if __name__ == '__main__':
    main()
