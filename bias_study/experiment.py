"""Construct, blind and evaluate a three-arm open-vocabulary idea experiment."""
import argparse
from difflib import SequenceMatcher
import hashlib
import math
from pathlib import Path
import random
import secrets
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'open_study'))
from study import EXTRACT, FIDELITY, Encoder, admitted, read, save
from selection import choose, project, rank_test

ARMS = ('control', 'weak', 'strong')
REWORD = (
    "Rewrite this piece of advice in natural prose, in your own words. Preserve each argument's reason, "
    "action and consequence. Keep the people, conditions, scope, timing, exceptions and causal links. "
    "You may change sentence structure, order and paragraphing. Check the draft against the original "
    "before returning your one final text. No heading, step labels or list."
)
POOL = EXTRACT
JUDGE = FIDELITY + ' Give a short explanation referring to specific parts of the two texts.'
MODEL = ROOT.parents[2] / 'work/models/qwen2.5-0.5b-instruct-4bit'


def packet(label, task, docs):
    path = ROOT / 'calls' / label / 'in.json'
    obj = {'instruction': task, 'documents': docs}
    if path.exists() and read(path) != obj:
        raise ValueError(f'Input packet changed: {label}')
    save(path, obj)


def answers(label):
    path = ROOT / 'calls' / label
    wanted = {d['id'] for d in read(path / 'in.json')['documents']}
    rows = read(path / 'out.json')['documents']
    mapped = {r['id']: r for r in rows}
    if len(rows) != len(mapped) or set(mapped) != wanted:
        raise ValueError(f'IDs differ: {label}')
    return mapped


def sources():
    return read(ROOT / 'sources_a.json') + read(ROOT / 'sources_b.json')


def pool_packets():
    for u, unit in enumerate(sources()):
        assert len(set(unit['passages'])) == 24
        docs = [dict(id=f's{j:02d}', problem=unit['problem'], text=source)
                for j, source in enumerate(unit['passages'])]
        packet(f'pool{u}', POOL, docs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    files = ['PROTOCOL.md', 'experiment.py', 'selection.py', 'sources_a.json', 'sources_b.json',
             '../open_study/study.py', '../real_model.py']
    paths = [ROOT / f for f in files]
    for u in range(len(sources())):
        answers(f'pool{u}')
        paths.extend((ROOT / 'calls' / f'pool{u}' / n for n in ('in.json', 'out.json')))
    record = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    path = ROOT / 'freeze.json'
    if path.exists() and read(path) != record:
        raise ValueError('Frozen input changed')
    save(path, record)


def step_vectors(enc, problem, steps):
    return np.stack([enc.vector(problem, [s]) for s in steps]) if steps else np.empty((0, 3*len(enc.encode(problem))))


def generate():
    freeze()
    units = sources()
    keyfile = ROOT / 'keys.json'
    if not keyfile.exists():
        save(keyfile, [[secrets.token_hex(32) for _ in range(200)] for _ in units])
    if (ROOT / 'generation.json').exists():
        raise ValueError('Generation already completed')
    keys = read(keyfile)
    enc = Encoder(MODEL)
    rows, vectors = [], {}
    for u, unit in enumerate(units):
        problem, passages = unit['problem'], unit['passages']
        readings = answers(f'pool{u}')
        pool = []
        weights = []
        for j, passage in enumerate(passages):
            steps = admitted(readings[f's{j:02d}'], passage)
            v = step_vectors(enc, problem, steps)
            vectors[f'u{u}c{j}'] = v
            pool.append(v)
            weights.append(enc.weight(problem, passage))
        projected = [project(matrix, keys[u][0], problem) for matrix in pool]
        rng = random.Random(77801 + u)
        noise = [-math.log(-math.log(rng.random())) for _ in passages]
        base = np.array([w['log_weight'] for w in weights]) + noise
        variants = {}
        for arm in ARMS:
            chosen = choose(arm, projected, base)
            text = '\n\n'.join(' '.join(passages[j] for j in chosen[i:i+4]) for i in range(0, len(chosen), 4))
            variants[arm] = dict(id=secrets.token_hex(8), selected=chosen, text=text)
        rows.append(dict(unit=u, problem=problem, weights=weights, noise=noise, variants=variants))
    np.savez_compressed(ROOT / 'pool_vectors.npz', **vectors)
    save(ROOT / 'generation.json', {'model': enc.lm.metadata, 'units': rows})


def groups():
    units = read(ROOT / 'generation.json')['units']
    for b in range(3):
        yield b, [(u, ARMS[(b + u['unit']) % 3]) for u in units]


def originals():
    for b, pairs in groups():
        docs = [dict(id=u['variants'][arm]['id'], problem=u['problem'], text=u['variants'][arm]['text']) for u, arm in pairs]
        packet(f'original{b}', EXTRACT, docs)
        packet(f'reword{b}', REWORD, docs)


def rewrites():
    for b, pairs in groups():
        if not (ROOT / 'calls' / f'reword{b}' / 'out.json').exists():
            continue
        out = answers(f'reword{b}')
        docs, compare = [], []
        for unit, arm in pairs:
            v = unit['variants'][arm]
            new = out[v['id']]['text']
            if not isinstance(new, str) or not new.strip() or new.strip() == v['text'].strip():
                raise ValueError('Expected a new wording')
            docs.append(dict(id=v['id'], problem=unit['problem'], text=new))
            compare.append(dict(id=v['id'], original=v['text'], rewrite=new))
        packet(f'rewritten{b}', EXTRACT, docs)
        for j in range(2):
            packet(f'judge{j}batch{b}', JUDGE, compare)


def rescore():
    keys = read(ROOT / 'keys.json')
    poolvec = np.load(ROOT / 'pool_vectors.npz')
    docvec = np.load(ROOT / 'document_vectors.npz')
    rows = []
    for b, pairs in groups():
        orig = answers(f'original{b}')
        rev = answers(f'rewritten{b}')
        word = answers(f'reword{b}')
        judges = [answers(f'judge{j}batch{b}') for j in range(2)]
        for unit, arm in pairs:
            u, p = unit['unit'], unit['problem']
            v = unit['variants'][arm]
            uid = v['id']
            new = word[uid]['text']
            methods = [r[uid]['status'] for r in judges]
            if not all(s in ('preserved', 'changed', 'uncertain') for s in methods):
                raise ValueError('Unknown preservation status')
            x = np.concatenate([poolvec[f'u{u}c{c}'] for c in v['selected']])
            row = dict(unit=u, arm=arm, id=uid, fidelity=methods, faithful=methods==['preserved','preserved'],
                       original_retained=len(admitted(orig[uid], v['text'])), rewrite_retained=len(admitted(rev[uid], new)),
                       word_edit_fraction=1-SequenceMatcher(None,v['text'].lower().split(),new.lower().split(),autojunk=False).ratio())
            for stage, matrix in [('plan',x),('original',docvec[f'o{uid}']),('rewrite',docvec[f'r{uid}'])]:
                row[stage] = rank_test(matrix, keys[u], p)
            rows.append(row)
    return sorted(rows, key=lambda r:(r['unit'], ARMS.index(r['arm'])))


def evaluate():
    enc = Encoder(MODEL)
    vectors = {}
    for b, pairs in groups():
        for label, prefix in [('original','o'),('rewritten','r')]:
            packet_name = f'{label}{b}'
            docs = read(ROOT / 'calls' / packet_name / 'in.json')['documents']
            out = answers(packet_name)
            for d in docs:
                steps = admitted(out[d['id']], d['text'])
                vectors[prefix+d['id']] = step_vectors(enc, d['problem'], steps)
    np.savez_compressed(ROOT / 'document_vectors.npz', **vectors)
    save(ROOT / 'results.json', rescore())


if __name__ == '__main__':
    tasks = dict(pool=pool_packets, generate=generate, originals=originals, rewrites=rewrites, evaluate=evaluate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=tasks)
    parser.add_argument('--model', type=Path, default=MODEL, help='Local path to the pinned Qwen checkpoint')
    args = parser.parse_args()
    MODEL = args.model
    tasks[args.phase]()
