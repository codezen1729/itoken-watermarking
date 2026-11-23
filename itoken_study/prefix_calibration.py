"""Choose a semantic-code length on historical training topics with fixed planes."""
import hashlib
import json

from canonical import canonical_step
from encoder import SentenceEncoder
from quantize import ARTIFACT, HERE, HOLDOUT, TRAIN, Hash, embed_steps, old_readings

LENGTHS = (4, 5, 6, 7, 8, 12)


def compare(groups, topics, length, use_sign):
    counts = {phase: dict(pairs=0, same=0, wrong_pairs=0, wrong_same=0)
              for phase in ('original', 'rewrite')}
    def token(value):
        bits, sign = value
        return (bits[:length], sign if use_sign else '')
    for (unit, candidate), readings in groups.items():
        if unit not in topics:
            continue
        for reading, candidate_token in readings.items():
            if reading == 'pool':
                continue
            cell = counts['rewrite' if reading.endswith('_rewrite') else 'original']
            query = token(candidate_token)
            cell['pairs'] += 1
            cell['same'] += query == token(readings['pool'])
            for (u, j), other in groups.items():
                if u != unit or j == candidate:
                    continue
                cell['wrong_pairs'] += 1
                cell['wrong_same'] += query == token(other['pool'])
    return counts


def main():
    raw, source_hash = old_readings()
    rows = [(key, reading, step) for key, views in sorted(raw.items()) for reading, step in views]
    steps = [step for _, _, step in rows]
    codes = Hash().codes(embed_steps(SentenceEncoder(), steps))
    groups = {}
    for (key, reading, step), bits in zip(rows, codes, strict=True):
        groups.setdefault(key, {})[reading] = (bits, canonical_step(step)['sign'])
    tables = {}
    for signed in (True, False):
        name = 'with_polarity' if signed else 'without_polarity'
        tables[name] = {str(n): {fold: compare(groups, topics, n, signed)
                                 for fold, topics in [('fit', TRAIN), ('held_out', HOLDOUT)]}
                        for n in LENGTHS}
    eligible = [n for n in LENGTHS if all(v['wrong_same'] <= .05*v['wrong_pairs']
                                           for v in tables['with_polarity'][str(n)]['fit'].values())]
    selected = min(eligible) if eligible else None
    data = dict(selected_semantic_bits=selected, append_action_polarity=True, candidate_lengths=LENGTHS,
                selection_rule='Smallest candidate number of greedy-ordered semantic bits with at most 5 percent same-ID collisions on different steps from the same problem, for both unchanged and rewritten historical training readouts, with the action polarity appended.',
                scope='These older fold outcomes are development diagnostics after viewing the original full-code calibration. The threshold uses training-topic pairs only. No cases, results or keys from the fresh cohort entered model selection.',
                training_topics=TRAIN, held_out_topics=HOLDOUT, public_axes_sha256=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
                canonical_parser_sha256=hashlib.sha256((HERE/'canonical.py').read_bytes()).hexdigest(),
                historical_source_sha256=source_hash, counts=tables)
    (HERE/'prefix_calibration.json').write_text(json.dumps(data,indent=2)+'\n')
    print('selected',selected)
    for n in LENGTHS:
        print(n,[[f"{x['same']}/{x['pairs']};{x['wrong_same']}/{x['wrong_pairs']}" for x in v.values()]
                 for v in tables['with_polarity'][str(n)].values()])


if __name__ == '__main__':
    main()
