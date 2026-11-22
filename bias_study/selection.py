"""Keyed concentration of a six-step set in an open semantic space."""
from functools import lru_cache
import hashlib
from itertools import combinations

import numpy as np

from study import direction

COUNT = 6
STRENGTH = {'control': 0.0, 'weak': 0.2, 'strong': 12.0}


def project(matrix, key, problem):
    directions = []
    for axis in range(3):
        seed = hashlib.sha256(b'idea-concentration-v1\0' + bytes.fromhex(key) + bytes([axis])).hexdigest()
        directions.append(direction(seed, problem, matrix.shape[1]))
    return (matrix @ np.asarray(directions).T) / 3.0


@lru_cache(maxsize=2)
def subsets(n):
    return np.asarray(list(combinations(range(n), COUNT)), dtype=int)


def choose(arm, projections, base):
    valid = np.array([i for i, values in enumerate(projections) if len(values)], dtype=int)
    if len(valid) < COUNT:
        raise ValueError('Fewer than six candidates with a supported reading')
    candidates = valid[subsets(len(valid))]
    counts = np.array([len(v) for v in projections])[candidates].sum(axis=1)
    totals = np.stack([v.sum(axis=0) for v in projections])[candidates].sum(axis=1)
    squares = np.stack([(v*v).sum(axis=0) for v in projections])[candidates].sum(axis=1)
    variances = squares / counts[:, None] - (totals / counts[:, None])**2
    reward = -variances.mean(axis=1)
    scale = reward.std()
    if scale:
        reward = (reward - reward.mean()) / scale
    utility = base[candidates].mean(axis=1)
    return candidates[np.argmax(utility + STRENGTH[arm] * reward)].tolist()


def rank_test(matrix, keys, problem):
    if len(matrix) < 2:
        return {'score': None, 'p': None, 'detected': None}
    scores = [-float(project(matrix, key, problem).var(axis=0).mean()) for key in keys]
    rank = 1 + sum(x >= scores[0] for x in scores[1:])
    p = rank / len(scores)
    return {'score': scores[0], 'p': p, 'detected': p <= 0.01}
