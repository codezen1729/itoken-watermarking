"""Frozen key-blind word-overlap diagnostic for natural-language recovery.

This intentionally ignores word order, agency and assertion scope. It is a
comparison, not a semantic validator. No threshold is learned from results.
"""
from __future__ import annotations
from collections import Counter
import math
import re

VERSION = 'per-decision-application-tfidf-v1'
TOKEN_PATTERN = re.compile(r'[a-z]+')
STOPWORDS = frozenset('a an and are as at be been but by for from had has have he her him his i in is it its of on or our she that the their them there these they this to was we were will with you your'.split())
TIE_TOLERANCE = 1e-12


def tokens(text):
    if not isinstance(text, str):
        raise TypeError('Text must be a string')
    return [word for word in TOKEN_PATTERN.findall(text.lower()) if word not in STOPWORDS]


def recover(problem, text):
    """Return all unique positive maxima, including possible false positives."""
    counts = Counter(tokens(text))
    choices, abstained, all_scores = {}, [], {}
    decisions = problem['decisions']
    if not decisions or len({d['id'] for d in decisions}) != len(decisions):
        raise ValueError('Decisions require nonempty unique identifiers')
    for decision in decisions:
        options = decision['options']
        if len(options) != 3 or len({o['id'] for o in options}) != 3:
            raise ValueError('Each decision must have exactly three unique options')
        vectors = [Counter(tokens(option['application'])) for option in options]
        vocabulary = set().union(*(set(v) for v in vectors))
        idf = {word: 1.0 + math.log(4.0 / (1.0 + sum(word in v for v in vectors)))
               for word in vocabulary}
        document_norm = math.sqrt(sum((counts[word] * idf[word]) ** 2 for word in vocabulary))
        scores = {}
        for option, vector in zip(options, vectors):
            option_norm = math.sqrt(sum((count * idf[word]) ** 2 for word, count in vector.items()))
            dot = sum(counts[word] * count * idf[word] ** 2 for word, count in vector.items())
            scores[option['id']] = dot / (document_norm * option_norm) if document_norm and option_norm else 0.0
        maximum = max(scores.values())
        winners = [option for option, score in scores.items() if abs(maximum - score) <= TIE_TOLERANCE]
        if maximum > 0.0 and len(winners) == 1:
            choices[decision['id']] = winners[0]
        else:
            abstained.append(decision['id'])
        all_scores[decision['id']] = scores
    return {'version': VERSION, 'choices': choices, 'abstained_decisions': abstained,
            'status': 'complete' if len(choices) == len(decisions) else 'partial' if choices else 'abstain',
            'scores': all_scores,
            'scope': 'word overlap only; no assertion, causal, role, or ordering judgment'}
