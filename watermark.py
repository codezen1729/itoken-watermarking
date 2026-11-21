"""Watermark selection and detection for canonical symbolic choices.

This module uses only the Python standard library and does not depend on the
symbolic domain.  A choice is a ``(context, candidate)`` pair of canonical
strings.  The caller must provide the same canonicalization at generation and
detection, and must keep the detection key secret.

Statistical contracts
---------------------
* ``bias_distribution`` implements q(a) proportional to p(a) exp(delta*g(a)).
* Sum ``log_likelihood_ratio`` along the generated path for the known-baseline
  likelihood ratio.  Under the exact baseline, its exponential is a null
  martingale, even when the successive states are dependent.  A threshold of
  1/alpha for the likelihood ratio gives conservative type-I error control.
* ``key_randomization_pvalue`` needs no independent-step assumption.  Its null
  is that the document is independent of a uniformly sampled secret key; the
  surrogate keys must be sampled independently from that same key distribution.
  Its guarantee averages over the key randomization.  It does not establish
  conditional validity for every fixed deployed key or for adaptive documents
  chosen using a detector oracle.  Duplicate structural pairs count once.
* ``binomial_tail`` is appropriate only when the preferred labels really are
  independent Bernoulli events.  Do not substitute it for the surrogate-key
  test on arbitrary traces.  A cryptographic PRF is not an exact random oracle.

All floating-point calculations use Python doubles.  Very large delta can
underflow the nonpreferred probabilities to zero, reaching the limiting tilted
distribution numerically.  The hash threshold has 256-bit resolution; gamma=.5
is represented exactly.

Run the embedded checks with ``python watermark.py``.  They cover validation,
normalization, numerical stability, null likelihood-ratio expectation, duplicate
invariance, conservative randomization ties, finite binomial sums, confidence
intervals, and tie-aware AUROC.  They are not evidence of real-LLM performance.
"""

from __future__ import annotations

import hmac
import json
import math
from collections.abc import Iterable, Mapping
from numbers import Real
from statistics import NormalDist


Choice = tuple[str, str]
_DOMAIN = b"itoken-watermark:v1\x00"
_HASH_RANGE = 1 << 256


def _real(value: object, name: str, *, finite: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if math.isnan(value) or (finite and not math.isfinite(value)):
        raise ValueError(f"{name} must be {'finite' if finite else 'non-NaN'}")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _key(key: bytes) -> bytes:
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if not key:
        raise ValueError("key must not be empty")
    return key


def _cutoff(gamma: float) -> int:
    gamma = _real(gamma, "gamma")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    numerator, denominator = gamma.as_integer_ratio()
    return numerator * _HASH_RANGE // denominator


def _payload(context: str, candidate: str) -> bytes:
    if not isinstance(context, str) or not isinstance(candidate, str):
        raise TypeError("context and candidate must be strings")
    # JSON's array boundary and string escaping make this encoding unambiguous.
    return _DOMAIN + json.dumps(
        [context, candidate], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _is_preferred(key: bytes, payload: bytes, cutoff: int) -> bool:
    value = int.from_bytes(hmac.digest(key, payload, "sha256"), "big")
    return value < cutoff


def _distribution(base: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(base, Mapping) or not base:
        raise ValueError(f"{name} must be a nonempty probability mapping")
    result: dict[str, float] = {}
    for candidate, probability in base.items():
        if not isinstance(candidate, str):
            raise TypeError(f"{name} candidate identifiers must be strings")
        probability = _real(probability, f"{name}[{candidate!r}]")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} probabilities must lie in [0, 1]")
        result[candidate] = probability
    total = math.fsum(result.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{name} probabilities must sum to one (got {total!r})")
    return {candidate: probability / total for candidate, probability in result.items()}


def keyed_preferred(
    key: bytes, context: str, candidate: str, gamma: float = 0.5
) -> bool:
    """Assign a reproducible keyed preferred label to a structural choice."""
    return _is_preferred(_key(key), _payload(context, candidate), _cutoff(gamma))


def bias_distribution(
    base: Mapping[str, float],
    key: bytes,
    context: str,
    delta: float,
    gamma: float = 0.5,
) -> dict[str, float]:
    """Exponentially tilt an admissible probability distribution.

    ``delta`` is finite and nonnegative.  Zero-probability candidates retain
    zero probability.  If all positive-probability choices share one label,
    the distribution remains unchanged and that state carries no watermark.
    """
    base = _distribution(base, "base")
    key, cutoff = _key(key), _cutoff(gamma)
    delta = _real(delta, "delta")
    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    labels = {
        candidate: _is_preferred(key, _payload(context, candidate), cutoff)
        for candidate in base
    }
    active_labels = {labels[candidate] for candidate, p in base.items() if p > 0.0}
    if delta == 0.0 or len(active_labels) == 1:
        return dict(base)
    # Divide every unnormalized weight by exp(delta).  This avoids overflow,
    # and preserves ratios *within* the preferred group even for huge delta.
    attenuation = math.exp(-delta)
    weights = {
        candidate: p if labels[candidate] else p * attenuation
        for candidate, p in base.items()
    }
    normalizer = math.fsum(weights.values())
    return {candidate: weight / normalizer for candidate, weight in weights.items()}


def log_likelihood_ratio(
    candidate: str,
    base: Mapping[str, float],
    watermarked: Mapping[str, float],
) -> float:
    """Return log(q(candidate)/p(candidate)) for one structural choice.

    Both mappings must be distributions and must contain the candidate.  A
    zero denominator with a positive numerator returns +infinity; a positive
    denominator with a zero numerator returns -infinity.  A 0/0 event is
    undefined and raises ValueError.
    """
    p = _distribution(base, "base")[candidate]
    q = _distribution(watermarked, "watermarked")[candidate]
    if p == 0.0 and q == 0.0:
        raise ValueError("the selected candidate has zero probability in both distributions")
    if p == 0.0:
        return math.inf
    if q == 0.0:
        return -math.inf
    return math.log(q) - math.log(p)


def _unique_payloads(choices: Iterable[Choice]) -> list[bytes]:
    payloads: list[bytes] = []
    seen: set[Choice] = set()
    for choice in choices:
        if not isinstance(choice, (tuple, list)) or len(choice) != 2:
            raise TypeError("each choice must be a (context, candidate) pair")
        context, candidate = choice
        payload = _payload(context, candidate)
        pair = (context, candidate)
        if pair not in seen:
            seen.add(pair)
            payloads.append(payload)
    return payloads


def score_counts(
    choices: Iterable[Choice], key: bytes, gamma: float = 0.5
) -> tuple[int, int]:
    """Return (preferred count, distinct pair count); ignore exact duplicates."""
    key, cutoff = _key(key), _cutoff(gamma)
    payloads = _unique_payloads(choices)
    return sum(_is_preferred(key, item, cutoff) for item in payloads), len(payloads)


def score(choices: Iterable[Choice], key: bytes, gamma: float = 0.5) -> int:
    """Return the preferred count over distinct structural pairs."""
    return score_counts(choices, key, gamma)[0]


def key_randomization_pvalue(
    choices: Iterable[Choice],
    key: bytes,
    surrogate_keys: Iterable[bytes],
    gamma: float = 0.5,
) -> tuple[int, float, int]:
    """Return (observed score, conservative Monte Carlo p-value, distinct n).

    For R independently sampled surrogate keys the p-value is
    ``(1 + number of surrogate scores >= observed score) / (R + 1)``.
    Including ties makes the test conservative; its smallest possible value
    is 1/(R+1).  Supply at least one surrogate key.  Random key generation is
    deliberately left to the caller so experiments can control their seeds.
    """
    key, cutoff = _key(key), _cutoff(gamma)
    payloads = _unique_payloads(choices)
    observed = sum(_is_preferred(key, item, cutoff) for item in payloads)
    at_least_observed = 0
    count = 0
    for surrogate_key in surrogate_keys:
        surrogate_key = _key(surrogate_key)
        surrogate_score = sum(
            _is_preferred(surrogate_key, item, cutoff) for item in payloads
        )
        at_least_observed += surrogate_score >= observed
        count += 1
    if count == 0:
        raise ValueError("supply at least one surrogate key")
    return observed, (1 + at_least_observed) / (count + 1), len(payloads)


def binomial_tail(n: int, k: int, p: float = 0.5) -> float:
    """Evaluate Pr[Binomial(n,p) >= k] by a finite sum, not a normal approximation.

    For p=.5, use an exact integer numerator and denominator followed by one
    floating-point division.  Otherwise use a stable log-space finite sum.
    """
    n, k = _integer(n, "n"), _integer(k, "k")
    p = _real(p, "p")
    if n < 0:
        raise ValueError("n must be nonnegative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if k <= 0:
        return 1.0
    if k > n or p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    if p == 0.5:
        return sum(math.comb(n, j) for j in range(k, n + 1)) / (1 << n)
    log_p, log_not_p = math.log(p), math.log1p(-p)
    logs = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * log_p + (n - j) * log_not_p
        for j in range(k, n + 1)
    ]
    largest = max(logs)
    result = math.exp(largest) * math.fsum(math.exp(x - largest) for x in logs)
    return min(1.0, max(0.0, result))


# Descriptive alias for callers who want to emphasize the finite-sum test.
exact_binomial_tail = binomial_tail


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson interval for independent Bernoulli trials; (0,1) when trials=0.

    This interval does not account for clustering by task or reused secret key.
    Use independent experimental units, or report that limitation explicitly.
    """
    successes = _integer(successes, "successes")
    trials = _integer(trials, "trials")
    confidence = _real(confidence, "confidence")
    if not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if trials == 0:
        return 0.0, 1.0
    # The lower quantile avoids rounding a confidence near one up to a CDF of 1.
    z = -NormalDist().inv_cdf((1.0 - confidence) / 2.0)
    z_squared = z * z
    observed = successes / trials
    denominator = 1.0 + z_squared / trials
    center = (observed + z_squared / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        observed * (1.0 - observed) / trials + z_squared / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def auroc(positive_scores: Iterable[float], negative_scores: Iterable[float]) -> float:
    """Tie-aware AUROC: Pr[positive > negative] + .5 Pr[equality].

    Higher scores must mean more evidence for the positive class.  Infinite
    likelihood-ratio scores are supported; NaNs are rejected.
    """
    positives = [_real(x, "positive score", finite=False) for x in positive_scores]
    negatives = [_real(x, "negative score", finite=False) for x in negative_scores]
    if not positives or not negatives:
        raise ValueError("AUROC needs at least one positive and one negative score")
    pooled = sorted([(x, 1) for x in positives] + [(x, 0) for x in negatives])
    rank_sum = 0.0
    start = 0
    while start < len(pooled):
        end = start + 1
        while end < len(pooled) and pooled[end][0] == pooled[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pooled[start:end])
        start = end
    n_positive, n_negative = len(positives), len(negatives)
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _run_tests() -> None:
    import unittest

    class WatermarkTests(unittest.TestCase):
        def setUp(self):
            self.key = b"test-key-not-for-production"
            self.context = "problem-1:goal-3"

        def split_candidates(self):
            preferred, other = [], []
            for i in range(100):
                candidate = f"route-{i}"
                (preferred if keyed_preferred(self.key, self.context, candidate) else other).append(candidate)
            self.assertGreaterEqual(len(preferred), 2)
            self.assertTrue(other)
            return preferred, other

        def test_hash_determinism_and_framing(self):
            self.assertEqual(keyed_preferred(self.key, "c", "a"), keyed_preferred(self.key, "c", "a"))
            self.assertNotEqual(_payload("ab", "c"), _payload("a", "bc"))

        def test_invalid_parameters(self):
            for gamma in (0, 1, -0.1, math.nan, math.inf):
                with self.assertRaises(ValueError):
                    keyed_preferred(self.key, "c", "a", gamma)
            for delta in (-1, math.inf, math.nan):
                with self.assertRaises(ValueError):
                    bias_distribution({"a": 1.0}, self.key, "c", delta)
            for base in ({}, {"a": 0.0}, {"a": 0.7}, {"a": -0.1, "b": 1.1}, {"a": math.nan}):
                with self.assertRaises(ValueError):
                    bias_distribution(base, self.key, "c", 1.0)

        def test_bias_and_likelihood_ratio(self):
            preferred, other = self.split_candidates()
            a, b = preferred[0], other[0]
            base = {a: 0.25, b: 0.75, "zero": 0.0}
            q = bias_distribution(base, self.key, self.context, 1.5)
            self.assertAlmostEqual(math.fsum(q.values()), 1.0)
            self.assertEqual(q["zero"], 0.0)
            self.assertAlmostEqual((q[a] / q[b]) / (base[a] / base[b]), math.exp(1.5))
            expected_ratio = math.fsum(
                p * math.exp(log_likelihood_ratio(candidate, base, q))
                for candidate, p in base.items() if p > 0.0
            )
            self.assertAlmostEqual(expected_ratio, 1.0)
            self.assertEqual(bias_distribution(base, self.key, self.context, 0.0), base)
            self.assertEqual(bias_distribution({a: 1.0}, self.key, self.context, 5.0), {a: 1.0})

        def test_large_delta_preserves_within_group_probabilities(self):
            preferred, other = self.split_candidates()
            a, b, c = preferred[0], preferred[1], other[0]
            q = bias_distribution({a: 0.1, b: 0.2, c: 0.7}, self.key, self.context, 1e300)
            self.assertAlmostEqual(q[a], 1 / 3)
            self.assertAlmostEqual(q[b], 2 / 3)
            self.assertEqual(q[c], 0.0)

        def test_duplicate_invariance_and_randomization_formula(self):
            choices = [("c1", "a1"), ("c2", "a2")]
            surrogates = [f"surrogate-{i}".encode() for i in range(19)]
            observed, pvalue, n = key_randomization_pvalue(choices, self.key, surrogates)
            expected = (1 + sum(score(choices, k) >= observed for k in surrogates)) / 20
            self.assertEqual(pvalue, expected)
            self.assertEqual(n, 2)
            self.assertEqual(
                key_randomization_pvalue(choices * 10, self.key, surrogates),
                (observed, pvalue, n),
            )
            self.assertEqual(key_randomization_pvalue([], self.key, surrogates), (0, 1.0, 0))
            self.assertEqual(key_randomization_pvalue(choices, self.key, [self.key])[1], 1.0)
            with self.assertRaises(ValueError):
                key_randomization_pvalue(choices, self.key, [])

        def test_binomial(self):
            self.assertEqual(binomial_tail(10, 10), 1 / 1024)
            self.assertEqual(binomial_tail(10, 5), 638 / 1024)
            self.assertAlmostEqual(binomial_tail(3, 2, 0.2), 0.104)
            self.assertEqual(binomial_tail(0, 0), 1.0)
            self.assertEqual(binomial_tail(3, 4), 0.0)
            self.assertEqual(binomial_tail(3, 1, 0.0), 0.0)
            self.assertEqual(binomial_tail(3, 3, 1.0), 1.0)

        def test_wilson(self):
            self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))
            low, high = wilson_interval(0, 100)
            self.assertAlmostEqual(low, 0.0)
            self.assertAlmostEqual(high, 0.03699349820698568)
            with self.assertRaises(ValueError):
                wilson_interval(2, 1)

        def test_auroc_ties_and_infinite_scores(self):
            self.assertEqual(auroc([2, 3], [0, 1]), 1.0)
            self.assertEqual(auroc([0, 1], [2, 3]), 0.0)
            self.assertEqual(auroc([1, 1], [1, 1]), 0.5)
            self.assertEqual(auroc([0, 1], [0, 1]), 0.5)
            self.assertEqual(auroc([math.inf], [-math.inf]), 1.0)
            with self.assertRaises(ValueError):
                auroc([math.nan], [1.0])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WatermarkTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
