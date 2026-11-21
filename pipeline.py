"""End-to-end reference experiment. No external model or service is used."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import random
from typing import Mapping

from core import (IToken, Problem, ROUTES, candidate_pairs, make_problem,
                  parse_proof, render_proof, validate_proof)
from watermark import bias_distribution, key_randomization_pvalue

PROTOCOL = "itoken-proof-v1"
BASE = {"mp": 0.55, "and": 0.30, "or": 0.15}


@lru_cache(maxsize=32)
def route_table(n: int):
    shape = make_problem(n)
    return tuple(candidate_pairs(shape, t) for t in range(n))


@lru_cache(maxsize=32)
def token_index(n: int):
    index = {}
    for t, pairs in enumerate(route_table(n)):
        for route, pair in pairs.items():
            for position, step in enumerate(pair):
                index[step] = (t, route, position)
    return index


def context(problem: Problem, stage: int) -> str:
    # This public task identifier must be supplied with the original problem.
    # No absolute text position or preceding wording enters the key context.
    return json.dumps({"protocol": PROTOCOL, "problem_id": problem.problem_id,
                       "start": f"p{stage}", "goal": f"p{stage + 1}"},
                      sort_keys=True, separators=(",", ":"))


def base_route_probabilities(previous: str | None = None) -> dict[str, float]:
    """History dependence intentionally exercises non-i.i.d. detection."""
    if previous is None:
        return dict(BASE)
    if previous not in BASE:
        raise ValueError("Unknown previous route")
    return {r: 0.75 * BASE[r] + 0.25 * (r == previous) for r in ROUTES}


def draw(distribution: Mapping[str, float], rng: random.Random) -> str:
    u = rng.random()
    cumulative = 0.0
    for key, probability in distribution.items():
        cumulative += probability
        if u < cumulative:
            return key
    return next(k for k in reversed(distribution) if distribution[k] > 0)


def steps_for_routes(problem: Problem, routes: list[str]) -> list[IToken]:
    if len(routes) != problem.n_stages or any(r not in ROUTES for r in routes):
        raise ValueError("Supply exactly one permitted route per stage")
    return [step for t, route in enumerate(routes)
            for step in route_table(problem.n_stages)[t][route]]


def generate(problem: Problem, key: bytes, delta: float, rng: random.Random):
    """Sample from exact induced iToken masses of the finite text predictor.

    The finite text model's nine outcomes aggregate analytically to these three
    masses; tests independently check the token-product aggregation. We use the
    analytic masses here rather than rebuild a trie for every simulated step.
    """
    routes, choices = [], []
    log_lr = 0.0
    previous = None
    for t, pairs in enumerate(route_table(problem.n_stages)):
        base = base_route_probabilities(previous)
        by_token = {pairs[r][0].canonical(): base[r] for r in ROUTES}
        ctx = context(problem, t)
        biased = bias_distribution(by_token, key, ctx, delta=delta)
        q = {r: biased[pairs[r][0].canonical()] for r in ROUTES}
        route = draw(q, rng)
        routes.append(route)
        choices.append((ctx, pairs[route][0].canonical()))
        log_lr += math.log(q[route] / base[route])
        previous = route
    return {"routes": routes, "steps": steps_for_routes(problem, routes),
            "choices": choices, "log_lr": log_lr}


@dataclass
class Extraction:
    status: str
    choices: list[tuple[str, str]]
    steps: list[IToken]
    routes: dict[int, str]
    complete: bool
    detail: str = ""


def extract(problem: Problem, text: str) -> Extraction:
    if not text.strip():
        return Extraction("insufficient_evidence", [], [], {}, False, "No steps supplied")
    try:
        steps = parse_proof(text)
    except (ValueError, TypeError) as exc:
        return Extraction("unsupported_text", [], [], {}, False, str(exc))
    seen_routes, first_steps = {}, {}
    index = token_index(problem.n_stages)
    for step in steps:
        item = index.get(step)
        if item is None:
            return Extraction("outside_domain", [], steps, {}, False,
                              "An inference is not an admitted step of this problem")
        t, route, position = item
        if t in seen_routes and seen_routes[t] != route:
            return Extraction("ambiguous_path", [], steps, {}, False,
                              "More than one route is expressed at a stage")
        seen_routes[t] = route
        if position == 0:
            first_steps[t] = step
    choices = [(context(problem, t), first_steps[t].canonical()) for t in sorted(first_steps)]
    complete = (len(first_steps) == problem.n_stages
                and validate_proof(problem, steps))
    status = "complete_proof" if complete else "partial_evidence"
    if not choices:
        status = "insufficient_evidence"
    return Extraction(status, choices, steps, seen_routes, complete)


def surrogate_keys(seed: int, count: int) -> list[bytes]:
    if count < 1:
        raise ValueError("At least one surrogate key is required")
    rng = random.Random(seed)
    return [rng.getrandbits(256).to_bytes(32, "big") for _ in range(count)]


def demo_key(seed: int = 20261121) -> bytes:
    """Public reproducibility key, unsuitable for deployment or attribution."""
    return hashlib.sha256(f"PUBLIC-DEMO-KEY:{seed}".encode()).digest()


def diagnose(problem: Problem, text: str, key: bytes,
             calibration_keys: list[bytes], delta: float = 2.0,
             alpha: float = 0.01) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    extracted = extract(problem, text)
    result = {"status": extracted.status, "proof_valid": extracted.complete,
              "n_choices": len(extracted.choices), "alpha": alpha,
              "detail": extracted.detail, "p_value": None, "detected": None,
              "log_likelihood_ratio": None, "likelihood_ratio_detected": None}
    if not extracted.choices:
        return result
    preferred, p_value, n = key_randomization_pvalue(
        extracted.choices, key, calibration_keys)
    result.update(preferred=preferred, n_choices=n, p_value=p_value,
                  detected=p_value <= alpha)
    if extracted.complete:
        log_lr = 0.0
        previous = None
        for t, pairs in enumerate(route_table(problem.n_stages)):
            route = extracted.routes[t]
            base = base_route_probabilities(previous)
            tokens = {pairs[r][0].canonical(): base[r] for r in ROUTES}
            q = bias_distribution(tokens, key, context(problem, t), delta=delta)
            selected_q = q[pairs[route][0].canonical()]
            log_lr += math.log(selected_q) - math.log(base[route]) if selected_q else -math.inf
            previous = route
        result.update(log_likelihood_ratio=log_lr,
                      likelihood_ratio_detected=log_lr >= math.log(1 / alpha))
    return result


def transform(problem: Problem, generated: dict, attack: str, rng: random.Random) -> str:
    steps = list(generated["steps"])
    if attack == "clean":
        return render_proof(steps, "plain")
    if attack == "controlled_rewording":
        return render_proof(steps, "paraphrase")
    if attack == "presentation_shuffle":
        rng.shuffle(steps)
        return render_proof(steps, "compact")
    if attack == "duplicate_steps":
        return render_proof(steps * 3, "plain")
    if attack == "delete_25_percent":
        count = max(1, round(len(steps) * 0.25))
        removed = set(rng.sample(range(len(steps)), count))
        return render_proof([s for j, s in enumerate(steps) if j not in removed], "plain")
    if attack in {"reroute_25_percent", "full_rewrite"}:
        routes = list(generated["routes"])
        count = problem.n_stages if attack == "full_rewrite" else max(1, round(problem.n_stages * 0.25))
        selected = set(rng.sample(range(problem.n_stages), count))
        previous = None
        for t in range(problem.n_stages):
            if t in selected:
                routes[t] = draw(base_route_probabilities(previous), rng)
            previous = routes[t]
        return render_proof(steps_for_routes(problem, routes), "paraphrase")
    raise ValueError(f"Unknown attack {attack!r}")
