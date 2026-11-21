"""Controlled logical iTokens and an exact finite autoregressive reference model.

This module is a deliberately restricted test bed, not a natural-language
semantic parser and not a trained language model.  The three supported text
styles denote exactly the same formal inference.  Unsupported prose is rejected.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import re
from typing import Iterable, Mapping, Sequence


ROUTES = ("mp", "and", "or")
STYLES = ("plain", "paraphrase", "compact")
DEFAULT_ROUTE_PROBABILITIES = {"mp": 0.55, "and": 0.30, "or": 0.15}
DEFAULT_STYLE_PROBABILITIES = {"plain": 0.50, "paraphrase": 0.30, "compact": 0.20}
EOS = "<END_OF_STEP>"
_ATOM = re.compile(r"[a-z][a-z0-9_]*\Z")
_TOKEN = re.compile(r"\s*(->|[()&|]|[a-z][a-z0-9_]*)")


def _parse_formula(text: str):
    """Parse propositional formulas with ->, |, &, and parentheses."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A formula must be a nonempty string")
    source = text.strip()
    tokens: list[str] = []
    pos = 0
    while pos < len(source):
        match = _TOKEN.match(source, pos)
        if match is None:
            raise ValueError(f"Invalid formula near {source[pos:]!r}")
        tokens.append(match.group(1))
        pos = match.end()
    cursor = 0

    def primary():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("Incomplete formula")
        token = tokens[cursor]
        cursor += 1
        if token == "(":
            expression = implication()
            if cursor >= len(tokens) or tokens[cursor] != ")":
                raise ValueError("Unmatched parenthesis")
            cursor += 1
            return expression
        if not _ATOM.fullmatch(token):
            raise ValueError(f"Expected atom, got {token!r}")
        return token

    def conjunction():
        nonlocal cursor
        left = primary()
        while cursor < len(tokens) and tokens[cursor] == "&":
            cursor += 1
            left = ("&", left, primary())
        return left

    def disjunction():
        nonlocal cursor
        left = conjunction()
        while cursor < len(tokens) and tokens[cursor] == "|":
            cursor += 1
            left = ("|", left, conjunction())
        return left

    def implication():
        nonlocal cursor
        left = disjunction()
        if cursor < len(tokens) and tokens[cursor] == "->":
            cursor += 1
            return ("->", left, implication())
        return left

    parsed = implication()
    if cursor != len(tokens):
        raise ValueError("Unexpected trailing formula tokens")
    return parsed


def _serialize_formula(tree) -> str:
    if isinstance(tree, str):
        return tree
    op, left, right = tree
    operands = [_serialize_formula(left), _serialize_formula(right)]
    if op in ("&", "|"):
        operands.sort()
    return f"({operands[0]}{op}{operands[1]})"


def canonical_formula(text: str) -> str:
    """Normalize syntax and binary AND/OR operand order, not all equivalence."""
    return _serialize_formula(_parse_formula(text))


def Atom(name: str) -> str:
    if not isinstance(name, str) or not _ATOM.fullmatch(name):
        raise ValueError(f"Invalid atom {name!r}")
    return name


def And(left: str, right: str) -> str:
    return canonical_formula(f"({left}&{right})")


def Or(left: str, right: str) -> str:
    return canonical_formula(f"({left}|{right})")


def Implies(left: str, right: str) -> str:
    return canonical_formula(f"({left}->{right})")


@dataclass(frozen=True)
class IToken:
    rule: str
    premises: tuple[str, ...]
    conclusion: str

    def __post_init__(self):
        object.__setattr__(self, "rule", self.rule.upper())
        object.__setattr__(
            self, "premises", tuple(sorted(set(canonical_formula(p) for p in self.premises)))
        )
        object.__setattr__(self, "conclusion", canonical_formula(self.conclusion))

    def canonical(self) -> str:
        return json.dumps(
            {"rule": self.rule, "premises": list(self.premises), "conclusion": self.conclusion},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )

    @property
    def stage(self) -> int:
        """Infer this domain's stage from formula atoms, never line position."""
        text = " ".join((*self.premises, self.conclusion))
        indices = {int(x) for x in re.findall(r"\b[mde](\d+)\b", text)}
        if len(indices) != 1:
            raise ValueError("Cannot identify a unique domain stage")
        return next(iter(indices))


@dataclass(frozen=True)
class Problem:
    n_stages: int
    problem_id: str
    premises: frozenset[str]
    goals: tuple[str, ...]

    @property
    def goal(self) -> str:
        return self.goals[-1]

    def canonical(self) -> str:
        return json.dumps(
            {"problem_id": self.problem_id, "premises": sorted(self.premises), "goal": self.goal},
            sort_keys=True, separators=(",", ":"),
        )


def make_problem(n: int, problem_id: str = "controlled-proof") -> Problem:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    facts = {"p0"}
    for t in range(n):
        p, nxt, m, d, e = f"p{t}", f"p{t+1}", f"m{t}", f"d{t}", f"e{t}"
        facts.update((d, Implies(p, m), Implies(m, nxt),
                      Implies(And(p, d), nxt), Implies(Or(p, e), nxt)))
    return Problem(n, str(problem_id), frozenset(facts), tuple(f"p{t+1}" for t in range(n)))


def candidate_pairs(problem: Problem, stage: int) -> dict[str, tuple[IToken, IToken]]:
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage < problem.n_stages:
        raise ValueError("Stage outside problem")
    p, nxt = f"p{stage}", f"p{stage+1}"
    m, d, e = f"m{stage}", f"d{stage}", f"e{stage}"
    conjunction, disjunction = And(p, d), Or(p, e)
    return {
        "mp": (
            IToken("MP", (p, Implies(p, m)), m),
            IToken("MP", (m, Implies(m, nxt)), nxt),
        ),
        "and": (
            IToken("AND_INTRO", (p, d), conjunction),
            IToken("MP", (conjunction, Implies(conjunction, nxt)), nxt),
        ),
        "or": (
            IToken("OR_INTRO", (p,), disjunction),
            IToken("MP", (disjunction, Implies(disjunction, nxt)), nxt),
        ),
    }


_RULE_LABELS = {"MP": "modus ponens", "AND_INTRO": "conjunction introduction", "OR_INTRO": "disjunction introduction"}
_LABEL_RULES = {value: key for key, value in _RULE_LABELS.items()}


def render_step(step: IToken, style: str = "plain") -> str:
    if step.rule not in _RULE_LABELS or not step.premises:
        raise ValueError("Unsupported inference rule or missing premises")
    if style == "plain":
        return f"From {' and '.join(step.premises)}, infer {step.conclusion} by {_RULE_LABELS[step.rule]}."
    if style == "paraphrase":
        return f"{step.conclusion} follows from {', '.join(step.premises)} using {_RULE_LABELS[step.rule]}."
    if style == "compact":
        return f"{'; '.join(step.premises)} |- {step.conclusion} [{step.rule}]."
    raise ValueError(f"Unknown controlled rendering style {style!r}")


def parse_step(text: str) -> IToken:
    """Accept exactly one of three controlled sentence grammars; fail closed."""
    if not isinstance(text, str) or "\n" in text.strip():
        raise ValueError("Expected one textual step")
    sentence = " ".join(text.strip().split())
    match = re.fullmatch(r"From (.+), infer (.+) by (.+)\.", sentence)
    if match and match[3] in _LABEL_RULES:
        return IToken(_LABEL_RULES[match[3]], tuple(match[1].split(" and ")), match[2])
    match = re.fullmatch(r"(.+) follows from (.+) using (.+)\.", sentence)
    if match and match[3] in _LABEL_RULES:
        return IToken(_LABEL_RULES[match[3]], tuple(match[2].split(", ")), match[1])
    match = re.fullmatch(r"(.+) \|- (.+) \[(MP|AND_INTRO|OR_INTRO)\]\.", sentence)
    if match:
        return IToken(match[3], tuple(match[1].split("; ")), match[2])
    raise ValueError(f"Unrecognized controlled step: {text!r}")


def render_proof(steps: Iterable[IToken], style: str = "plain") -> str:
    return "\n".join(render_step(step, style) for step in steps)


def parse_proof(text: str) -> list[IToken]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected a nonempty proof")
    return [parse_step(line) for line in text.splitlines() if line.strip()]


def valid_inference(step: IToken) -> bool:
    """Check the actual logical inference independently of candidate templates."""
    premises = set(step.premises)
    conclusion = _parse_formula(step.conclusion)
    if step.rule == "MP" and len(premises) == 2:
        for premise in premises:
            parsed = _parse_formula(premise)
            if isinstance(parsed, tuple) and parsed[0] == "->":
                antecedent, consequent = _serialize_formula(parsed[1]), _serialize_formula(parsed[2])
                if antecedent in premises and consequent == step.conclusion:
                    return True
        return False
    if step.rule == "AND_INTRO" and isinstance(conclusion, tuple) and conclusion[0] == "&":
        return premises == {_serialize_formula(conclusion[1]), _serialize_formula(conclusion[2])}
    if step.rule == "OR_INTRO" and len(premises) == 1 and isinstance(conclusion, tuple) and conclusion[0] == "|":
        return next(iter(premises)) in {_serialize_formula(conclusion[1]), _serialize_formula(conclusion[2])}
    return False


def validate_proof(problem: Problem, steps: Sequence[IToken]) -> bool:
    """Validate an inference graph, independent of textual presentation order.

    Every step must be sound and justified by initial facts or other supplied
    steps.  Unsupported cycles are rejected.  Duplicate justified steps are
    harmless but do not create additional evidence for a detector.
    """
    try:
        pending = set(steps)
        if not pending or any(not valid_inference(step) for step in pending):
            return False
        available = set(problem.premises)
        while pending:
            ready = {step for step in pending if set(step.premises) <= available}
            if not ready:
                return False
            available.update(step.conclusion for step in ready)
            pending.difference_update(ready)
        return problem.goal in available
    except (TypeError, ValueError, AttributeError):
        return False


class FiniteTokenPredictor:
    """Exact finite autoregressive predictor derived from weighted text strings.

    This is an analytic reference distribution, not a model trained on data.
    Whitespace-delimited tokens plus EOS define disjoint complete step outcomes.
    The next-token probability is mass(prefix + token) / mass(prefix).
    """

    def __init__(self, weighted_texts: Mapping[str, float]):
        if not weighted_texts:
            raise ValueError("At least one textual outcome is required")
        merged: dict[str, float] = defaultdict(float)
        for text, weight in weighted_texts.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Textual outcomes must be nonempty strings")
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Weights must be finite and nonnegative")
            normalized = " ".join(text.split())
            if EOS in normalized.split():
                raise ValueError("EOS is reserved")
            merged[normalized] += float(weight)
        total = math.fsum(merged.values())
        if total <= 0:
            raise ValueError("Total weight must be positive")
        self.text_probabilities = {text: weight / total for text, weight in merged.items() if weight > 0}
        self._prefix_mass: dict[tuple[str, ...], float] = defaultdict(float)
        self._next_mass: dict[tuple[str, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for text, probability in self.text_probabilities.items():
            tokens = (*text.split(), EOS)
            for index, token in enumerate(tokens):
                prefix = tokens[:index]
                self._prefix_mass[prefix] += probability
                self._next_mass[prefix][token] += probability

    @classmethod
    def from_weighted_texts(cls, mapping: Mapping[str, float]) -> "FiniteTokenPredictor":
        return cls(mapping)

    def token_probs(self, prefix: Sequence[str] | str = ()) -> dict[str, float]:
        prefix = tuple(prefix.split()) if isinstance(prefix, str) else tuple(prefix)
        mass = self._prefix_mass.get(prefix, 0.0)
        if mass <= 0:
            return {}
        return {token: value / mass for token, value in self._next_mass[prefix].items()}

    def sequence_probability(self, text: str) -> float:
        prefix: tuple[str, ...] = ()
        probability = 1.0
        for token in (*text.split(), EOS):
            probability *= self.token_probs(prefix).get(token, 0.0)
            if probability == 0:
                return 0.0
            prefix += (token,)
        return probability

    def itoken_probabilities(self) -> dict[IToken, float]:
        """Aggregate probabilities for predicting each next canonical iToken."""
        totals: dict[IToken, float] = defaultdict(float)
        for text in self.text_probabilities:
            totals[parse_step(text)] += self.sequence_probability(text)
        return dict(totals)


def make_step_predictor(
    problem: Problem,
    stage: int,
    route_probabilities: Mapping[str, float] | None = None,
    style_probabilities: Mapping[str, float] | None = None,
) -> FiniteTokenPredictor:
    """Give each first-step iToken three textual realizations of known mass."""
    route_p = dict(DEFAULT_ROUTE_PROBABILITIES if route_probabilities is None else route_probabilities)
    style_p = dict(DEFAULT_STYLE_PROBABILITIES if style_probabilities is None else style_probabilities)
    if set(route_p) != set(ROUTES) or set(style_p) != set(STYLES):
        raise ValueError("Supply exactly the three route/style probabilities")
    for probabilities in (route_p, style_p):
        if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
            raise ValueError("Probabilities must be finite and nonnegative")
        if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-12):
            raise ValueError("Probabilities must sum to one")
    pairs = candidate_pairs(problem, stage)
    outcomes = {
        render_step(pairs[route][0], style): route_p[route] * style_p[style]
        for route in ROUTES for style in STYLES
    }
    return FiniteTokenPredictor.from_weighted_texts(outcomes)


if __name__ == "__main__":
    problem = make_problem(3)
    proof = [step for stage, route in enumerate(ROUTES) for step in candidate_pairs(problem, stage)[route]]
    assert validate_proof(problem, proof)
    assert validate_proof(problem, list(reversed(proof)))
    for style in STYLES:
        assert parse_proof(render_proof(proof, style)) == proof
    predictor = make_step_predictor(problem, 0)
    print(render_proof(proof))
    print("Next-iToken prediction probabilities:")
    for step, probability in predictor.itoken_probabilities().items():
        print(step.rule, probability)
