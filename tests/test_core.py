"""Independent correctness checks for the restricted symbolic reference domain."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (
    And, FiniteTokenPredictor, IToken, Problem, ROUTES, STYLES,
    candidate_pairs, make_problem, make_step_predictor, parse_proof,
    parse_step, render_proof, render_step, valid_inference, validate_proof,
)


def truth_value(formula: str, assignment: dict[str, bool]) -> bool:
    """Independent truth-table evaluator for fully parenthesized test formulas.

    Deliberately does not use core's parser, inference checker, or factories.
    It supplies a semantic oracle rather than reproducing the inference rules.
    """
    if not formula.startswith("("):
        return assignment[formula]
    if not formula.endswith(")"):
        raise ValueError("Unbalanced test formula")
    body = formula[1:-1]
    depth = 0
    for index, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and (char in "&|" or body[index:index + 2] == "->"):
            operation = "->" if char == "-" else char
            left = truth_value(body[:index], assignment)
            right = truth_value(body[index + len(operation):], assignment)
            return {"&": left and right, "|": left or right, "->": (not left) or right}[operation]
    raise ValueError("No top-level connective in compound test formula")


def semantic_counterexamples(step: IToken):
    atoms = sorted(set(re.findall(r"[a-z][a-z0-9_]*", " ".join((*step.premises, step.conclusion)))))
    failures = []
    for values in itertools.product((False, True), repeat=len(atoms)):
        assignment = dict(zip(atoms, values))
        if all(truth_value(premise, assignment) for premise in step.premises):
            if not truth_value(step.conclusion, assignment):
                failures.append(assignment)
    return failures


class LogicalCoreTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_problem(3, "test-proof")

    def test_generated_inferences_are_truth_table_sound(self):
        for stage in range(self.problem.n_stages):
            for route, pair in candidate_pairs(self.problem, stage).items():
                for step in pair:
                    with self.subTest(stage=stage, route=route, rule=step.rule):
                        self.assertFalse(semantic_counterexamples(step))
                        self.assertTrue(valid_inference(step))

    def test_independent_oracle_detects_invalid_inferences(self):
        invalid = (
            IToken("MP", ("p0", "(d0->m0)"), "m0"),
            IToken("AND_INTRO", ("p0", "d0"), "(p0&e0)"),
            IToken("OR_INTRO", ("p0",), "(d0|e0)"),
        )
        for step in invalid:
            with self.subTest(rule=step.rule):
                self.assertTrue(semantic_counterexamples(step))
                self.assertFalse(valid_inference(step))

    def test_all_small_paths_validate_as_graphs(self):
        for routes in itertools.product(ROUTES, repeat=3):
            steps = [step for t, route in enumerate(routes)
                     for step in candidate_pairs(self.problem, t)[route]]
            with self.subTest(routes=routes):
                self.assertEqual(len(steps), 6)
                self.assertTrue(validate_proof(self.problem, steps))
                self.assertTrue(validate_proof(self.problem, list(reversed(steps))))
                self.assertTrue(validate_proof(self.problem, steps * 2))

    def test_missing_antecedent_is_not_recovered_from_final_claim(self):
        steps = [step for t in range(3) for step in candidate_pairs(self.problem, t)["mp"]]
        self.assertFalse(validate_proof(self.problem, steps[1:]))
        self.assertFalse(validate_proof(self.problem, [steps[-1]]))

    def test_unsupported_cycle_is_not_a_proof(self):
        problem = Problem(1, "cycle", frozenset(("(x->y)", "(y->x)")), ("y",))
        cycle = [IToken("MP", ("x", "(x->y)"), "y"),
                 IToken("MP", ("y", "(y->x)"), "x")]
        self.assertTrue(all(valid_inference(step) for step in cycle))
        self.assertFalse(validate_proof(problem, cycle))

    def test_rendering_grammars_preserve_itokens(self):
        for stage in range(3):
            for pair in candidate_pairs(self.problem, stage).values():
                for style in STYLES:
                    self.assertEqual(parse_proof(render_proof(pair, style)), list(pair))
                    for step in pair:
                        self.assertEqual(parse_step(render_step(step, style)), step)
                        self.assertEqual(step.stage, stage)

    def test_premise_and_operand_order_canonicalization(self):
        left = IToken("AND_INTRO", ("p0", "d0"), "(p0&d0)")
        right = IToken("AND_INTRO", ("d0", "p0"), "( d0 & p0 )")
        self.assertEqual(left, right)
        self.assertEqual(left.canonical(), right.canonical())
        self.assertEqual(And("p0", "d0"), And("d0", "p0"))
        ordered = "From p0 and d0, infer (p0&d0) by conjunction introduction."
        reversed_premises = "From d0 and p0, infer (d0&p0) by conjunction introduction."
        self.assertEqual(parse_step(ordered), parse_step(reversed_premises))

    def test_unsupported_prose_and_malformed_formula_fail_closed(self):
        for text in ("Clearly the answer follows.", "From p0, infer p1 by intuition.",
                     "From p0, infer (p0|) by disjunction introduction.", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_step(text)
        with self.assertRaises(ValueError):
            parse_proof("We now skip several obvious steps.")


class TokenPredictionTests(unittest.TestCase):
    def test_conditional_token_products_recover_text_probabilities(self):
        predictor = make_step_predictor(make_problem(1), 0)
        for text, expected in predictor.text_probabilities.items():
            self.assertAlmostEqual(predictor.sequence_probability(text), expected, places=14)
            words = text.split()
            for length in range(len(words) + 1):
                self.assertAlmostEqual(sum(predictor.token_probs(words[:length]).values()), 1.0)

    def test_termination_disambiguates_prefix_overlapping_strings(self):
        predictor = FiniteTokenPredictor.from_weighted_texts({"a": 0.4, "a b": 0.6})
        self.assertAlmostEqual(predictor.sequence_probability("a"), 0.4)
        self.assertAlmostEqual(predictor.sequence_probability("a b"), 0.6)
        self.assertEqual(predictor.sequence_probability("a b c"), 0.0)
        after_a = predictor.token_probs(("a",))
        self.assertAlmostEqual(after_a["b"], 0.6)
        self.assertAlmostEqual(after_a["<END_OF_STEP>"], 0.4)

    def test_aggregation_predicts_next_itoken_independent_of_wording_mass(self):
        problem = make_problem(1)
        route_p = {"mp": 0.27, "and": 0.12, "or": 0.61}
        pair = candidate_pairs(problem, 0)
        for style_p in (
            {"plain": 0.1, "paraphrase": 0.2, "compact": 0.7},
            {"plain": 0.8, "paraphrase": 0.1, "compact": 0.1},
        ):
            predictor = make_step_predictor(problem, 0, route_p, style_p)
            inferred = predictor.itoken_probabilities()
            self.assertEqual(len(inferred), 3)
            self.assertAlmostEqual(math.fsum(inferred.values()), 1.0)
            for route, expected in route_p.items():
                self.assertAlmostEqual(inferred[pair[route][0]], expected, places=14)

    def test_bad_probability_support_is_rejected(self):
        for weights in ({}, {"a": -1.0}, {"a": math.nan}, {"a": 0.0}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                FiniteTokenPredictor.from_weighted_texts(weights)


if __name__ == "__main__":
    unittest.main()
