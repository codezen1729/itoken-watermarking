"""End-to-end detector tests with exact known baselines, no external model."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ROUTES, candidate_pairs, make_problem, render_proof, validate_proof
from pipeline import (
    base_route_probabilities, context, demo_key, diagnose, draw, extract,
    generate, steps_for_routes, surrogate_keys, transform,
)
from watermark import bias_distribution


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_problem(6, "pipeline-test")
        self.key = demo_key(12)
        self.surrogates = surrogate_keys(81, 19)
        self.generated = generate(self.problem, self.key, 2.0, random.Random(25))

    def test_controlled_reword_shuffle_and_duplicate_preserve_detection(self):
        original = extract(self.problem, render_proof(self.generated["steps"]))
        reference = None
        for attack in ("clean", "controlled_rewording", "presentation_shuffle", "duplicate_steps"):
            text = transform(self.problem, self.generated, attack, random.Random(30))
            extracted = extract(self.problem, text)
            diagnosed = diagnose(self.problem, text, self.key, self.surrogates, alpha=0.05)
            with self.subTest(attack=attack):
                self.assertTrue(extracted.complete)
                self.assertEqual(extracted.choices, original.choices)
                self.assertEqual(extracted.routes, original.routes)
                self.assertEqual(diagnosed["n_choices"], self.problem.n_stages)
                comparison = (diagnosed["p_value"], diagnosed["preferred"],
                              diagnosed["log_likelihood_ratio"])
                if reference is None:
                    reference = comparison
                self.assertEqual(comparison, reference)

    def test_generation_and_detection_likelihood_ratios_agree(self):
        text = render_proof(self.generated["steps"])
        diagnosed = diagnose(self.problem, text, self.key, self.surrogates, delta=2.0)
        self.assertTrue(diagnosed["proof_valid"])
        self.assertAlmostEqual(diagnosed["log_likelihood_ratio"], self.generated["log_lr"], places=13)

    def test_conflicting_routes_abstain_even_if_graph_is_logically_valid(self):
        other = next(route for route in ROUTES if route != self.generated["routes"][0])
        steps = self.generated["steps"] + list(candidate_pairs(self.problem, 0)[other])
        self.assertTrue(validate_proof(self.problem, steps))
        text = render_proof(steps)
        result = diagnose(self.problem, text, self.key, self.surrogates)
        self.assertEqual(result["status"], "ambiguous_path")
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["detected"])
        self.assertIsNone(result["log_likelihood_ratio"])

    def test_deletion_never_gets_complete_path_likelihood_ratio(self):
        text = transform(self.problem, self.generated, "delete_25_percent", random.Random(11))
        result = diagnose(self.problem, text, self.key, self.surrogates)
        self.assertFalse(result["proof_valid"])
        self.assertIn(result["status"], ("partial_evidence", "insufficient_evidence"))
        self.assertIsNone(result["log_likelihood_ratio"])
        self.assertIsNone(result["likelihood_ratio_detected"])
        self.assertLessEqual(result["n_choices"], self.problem.n_stages)

    def test_no_choice_steps_and_unrecognized_text_abstain(self):
        forced_only = render_proof(self.generated["steps"][1::2])
        for text, expected_status in (
            (forced_only, "insufficient_evidence"),
            ("", "insufficient_evidence"),
            ("The conclusion follows by another line of reasoning.", "unsupported_text"),
        ):
            result = diagnose(self.problem, text, self.key, self.surrogates)
            with self.subTest(expected_status=expected_status):
                self.assertEqual(result["status"], expected_status)
                self.assertIsNone(result["p_value"])
                self.assertIsNone(result["detected"])

    def test_problem_identity_changes_structural_context(self):
        renamed = make_problem(6, "different-task-identifier")
        for stage in range(6):
            self.assertNotEqual(context(self.problem, stage), context(renamed, stage))
        self.assertNotEqual(context(self.problem, 0), context(self.problem, 1))

    def test_delta_zero_matches_baseline_and_ignores_key(self):
        first = generate(self.problem, self.key, 0.0, random.Random(80))
        second = generate(self.problem, demo_key(99), 0.0, random.Random(80))
        self.assertEqual(first["routes"], second["routes"])
        rng, previous, reference = random.Random(80), None, []
        for _ in range(6):
            previous = draw(base_route_probabilities(previous), rng)
            reference.append(previous)
        self.assertEqual(first["routes"], reference)
        self.assertAlmostEqual(first["log_lr"], 0.0, places=14)
        diagnosed = diagnose(self.problem, render_proof(first["steps"]),
                             self.key, self.surrogates, delta=0.0)
        self.assertAlmostEqual(diagnosed["log_likelihood_ratio"], 0.0, places=14)

    def test_exact_small_dependent_path_null_likelihood_expectation(self):
        """Enumerate all paths; verify normalization under a dependent baseline."""
        problem = make_problem(3, "enumerated-null")
        total_baseline, expected_likelihood = 0.0, 0.0
        for routes in itertools.product(ROUTES, repeat=3):
            previous, path_p = None, 1.0
            for route in routes:
                path_p *= base_route_probabilities(previous)[route]
                previous = route
            text = render_proof(steps_for_routes(problem, list(routes)))
            result = diagnose(problem, text, self.key, self.surrogates, delta=1.7)
            total_baseline += path_p
            expected_likelihood += path_p * math.exp(result["log_likelihood_ratio"])
        self.assertAlmostEqual(total_baseline, 1.0, places=13)
        self.assertAlmostEqual(expected_likelihood, 1.0, places=13)

    def test_rerouting_produces_valid_proofs_with_same_goal(self):
        for attack in ("reroute_25_percent", "full_rewrite"):
            text = transform(self.problem, self.generated, attack, random.Random(53))
            result = extract(self.problem, text)
            self.assertTrue(result.complete)
            self.assertEqual(len(result.routes), self.problem.n_stages)
            self.assertIn(self.problem.goal, {step.conclusion for step in result.steps})


if __name__ == "__main__":
    unittest.main()
