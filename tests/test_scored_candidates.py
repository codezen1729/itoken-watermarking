"""Synthetic analytic checks only; these fixtures are not real LLM scores."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import candidate_pairs, make_problem, render_step
from scored_candidates import aggregate_scored_candidates


class ScoredCandidateTests(unittest.TestCase):
    def setUp(self):
        pairs = candidate_pairs(make_problem(1), 0)
        self.mp = pairs["mp"][0]
        self.other = pairs["and"][0]

    def record(self, step, probability, style="plain"):
        # One synthetic scored event and a deterministic terminating token.
        return {"text": render_step(step, style),
                "token_logprobs": [math.log(probability), 0.0],
                "includes_termination": True}

    def aggregate(self, records):
        return aggregate_scored_candidates(records, context_id="synthetic-context", tokenizer_id="synthetic-tokenizer")

    def test_analytic_aggregation_is_conditional(self):
        report = self.aggregate([self.record(self.mp, .1),
                                 self.record(self.mp, .2, "paraphrase"),
                                 self.record(self.other, .1)])
        self.assertAlmostEqual(report["covered_probability_mass"], .4)
        masses = {item["canonical"]: item["probability"] for item in report["itokens"]}
        self.assertAlmostEqual(masses[self.mp.canonical()], .75)
        self.assertAlmostEqual(masses[self.other.canonical()], .25)
        self.assertEqual(report["distribution_scope"], "conditional_on_listed_candidates")
        self.assertEqual(report["support_status"], "not_certified_complete")
        self.assertFalse(report["model_invoked"])

    def test_tiny_raw_mass_preserves_conditional_distribution(self):
        a, b = self.record(self.mp, .1), self.record(self.other, .1)
        a["token_logprobs"], b["token_logprobs"] = [-10000.0, 0.0], [-10001.0, 0.0]
        report = self.aggregate([a, b])
        self.assertEqual(report["covered_probability_mass"], 0.0)
        self.assertTrue(math.isfinite(report["log_covered_probability_mass"]))
        masses = {item["canonical"]: item["probability"] for item in report["itokens"]}
        self.assertAlmostEqual(masses[self.mp.canonical()], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(sum(masses.values()), 1.0)

    def test_rejects_duplicate_and_whitespace_variants(self):
        a = self.record(self.mp, .1)
        for text in (a["text"], "  " + a["text"].replace(" ", "  ") + "  "):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.aggregate([a, dict(a, text=text)])

    def test_extreme_log_masses_do_not_lose_normalization(self):
        a, b = self.record(self.mp, .1), self.record(self.other, .1)
        a["token_logprobs"] = b["token_logprobs"] = [-1e300, 0.0]
        report = self.aggregate([a, b])
        self.assertEqual([item["probability"] for item in report["itokens"]], [.5, .5])

    def test_rejects_incompatible_mass(self):
        with self.assertRaisesRegex(ValueError, "exceed one"):
            self.aggregate([self.record(self.mp, .8), self.record(self.other, .8)])

    def test_rejects_bad_scores_and_termination(self):
        for values in ([], [1e-8], [math.inf], [-math.inf], [math.nan], [True], ["0"]):
            record = self.record(self.mp, .1)
            record["token_logprobs"] = values
            with self.assertRaises(ValueError):
                self.aggregate([record])
        for flag in (None, False, 1, "true"):
            with self.assertRaises(ValueError):
                self.aggregate([dict(self.record(self.mp, .1), includes_termination=flag)])

    def test_context_and_tokenizer_are_fixed(self):
        record = self.record(self.mp, .1)
        with self.assertRaisesRegex(ValueError, "context_id"):
            aggregate_scored_candidates([record])
        for field in ("context_id", "tokenizer_id"):
            with self.assertRaisesRegex(ValueError, field):
                self.aggregate([dict(record, **{field: "conflict"})])
        declared = dict(record, context_id="c", tokenizer_id="t")
        report = aggregate_scored_candidates([declared])
        self.assertEqual((report["context_id"], report["tokenizer_id"]), ("c", "t"))

    def test_rejects_unsound_inference_and_unsupported_prose(self):
        for text in ("This is an arbitrary idea.", "From p0 and (p0->m0), infer q0 by modus ponens."):
            with self.assertRaises(ValueError):
                self.aggregate([dict(self.record(self.mp, .1), text=text)])

    def test_cli_reads_jsonl_and_emits_honest_scope(self):
        script = Path(__file__).resolve().parents[1] / "scored_candidates.py"
        result = subprocess.run(
            [sys.executable, str(script), "-", "--context-id", "c", "--tokenizer", "t"],
            input=json.dumps(self.record(self.mp, .25)) + "\n", text=True,
            capture_output=True, check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["distribution_scope"], "conditional_on_listed_candidates")
        self.assertAlmostEqual(report["covered_probability_mass"], .25)
        self.assertEqual(report["itokens"][0]["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
