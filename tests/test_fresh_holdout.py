"""Validate holdout structure without exposing its example text in test output."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unittest

from core import IToken, candidate_pairs, make_problem, parse_proof, valid_inference, validate_proof


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "58212368fe99fe80f3ed4a015c0c4545a11d867c6544413b6416306090d6752f"


class FrozenHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "fresh_holdout.json").read_text())

    def test_fixed_cohort_and_no_old_text_reuse(self):
        self.assertEqual(hashlib.sha256((ROOT / "fresh_holdout.json").read_bytes()).hexdigest(), EXPECTED_SHA256)
        cases = self.data["cases"]
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({c["id"] for c in cases}), 32)
        self.assertEqual({c["split"] for c in cases}, {"test"})
        self.assertEqual(Counter(c["expected_status"] for c in cases), {"complete": 12, "partial": 8, "abstain": 12})
        old = json.loads((ROOT / "research_eval_cases.json").read_text())
        self.assertFalse({c["text"] for c in cases} & {c["text"] for c in old["cases"]})

    def test_gold_steps_are_sound_admitted_and_have_correct_graph_status(self):
        total = 0
        for case in self.data["cases"]:
            problem = make_problem(case["n_stages"], case["problem_id"])
            steps = [IToken(s["rule"], tuple(s["premises"]), s["conclusion"]) for s in case["expected_steps"]]
            admitted = {s for t in range(problem.n_stages) for pair in candidate_pairs(problem, t).values() for s in pair}
            self.assertEqual(len(set(steps)), len(steps))
            self.assertTrue(all(valid_inference(s) and s in admitted for s in steps))
            self.assertEqual(validate_proof(problem, steps), case["expected_status"] == "complete")
            self.assertEqual(sorted(s.canonical() for s in steps), case["expected_canonical"])
            if case["expected_status"] == "abstain":
                self.assertFalse(steps)
            total += len(steps)
        self.assertEqual(total, 47)

    def test_evidence_offsets_and_atom_anchors_are_exact(self):
        for case in self.data["cases"]:
            self.assertEqual(len(case["evidence"]), len(case["expected_steps"]))
            for item in case["evidence"]:
                self.assertEqual(case["text"][item["start"]:item["end"]], item["quote"])
                step = case["expected_steps"][item["step_index"]]
                atoms = set(re.findall(r"\b[pmed]\d+\b", " ".join(step["premises"] + [step["conclusion"]])))
                self.assertLessEqual(atoms, set(re.findall(r"\b[pmed]\d+\b", item["quote"])))

    def test_branch_gold_excludes_unobserved_choices_and_old_renderer(self):
        forced_only = 0
        for case in self.data["cases"]:
            problem = make_problem(case["n_stages"], case["problem_id"])
            observed = set(case["expected_canonical"])
            expected = {str(t): route for t in range(problem.n_stages)
                        for route, pair in candidate_pairs(problem, t).items() if pair[0].canonical() in observed}
            self.assertEqual(expected, case["expected_routes"])
            if case["expected_status"] == "partial" and not expected:
                forced_only += 1
            with self.assertRaises(ValueError):
                parse_proof(case["text"])
        self.assertEqual(forced_only, 3)


if __name__ == "__main__":
    unittest.main()
