"""Development tests for a compact output schema, not model accuracy evidence."""
import json
import unittest

from core import candidate_pairs, make_problem
from semantic_recovery import parse_extraction_output as parse_full
from semantic_recovery_ids import PROMPT_VERSION, inference_ids, make_extraction_prompt, parse_extraction_output


class SemanticRecoveryIDTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_problem(1, "id-interface-development")
        self.ids = inference_ids(self.problem)
        self.pair = candidate_pairs(self.problem, 0)["mp"]
        self.names = [name for step in self.pair for name, known in self.ids.items() if known == step]
        self.clauses = ["p0 establishes m0 using the supplied implication.",
                        "The established m0 then gives p1."]
        self.text = " ".join(self.clauses)

    def result(self, entries=None, text=None, status="extracted"):
        if entries is None:
            entries = [{"id": name, "evidence": clause} for name, clause in zip(self.names, self.clauses)]
        return parse_extraction_output(self.problem, self.text if text is None else text,
                                       json.dumps({"status": status, "steps": entries}))

    def test_complete_recovery_matches_frozen_full_schema_validator(self):
        result = self.result()
        full = parse_full(self.problem, self.text, json.dumps({
            "status": "extracted", "steps": [
                {**json.loads(step.canonical()), "evidence": clause}
                for step, clause in zip(self.pair, self.clauses)]}))
        self.assertEqual(result.status, full.status)
        self.assertEqual(result.steps, full.steps)
        self.assertTrue(result.complete)
        self.assertEqual(result.prompt_version, PROMPT_VERSION)
        self.assertIn('"id"', result.model_output)
        self.assertFalse(result.semantic_entailment_verified)

    def test_id_mapping_is_complete_and_independent_of_problem_identifier(self):
        self.assertEqual(len(self.ids), 6)
        self.assertEqual(self.ids, inference_ids(make_problem(1, "different-public-id")))
        self.assertEqual(len(set(self.ids.values())), 6)

    def test_unknown_id_is_not_repaired(self):
        self.assertEqual(self.result([{"id": "I999", "evidence": self.clauses[0]}]).status,
                         "invalid_model_output")

    def test_unknown_formula_fields_are_not_accepted_as_another_schema(self):
        entries = [{"id": self.names[0], "evidence": self.clauses[0], "conclusion": "m0"}]
        self.assertEqual(self.result(entries).status, "invalid_model_output")

    def test_evidence_and_negation_checks_remain_active(self):
        result = self.result([{"id": self.names[0], "evidence": "invented"}])
        self.assertEqual(result.status, "unsupported_evidence")
        negative = "We did not infer m0 from p0."
        result = self.result([{"id": self.names[0], "evidence": negative}], text=negative)
        self.assertEqual(result.status, "unsupported_evidence")

    def test_forced_completion_remains_partial_recovery(self):
        result = self.result([{"id": self.names[1], "evidence": self.clauses[1]}])
        self.assertEqual(result.status, "partial_evidence")
        self.assertEqual(result.steps, [self.pair[1]])

    def test_conflicting_ids_do_not_bypass_route_consistency(self):
        and_step = candidate_pairs(self.problem, 0)["and"][0]
        and_id = next(name for name, step in self.ids.items() if step == and_step)
        clause = "We join p0 and d0 into their conjunction."
        result = self.result([{"id": self.names[0], "evidence": self.clauses[0]},
                              {"id": and_id, "evidence": clause}],
                             text=self.text + " " + clause)
        self.assertEqual(result.status, "ambiguous_path")

    def test_abstention_is_preserved_without_gold_completion(self):
        self.assertEqual(self.result([], text="The goal is established.", status="abstain").status,
                         "abstained")

    def test_prompt_uses_neutral_placeholder_and_no_key_information(self):
        prompt = make_extraction_prompt(self.problem, self.text)
        self.assertIn('"id":"ID_FROM_LIST"', prompt)
        self.assertIn("Do not solve the problem", prompt)
        self.assertNotIn("preferred", prompt)
        self.assertNotIn("secret", prompt)
        self.assertTrue(all(name in prompt for name in self.ids))

    def test_duplicate_json_fields_are_rejected(self):
        raw = '{"status":"extracted","status":"abstain","steps":[]}'
        self.assertEqual(parse_extraction_output(self.problem, self.text, raw).status,
                         "invalid_model_output")


if __name__ == "__main__":
    unittest.main()
