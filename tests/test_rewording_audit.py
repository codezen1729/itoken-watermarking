"""Independent accounting regressions; fixtures are not holdout examples."""
import copy
import json
import unittest

from core import candidate_pairs, make_problem
from rewording_audit import recovery_metrics, summarize_recovery, summarize_pairs


def fixture_case(label="complete", ident="unit", positions=(0, 1)):
    problem = make_problem(1, "accounting-unit")
    steps = [candidate_pairs(problem, 0)["mp"][i] for i in positions] if label != "abstain" else []
    return {"id": ident, "n_stages": 1, "problem_id": problem.problem_id,
            "expected_status": label, "expected_steps": [json.loads(s.canonical()) for s in steps],
            "expected_routes": {"0": "mp"} if label != "abstain" and 0 in positions else {}}


def fixture_recovery(route="mp", positions=(0, 1), complete=True, accepted=True):
    problem = make_problem(1, "accounting-unit")
    steps = [candidate_pairs(problem, 0)[route][i] for i in positions] if accepted else []
    return {"accepted": accepted, "complete": complete if accepted else False,
            "status": "complete_proof" if complete and accepted else "partial_evidence" if accepted else "abstained",
            "steps": [json.loads(s.canonical()) for s in steps]}


def pair_rows(units=2):
    return [{"unit_id": str(unit), "condition": condition, "variant": variant,
             "planned_detected": condition == "marked",
             "detection": {"detected": condition == "marked", "p_value": .005 if condition == "marked" else .5,
                           "n_choices": 32},
             "recovery": {"accepted": True},
             "fidelity": {"status": "preserved" if variant == "reworded" else "not_assessed"}}
            for unit in range(units) for condition in ("unmarked", "marked") for variant in ("original", "reworded")]


class RecoveryAccountingTests(unittest.TestCase):
    def test_abstain_all_cannot_improve_positive_recall(self):
        cases = [fixture_case(ident="positive"), fixture_case("abstain", ident="negative")]
        summary = summarize_recovery(cases, {c["id"]: fixture_recovery(accepted=False) for c in cases})
        self.assertEqual(summary["strict_successes"], 1)
        self.assertEqual(summary["step_recall"], 0)
        self.assertIsNone(summary["step_precision"])
        self.assertEqual(summary["complete_strict_successes"], 0)

    def test_valid_but_wrong_route_is_still_wrong(self):
        result = recovery_metrics(fixture_case(), fixture_recovery("and"))
        self.assertFalse(result["strict_success"])
        self.assertFalse(result["exact_explicit_routes"])
        self.assertEqual(result["correct_steps"], 0)
        self.assertTrue(result["accepted_graph_complete"])

    def test_forced_completion_is_partial_without_branch_evidence(self):
        case = fixture_case("partial", positions=(1,))
        answer = fixture_recovery(positions=(1,), complete=False)
        answer["steps"] *= 3
        result = recovery_metrics(case, answer)
        self.assertTrue(result["strict_success"])
        self.assertEqual(result["predicted_steps"], 1)
        self.assertEqual(result["predicted_routes"], {})

    def test_false_acceptance_is_included_in_precision(self):
        case = fixture_case("abstain")
        summary = summarize_recovery([case], {case["id"]: fixture_recovery()})
        self.assertEqual(summary["false_acceptances"], 1)
        self.assertEqual(summary["predicted_steps"], 2)
        self.assertEqual(summary["step_precision"], 0)

    def test_missing_case_is_not_silently_excluded(self):
        with self.assertRaises(ValueError):
            summarize_recovery([fixture_case()], {})


class PairAccountingTests(unittest.TestCase):
    def test_abstention_remains_in_joint_detection_denominator(self):
        rows = pair_rows()
        target = next(r for r in rows if (r["unit_id"], r["condition"], r["variant"]) == ("0", "marked", "reworded"))
        target["detection"] = {"detected": None, "p_value": None, "n_choices": 0}
        target["recovery"] = {"accepted": False}
        result = summarize_pairs(rows, expected_units=2)["marked"]
        self.assertEqual(result["reworded_joint_detection"], {"successes": 1, "trials": 2, "fraction": .5})
        self.assertEqual(result["survival_from_original_detected"]["fraction"], .5)
        self.assertEqual(result["reworded_detector_abstentions"], 1)

    def test_zero_survival_denominator_is_undefined(self):
        rows = pair_rows()
        for row in rows:
            row["planned_detected"] = False
            row["detection"] = {"detected": False, "p_value": .5, "n_choices": 32}
        result = summarize_pairs(rows, expected_units=2)["marked"]
        self.assertIsNone(result["survival_from_planned_detected"]["fraction"])
        self.assertIsNone(result["survival_from_original_detected"]["fraction"])
        self.assertEqual(result["reworded_joint_detection"]["fraction"], 0)

    def test_detector_success_is_not_fidelity(self):
        rows = pair_rows()
        for row in rows:
            if row["condition"] == "marked" and row["variant"] == "reworded":
                row["fidelity"]["status"] = "changed"
        result = summarize_pairs(rows, expected_units=2)["marked"]
        self.assertEqual(result["reworded_joint_detection"]["fraction"], 1)
        self.assertEqual(result["faithful_reworded_joint_detection"]["fraction"], 0)
        self.assertIsNone(result["detection_given_assessed_faithfulness"]["fraction"])

    def test_missing_or_duplicate_document_is_rejected(self):
        rows = pair_rows()
        with self.assertRaises(ValueError):
            summarize_pairs(rows[:-1], expected_units=2)
        with self.assertRaises(ValueError):
            summarize_pairs(rows+[copy.deepcopy(rows[0])], expected_units=2)

    def test_pvalue_and_declared_decision_must_agree(self):
        rows = pair_rows()
        rows[0]["detection"]["detected"] = True
        with self.assertRaises(ValueError):
            summarize_pairs(rows, expected_units=2)
        rows = pair_rows()
        rows[0]["detection"]["p_value"] = .0075
        rows[0]["detection"]["detected"] = True
        with self.assertRaises(ValueError):
            summarize_pairs(rows, expected_units=2)


if __name__ == "__main__":
    unittest.main()
