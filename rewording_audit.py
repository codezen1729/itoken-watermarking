"""Independent accounting helpers for fresh recovery and paired rewording.

These functions do not invoke a model or certify textual entailment. They
require explicit failed/abstained records, so dropping an unfavorable document
cannot silently improve a survival denominator.
"""
from __future__ import annotations

from collections import Counter
import json
import math

from core import IToken, candidate_pairs, make_problem, valid_inference, validate_proof


def _tokens(values):
    if not isinstance(values, list):
        raise ValueError("Step records must be a list")
    result = set()
    for value in values:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict) or set(value) != {"rule", "premises", "conclusion"}:
            raise ValueError("A canonical step needs exactly rule, premises, conclusion")
        if not isinstance(value["premises"], list):
            raise ValueError("Canonical premises must be a list")
        result.add(IToken(value["rule"], tuple(value["premises"]), value["conclusion"]))
    return result


def recovery_metrics(case, recovered):
    """Score admitted output, never a rejected proposal hidden in raw output."""
    if type(recovered.get("accepted")) is not bool or type(recovered.get("complete")) is not bool:
        raise ValueError("Recovery needs explicit accepted and complete booleans")
    if recovered["complete"] and not recovered["accepted"]:
        raise ValueError("A rejected recovery cannot be complete")
    accepted = recovered["accepted"]
    pred = _tokens(recovered.get("steps", [])) if accepted else set()
    gold = _tokens(case["expected_steps"])
    problem = make_problem(case["n_stages"], case["problem_id"])
    admitted = {s for stage in range(problem.n_stages)
                for pair in candidate_pairs(problem, stage).values() for s in pair}
    branches = {}
    for stage in range(problem.n_stages):
        found = [route for route, pair in candidate_pairs(problem, stage).items() if pair[0] in pred]
        if found:
            branches[str(stage)] = found[0] if len(found) == 1 else found
    label = case["expected_status"]
    if label not in {"complete", "partial", "abstain"}:
        raise ValueError("Unknown gold status")
    status_correct = ((accepted and recovered["complete"]) if label == "complete" else
                      (accepted and not recovered["complete"]) if label == "partial" else not accepted)
    return {
        "exact_steps": pred == gold, "status_correct": status_correct,
        "strict_success": pred == gold and status_correct,
        "gold_steps": len(gold), "predicted_steps": len(pred), "correct_steps": len(pred & gold),
        "missing_steps": sorted(s.canonical() for s in gold - pred),
        "extra_steps": sorted(s.canonical() for s in pred - gold),
        "predicted_routes": branches, "exact_explicit_routes": branches == case["expected_routes"],
        "false_acceptance": label == "abstain" and accepted,
        "accepted_steps_rule_valid": all(valid_inference(s) for s in pred),
        "accepted_steps_admitted": pred <= admitted,
        "accepted_graph_complete": accepted and validate_proof(problem, list(pred)),
    }


def summarize_recovery(cases, recoveries):
    """Require every declared case exactly once, including failure records."""
    identifiers = [c["id"] for c in cases]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(recoveries):
        raise ValueError("Recovery identifiers must match the full declared cohort")
    rows = [(case, recoveries[case["id"]], recovery_metrics(case, recoveries[case["id"]]))
            for case in cases]
    predicted = sum(row[2]["predicted_steps"] for row in rows)
    gold = sum(row[2]["gold_steps"] for row in rows)
    correct = sum(row[2]["correct_steps"] for row in rows)
    summary = {
        "cases": len(rows), "strict_successes": sum(row[2]["strict_success"] for row in rows),
        "gold_steps": gold, "predicted_steps": predicted, "correct_steps": correct,
        "step_precision": correct / predicted if predicted else None,
        "step_recall": correct / gold if gold else None,
        "accepted_cases": sum(row[1]["accepted"] for row in rows),
        "false_acceptances": sum(row[2]["false_acceptance"] for row in rows),
        "statuses": dict(Counter(row[1].get("status", "unreported") for row in rows)),
    }
    for status in ("complete", "partial", "abstain"):
        group = [row for row in rows if row[0]["expected_status"] == status]
        summary[status + "_cases"] = len(group)
        summary[status + "_strict_successes"] = sum(row[2]["strict_success"] for row in group)
    evidence = [row for row in rows if row[0]["expected_routes"]]
    summary["cases_with_explicit_branch_choices"] = len(evidence)
    summary["exact_route_recoveries"] = sum(row[2]["exact_explicit_routes"] for row in evidence)
    return summary


def _fraction(successes, trials):
    return {"successes": successes, "trials": trials,
            "fraction": successes / trials if trials else None}


def summarize_pairs(rows, *, expected_units=8, alpha=0.01, surrogate_count=199):
    """Descriptive paired counts, without independent-binomial assumptions.

    Each row needs unit_id, condition (unmarked/marked), variant
    (original/reworded), planned_detected, detection, recovery, and fidelity.
    Fidelity is an externally supplied assessment, never inferred from detector
    success or planned/recovered path equality.
    """
    if not 0 < alpha < 1 or surrogate_count < 1:
        raise ValueError("Invalid detector settings")
    indexed = {}
    for row in rows:
        identity = (row["unit_id"], row["condition"], row["variant"])
        if identity in indexed:
            raise ValueError("Duplicate paired document")
        if identity[1] not in {"unmarked", "marked"} or identity[2] not in {"original", "reworded"}:
            raise ValueError("Unknown condition or variant")
        if type(row["planned_detected"]) is not bool:
            raise ValueError("Planned detection must be explicit")
        detection = row["detection"]
        p, detected, n = detection.get("p_value"), detection.get("detected"), detection.get("n_choices")
        if type(n) is not int or n < 0:
            raise ValueError("Choice count must be a nonnegative integer")
        if p is None:
            if detected is not None:
                raise ValueError("An unscored detection must explicitly abstain")
        else:
            if (isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
                    or not 1 / (surrogate_count + 1) <= p <= 1 or n == 0
                    or type(detected) is not bool or detected != (p <= alpha)):
                raise ValueError("Inconsistent surrogate-key detector record")
            rank = p * (surrogate_count + 1)
            if not math.isclose(rank, round(rank), abs_tol=1e-8):
                raise ValueError("p-value does not match the declared surrogate-key resolution")
        if type(row["recovery"].get("accepted")) is not bool:
            raise ValueError("Recovery acceptance must be explicit")
        if not row["recovery"]["accepted"] and p is not None:
            raise ValueError("Rejected recovery cannot supply scored branching evidence")
        if row["fidelity"].get("status") not in {"preserved", "changed", "uncertain", "not_assessed"}:
            raise ValueError("Unknown fidelity assessment")
        indexed[identity] = row
    units = sorted({identity[0] for identity in indexed})
    if len(units) != expected_units or len(indexed) != 4 * expected_units:
        raise ValueError("The complete declared paired cohort is required")
    expected = {(unit, condition, variant) for unit in units
                for condition in ("unmarked", "marked") for variant in ("original", "reworded")}
    if set(indexed) != expected:
        raise ValueError("Missing paired document")
    result = {"units": expected_units, "documents": len(rows), "alpha": alpha,
              "surrogate_keys": surrogate_count, "uncertainty_scope": "descriptive paired pilot; no independent-binomial interval"}
    for condition in ("unmarked", "marked"):
        original = [indexed[(u, condition, "original")] for u in units]
        rewritten = [indexed[(u, condition, "reworded")] for u in units]
        for a, b in zip(original, rewritten):
            if a["planned_detected"] != b["planned_detected"]:
                raise ValueError("A rewrite cannot change the planned-path baseline record")
        positive = lambda r: r["detection"]["detected"] is True
        planned_denominator = sum(r["planned_detected"] for r in original)
        source_denominator = sum(positive(r) for r in original)
        faithful = [r for r in rewritten if r["fidelity"]["status"] == "preserved"]
        result[condition] = {
            "planned_detection": _fraction(planned_denominator, expected_units),
            "original_joint_detection": _fraction(source_denominator, expected_units),
            "reworded_joint_detection": _fraction(sum(positive(r) for r in rewritten), expected_units),
            "survival_from_planned_detected": _fraction(
                sum(a["planned_detected"] and positive(b) for a, b in zip(original, rewritten)), planned_denominator),
            "survival_from_original_detected": _fraction(
                sum(positive(a) and positive(b) for a, b in zip(original, rewritten)), source_denominator),
            "faithful_reworded_joint_detection": _fraction(sum(positive(r) for r in faithful), expected_units),
            "detection_given_assessed_faithfulness": _fraction(sum(positive(r) for r in faithful), len(faithful)),
            "rewording_fidelity": dict(Counter(r["fidelity"]["status"] for r in rewritten)),
            "original_recovery_acceptances": sum(r["recovery"]["accepted"] for r in original),
            "reworded_recovery_acceptances": sum(r["recovery"]["accepted"] for r in rewritten),
            "original_detector_abstentions": sum(r["detection"]["detected"] is None for r in original),
            "reworded_detector_abstentions": sum(r["detection"]["detected"] is None for r in rewritten),
            "original_recovered_choices": [r["detection"]["n_choices"] for r in original],
            "reworded_recovered_choices": [r["detection"]["n_choices"] for r in rewritten],
        }
    return result
