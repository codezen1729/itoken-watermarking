"""Accounting and isolation checks; no trained model is invoked in these tests."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core import candidate_pairs, make_problem, parse_step, render_step
import research_runner as runner
from semantic_recovery import RecoveryResult


def case(label="complete", route="mp", *, ident="unit-dev", split="dev", positions=(0, 1)):
    problem = make_problem(1, "unit-problem")
    steps = [candidate_pairs(problem, 0)[route][j] for j in positions] if label != "abstain" else []
    return {
        "id": ident, "split": split, "n_stages": 1, "problem_id": problem.problem_id,
        "category": "unit-only", "text": "Unit source text; no gold fields belong in a model prompt.",
        "expected_status": label, "expected_canonical": [s.canonical() for s in steps],
        "expected_routes": {"0": route} if steps and 0 in positions else {},
        "source_routes": {"private_marker": "GOLD_PRIVATE_MARKER_918"},
        "reason": "GOLD_PRIVATE_REASON_918",
    }


def recovery(route="mp", positions=(0, 1), complete=True):
    problem = make_problem(1, "unit-problem")
    steps = [candidate_pairs(problem, 0)[route][j] for j in positions]
    return RecoveryResult(status="complete_proof" if complete else "partial_evidence",
                          steps=steps, complete=complete)


def row(gold, result):
    return {"gold": gold, "recovery": result.to_dict(),
            "metrics": runner.result_metrics(gold, result),
            "generation": {"finish_reason": "stop", "wall_seconds": 0.0}}


class MetricAccountingTests(unittest.TestCase):
    def test_abstain_all_has_zero_positive_recall(self):
        no = RecoveryResult(status="abstained")
        rows = [row(case(), no),
                row(case("partial", positions=(0,)), no),
                row(case("abstain"), no)]
        summary = runner.recovery_summary(rows)
        self.assertEqual(summary["positive_cases"], 2)
        self.assertEqual(summary["positive_exact_recoveries"], 0)
        self.assertEqual(summary["step_recall"], 0.0)
        self.assertIsNone(summary["step_precision"])
        self.assertEqual(summary["strict_successes"], 1)
        self.assertEqual(summary["false_acceptances"], 0)

    def test_wrong_valid_route_is_not_correct_recovery(self):
        result = runner.result_metrics(case(route="mp"), recovery(route="and"))
        self.assertFalse(result["strict_success"])
        self.assertFalse(result["exact_explicit_routes"])
        self.assertEqual(result["correct_steps"], 0)
        self.assertEqual(result["predicted_steps"], 2)
        self.assertEqual(len(result["extra_steps"]), 2)
        self.assertEqual(len(result["missing_steps"]), 2)

    def test_valid_route_on_rejection_case_is_false_acceptance(self):
        summary = runner.recovery_summary([row(case("abstain"), recovery())])
        self.assertEqual(summary["false_acceptances"], 1)
        self.assertEqual(summary["step_precision"], 0.0)
        self.assertEqual(summary["strict_successes"], 0)

    def test_correct_partial_is_scored_as_partial(self):
        gold = case("partial", route="and", positions=(0,))
        result = runner.result_metrics(gold, recovery("and", (0,), complete=False))
        self.assertTrue(result["strict_success"])
        self.assertEqual(result["correct_steps"], 1)
        self.assertEqual(result["predicted_routes"], {"0": "and"})

    def test_completion_only_does_not_invent_a_branching_choice(self):
        gold = case("partial", positions=(1,))
        result = runner.result_metrics(gold, recovery(positions=(1,), complete=False))
        self.assertTrue(result["strict_success"])
        self.assertEqual(result["predicted_routes"], {})

    def test_duplicate_steps_do_not_inflate_counts(self):
        answer = recovery()
        answer.steps *= 4
        metrics = runner.result_metrics(case(), answer)
        self.assertTrue(metrics["strict_success"])
        self.assertEqual(metrics["correct_steps"], 2)
        self.assertEqual(metrics["predicted_steps"], 2)

    def test_candidate_contexts_have_three_checked_units_not_nine_units(self):
        for specification in runner.context_cases():
            _, candidates, messages = runner.scoring_request(specification)
            self.assertEqual(len(candidates), 9)
            self.assertEqual(len({c["itoken_id"] for c in candidates}), 3)
            self.assertEqual(len({c["text"] for c in candidates}), 9)
            self.assertTrue(messages)
            for candidate in candidates:
                self.assertEqual(parse_step(candidate["text"]).canonical(), candidate["itoken_id"])


class RecordingGenerator:
    """A fixed-output stub for runner I/O tests, never probability evidence."""
    metadata = {
        "checkpoint": {"manifest_sha256": "unit-checkpoint"},
        "revision": "unit-revision", "packages": {"runtime": "unit-v1"},
        "device": "unit-cpu",
    }

    def __init__(self, output=None, finish_reason="stop"):
        self.calls = []
        self.output = output or '{"status":"abstain","steps":[],"reason":"unit stub"}'
        self.finish_reason = finish_reason

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"text": self.output, "finish_reason": self.finish_reason,
                "generated_tokens": 1, "wall_seconds": 0.0}


class RunnerIsolationTests(unittest.TestCase):
    def make_dataset(self, root):
        dev = case(ident="unit-dev", split="dev")
        dev["text"] = "DEV_INPUT_ONLY_918"
        test = case(ident="unit-test", split="test")
        test["text"] = "TEST_INPUT_ONLY_918"
        (root / "research_eval_cases.json").write_text(json.dumps({"cases": [dev, test]}))

    def test_split_selection_and_gold_are_not_sent_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            model = RecordingGenerator()
            with patch.object(runner, "ROOT", root), patch.object(runner, "freeze_record", return_value={"unit": True}), redirect_stdout(io.StringIO()):
                runner.run_recovery(model, root / "out", "dev", max_tokens=64)
            self.assertEqual(len(model.calls), 1)
            prompt = model.calls[0]["messages"][0]["content"]
            self.assertIn("DEV_INPUT_ONLY_918", prompt)
            for forbidden in ("TEST_INPUT_ONLY_918", "GOLD_PRIVATE_MARKER_918", "GOLD_PRIVATE_REASON_918", "expected_canonical", "source_routes"):
                self.assertNotIn(forbidden, prompt)
            stored = json.loads((root / "out/recovery/unit-dev.json").read_text())
            self.assertEqual(stored["generation"]["text"], model.output)
            self.assertIn("gold", stored)  # Gold is retained only for offline auditing.
            self.assertFalse((root / "out/recovery/unit-test.json").exists())

    def test_test_phase_requires_recorded_development_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            model = RecordingGenerator()
            with patch.object(runner, "ROOT", root), patch.object(runner, "freeze_record", return_value={"unit": True}):
                with self.assertRaises(ValueError):
                    runner.run_recovery(model, root / "out", "test", max_tokens=64)
            self.assertEqual(model.calls, [])

    def test_resume_reuses_raw_record_instead_of_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            model = RecordingGenerator()
            with patch.object(runner, "ROOT", root), patch.object(runner, "freeze_record", return_value={"unit": True}), redirect_stdout(io.StringIO()):
                runner.run_recovery(model, root / "out", "dev", max_tokens=64)
                runner.run_recovery(model, root / "out", "dev", max_tokens=64)
            self.assertEqual(len(model.calls), 1)

    def test_changed_core_invalidates_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("semantic_recovery.py", "real_model.py", "core.py"):
                (root / name).write_text("# unit source\n")
            dataset = root / "cases.json"
            dataset.write_text('{"cases":[]}')
            with patch.object(runner, "ROOT", root):
                runner.freeze_record(root / "out", RecordingGenerator(), dataset, 64)
                (root / "core.py").write_text("# changed semantic checker\n")
                with self.assertRaises(ValueError):
                    runner.freeze_record(root / "out", RecordingGenerator(), dataset, 64)

    def test_changed_runtime_invalidates_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("semantic_recovery.py", "real_model.py", "core.py"):
                (root / name).write_text("# unit source\n")
            dataset = root / "cases.json"
            dataset.write_text('{"cases":[]}')
            model = RecordingGenerator()
            model.metadata = dict(model.metadata)
            with patch.object(runner, "ROOT", root):
                runner.freeze_record(root / "out", model, dataset, 64)
                model.metadata = {**model.metadata, "packages": {"runtime": "unit-v2"}}
                with self.assertRaises(ValueError):
                    runner.freeze_record(root / "out", model, dataset, 64)

    def test_truncated_complete_json_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = case()
            problem = make_problem(1, gold["problem_id"])
            steps = candidate_pairs(problem, 0)["mp"]
            phrases = [render_step(step) for step in steps]
            gold["text"] = "\n".join(phrases)
            (root / "research_eval_cases.json").write_text(json.dumps({"cases": [gold]}))
            output = json.dumps({"status": "extracted", "steps": [
                {**json.loads(step.canonical()), "evidence": text}
                for step, text in zip(steps, phrases)
            ], "reason": "unit output"})
            model = RecordingGenerator(output=output, finish_reason="length")
            with patch.object(runner, "ROOT", root), patch.object(runner, "freeze_record", return_value={"unit": True}), redirect_stdout(io.StringIO()):
                runner.run_recovery(model, root / "out", "dev", max_tokens=64)
            stored = json.loads((root / "out/recovery/unit-dev.json").read_text())
            self.assertEqual(stored["generation"]["text"], output)
            self.assertFalse(stored["recovery"]["accepted"])
            self.assertFalse(stored["metrics"]["strict_success"])


if __name__ == "__main__":
    unittest.main()
