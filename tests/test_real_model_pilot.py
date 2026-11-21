"""Pilot accounting tests with a fixed fake backend, not real-model evidence."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core import IToken, candidate_pairs, make_problem, parse_proof, render_proof
from pipeline import context, surrogate_keys
import real_model_pilot as pilot
from watermark import keyed_preferred


class FixedBackend:
    """Known conditional distributions and fixed outputs for interface tests."""
    metadata = {"source_repo": "unit-only", "revision": "unit-revision",
                "checkpoint": {"manifest_sha256": "unit-manifest"},
                "packages": {"runtime": "unit-v1"}, "device": "unit-cpu",
                "model_invoked": False}

    def __init__(self, *, empty_verbalization=False, extraction_truncated=False,
                 verbalization_truncated=False):
        self.score_calls = []
        self.generation_calls = []
        self.empty_verbalization = empty_verbalization
        self.extraction_truncated = extraction_truncated
        self.verbalization_truncated = verbalization_truncated

    def score_candidates(self, messages, candidates, **kwargs):
        prompt = messages[0]["content"]
        state = json.loads(prompt.split("\nCurrent problem and path:\n", 1)[1])
        stage = int(state["next_goal"][1:]) - 1
        assert len(state["steps_already_taken"]) == 2 * stage
        previous = state["steps_already_taken"][-2]["rule"] if stage else None
        # Deliberately different from the earlier synthetic sampler, and
        # dependent on the actual selected preceding path.
        if previous == "AND_INTRO":
            mass = {"MP": .10, "AND_INTRO": .20, "OR_INTRO": .70}
        elif previous == "OR_INTRO":
            mass = {"MP": .20, "AND_INTRO": .70, "OR_INTRO": .10}
        else:
            mass = {"MP": .70, "AND_INTRO": .10, "OR_INTRO": .20}
        probabilities = {c["itoken_id"]: mass[json.loads(c["itoken_id"])["rule"]]
                         for c in candidates}
        self.score_calls.append({"messages": messages, "state": state, "probabilities": probabilities})
        return {"itoken_probabilities": probabilities,
                "distribution_scope": "unit-test-only-not-model-probabilities",
                "model_invoked": False}

    def generate(self, messages, **kwargs):
        prompt = messages[0]["content"]
        self.generation_calls.append({"messages": messages, "kwargs": kwargs})
        verbalize = prompt.startswith("Write a short natural-language paragraph")
        if verbalize:
            objects = json.loads(prompt[prompt.index("["):])
            steps = [IToken(item["rule"], tuple(item["premises"]), item["conclusion"]) for item in objects]
            text = "  " if self.empty_verbalization else render_proof(steps)
            truncated = self.verbalization_truncated
        else:
            payload = json.loads(prompt.split("The following JSON is the problem and source evidence:\n", 1)[1])
            lines = [line for line in payload["source_text"].splitlines() if line.strip()]
            steps = parse_proof(payload["source_text"])
            text = json.dumps({"status": "extracted", "steps": [
                {**json.loads(step.canonical()), "evidence": line}
                for step, line in zip(steps, lines)
            ], "reason": "unit stub"})
            truncated = self.extraction_truncated
        return {"text": text, "finish_reason": "length" if truncated else "stop",
                "generated_tokens": 1, "wall_seconds": 0.0, "model_invoked": False}


class StructuralDetectorTests(unittest.TestCase):
    def test_duplicate_structural_evidence_does_not_add_confidence(self):
        problem = make_problem(1, "unit-detector")
        steps = list(candidate_pairs(problem, 0)["mp"])
        key = b"unit-key"
        bank = surrogate_keys(17, 19)
        first = pilot.detect_structural(problem, steps, key, bank)
        duplicated = pilot.detect_structural(problem, steps * 5, key, bank)
        self.assertEqual(first, duplicated)
        self.assertEqual(first["n_choices"], 1)
        self.assertNotIn("log_likelihood_ratio", first)

    def test_conflicting_routes_abstain(self):
        problem = make_problem(1, "unit-detector")
        pairs = candidate_pairs(problem, 0)
        result = pilot.detect_structural(problem, list(pairs["mp"] + pairs["and"]), b"key", surrogate_keys(2, 19))
        self.assertEqual(result["status"], "ambiguous_path")
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["detected"])

    def test_partial_evidence_gets_only_structural_randomization(self):
        problem = make_problem(2, "unit-detector")
        steps = [candidate_pairs(problem, 0)["or"][0]]
        result = pilot.detect_structural(problem, steps, b"key", surrogate_keys(2, 19))
        self.assertEqual(result["status"], "partial_evidence")
        self.assertEqual(result["n_choices"], 1)
        self.assertNotIn("log_likelihood_ratio", result)


class PilotAccountingTests(unittest.TestCase):
    def run_pilot(self, model, destination, *, stages=2, seed=17):
        with patch("pipeline.base_route_probabilities", side_effect=AssertionError("Synthetic predictor used")), redirect_stdout(io.StringIO()):
            pilot.run(model, destination, stages=stages, seed=seed)
        return json.loads((destination / "pilot_summary.json").read_text())

    def test_planned_lr_uses_actual_conditional_candidate_masses(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pilot"
            model = FixedBackend()
            summaries = self.run_pilot(model, destination, stages=3)["traces"]
            self.assertEqual(len(model.score_calls), 6)
            self.assertIsNone(json.loads((destination / "pilot_summary.json").read_text())["power_estimate"])
            for summary in summaries:
                key = bytes.fromhex(summary["public_demonstration_key_hex"])
                problem = make_problem(3, "real-pilot-17")
                expected = 0.0
                for stage in range(3):
                    row = json.loads((destination / summary["mode"] / f"stage_{stage:02d}.json").read_text())
                    p = row["scores"]["itoken_probabilities"]
                    selected = IToken(**{**row["planned_steps"][0], "premises": tuple(row["planned_steps"][0]["premises"])}).canonical()
                    labels = {candidate: keyed_preferred(key, context(problem, stage), candidate) for candidate in p}
                    normalizer = sum(probability * math.exp(4 * labels[candidate]) for candidate, probability in p.items())
                    expected += 4 * labels[selected] - math.log(normalizer)
                self.assertAlmostEqual(summary["planned_policy_log_likelihood_ratio"], expected, places=12)
                self.assertTrue(summary["planned_proof_valid"])
                self.assertTrue(summary["recovered_proof_valid"])
                self.assertTrue(summary["exact_representation_agreement"])
                self.assertNotIn("log_likelihood_ratio", summary["recovered_detection"])
            key_hex = hashlib.sha256(b"PUBLIC-REAL-PILOT-KEY:17").hexdigest()
            for call in model.score_calls + model.generation_calls:
                serialized = json.dumps(call["messages"])
                self.assertNotIn(key_hex, serialized)
                self.assertNotIn("PUBLIC-REAL-PILOT-KEY", serialized)

    def test_existing_partial_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pilot"
            old = destination / "unmarked" / "stage_00.json"
            old.parent.mkdir(parents=True)
            old.write_text('{"sentinel":"preserve original raw output"}')
            model = FixedBackend()
            with self.assertRaises(ValueError):
                self.run_pilot(model, destination)
            self.assertEqual(model.score_calls, [])
            self.assertIn("preserve original", old.read_text())

    def test_empty_verbalization_is_retained_and_abstains(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pilot"
            model = FixedBackend(empty_verbalization=True)
            summaries = self.run_pilot(model, destination, stages=1)["traces"]
            self.assertEqual(len(model.generation_calls), 2)  # No extraction call on empty text.
            for summary in summaries:
                row = json.loads((destination / summary["mode"] / "stage_00.json").read_text())
                self.assertEqual(row["verbalization"]["text"], "  ")
                self.assertEqual(row["recovery"]["status"], "empty_verbalization")
                self.assertIsNone(row["extraction_prompt"])
                self.assertIsNone(row["extraction_generation"])
                self.assertEqual(summary["recovered_steps"], 0)
                self.assertIsNone(summary["recovered_detection"]["detected"])

    def test_truncated_extractor_does_not_recover_even_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pilot"
            summaries = self.run_pilot(FixedBackend(extraction_truncated=True), destination, stages=1)["traces"]
            for summary in summaries:
                row = json.loads((destination / summary["mode"] / "stage_00.json").read_text())
                self.assertEqual(row["recovery"]["status"], "truncated_generation")
                self.assertEqual(summary["recovered_steps"], 0)
                self.assertFalse(summary["recovered_proof_valid"])

    def test_manifest_is_recorded_and_verbalization_truncation_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pilot"
            summaries = self.run_pilot(FixedBackend(verbalization_truncated=True), destination, stages=1)["traces"]
            manifest = json.loads((destination / "pilot_manifest.json").read_text())
            self.assertTrue(manifest)
            self.assertIn("unit-manifest", json.dumps(manifest))
            for summary in summaries:
                self.assertEqual(summary["verbalization_truncations"], 1)

    def test_completed_calls_survive_later_infrastructure_failure(self):
        class FailsDuringExtraction(FixedBackend):
            def generate(self, messages, **kwargs):
                if not messages[0]["content"].startswith("Write a short natural-language paragraph"):
                    raise RuntimeError("unit extraction infrastructure failure")
                return super().generate(messages, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "pilot"
            with self.assertRaises(RuntimeError):
                self.run_pilot(FailsDuringExtraction(), destination, stages=1)
            stage_file = destination / "unmarked/stage_00.json"
            self.assertTrue(stage_file.exists(), "Finished scoring/verbalization must remain auditable")
            record = json.loads(stage_file.read_text())
            self.assertIn("scores", record)
            self.assertTrue(record["verbalization"]["text"])


if __name__ == "__main__":
    unittest.main()
