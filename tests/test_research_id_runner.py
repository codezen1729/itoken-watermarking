"""Inference-ID evaluation isolation/accounting tests; no real model is called."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import research_id_runner as runner
from core import candidate_pairs, make_problem, render_step
from semantic_recovery_ids import PROMPT_VERSION, inference_ids
from tests.test_research_runner import RecordingGenerator, case


class IDRunnerTests(unittest.TestCase):
    def make_dataset(self, root):
        dev, test = case(ident="unit-dev", split="dev"), case(ident="unit-test", split="test")
        dev["text"], test["text"] = "DEV_INPUT_ONLY_918", "TEST_INPUT_ONLY_918"
        (root / "research_eval_cases.json").write_text(json.dumps({"cases": [dev, test]}))
        return dev, test

    def create_sources(self, root):
        for name in ("semantic_recovery_ids.py", "semantic_recovery.py", "core.py", "real_model.py", "research_runner.py"):
            (root / name).write_text("# unit source, not model execution\n")

    def run_dev(self, root, model):
        with patch.object(runner, "ROOT", root), patch.object(runner, "freeze_record", return_value={"unit": True}), redirect_stdout(io.StringIO()):
            runner.run_recovery(model, root / "out", "dev", max_tokens=64)
        return json.loads((root / "out/recovery/unit-dev.json").read_text())

    def test_only_selected_split_source_reaches_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            model = RecordingGenerator()
            stored = self.run_dev(root, model)
            self.assertEqual(len(model.calls), 1)
            prompt = model.calls[0]["messages"][0]["content"]
            self.assertIn("DEV_INPUT_ONLY_918", prompt)
            self.assertIn("PUBLIC INFERENCE LIST", prompt)
            for forbidden in ("TEST_INPUT_ONLY_918", "GOLD_PRIVATE_MARKER_918", "GOLD_PRIVATE_REASON_918", "expected_canonical", "source_routes"):
                self.assertNotIn(forbidden, prompt)
            self.assertEqual(stored["generation"]["text"], model.output)
            self.assertEqual(stored["recovery"]["prompt_version"], PROMPT_VERSION)
            self.assertFalse((root / "out/recovery/unit-test.json").exists())

    def test_resume_reuses_raw_model_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            model = RecordingGenerator()
            self.run_dev(root, model)
            self.run_dev(root, model)
            self.assertEqual(len(model.calls), 1)

    def test_test_requires_dev_and_bad_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            model = RecordingGenerator()
            with patch.object(runner, "ROOT", root), patch.object(runner, "freeze_record", return_value={"unit": True}):
                for split in ("test", "other"):
                    with self.assertRaises(ValueError):
                        runner.run_recovery(model, root / "out", split)
            self.assertEqual(model.calls, [])

    def test_freeze_tracks_validator_and_metrics_dependencies(self):
        for filename in ("semantic_recovery.py", "research_runner.py", "semantic_recovery_ids.py"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.create_sources(root)
                self.make_dataset(root)
                with patch.object(runner, "ROOT", root):
                    frozen = runner.freeze_record(root / "out", RecordingGenerator(), root / "research_eval_cases.json", 64)
                    self.assertEqual(frozen["extractor_protocol"], "inference_ids")
                    self.assertIn("metrics_runner_sha256", frozen)
                    self.assertIn("validator_sha256", frozen)
                    (root / filename).write_text("# altered dependency\n")
                    with self.assertRaises(ValueError):
                        runner.freeze_record(root / "out", RecordingGenerator(), root / "research_eval_cases.json", 64)

    def test_truncated_valid_ids_are_rejected_and_keep_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = case(ident="unit-dev")
            problem = make_problem(1, gold["problem_id"])
            steps = candidate_pairs(problem, 0)["mp"]
            phrases = [render_step(step) for step in steps]
            identifiers = {step: name for name, step in inference_ids(problem).items()}
            gold["text"] = "\n".join(phrases)
            (root / "research_eval_cases.json").write_text(json.dumps({"cases": [gold]}))
            output = json.dumps({"status": "extracted", "steps": [
                {"id": identifiers[step], "evidence": text} for step, text in zip(steps, phrases)]})
            model = RecordingGenerator(output, finish_reason="length")
            stored = self.run_dev(root, model)
            self.assertEqual(stored["generation"]["text"], output)
            self.assertEqual(stored["recovery"]["status"], "truncated_generation")
            self.assertEqual(stored["recovery"]["prompt_version"], PROMPT_VERSION)
            self.assertFalse(stored["recovery"]["accepted"])
            self.assertFalse(stored["metrics"]["strict_success"])

    def test_unknown_id_is_not_repaired_into_a_valid_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            model = RecordingGenerator('{"status":"extracted","steps":[{"id":"I999","evidence":"DEV_INPUT_ONLY_918"}]}')
            stored = self.run_dev(root, model)
            self.assertEqual(stored["recovery"]["status"], "invalid_model_output")
            self.assertEqual(stored["generation"]["text"], model.output)
            self.assertEqual(stored["metrics"]["predicted_steps"], 0)


if __name__ == "__main__":
    unittest.main()
