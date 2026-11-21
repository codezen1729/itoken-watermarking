"""Protocol-boundary tests; the scripted backend is not model evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from core import candidate_pairs, make_problem, render_step
from research_native_samples import _single_first_route, run
from semantic_recovery import RecoveryResult


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


class ScriptedBackend:
    metadata = {"test_backend": True, "model_invoked": False}
    eos_token_id = 100000
    stop_token_ids = {100000}
    tokenizer = CharacterTokenizer()

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def _prompt(self, messages):
        return "PROMPT\n", self.tokenizer.encode("PROMPT\n"), "unit-test-context"

    def generate(self, messages, **settings):
        self.calls.append(settings)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        text, reason = value
        return {"text": text, "finish_reason": reason,
                "token_ids": self.tokenizer.encode(text) + ([self.eos_token_id] if reason == "stop" else []),
                "model_invoked": False}


class NativeSampleBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_problem(1, "native-unit")
        self.pair = candidate_pairs(self.problem, 0)["mp"]
        self.text = render_step(self.pair[0])

    def test_only_one_first_step_receives_route(self):
        self.assertEqual(_single_first_route(self.problem, RecoveryResult(status="partial_evidence", steps=[self.pair[0]]))[0], "mp")
        self.assertIsNone(_single_first_route(self.problem, RecoveryResult(status="complete_proof", steps=list(self.pair)))[0])
        self.assertIsNone(_single_first_route(self.problem, RecoveryResult(status="partial_evidence", steps=[self.pair[1]]))[0])

    def test_empty_native_text_abstains_without_extraction(self):
        backend = ScriptedBackend([("  ", "stop")])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory) / "result"
            summary = run(backend, out, count=1)
            row = json.loads((out / "sample_00.json").read_text())
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(row["recovery"]["status"], "abstain")
        self.assertIsNone(row["extraction_generation"])
        self.assertEqual(summary["count_completed"], 1)

    def test_complete_json_is_rejected_if_extraction_did_not_stop(self):
        response = json.dumps({"status": "extracted", "steps": [{"id": "I000", "evidence": self.text}]})
        backend = ScriptedBackend([(self.text, "stop"), (response, "length")])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory) / "result"
            summary = run(backend, out, count=1)
            row = json.loads((out / "sample_00.json").read_text())
        self.assertEqual(row["recovery"]["status"], "truncated_generation")
        self.assertIsNone(row["single_first_step_route"])
        self.assertEqual(summary["extraction_truncations"], 1)
        self.assertTrue(row["exact_listed_terminated_sequence"])
        self.assertEqual(backend.calls[0], {"max_tokens": 128, "temperature": 1.0, "seed": 20261121})
        self.assertEqual(backend.calls[1], {"max_tokens": 1024, "temperature": 0.0, "seed": 0})

    def test_native_call_is_saved_before_failed_extraction(self):
        backend = ScriptedBackend([(self.text, "stop"), RuntimeError("scripted failure")])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "result"
            with self.assertRaisesRegex(RuntimeError, "scripted failure"):
                run(backend, out, count=1)
            row = json.loads((out / "sample_00.json").read_text())
            self.assertFalse((out / "native_summary.json").exists())
            with self.assertRaises(ValueError):
                run(backend, out, count=1)
        self.assertEqual(row["generation"]["text"], self.text)
        self.assertEqual(row["phase"], "extraction_failed_or_interrupted")
        self.assertEqual(row["exception_type"], "RuntimeError")

    def test_truncated_native_prefix_never_receives_primary_route(self):
        response = json.dumps({"status": "extracted", "steps": [{"id": "I000", "evidence": self.text}]})
        backend = ScriptedBackend([(self.text, "length"), (response, "stop")])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory) / "result"
            summary = run(backend, out, count=1)
            row = json.loads((out / "sample_00.json").read_text())
        self.assertTrue(row["recovery"]["accepted"])
        self.assertIsNone(row["single_first_step_route"])
        self.assertEqual(row["prefix_single_first_step_route"], "mp")
        self.assertEqual(row["route_recovery_status"], "native_generation_truncated")
        self.assertFalse(row["exact_listed_terminated_sequence"])
        self.assertEqual(summary["single_first_step_route_counts"], {"unknown_or_multiple": 1})


if __name__ == "__main__":
    unittest.main()
