"""Planner bookkeeping and null-isolation checks; no trained model is invoked."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from core import candidate_pairs, make_problem, parse_step, valid_inference, validate_proof
import local_state_planner as planner
from pipeline import context, draw
from watermark import bias_distribution


class RecordedProbabilityStub:
    metadata = {"checkpoint": {"manifest_sha256": "unit-checkpoint"}, "revision": "unit-revision"}

    def __init__(self, route_mass=None, difference=.00001):
        self.calls = []
        self.route_mass = route_mass or {"mp": .01, "and": .01, "or": .98}
        self.difference = difference

    def score_candidates(self, messages, candidates, **kwargs):
        self.calls.append((messages, candidates, kwargs))
        masses = {c["itoken_id"]: self.route_mass[c["route"]] for c in candidates}
        return {"itoken_probabilities": masses, "covered_probability_mass": .125,
                "uncached_crosschecks": [{"max_absolute_token_logprob_difference": self.difference}],
                "candidates": [{"text": c["text"], "itoken_id": c["itoken_id"]} for c in candidates]}


def deterministic_entropy(label):
    counter = 0
    def draw_bytes(n):
        nonlocal counter
        if n != 32:
            raise AssertionError("Expected 256-bit key draw")
        result = hashlib.sha256(f"{label}:{counter}".encode()).digest()
        counter += 1
        return result
    return draw_bytes


class LocalStatePlannerTests(unittest.TestCase):
    def make_bank(self, root, *, stages=2, model=None):
        model = model or RecordedProbabilityStub()
        bank = root / "bank"
        with redirect_stdout(io.StringIO()):
            planner.build_bank(model, bank, stages=stages)
        return bank, model

    def sample(self, bank, out, *, label="keys-a", units=2, stages=2, seed=4):
        with (patch.object(planner.secrets, "token_bytes", side_effect=deterministic_entropy(label)) as key_draw,
              patch.object(planner.secrets, "token_hex", side_effect=[f"opaque-{i}" for i in range(2*units)])):
            rows = planner.sample_cohort(bank, out, units=units, stages=stages, delta=6, seed=seed)
        return rows, key_draw.call_count

    def test_context_has_only_current_local_state_and_valid_candidate_labels(self):
        candidates, messages = planner.request(7)
        text = messages[0]["content"]
        state = json.loads(text.split("LOCAL STATE (the entire model context):\n")[1])
        self.assertEqual(state["current_fact"], "p7")
        self.assertEqual(state["next_goal"], "p8")
        self.assertEqual(len(state["local_given_premises"]), 6)
        for forbidden in ("p0", "p6", "m6", "problem_id", "steps_already_taken", "watermark", "secret"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(len(candidates), 9)
        self.assertEqual(len({c["itoken_id"] for c in candidates}), 3)
        for candidate in candidates:
            parsed = parse_step(candidate["text"])
            self.assertEqual(parsed.canonical(), candidate["itoken_id"])
            self.assertTrue(valid_inference(parsed))

    def test_bank_reuses_frozen_calls_and_records_all_behavioral_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank, model = self.make_bank(root)
            with redirect_stdout(io.StringIO()):
                planner.build_bank(model, bank, stages=2)
            self.assertEqual(len(model.calls), 2)
            self.assertTrue(model.calls[0][2]["verify_all_uncached"])
            self.assertFalse(model.calls[1][2]["verify_all_uncached"])
            manifest = json.loads((bank / "bank_manifest.json").read_text())
            self.assertEqual(set(manifest["source_sha256"]), set(planner.DEPENDENCIES))
            self.assertTrue({"pipeline.py", "watermark.py", "real_model_pilot.py", "research_runner.py"}
                            <= set(manifest["source_sha256"]))
            for name, sha in manifest["source_sha256"].items():
                self.assertEqual(sha, hashlib.sha256((Path(planner.__file__).parent/name).read_bytes()).hexdigest())
            for stage, (messages, candidates, kwargs) in enumerate(model.calls):
                self.assertEqual((candidates, messages), planner.request(stage))
                self.assertTrue(kwargs["verify_first_uncached"])

    def test_changed_model_metadata_invalidates_bank_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank, model = self.make_bank(Path(tmp))
            model.metadata = {**model.metadata, "revision": "changed-revision"}
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                planner.build_bank(model, bank, stages=2)
            self.assertEqual(len(model.calls), 2)

    def test_actual_recorded_probabilities_drive_both_policies_and_fixed_alternative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank, _ = self.make_bank(root)
            with patch("pipeline.base_route_probabilities", side_effect=AssertionError("Synthetic predictor used")):
                rows, draws = self.sample(bank, root / "cohort")
            self.assertEqual(draws, 2 * 200)
            for row in rows:
                key = bytes.fromhex(row["public_demonstration_key_hex"])
                problem = make_problem(2, row["problem_id"])
                rng = random.Random(row["sampling_seed"])
                total = 0
                for stage, policy in enumerate(row["policy_rows"]):
                    p = policy["p"]
                    expected_p = {pair[0].canonical(): {"mp": .01, "and": .01, "or": .98}[route]
                                  for route, pair in candidate_pairs(problem, stage).items()}
                    self.assertEqual(p, expected_p)
                    q = bias_distribution(p, key, context(problem, stage), delta=row["delta"])
                    alt = bias_distribution(p, key, context(problem, stage), delta=6)
                    self.assertEqual(policy["q"], q)
                    self.assertEqual(policy["selected"], draw(q, rng))
                    total += math.log(alt[policy["selected"]] / p[policy["selected"]])
                self.assertAlmostEqual(total, row["planned_log_likelihood_ratio"])
                if row["condition"] == "unmarked":
                    self.assertEqual(row["routes"], ["or", "or"])
                    self.assertEqual(row["delta"], 0)
                else:
                    self.assertEqual(row["delta"], 6)
                from core import IToken
                steps = [IToken(**{**json.loads(s), "premises": tuple(json.loads(s)["premises"])})
                         for s in row["planned_canonical"]]
                self.assertTrue(validate_proof(problem, steps))

    def test_null_routes_ignore_keys_and_pairs_share_only_uniforms_and_surrogate_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank, _ = self.make_bank(root)
            first, _ = self.sample(bank, root / "a", label="keys-a")
            second, _ = self.sample(bank, root / "b", label="keys-b")
            null_a = [r for r in first if r["condition"] == "unmarked"]
            null_b = [r for r in second if r["condition"] == "unmarked"]
            self.assertEqual([r["routes"] for r in null_a], [r["routes"] for r in null_b])
            self.assertNotEqual([r["public_demonstration_key_hex"] for r in null_a],
                                [r["public_demonstration_key_hex"] for r in null_b])
            self.assertEqual(len({r["public_demonstration_key_hex"] for r in null_a}), 2)
            for unit in (0, 1):
                a, b = [r for r in first if r["unit_id"] == unit]
                self.assertEqual(a["sampling_seed"], b["sampling_seed"])
                self.assertEqual(a["surrogate_keys_hex"], b["surrogate_keys_hex"])
                self.assertEqual(len(a["surrogate_keys_hex"]), 199)
                self.assertNotIn(a["public_demonstration_key_hex"], a["surrogate_keys_hex"])

    def test_nonempty_cohort_refuses_reroll_before_key_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank, _ = self.make_bank(root)
            self.sample(bank, root / "cohort")
            with patch.object(planner.secrets, "token_bytes") as draws, self.assertRaises(ValueError):
                planner.sample_cohort(bank, root / "cohort", units=2, stages=2)
            draws.assert_not_called()

    def test_bank_tampering_fails_before_any_keys_are_drawn(self):
        edits = {
            "wrong_stage": lambda b: b["stages"][0].update(stage=1),
            "bad_probability": lambda b: b["stages"][0]["itoken_probabilities"].update(
                {next(iter(b["stages"][0]["itoken_probabilities"])): .9}),
            "bad_numeric": lambda b: b["stages"][0].update(max_checked_logprob_difference=.1),
            "summary_mass": lambda b: b["stages"][0].update(covered_probability_mass=.9),
            "manifest": lambda b: b["freeze"].update(protocol="different"),
        }
        for label, edit in edits.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bank, _ = self.make_bank(root)
                target = bank / "bank_summary.json"
                data = json.loads(target.read_text())
                edit(data)
                target.write_text(json.dumps(data))
                with patch.object(planner.secrets, "token_bytes") as draws, self.assertRaises(ValueError):
                    planner.sample_cohort(bank, root / "cohort", units=2, stages=2)
                draws.assert_not_called()

    def test_invalid_settings_fail_before_files_or_randomness(self):
        for kwargs in ({"units": 0}, {"units": True}, {"stages": 0}, {"delta": -1}, {"delta": math.nan}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(planner.secrets, "token_bytes") as draws, self.assertRaises(ValueError):
                    planner.sample_cohort(root / "missing-bank", root / "cohort", **kwargs)
                draws.assert_not_called()
                self.assertFalse((root / "cohort").exists())


if __name__ == "__main__":
    unittest.main()
