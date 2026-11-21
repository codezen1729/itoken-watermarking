"""Numerical/token-boundary tests only; these do not count as real LLM runs."""
import math
import unittest

from real_model import aggregate_sequence_scores, continuation_tokens, selected_logprobs


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class MergingTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Boundary-changing toy tokenizer: AB is one token instead of A then B.
        return [999] if text == "AB" else [ord(c) for c in text]


class RealModelProbabilityTests(unittest.TestCase):
    def test_causal_indexing_scores_first_completion_and_eos(self):
        sequence = [0, 1, 2, 0]  # prefix two tokens; body token 2; EOS token 0
        logits = [[math.log(.1), math.log(.8), math.log(.1)],
                  [math.log(.2), math.log(.3), math.log(.5)],
                  [math.log(.7), math.log(.2), math.log(.1)]]
        scores = selected_logprobs(logits, sequence, 2)
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(math.exp(sum(scores)), .5 * .7)
        self.assertNotAlmostEqual(math.exp(sum(scores)), .8 * .5 * .7)

    def test_termination_makes_prefix_candidates_disjoint(self):
        tokenizer = CharacterTokenizer()
        p, shorter = continuation_tokens(tokenizer, "P:", "a", 0, {0})
        _, longer = continuation_tokens(tokenizer, "P:", "ab", 0, {0})
        self.assertEqual(p, [80, 58])
        self.assertEqual(shorter, [97, 0])
        self.assertEqual(longer, [97, 98, 0])
        self.assertNotEqual(shorter, longer[:len(shorter)])

    def test_boundary_merge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            continuation_tokens(MergingTokenizer(), "A", "B", 0, {0})

    def test_early_stop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stopping"):
            continuation_tokens(CharacterTokenizer(), "P:", "a\0b", 0, {0})

    def test_aggregation_sums_realizations_without_length_normalizing(self):
        records = [
            {"token_ids": [1, 0], "token_logprobs": [math.log(.2), math.log(.5)], "itoken_id": "A"},
            {"token_ids": [2, 3, 0], "token_logprobs": [math.log(.2), math.log(.5), math.log(.5)], "itoken_id": "A"},
            {"token_ids": [4, 0], "token_logprobs": [math.log(.1), math.log(.5)], "itoken_id": "B"},
        ]
        result = aggregate_sequence_scores(records)
        self.assertAlmostEqual(result["covered_probability_mass"], .2)
        self.assertAlmostEqual(result["itoken_probabilities"]["A"], .75)
        self.assertAlmostEqual(result["itoken_probabilities"]["B"], .25)
        self.assertEqual(result["support_status"], "not_certified_complete")

    def test_duplicate_token_sequences_are_rejected_even_with_other_labels(self):
        records = [{"token_ids": [1, 0], "token_logprobs": [-1, -1], "itoken_id": label}
                   for label in ["A", "B"]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_sequence_scores(records)

    def test_invalid_mass_or_unscored_eos_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds one"):
            aggregate_sequence_scores([{"token_ids": [t, 0], "token_logprobs": [0, 0], "itoken_id": str(t)}
                                       for t in [1, 2]])
        with self.assertRaisesRegex(ValueError, "Every token"):
            aggregate_sequence_scores([{"token_ids": [1, 0], "token_logprobs": [-1], "itoken_id": "A"}])

    def test_extreme_log_masses_keep_relative_probability(self):
        result = aggregate_sequence_scores([
            {"token_ids": [t, 0], "token_logprobs": [-1e300, 0], "itoken_id": str(t)}
            for t in [1, 2]])
        self.assertEqual(result["covered_probability_mass"], 0.0)
        self.assertEqual(result["itoken_probabilities"], {"1": .5, "2": .5})

    def test_invalid_indexing_is_rejected(self):
        for prefix in [0, 2, True]:
            with self.assertRaises(ValueError):
                selected_logprobs([[0, 0]], [0, 1], prefix)


if __name__ == "__main__":
    unittest.main()
