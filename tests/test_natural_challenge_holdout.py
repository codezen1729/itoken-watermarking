"""Frozen diagnostic-data checks, not tests of LLM semantic competence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHA256 = '0439100321c8105f8dcbcb121689d7b77e13daf9f8bf994b0d508274c9bcdd82'


class NaturalChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'natural_challenge_holdout.json'
        cls.data = json.loads(cls.path.read_text())

    def test_frozen_hash_counts_and_admitted_gold(self):
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), SHA256)
        cases = self.data['cases']
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c['id'] for c in cases}), 24)
        self.assertEqual(Counter(c['expected_status'] for c in cases),
                         {'complete': 4, 'partial': 12, 'abstain': 6, 'ambiguous': 2})
        vocabulary = {d['id']: {o['id'] for o in d['options']}
                      for d in self.data['public_problem']['decisions']}
        for c in cases:
            for decision, option in c['expected_choices'].items():
                self.assertIn(option, vocabulary[decision])
            self.assertLessEqual(set(c['expected_ambiguous_decisions']), set(vocabulary))
            self.assertFalse(set(c['expected_choices']) & set(c['expected_ambiguous_decisions']))
            if c['expected_status'] == 'complete':
                self.assertEqual(set(c['expected_choices']), set(vocabulary))
            elif c['expected_status'] == 'partial':
                self.assertTrue(0 < len(c['expected_choices']) < len(vocabulary))
            else:
                self.assertFalse(c['expected_choices'])
            self.assertEqual(bool(c['expected_ambiguous_decisions']), c['expected_status'] == 'ambiguous')
        self.assertEqual(sum(len(c['expected_choices']) for c in cases), 28)

    def test_counterfactual_pairs_have_identical_word_multisets_and_different_gold(self):
        by_id = {c['id']: c for c in self.data['cases']}
        pairs = self.data['same_word_multiset_pairs']
        self.assertEqual(len(pairs), 4)
        for a, b in pairs:
            self.assertEqual(Counter(re.findall(r'[a-z]+', by_id[a]['text'].lower())),
                             Counter(re.findall(r'[a-z]+', by_id[b]['text'].lower())))
            self.assertNotEqual(by_id[a]['expected_choices'], by_id[b]['expected_choices'])

    def test_no_artificial_atom_or_catalog_ids_in_source_text(self):
        for c in self.data['cases']:
            self.assertIsNone(re.search(r'\b[pmed]\d+\b|\bI\d{3}\b', c['text']))
            for d in self.data['public_problem']['decisions']:
                self.assertNotIn(d['id'], c['text'])
                for option in d['options']:
                    self.assertNotIn(option['id'], c['text'])


if __name__ == '__main__':
    unittest.main()
