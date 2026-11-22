import unittest

from natural_lexical_baseline import recover, tokens


def fixture():
    return {'decisions': [{'id': 'proposal', 'options': [
        {'id': 'host', 'application': 'The host sends the volunteer a preferred slot; the volunteer confirms it with the host.'},
        {'id': 'volunteer', 'application': 'The volunteer sends the host a preferred slot; the host confirms it with the volunteer.'},
        {'id': 'overlap', 'application': 'Everyone shares availability and selects a common overlap.'},
    ]}]}


class NaturalLexicalTests(unittest.TestCase):
    def test_same_bag_role_reversal_is_indistinguishable_and_abstains(self):
        problem = fixture()
        first, second = [o['application'] for o in problem['decisions'][0]['options'][:2]]
        a, b = recover(problem, first), recover(problem, second)
        self.assertEqual(a, b)
        self.assertEqual(a['choices'], {})
        self.assertEqual(a['abstained_decisions'], ['proposal'])

    def test_distinct_words_can_select_without_semantic_judgment(self):
        value = recover(fixture(), 'Everyone shares availability and selects a common overlap.')
        self.assertEqual(value['choices'], {'proposal': 'overlap'})
        self.assertEqual(value['status'], 'complete')

    def test_unknown_or_empty_words_abstain(self):
        for text in ('', 'Zephyrs shimmering everywhere.'):
            self.assertEqual(recover(fixture(), text)['status'], 'abstain')

    def test_tokenization_retains_negation_but_ignores_order(self):
        self.assertIn('not', tokens('This is NOT adopted.'))
        self.assertIn('no', tokens('No one adopted it.'))
        self.assertEqual(tokens('The TEST-Case 42'), ['test', 'case'])

    def test_duplicates_and_invalid_option_count_rejected(self):
        p = fixture();p['decisions'].append(p['decisions'][0])
        with self.assertRaises(ValueError): recover(p, 'text')
        p = fixture();p['decisions'][0]['options'].pop()
        with self.assertRaises(ValueError): recover(p, 'text')


if __name__ == '__main__':
    unittest.main()
