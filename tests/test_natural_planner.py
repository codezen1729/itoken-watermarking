"""Boundary checks for natural candidate policy; no model accuracy fixtures."""
import copy
import json
import unittest

from natural_planner import canonical,context,request,detect,validate_catalog
from watermark import bias_distribution

class NaturalPlannerTests(unittest.TestCase):
    def setUp(self):
        self.problem={'id':'internal_problem','title':'Shared bench','problem':'A group shares a workbench.',
          'goal':'Agree who confirms a booking.','decisions':[{'id':'internal_decision','question':'Who confirms a booking?',
          'context':'A visitor and a host can both propose times.', 'options':[
          {'id':'internal_a','mechanism':'The visitor proposes a time and the host confirms it.','consequence':'Both agree on the booking.','application':'The visitor proposes a time and the host confirms it, giving both an agreed booking.'},
          {'id':'internal_b','mechanism':'The host proposes a time and the visitor confirms it.','consequence':'Both agree on the booking.','application':'The host proposes a time and the visitor confirms it, giving both an agreed booking.'},
          {'id':'internal_c','mechanism':'Both independently list times and confirm the first match.','consequence':'Both agree on the booking.','application':'Both independently list times and confirm the first match to agree on the booking.'}]}]}

    def test_canonical_identity_tracks_meaning_fields(self):
        d=self.problem['decisions'][0]
        self.assertNotEqual(canonical(d,d['options'][0]),canonical(d,d['options'][1]))
        altered=copy.deepcopy(d['options'][0]);altered['id']='renamed-internal-id'
        self.assertEqual(canonical(d,d['options'][0]),canonical(d,altered))

    def test_scoring_prompt_has_no_internal_option_ids(self):
        d=self.problem['decisions'][0];c,m=request(self.problem,d)
        self.assertEqual(len(c),3)
        self.assertNotIn('internal_',m[0]['content'])
        self.assertEqual(request(self.problem,d),(c,m))

    def test_catalog_rejects_duplicate_applications(self):
        validate_catalog({'problems':[self.problem]})
        p=copy.deepcopy(self.problem);p['decisions'][0]['options'][1]['application']=p['decisions'][0]['options'][0]['application']
        with self.assertRaises(ValueError):validate_catalog({'problems':[p]})

    def test_zero_bias_does_not_depend_on_target_key(self):
        p={'a':.2,'b':.3,'c':.5}
        self.assertEqual(bias_distribution(p,b'one','public-context',0),p)
        self.assertEqual(bias_distribution(p,b'two','public-context',0),p)

    def test_detection_cannot_score_unknown_or_empty_choices(self):
        p=self.problem;key=b'target';bank=[b'one',b'two']
        self.assertIsNone(detect(p,{},key,bank)['p_value'])
        with self.assertRaises(ValueError):detect(p,{'unknown':'internal_a'},key,bank)
        with self.assertRaises(ValueError):detect(p,{'internal_decision':'unknown'},key,bank)
        r=detect(p,{'internal_decision':'internal_a'},key,bank)
        self.assertEqual(r['n_choices'],1)

    def test_context_changes_when_public_problem_changes(self):
        p=copy.deepcopy(self.problem);p['goal']='Another goal'
        self.assertNotEqual(context(p,p['decisions'][0]),context(self.problem,self.problem['decisions'][0]))

if __name__=='__main__':unittest.main()
