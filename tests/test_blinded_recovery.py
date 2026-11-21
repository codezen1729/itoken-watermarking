"""Boundary tests for file transport; fixtures do not measure LLM accuracy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import blinded_recovery as b
from core import make_problem, candidate_pairs
from recovery_v2 import inference_ids
from research_runner import save

class BlindedTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)/'batch'
        self.case = {'id':'private-id','n_stages':1,'problem_id':'public-problem',
                     'text':'Using p0, we obtain m0. This m0 establishes p1.'}
        b.prepare([self.case], self.out, label='private-label', batch_group='private-group')
        self.requests=json.loads((self.out/'proposals/requests.json').read_text())['requests']
        self.rid=self.requests[0]['request_id']

    def proposal(self, response):
        save(self.out/'proposals/responses.json', {'responses':[{'request_id':self.rid,'response':response}]})

    def test_blind_packet_excludes_private_metadata(self):
        text=(self.out/'proposals/requests.json').read_text()
        for value in ('private-id','private-label','private-group'):
            self.assertNotIn(value,text)
        self.assertIn('public_problem',text)
        self.assertIn('possible_inferences_not_observed_steps',text)
        with self.assertRaises(ValueError):
            b.prepare([self.case],self.out,label='x',batch_group='x')

    def test_missing_duplicate_and_unknown_responses_fail(self):
        for records in ([], [{'request_id':'unknown','response':'{}'}],
                        [{'request_id':self.rid,'response':'{}'}]*2):
            save(self.out/'proposals/responses.json',{'responses':records})
            with self.assertRaises(ValueError): b.prepare_verification(self.out)

    def test_early_abstention_retained_and_no_reroll(self):
        self.proposal('{"status":"abstain","steps":[]}')
        self.assertEqual(b.prepare_verification(self.out),0)
        with self.assertRaises(ValueError): b.prepare_verification(self.out)
        rows=b.finalize(self.out)
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['recovery']['accepted'])
        with self.assertRaises(ValueError): b.finalize(self.out)

    def test_invalid_proposal_retained(self):
        self.proposal('not JSON')
        self.assertEqual(b.prepare_verification(self.out),0)
        self.assertEqual(b.finalize(self.out)[0]['recovery']['status'],'invalid_model_output')

    def test_source_and_prompt_drift_detected(self):
        with patch.object(b,'digest',return_value='changed'):
            with self.assertRaises(ValueError): b._load(self.out)
        path=self.out/'proposals/requests.json'
        value=json.loads(path.read_text());value['requests'][0]['prompt']+=' drift'
        save(path,value)
        with self.assertRaises(ValueError): b._load(self.out)

    def test_response_drift_after_verifier_packet_detected(self):
        self.proposal('{"status":"abstain","steps":[]}')
        b.prepare_verification(self.out)
        self.proposal('{"status":"ambiguous","steps":[]}')
        with self.assertRaises(ValueError): b.finalize(self.out)

    def test_verified_pair_is_the_only_admitted_evidence(self):
        problem=make_problem(1,'public-problem')
        ids=inference_ids(problem)
        names=[next(k for k,v in ids.items() if v==s) for s in candidate_pairs(problem,0)['mp']]
        self.proposal(json.dumps({'status':'extracted','steps':[{'id':n,'sentences':[i+1]} for i,n in enumerate(names)]}))
        self.assertEqual(b.prepare_verification(self.out),1)
        response=json.dumps({'path_status':'unique','verdicts':[{'id':n,'verdict':'supported','sentences':[i+1]} for i,n in enumerate(names)]})
        save(self.out/'verification/responses.json',{'responses':[{'request_id':self.rid,'response':response}]})
        row=b.finalize(self.out)[0]
        self.assertTrue(row['recovery']['complete'])
        self.assertFalse(row['recovery']['semantic_entailment_verified'])
        self.assertNotIn('finish_reason',row)

if __name__=='__main__': unittest.main()
