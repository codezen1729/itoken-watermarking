"""Natural-language file-transport integrity checks using invented fixtures only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import natural_transport as transport
from natural_recovery import choice_bindings, decision_bindings
from research_runner import save
from test_natural_recovery import fixture


class NaturalTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'batch'
        self.case={'id':'PRIVATE-CASE','problem':fixture(),
                   'text':'Guests choose a time before staff check it. Let them try before showing how, so their difficulties guide the explanation.',
                   'expected_choices':{'access':'PRIVATE-GOLD'},'condition':'PRIVATE-CONDITION','key':'PRIVATE-KEY'}
        transport.prepare([self.case],self.out,label='PRIVATE-LABEL',batch_group='PRIVATE-GROUP')
        self.req=json.loads((self.out/'proposals/requests.json').read_text())['requests']
        self.rid=self.req[0]['request_id']
        self.ids={(v['decision_id'],v['option_id']):k for k,v in choice_bindings(self.case['problem']).items()}
        self.decisions={v:k for k,v in decision_bindings(self.case['problem']).items()}
        self.names=[self.ids[('access','visitor-first')],self.ids[('instruction','practice-first')]]

    def write_proposal(self,value):
        save(self.out/'proposals/responses.json',{'responses':[{'request_id':self.rid,'response':value}]})

    def proposed(self,ambiguous=None):
        return json.dumps({'status':'ambiguous' if ambiguous else 'extracted',
            'choices':[{'id':name,'sentences':[i+1]} for i,name in enumerate(self.names)],
            'ambiguous_decisions':ambiguous or []})

    def write_verification(self,records=None):
        verdicts=records if records is not None else [{'id':name,'verdict':'supported','sentences':[i+1]} for i,name in enumerate(self.names)]
        save(self.out/'verification/responses.json',{'responses':[{'request_id':self.rid,
            'response':json.dumps({'verdicts':verdicts,'ambiguous_decisions':[]})}]})

    def test_public_packet_excludes_private_case_fields(self):
        prompt=self.req[0]['prompt']
        for value in ('PRIVATE-CASE','PRIVATE-GOLD','PRIVATE-CONDITION','PRIVATE-KEY','PRIVATE-LABEL','PRIVATE-GROUP'):
            self.assertNotIn(value,prompt)
        with self.assertRaises(ValueError):transport.prepare([self.case],self.out,label='x',batch_group='x')

    def test_every_assigned_response_is_required_exactly_once(self):
        for responses in ([],[{'request_id':'unknown','response':'{}'}],
                          [{'request_id':self.rid,'response':'{}'}]*2):
            save(self.out/'proposals/responses.json',{'responses':responses})
            with self.assertRaises(ValueError):transport.prepare_verification(self.out)

    def test_verified_result_replays_exactly_and_cannot_be_refinalized(self):
        self.write_proposal(self.proposed())
        self.assertEqual(transport.prepare_verification(self.out),1)
        self.write_verification()
        rows=transport.finalize(self.out)
        self.assertTrue(rows[0]['recovery']['complete'])
        self.assertEqual(rows[0]['recovery']['choices'],{'access':'visitor-first','instruction':'practice-first'})
        self.assertFalse(rows[0]['recovery']['semantic_entailment_verified'])
        self.assertEqual(transport.replay(self.out),json.loads((self.out/'results.json').read_text()))
        with self.assertRaises(ValueError):transport.finalize(self.out)
        with self.assertRaises(ValueError):transport.prepare_verification(self.out)

    def test_ambiguous_proposal_still_verifies_unrelated_choices(self):
        self.write_proposal(self.proposed([self.decisions['access']]))
        self.assertEqual(transport.prepare_verification(self.out),1)
        self.write_verification()
        row=transport.finalize(self.out)[0]
        self.assertEqual(row['recovery']['choices'],{'instruction':'practice-first'})
        self.assertEqual(row['recovery']['ambiguous_decisions'],['access'])
        self.assertEqual(row['recovery']['status'],'partial_evidence')

    def test_abstention_and_invalid_first_answers_remain_in_results(self):
        for index,answer in enumerate(('{"status":"abstain","choices":[],"ambiguous_decisions":[]}','broken JSON')):
            out=Path(self.tmp.name)/f'early-{index}'
            transport.prepare([self.case],out,label='test',batch_group='test')
            rid=json.loads((out/'proposals/requests.json').read_text())['requests'][0]['request_id']
            save(out/'proposals/responses.json',{'responses':[{'request_id':rid,'response':answer}]})
            self.assertEqual(transport.prepare_verification(out),0)
            row=transport.finalize(out)[0]
            self.assertEqual(row['proposal_response'],answer)
            self.assertFalse(row['recovery']['accepted'])
            self.assertEqual(row['recovery']['status'],'abstained' if index==0 else 'invalid_model_output')

    def test_case_binding_swap_is_detected_even_when_case_and_request_sets_match(self):
        out=Path(self.tmp.name)/'two-cases'
        other=copy.deepcopy(self.case);other['id']='SECOND';other['text']='Staff assign a time before guests review it.'
        transport.prepare([self.case,other],out,label='test',batch_group='test')
        path=out/'private_manifest.json';value=json.loads(path.read_text())
        rows=value['bindings'];rows[0]['request_id'],rows[1]['request_id']=rows[1]['request_id'],rows[0]['request_id']
        save(path,value)
        with self.assertRaises(ValueError):transport.load(out)

    def test_case_and_prompt_and_source_drift_are_detected(self):
        with patch.object(transport,'digest',return_value='changed'):
            with self.assertRaises(ValueError):transport.load(self.out)
        path=self.out/'proposals/requests.json';saved=path.read_text();obj=json.loads(saved)
        obj['requests'][0]['prompt']+=' Changed prompt.';save(path,obj)
        with self.assertRaises(ValueError):transport.load(self.out)
        path.write_text(saved)
        path=self.out/'private_manifest.json';obj=json.loads(path.read_text());obj['bindings'][0]['case']['text']+=' Changed source.';save(path,obj)
        with self.assertRaises(ValueError):transport.load(self.out)

    def test_proposer_cannot_change_after_verifier_export(self):
        self.write_proposal(self.proposed());transport.prepare_verification(self.out)
        self.write_proposal('{"status":"abstain","choices":[],"ambiguous_decisions":[]}')
        with self.assertRaises(ValueError):transport.replay(self.out)

    def test_verifier_packet_and_response_identity_are_checked(self):
        self.write_proposal(self.proposed());transport.prepare_verification(self.out)
        save(self.out/'verification/responses.json',{'responses':[{'request_id':'unknown','response':'{}'}]})
        with self.assertRaises(ValueError):transport.replay(self.out)
        self.write_verification()
        path=self.out/'verification/requests.json';obj=json.loads(path.read_text());obj['requests'][0]['prompt']+=' drift';save(path,obj)
        with self.assertRaises(ValueError):transport.replay(self.out)

    def test_missing_verifier_is_not_converted_into_a_success(self):
        self.write_proposal(self.proposed());transport.prepare_verification(self.out)
        with self.assertRaises(FileNotFoundError):transport.finalize(self.out)
        self.assertFalse((self.out/'results.json').exists())
        self.write_verification([{'id':self.names[0],'verdict':'supported','sentences':[1]}])
        row=transport.finalize(self.out)[0]
        self.assertFalse(row['recovery']['accepted'])
        self.assertEqual(row['recovery']['status'],'invalid_model_output')


if __name__=='__main__':unittest.main()
