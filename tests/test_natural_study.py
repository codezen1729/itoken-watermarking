"""Boundary and context-separation tests for the natural-language study.

Invented fixtures are used throughout; no model judgments are simulated as
accuracy measurements.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import natural_failures as failures
import natural_study as study
from research_runner import save,digest
from test_natural_recovery import fixture


class NaturalFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'batch'
        self.case={'id':'normal','problem':fixture(),'text':'The text offers no plan.'}

    def _abstain(self):
        requests=json.loads((self.out/'proposals/requests.json').read_text())['requests']
        save(self.out/'proposals/responses.json',{'responses':[{'request_id':r['request_id'],
            'response':'{"status":"abstain","choices":[],"ambiguous_decisions":[]}'} for r in requests]})
        failures.prepare_verification(self.out)

    def test_nonempty_delegation_preserves_transport_result(self):
        failures.prepare_recovery([self.case],self.out,label='test',batch_group='test')
        self._abstain()
        expected=failures.transport.replay(self.out)
        self.assertEqual(failures.replay_recovery(self.out),expected)
        self.assertEqual(failures.finalize_recovery(self.out),expected['rows'])
        with self.assertRaises(ValueError):failures.finalize_recovery(self.out)

    def test_mixed_batch_keeps_raw_blank_in_original_assignment_order(self):
        blank={**self.case,'id':'blank','text':' \n\t '}
        failures.prepare_recovery([blank,self.case],self.out,label='test',batch_group='test')
        requests=json.loads((self.out/'proposals/requests.json').read_text())['requests']
        self.assertEqual(len(requests),1)
        self._abstain();rows=failures.finalize_recovery(self.out)
        self.assertEqual([r['id'] for r in rows],['blank','normal'])
        self.assertEqual(rows[0]['text'],' \n\t ')
        self.assertEqual(rows[0]['recovery']['status'],'empty_source')
        self.assertFalse(rows[0]['recovery']['accepted'])
        self.assertIsNone(rows[0]['proposal_response'])
        self.assertEqual(failures.replay_recovery(self.out),json.loads((self.out/'results.json').read_text()))

    def test_all_empty_finalizes_without_responses_or_model_calls(self):
        cases=[{**self.case,'id':str(i),'text':text} for i,text in enumerate(('', ' \n'))]
        with patch.object(failures.transport,'prepare',side_effect=AssertionError('no model transport')):
            failures.prepare_recovery(cases,self.out,label='empty',batch_group='empty')
        self.assertEqual(failures.prepare_verification(self.out),0)
        with self.assertRaises(ValueError):failures.prepare_verification(self.out)
        self.assertFalse((self.out/'proposals/responses.json').exists())
        self.assertEqual(len(failures.finalize_recovery(self.out)),2)

    def test_blank_assignment_cannot_be_changed_to_hide_a_failure(self):
        blank={**self.case,'id':'blank','text':''}
        failures.prepare_recovery([self.case,blank],self.out,label='x',batch_group='x')
        path=self.out/'assignment_manifest.json';value=json.loads(path.read_text())
        value['assignment']['cases'].pop();save(path,value)
        with self.assertRaises(ValueError):failures.replay_recovery(self.out)

    def test_wrong_text_type_is_not_reinterpreted_as_empty(self):
        with self.assertRaises(ValueError):
            failures.prepare_recovery([{**self.case,'text':None}],self.out,label='x',batch_group='x')


class NaturalStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.catalog=self.base/'catalog.json';self.cohort=self.base/'cohort';self.jobs=self.base/'jobs'
        self.problem=fixture();save(self.catalog,{'problems':[self.problem]})
        documents=[]
        for condition in ('unmarked','marked'):
            for unit in range(8):
                ident=f'{condition}-{unit}'
                plan={'document_id':ident,'problem_id':self.problem['id'],'condition':condition,
                      'choices':{d['id']:d['options'][0]['id'] for d in self.problem['decisions']}}
                save(self.cohort/'plans'/f'{ident}.json',plan)
                documents.append({'document_id':ident,'condition':condition,'problem_id':self.problem['id']})
        save(self.cohort/'index.json',{'config':{'catalog_sha256':digest(self.catalog)},'documents':documents})
        study.prepare_authors(self.catalog,self.cohort,self.jobs)

    def test_author_packets_separate_conditions_and_omit_internal_choice_ids(self):
        for n in range(1,5):
            packet=json.loads((self.jobs/f'author_{n:02d}'/'requests.json').read_text())
            names=[d['document_id'] for d in packet['documents']]
            self.assertEqual(len({name.split('-')[0] for name in names}),1)
            for row in packet['documents']:
                self.assertEqual(set(row),{'document_id','problem','goal','selected_applications'})
                for application in row['selected_applications']:
                    self.assertEqual(set(application),{'question','context','application'})
        with self.assertRaises(ValueError):study.prepare_authors(self.catalog,self.cohort,self.jobs)

    def test_response_bundle_cannot_omit_duplicate_or_substitute_documents(self):
        p=self.base/'response.json'
        for records in ([],[{'document_id':'x'}],[{'document_id':'a'},{'document_id':'a'}]):
            save(p,{'documents':records})
            with self.assertRaises(ValueError):study.complete_map(p,'documents',['a','b'])

    def test_rewrite_and_fidelity_inputs_do_not_receive_intended_selections(self):
        packet=json.loads((self.jobs/'author_01/requests.json').read_text())
        originals=[{'document_id':d['document_id'],'text':'Guests select a time, which staff then confirm.'} for d in packet['documents']]
        save(self.jobs/'author_01/responses.json',{'documents':originals})
        study.import_authors(self.catalog,self.cohort,self.jobs)
        rewrite=json.loads((self.jobs/'rewrite_01/requests.json').read_text())
        self.assertEqual(rewrite['documents'],originals)
        rewritten=[{**d,'text':'Staff confirm the time that guests have selected.'} for d in originals]
        save(self.jobs/'rewrite_01/responses.json',{'documents':rewritten})
        study.import_rewrites(self.catalog,self.cohort,self.jobs)
        source_check=json.loads((self.jobs/'authorship_01/requests.json').read_text())
        rewrite_check=json.loads((self.jobs/'fidelity_01/requests.json').read_text())
        for row in source_check['documents']:self.assertIn('intended_applications',row)
        for row in rewrite_check['documents']:
            self.assertEqual(set(row),{'document_id','decision_topics','original','rewrite'})
            for topic in row['decision_topics']:self.assertEqual(set(topic),{'decision_id','question'})

    def test_frozen_instructions_cannot_silently_change(self):
        with patch.object(study,'REWRITE_INSTRUCTION','Changed instruction'):
            with self.assertRaises(ValueError):study.import_authors(self.catalog,self.cohort,self.jobs)


if __name__=='__main__':unittest.main()
