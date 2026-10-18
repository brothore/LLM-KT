import json
from pathlib import Path
import shutil
import uuid
import unittest
import yaml
from llm_kt.config import load_config
from llm_kt.decision_prompts import load_decision_prompt
from llm_kt.decision_prompts import render_decision,recover_facts,statistical_reference
from llm_kt.domain import ModelInput
from llm_kt.providers import Decisions


class DecisionPromptTests(unittest.TestCase):
    def test_independent_typed_request_contains_only_allowed_facts(self):
        prompt={'version':'synthetic','instructions':'Will `current_item` be correct?',
                'criteria':{'true':'correct','false':'incorrect'},'evidence':'history',
                'state_description':{'ids':'opaque'}}
        model={'model_id':'synthetic','probability_strategy':'noul','parameters':{},'decision_snapshot':prompt}
        item=ModelInput('private-sample',[{'question_id':'q1','concept_ids':['c'],'correct':True}],
                        {'question_id':'q2','concept_ids':['c']})
        payload=Decisions(model).request(item)
        self.assertEqual(set(payload),{'model','state','questions'})
        self.assertEqual(payload['state']['history'],item.history)
        self.assertEqual(payload['state']['evidence']['overall']['n'],1)
        self.assertNotIn('private-sample',json.dumps(payload))
        self.assertNotIn('messages',payload)
        self.assertEqual(payload['questions']['will_answer_correctly']['type'],'noul')
        self.assertEqual(Decisions(model).probability({'answers':{'will_answer_correctly':{'type':'noul','noul':.6237}}}),.6237)

    def test_chat_template_and_decision_template_cannot_be_mixed(self):
        root=Path(__file__).parent/('work_'+uuid.uuid4().hex)
        root.mkdir()
        try:
            template=root/'decision.yaml'
            template.write_text(yaml.safe_dump({'version':'s','instructions':'yes?',
                'criteria':{'true':'yes','false':'no'},'evidence':'history'}))
            p=root/'config.yaml'
            p.write_text(yaml.safe_dump({'model':{'adapter':'openai_compatible_chat','decision_file':'decision.yaml'}}))
            with self.assertRaises(ValueError):load_config(p)
            p.write_text(yaml.safe_dump({'model':{'adapter':'openrouter_decisions','prompt_file':'anything.yaml'}}))
            with self.assertRaises(ValueError):load_config(p)
            bad=root/'chat.yaml';bad.write_text(yaml.safe_dump({'version':'s','system':'chat only'}))
            with self.assertRaises(ValueError):load_decision_prompt(bad)
        finally:
            if root.resolve().parent!=Path(__file__).parent.resolve() or not root.name.startswith('work_'):
                raise RuntimeError('Unsafe test cleanup path')
            shutil.rmtree(root.resolve())

    def test_noul_does_not_accept_out_of_range_or_wrong_type(self):
        provider=Decisions({'probability_strategy':'noul'})
        for answer in [{'type':'noul','noul':1.2},{'type':'noul','noul':True},{'type':'score','score':.6}]:
            with self.assertRaises(ValueError):provider.probability({'answers':{'will_answer_correctly':answer}})

    def test_decision_card_table_preserves_all_history_and_features(self):
        item=ModelInput('s',[{'question_id':'001','concept_ids':['2','3'],'correct':False}],{'question_id':'4','concept_ids':['2']})
        prompt={'criteria':{'true':'correct','false':'incorrect'},'instructions':{'question':'Will current response be correct?'},'evidence':'history'}
        base,_=render_decision(item,prompt,'noul')
        # Synthetic training prior injection avoids external files.
        from unittest.mock import patch
        facts={**base['evidence'],'training_population':{'global':{'p':.5},'question':{'p':.7},'concepts':[]}}
        with patch('llm_kt.decision_prompts.evidence',return_value=facts):
            state,_=render_decision(item,{**prompt,'state_layout':'decision_card','history_encoding':'table','reference_mode':'odds_v1'},'noul')
        self.assertEqual(recover_facts(state),{'history':item.history,'current_item':item.current_item,'evidence':facts})
        self.assertTrue(0<state['statistical_reference']['student_adjusted_item_p']<1)
        self.assertAlmostEqual(state['statistical_reference']['student_adjusted_item_p']+state['statistical_reference']['student_adjusted_item_incorrect_p'],1)

    def test_negative_question_polarity_is_explicitly_converted(self):
        p=Decisions({'probability_strategy':'noul','decision_snapshot':{'positive_class':'incorrect'}})
        self.assertAlmostEqual(p.probability({'answers':{'will_answer_correctly':{'type':'noul','noul':.2}}}),.8)
        p=Decisions({'probability_strategy':'choice','decision_snapshot':{'positive_class':'incorrect'}})
        self.assertEqual(p.probability({'answers':{'will_answer_correctly':{'type':'choice','probabilities':{'correct':.7,'incorrect':.3}}}}),.7)
        p=Decisions({'probability_strategy':'noul','decision_snapshot':{'positive_class':'incorrect','question_key':'will_answer_incorrectly'}})
        self.assertAlmostEqual(p.probability({'answers':{'will_answer_incorrectly':{'type':'noul','noul':.3}}}),.7)
