import unittest
from llm_kt.domain import ModelInput
from llm_kt.providers import Chat
from llm_kt.prompts import evidence, render


class PromptTests(unittest.TestCase):
    def test_evidence_uses_only_visible_relevant_history(self):
        item = ModelInput('s', [{'question_id':'a','concept_ids':['c'],'correct':True},
                                {'question_id':'b','concept_ids':['d'],'correct':False}],
                          {'question_id':'a','concept_ids':['c']})
        result = evidence(item, {'evidence':'history'})
        self.assertEqual(result['overall']['n'], 2)
        self.assertEqual(result['shared_concept']['n'], 1)
        self.assertEqual(result['shared_concept']['rate'], 1)
        self.assertTrue(result['last_same_question_correct'])
        text = render(item, {'system':'test','evidence':'history'})[1]['content']
        self.assertNotIn('y_true', text)
        self.assertTrue(text.startswith('{"history":'))

    def test_single_json_fence_is_normalized_without_accepting_commentary(self):
        provider = Chat({})
        def response(text):
            return {'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        self.assertEqual(provider.probability(response('```json\n{"p_correct":0.6}\n```')), .6)
        with self.assertRaises(ValueError):
            provider.probability(response('Explanation\n```json\n{"p_correct":0.6}\n```'))
        with self.assertRaises(ValueError):
            provider.probability(response('{"p_correct":0.6,"extra":1}'))
