import json
import unittest
from llm_kt.domain import ModelInput
from llm_kt.prompts import render,decode_visible,pack_evidence,unpack_evidence
from llm_kt.providers import Chat


class CompactHistoryTests(unittest.TestCase):
    def test_lossless_multi_concept_and_response_encoding(self):
        item = ModelInput('x', [dict(question_id='001',concept_ids=['7','12'],correct=False),
                                dict(question_id='9',concept_ids=['7'],correct=True)],
                          dict(question_id='003',concept_ids=['12','0']))
        data = json.loads(render(item,dict(system='schema',history_encoding='tuples_v1'))[1]['content'])
        restored=[dict(question_id=q,concept_ids=c,correct=bool(r)) for q,c,r in data['history']]
        self.assertEqual(restored,item.history)
        self.assertEqual(dict(question_id=data['current_item'][0],concept_ids=data['current_item'][1]),item.current_item)
        self.assertEqual(data['history'][0][0],'001')

    def test_default_preserves_object_representation(self):
        item=ModelInput('x',[],dict(question_id='2',concept_ids=['1']))
        data=json.loads(render(item,dict(system='original'))[1]['content'])
        self.assertEqual(data,dict(history=[],current_item=item.current_item))

    def test_rows_roundtrip_and_delimiter_rejection(self):
        item=ModelInput('x',[dict(question_id='001',concept_ids=['02','3'],correct=True)],dict(question_id='2',concept_ids=['3']))
        prompt=dict(system='schema',history_encoding='rows_v1')
        decoded=decode_visible(render(item,prompt)[1]['content'],prompt)
        self.assertEqual(decoded,dict(history=item.history,current_item=item.current_item))
        bad=ModelInput('x',[dict(question_id='bad|id',concept_ids=['2'],correct=False)],item.current_item)
        with self.assertRaises(ValueError):render(bad,prompt)

    def test_packed_evidence_roundtrip_with_null_and_multiple_concepts(self):
        facts={'overall':{'n':2,'correct':1,'rate':.5},'per_current_concept':{'01':{'n':0,'correct':0,'rate':None}},
               'last_same_question_correct':False,'training_population':{'global':{'n':20,'correct':11,'p':.55},
               'question':None,'concepts':[{'id':'01','n':9,'correct':3,'p':.4}], 'smoothing_strength':20}}
        self.assertEqual(unpack_evidence(pack_evidence(facts)),facts)
        facts['training_population']['question']={'id':'009','n':3,'correct':2,'p':.6}
        self.assertEqual(unpack_evidence(pack_evidence(facts)),facts)

    def test_integer_probability_strict_parsing(self):
        provider=Chat({'probability_strategy':'scaled_integer_probability'})
        def response(value):return {'choices':[{'finish_reason':'stop','message':{'content':value}}]}
        for text,p in [('0',0),('10000',1),('6237',.6237)]:
            self.assertEqual(provider.probability(response(text)),p)
        for text in ['10001','-1','0.5','true','01','{"p_correct":0.5}','5000 explanation']:
            with self.assertRaises(ValueError):provider.probability(response(text))

    def test_numeric_ids_roundtrip_without_changing_leading_zeros(self):
        item=ModelInput('x',[dict(question_id='001',concept_ids=['02','3'],correct=True)],dict(question_id='20',concept_ids=['3']))
        prompt=dict(system='schema',history_encoding='numeric_tuples_v1')
        encoded=render(item,prompt)[1]['content']
        self.assertEqual(json.loads(encoded)['current_item'][0],20)
        self.assertEqual(decode_visible(encoded,prompt),dict(history=item.history,current_item=item.current_item))

    def test_decimal_probability_rejects_nonprobabilities(self):
        provider=Chat({'probability_strategy':'decimal_probability'})
        def response(text):return {'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        self.assertEqual(provider.probability(response('0.6237')),.6237)
        for text in ['NaN','1.01','-0.1','6237','{"p_correct":0.5}','0.5 explanation']:
            with self.assertRaises(ValueError):provider.probability(response(text))
