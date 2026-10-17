import importlib.util
import unittest
from decimal import Decimal
from pathlib import Path

spec = importlib.util.spec_from_file_location('adjacent_cache', Path(__file__).parents[1] / 'tools/explore_adjacent_cache.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AdjacentCacheTests(unittest.TestCase):
    def test_unknown_attempt_retains_reservation(self):
        ledger = {'limit_usd': '0.50', 'attempts': []}
        a = module.reserve(ledger, 'first', Decimal('0.30'))
        with self.assertRaises(ValueError):
            module.reserve(ledger, 'second', Decimal('0.21'))
        a['reported_cost_usd'] = 0.01
        module.reserve(ledger, 'third', Decimal('0.40'))
        self.assertEqual(module.budget_totals(ledger), (Decimal('0.01'), Decimal('0.40')))

    def test_budget_includes_exact_boundary(self):
        ledger = {'limit_usd': '0.50', 'attempts': []}
        module.reserve(ledger, 'first', Decimal('0.50'))
        with self.assertRaises(ValueError):
            module.reserve(ledger, 'extra', Decimal('0.00000001'))

    def test_history_first_preserves_growing_prefix_and_hides_label(self):
        h = {'question_id': 'a', 'concept_ids': ['x'], 'correct': True}
        sample = {'history': [h], 'current_item': {'question_id': 'b', 'concept_ids': ['y']}, 'y_true': 0, 'student_id': 'private-local-id'}
        first = module.messages(sample, 'history_first')[1]['content']
        later = {**sample, 'history': [h, {**sample['current_item'], 'correct': False}],
                 'current_item': {'question_id': 'c', 'concept_ids': ['z']}}
        second = module.messages(later, 'history_first')[1]['content']
        prefix = first.split('\nCurrent item:')[0]
        self.assertTrue(second.startswith(prefix))
        self.assertNotIn('y_true', first)
        self.assertNotIn('private-local-id', first)
        self.assertNotIn('correct', first.split('Current item:')[1])

    def test_reservation_accounts_for_expensive_cache_writes(self):
        payload = {'max_tokens': 128, 'messages': []}
        endpoint = {'pricing': {'prompt': '0.0000001', 'completion': '0.000001', 'input_cache_write': '0.0000005'}}
        estimate = module.reserve_cost(payload, endpoint)
        self.assertGreater(estimate, Decimal('0.008'))

    def test_compact_variant_changes_only_top_level_key_order(self):
        import json
        sample = {'history': [{'question_id': 'q', 'concept_ids': ['c'], 'correct': True}],
                  'current_item': {'question_id': 'next', 'concept_ids': ['c']}}
        original = module.messages(sample, 'current_first')[1]['content']
        compact = module.messages(sample, 'history_first_json')[1]['content']
        self.assertEqual(json.loads(original), json.loads(compact))
        self.assertEqual(len(original), len(compact))
        self.assertTrue(compact.startswith('{"history":'))
