import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from llm_kt.common import write_json, write_jsonl
from llm_kt.config import load_config
from llm_kt.data import import_snapshot, read_sequences
from llm_kt.domain import ModelInput
from llm_kt.providers import make_provider
from llm_kt.costs import request_character_totals, cost_for_endpoint, estimate, calibration
from llm_kt.vendor.pykt.sequence import target_ranges

PROJECT = Path(__file__).parents[1]


class CostTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / 'tests' / ('work_cost_' + uuid.uuid4().hex)
        self.root.mkdir()
        import_snapshot(PROJECT / 'tests/fixtures/test_quelevel.csv', self.root / 'snapshot', 'demo')
        self.path = self.root / 'config.json'
        self.raw = {'dataset': {'snapshot': 'snapshot'}, 'task': {'maxlen': 4},
                    'model': {'pricing_file': str(PROJECT / 'configs/pricing/jev-1.14.json')}}
        write_json(self.path, self.raw)
        self.config = load_config(self.path)

    def tearDown(self):
        assert self.root.resolve().parent == (PROJECT / 'tests').resolve() and self.root.name.startswith('work_cost_')
        shutil.rmtree(self.root)

    def test_linear_character_accounting_matches_real_requests(self):
        rows = read_sequences(self.root / 'snapshot/test.csv', 'question_native')
        for adapter in ('openrouter_decisions', 'openai_compatible_chat'):
            self.config['model']['adapter'] = adapter
            self.config['model']['parameters'] = {'response_format': {'type': 'json_object'}} if adapter.endswith('chat') else {}
            for protocol in ('pykt_window', 'pykt_segmented', 'full_prefix'):
                self.config['task']['protocol'] = protocol
                total = count = 0
                provider = make_provider(self.config['model'])
                for row in rows:
                    events = row['events']
                    for start, index in target_ranges(len(events), protocol, 4, 3):
                        history = [{'question_id': e['question_id'], 'concept_ids': e['concept_ids'], 'correct': bool(e['response'])} for e in events[start:index]]
                        current = {k: events[index][k] for k in ('question_id', 'concept_ids')}
                        payload = provider.request(ModelInput('', history, current))
                        if adapter == 'openai_compatible_chat':
                            total += sum(len(m['content']) for m in payload['messages']) + len(json.dumps(payload['response_format'], sort_keys=True, separators=(',', ':')))
                        else:
                            total += len(json.dumps({'state': payload['state'], 'questions': payload['questions']}, sort_keys=True, separators=(',', ':')))
                        count += 1
                actual = request_character_totals(self.config)
                self.assertEqual(actual['input_character_proxy'], total)
                self.assertEqual(actual['n_requests'], count)

    def test_known_costs_and_cache_arithmetic(self):
        endpoint = {'provider': 'test', 'input_per_million_usd': 1, 'output_per_million_usd': 2,
                    'cache_read_per_million_usd': 0.1, 'cache_write_per_million_usd': 1.25, 'supports_implicit_caching': True}
        self.assertAlmostEqual(cost_for_endpoint(endpoint, 1e6, 1e5, 0.5)['total_usd'], 0.75)
        self.assertAlmostEqual(cost_for_endpoint(endpoint, 1e6, 0, 0, 0.5)['total_usd'], 1.125)
        with self.assertRaises(ValueError):
            cost_for_endpoint(self.config['model']['pricing_snapshot']['endpoints'][0], 1e6, 0, 0.5)

    def test_estimate_offline_no_credentials_and_zero_output_price(self):
        with patch('httpx.AsyncClient.post', side_effect=AssertionError('No API calls')), patch('llm_kt.runtime.resolve_api_key', side_effect=AssertionError('No secret reads')):
            result = estimate(self.config, output_tokens=10000)
        self.assertEqual(result['volume']['n_requests'], 10)
        self.assertEqual(result['endpoint_estimates'][0]['central']['output_usd'], 0)
        expected = result['volume']['input_character_proxy'] / 3 * 0.042 / 1e6
        self.assertAlmostEqual(result['endpoint_estimates'][0]['central']['total_usd'], expected)
        all_cached = estimate(self.config, local_hit_fraction=1)
        self.assertEqual(all_cached['central_cost_range_usd']['max'], 0)

    def test_prices_are_loaded_into_config_and_model_must_match(self):
        self.assertEqual(self.config['model']['pricing_snapshot']['endpoints'][0]['input_per_million_usd'], 0.042)
        self.raw['model']['model_id'] = 'wrong/model'
        write_json(self.path, self.raw)
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_provider_range_selection_and_invalid_assumptions(self):
        config = load_config(PROJECT / 'configs/chat.yaml')
        config['dataset']['snapshot'] = str(self.root / 'snapshot')
        result = estimate(config, provider='darkbloom/fp4', output_tokens=64)
        self.assertEqual(len(result['endpoint_estimates']), 1)
        self.assertEqual(result['endpoint_estimates'][0]['provider'], 'Darkbloom')
        self.assertGreater(result['endpoint_estimates'][0]['central']['output_usd'], 0)
        for kwargs in ({'chars_per_token': 0}, {'cache_read_fraction': 0.8, 'cache_write_fraction': 0.4}, {'output_tokens': -1}, {'provider': 'unknown'}):
            with self.assertRaises(ValueError):
                estimate(self.config, **kwargs)

    def test_calibration_uses_all_input_samples_with_missing_output_usage(self):
        run = self.root / 'run'
        write_json(run / 'manifest.json', {'identity': {'config': self.config}})
        write_json(run / 'requests/a.json', {'state': {'x': 'abc'}, 'questions': {}})
        write_json(run / 'requests/b.json', {'state': {'x': 'def'}, 'questions': {}})
        write_jsonl(run / 'predictions.jsonl', [
            {'local_request_id': 'a', 'usage': {'input_tokens': 10}},
            {'local_request_id': 'b', 'usage': {'input_tokens': 10, 'output_tokens': 4}}])
        result = calibration(self.config, run)
        self.assertEqual(result['samples'], 2)
        self.assertIsNone(result['mean_output_tokens'])


if __name__ == '__main__':
    unittest.main()
