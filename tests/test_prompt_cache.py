import importlib.util
import shutil
import uuid
import unittest
from pathlib import Path

from llm_kt.common import write_json, write_jsonl

spec = importlib.util.spec_from_file_location('cache_benchmark', Path(__file__).parents[1] / 'tools/benchmark_prompt_cache.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PromptCacheAccountingTests(unittest.TestCase):
    def collect(self, usage):
        root = Path(__file__).parent / ('work_' + uuid.uuid4().hex)
        root.mkdir()
        try:
            write_jsonl(root / 'predictions.jsonl', [{'sample_id': 'synthetic', 'request_hash': 'hash',
                        'local_request_id': 'attempt_test', 'p_correct': 0.6, 'latency_ms': 20}])
            write_json(root / 'responses/attempt_test.json', {'provider': 'Synthetic', 'usage': usage})
            return benchmark.collect(root, {'input_per_million_usd': 0.15, 'cache_read_per_million_usd': 0.05})[0]
        finally:
            if root.resolve().parent != Path(__file__).parent.resolve() or not root.name.startswith('work_'):
                raise RuntimeError('Unsafe cleanup path')
            shutil.rmtree(root)

    def test_cached_token_savings_exclude_output_cost(self):
        row = self.collect({'prompt_tokens': 2000, 'completion_tokens': 10, 'cost': 0.00021,
                            'prompt_tokens_details': {'cached_tokens': 1000, 'cache_write_tokens': 0},
                            'cost_details': {'upstream_inference_prompt_cost': 0.0002}})
        self.assertAlmostEqual(row['no_cache_prompt_cost_usd'], 0.0003)
        self.assertAlmostEqual(row['cache_saving_from_tokens_usd'], 0.0001)
        self.assertAlmostEqual(row['reported_prompt_cost_usd'], 0.0002)

    def test_missing_cache_fields_are_unknown_not_zero(self):
        row = self.collect({'prompt_tokens': 2000})
        self.assertIsNone(row['cached_tokens'])
        self.assertIsNone(row['cache_saving_from_tokens_usd'])
        self.assertIsNone(row['cost_usd'])

    def test_reported_zero_is_a_known_miss(self):
        row = self.collect({'prompt_tokens': 2000, 'prompt_tokens_details': {'cached_tokens': 0}})
        self.assertEqual(row['cached_tokens'], 0)
        self.assertEqual(row['cache_saving_from_tokens_usd'], 0)
