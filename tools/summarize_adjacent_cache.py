"""Offline summary of adjacent-cache evidence; never reads API credentials."""
from collections import defaultdict
from pathlib import Path
import argparse
import importlib.util

from llm_kt.artifacts import finish_stage, verify_stage
from llm_kt.common import digest, file_hash, read_json, write_json


def summarize(root):
    root = Path(root)
    ledger = read_json(root / 'budget.json')
    plans = {p.stem: read_json(p) for p in (root / 'plans').glob('*.json')}
    metadata = {}
    for name in ('qwen36', 'qwen38'):
        import json
        data = json.loads((root / 'metadata' / (name + '.json')).read_text(encoding='utf-8-sig'))['data']
        metadata[data['id']] = {e['tag']: e for e in data['endpoints']}
    groups, rows = defaultdict(list), []
    for attempt in ledger['attempts']:
        verify_stage(root / 'attempts' / attempt['attempt_id'])
        evidence = read_json(root / 'attempts' / attempt['attempt_id'] / 'result.json')
        if attempt != evidence:
            raise ValueError('Budget ledger differs from request evidence')
        endpoint = metadata[attempt['model']][attempt['provider_tag']]
        rates = endpoint['pricing']
        usage = attempt.get('usage', {})
        native_cost = usage.get('cost_details', {}).get('upstream_inference_prompt_cost')
        tokens = attempt.get('prompt_tokens')
        baseline = tokens * float(rates['prompt']) if isinstance(tokens, int) else None
        write = usage.get('prompt_tokens_details', {}).get('cache_write_tokens')
        case = next(c for c in plans[attempt['plan_id']]['cases'] if c['name'] == attempt['case'])
        indices = case['indices']
        control = attempt['position'] > 0 and indices[attempt['position']] == indices[attempt['position'] - 1]
        row = {**attempt, 'exact_repeat_control': control, 'uncached_prompt_cost_usd': baseline,
               'actual_prompt_cost_usd': native_cost, 'cache_write_tokens': write,
               'prompt_saving_usd': baseline - native_cost if baseline is not None and native_cost is not None else None}
        rows.append(row)
        groups[attempt['case']].append(row)
    results = []
    for case, records in groups.items():
        unique = [r for r in records if not r['exact_repeat_control']]
        valid = [r for r in unique if r['status'] == 'success']
        measured = [r for r in valid if r['actual_prompt_cost_usd'] is not None and r['uncached_prompt_cost_usd'] is not None]
        baseline = sum(r['uncached_prompt_cost_usd'] for r in measured)
        actual = sum(r['actual_prompt_cost_usd'] for r in measured)
        results.append({'case': case, 'model': records[0]['model'], 'provider': records[0]['provider_tag'],
                        'layout': records[0]['layout'], 'n_unique_targets': len(unique), 'n_success': len(valid),
                        'n_adjacent_hits': sum((r.get('cached_tokens') or 0) > 0 for r in valid[1:]),
                        'indices': [r['event_index'] for r in unique],
                        'cached_tokens': [r.get('cached_tokens') for r in unique],
                        'prompt_tokens': [r.get('prompt_tokens') for r in unique],
                        'uncached_prompt_cost_usd': baseline, 'actual_prompt_cost_usd': actual,
                        'input_savings_fraction': (baseline - actual) / baseline if baseline else None,
                        'unique_target_total_cost_usd': sum(r.get('reported_cost_usd') or 0 for r in unique),
                        'exact_repeat_controls': len(records) - len(unique)})
    known = sum(r['reported_cost_usd'] for r in rows if r.get('reported_cost_usd') is not None)
    unknown = [r for r in rows if r.get('reported_cost_usd') is None]
    reserved = sum(float(r['reserved_usd']) for r in unknown)
    report = {'budget_limit_usd': float(ledger['limit_usd']), 'known_cost_usd': known,
              'unknown_cost_attempts': len(unknown), 'unknown_reserved_usd': reserved,
              'n_attempts': len(rows), 'n_success': sum(r['status'] == 'success' for r in rows),
              'cases': results, 'rows': rows,
              'source_budget_sha256': file_hash(root / 'budget.json'),
              'limitations': ['First observation is not proven cold cache', 'One student; small exploratory sample',
                              'Input-layout variants change token counts and can change probabilities',
                              'Input savings use actual per-request token counts and dated endpoint rates',
                              'Exact repeats are controls, excluded from unique-target aggregates',
                              'Provider pinning and session affinity do not pin the upstream worker']}
    write_json(root / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    for c in report['cases']:
        print(c['case'], c['cached_tokens'], 'input_saving', round(100 * (c['input_savings_fraction'] or 0), 2), 'percent')
    print('attempts', report['n_attempts'], 'known_cost_usd', report['known_cost_usd'], 'unknown', report['unknown_cost_attempts'])
