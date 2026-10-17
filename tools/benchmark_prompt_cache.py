"""Compare two remote passes over at most two prepared targets, then local resume.

Uses ordinary inference artifacts; separate run directories prevent the local
result cache from hiding remote prompt-cache behavior. No input reformatting.
"""
import argparse
import asyncio
from pathlib import Path

from llm_kt.artifacts import finish_stage, verify_stage
from llm_kt.common import read_json, read_jsonl, write_json
from llm_kt.config import load_config
from llm_kt.protocols import load_prepared
from llm_kt.runtime import infer, verify_run


def collect(run, endpoint):
    rows = []
    for prediction in read_jsonl(run / 'predictions.jsonl'):
        response = read_json(run / 'responses' / (prediction['local_request_id'] + '.json'))
        usage = response.get('usage') or {}
        details = usage.get('prompt_tokens_details') or {}
        tokens, output = usage.get('prompt_tokens'), usage.get('completion_tokens')
        cached = details.get('cached_tokens')
        prompt_cost = (usage.get('cost_details') or {}).get('upstream_inference_prompt_cost')
        baseline = tokens * endpoint['input_per_million_usd'] / 1e6 if isinstance(tokens, int) else None
        token_saving = cached * (endpoint['input_per_million_usd'] - endpoint['cache_read_per_million_usd']) / 1e6 if isinstance(cached, int) else None
        rows.append({'sample_id': prediction['sample_id'], 'request_hash': prediction['request_hash'],
                     'attempt_id': prediction['local_request_id'], 'provider': response.get('provider'),
                     'prompt_tokens': tokens, 'completion_tokens': output, 'cached_tokens': cached,
                     'cache_write_tokens': details.get('cache_write_tokens'), 'cost_usd': usage.get('cost'),
                     'reported_prompt_cost_usd': prompt_cost, 'no_cache_prompt_cost_usd': baseline,
                     'cache_saving_from_tokens_usd': token_saving,
                     'cache_discount': response.get('cache_discount', usage.get('cache_discount')),
                     'p_correct': prediction['p_correct'], 'latency_ms': prediction['latency_ms']})
    return rows


async def benchmark(config, prepared, output):
    task, _, inputs = load_prepared(prepared, config)
    if not 1 <= len(inputs) <= 2:
        raise ValueError('Benchmark is limited to one or two prepared targets')
    if config['model']['adapter'] != 'openai_compatible_chat':
        raise ValueError('This benchmark measures the Chat API only')
    route = config['model']['parameters'].get('provider', {})
    if len(route.get('only', [])) != 1 or route.get('allow_fallbacks') is not False:
        raise ValueError('Pin exactly one provider and disable fallbacks')
    tag = route['only'][0]
    endpoint = next(e for e in config['model']['pricing_snapshot']['endpoints'] if e['tag'] == tag)
    if not endpoint['supports_implicit_caching'] or endpoint['cache_read_per_million_usd'] is None:
        raise ValueError('Endpoint must advertise implicit caching and cache-read pricing')
    root = Path(output).resolve()
    identity = {'config': config, 'task_id': task['task_id'], 'input_hash': task['input_hash']}
    if (root / 'artifact_manifest.json').exists():
        stage = verify_stage(root)
        if stage['identity'] != identity:
            raise ValueError('Benchmark output belongs to a different experiment')
        for name in ('pass_1', 'pass_2'):
            verify_run(root / name)
        return read_json(root / 'report.json')
    ex = config['execution']
    if ex['max_attempts'] != 1 or ex['max_requests'] != len(inputs) or ex['max_concurrency'] != 1:
        raise ValueError('Use one attempt, sequential dispatch, and max_requests equal to target count')
    write_json(root / 'experiment.json', identity)
    passes = []
    for number in (1, 2):
        run = root / f'pass_{number}'
        summary = await infer(config, prepared, run)
        rows = collect(run, endpoint)
        passes.append({'pass': number, 'execution': summary, 'rows': rows})
        if not rows:
            write_json(root / 'incomplete.json', {'passes': passes})
            raise ValueError('Remote pass incomplete; inspect local failure evidence before retrying')
        if any(r['provider'].casefold() != endpoint['provider'].casefold() for r in rows):
            raise ValueError('Returned provider differs from the pinned provider')
    first = {r['sample_id']: r for r in passes[0]['rows']}
    second = {r['sample_id']: r for r in passes[1]['rows']}
    common = sorted(first.keys() & second.keys())
    if not common or any(first[s]['request_hash'] != second[s]['request_hash'] for s in common):
        raise ValueError('No identical successful requests available for paired comparison')
    local = await infer(config, prepared, root / 'pass_2')
    if local['attempts_this_invocation'] != 0:
        raise ValueError('Local resume unexpectedly dispatched requests')
    for name in ('pass_1', 'pass_2'):
        verify_run(root / name)
    report = {'task_id': task['task_id'], 'endpoint': endpoint, 'passes': passes,
              'n_common_successful_samples': len(common),
              'paired_rows': [{'first': first[s], 'repeat': second[s]} for s in common],
              'local_resume': local, 'local_resume_incremental_api_cost_usd': 0,
              'total_benchmark_cost_usd': sum(r['cost_usd'] for p in passes for r in p['rows'] if isinstance(r['cost_usd'], (int, float))),
              'unknown_cost_attempts': sum(p['execution']['unknown_cost_attempts'] for p in passes),
              'limitations': ['Pass 1 is the first observed pass, not a guaranteed cold cache',
                              'Exact repeats test reuse, not adjacent-target cache behavior',
                              'Missing usage fields remain unknown; output cost can differ independently',
                              'Two targets cannot establish production hit rate or latency distributions']}
    write_json(root / 'report.json', report)
    finish_stage(root, 'prompt_cache_benchmark', identity, ['experiment.json', 'report.json'], [task['task_id']])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = asyncio.run(benchmark(load_config(args.config), args.prepared, args.output))
    # Payloads and student identifiers remain in ignored local artifacts.
    for p in result['passes']:
        print('pass', p['pass'], [{k: r[k] for k in ('provider', 'prompt_tokens', 'cached_tokens', 'completion_tokens', 'cost_usd', 'reported_prompt_cost_usd')} for r in p['rows']])
    print('total_benchmark_cost_usd', result['total_benchmark_cost_usd'])
    print('local_resume_new_requests', result['local_resume']['attempts_this_invocation'])


if __name__ == '__main__':
    main()
