"""Budgeted, sequential KT prefix-cache experiments, isolated from production.

All model calls are ledgered before dispatch. Unknown charges retain the full
conservative reservation. Running attempts are never automatically resent.
"""
import argparse
import asyncio
import json
import math
import os
import re
import shutil
import time
from decimal import Decimal
from pathlib import Path

import httpx
from llm_kt.artifacts import finish_stage, unique_id, utc_now
from llm_kt.common import canonical, digest, file_hash, read_json, read_jsonl, write_json
from llm_kt.credentials import resolve_api_key
from llm_kt.data import iter_sequences
from llm_kt.providers import Chat, TASK


def messages(sample, layout):
    if layout == 'current_first':
        content = canonical({'history': sample['history'], 'current_item': sample['current_item']})
    elif layout == 'history_first_json':
        # Preserve the exact compact JSON facts; change top-level key order only.
        content = '{"history":' + canonical(sample['history']) + ',"current_item":' + canonical(sample['current_item']) + '}'
    elif layout == 'history_first':
        # Appendable JSONL: no changing sequence length header before the prefix.
        content = 'Observed history (one JSON record per line):\n' + '\n'.join(canonical(h) for h in sample['history'])
        content += '\nCurrent item:\n' + canonical(sample['current_item'])
    else:
        raise ValueError('Unknown layout')
    return [{'role': 'system', 'content': TASK}, {'role': 'user', 'content': content}]


def reserve_cost(payload, endpoint):
    # ASCII IDs/bools in this experiment: byte count substantially exceeds
    # native input tokens. Extra framing allowance and 4x rate safety margin.
    rates = endpoint['pricing']
    input_rate = max(Decimal(rates['prompt']), Decimal(rates.get('input_cache_write', rates['prompt'])))
    return (Decimal(len(canonical(payload).encode('utf-8')) + 4096) * input_rate
            + Decimal(payload['max_tokens']) * Decimal(rates['completion'])) * 4


def budget_totals(ledger):
    known, uncertain = Decimal('0'), Decimal('0')
    for row in ledger['attempts']:
        cost = row.get('reported_cost_usd')
        if cost is None:
            uncertain += Decimal(row['reserved_usd'])
        else:
            known += Decimal(str(cost))
    return known, uncertain


def reserve(ledger, request_id, ceiling):
    known, uncertain = budget_totals(ledger)
    if known + uncertain + ceiling > Decimal(ledger['limit_usd']):
        raise ValueError('Budget limit reached before dispatch')
    entry = {'attempt_id': request_id, 'reserved_usd': str(ceiling), 'reported_cost_usd': None,
             'status': 'reserved', 'started_at': utc_now()}
    ledger['attempts'].append(entry)
    return entry


def prepare_samples(root):
    path = root / 'samples.json'
    if path.exists():
        record = read_json(path)
        if record['source_sha256'] != file_hash(record['source_path']):
            raise ValueError('Source sequences changed')
        return record
    source = Path('data/snapshots/nips_task34/test.csv').resolve()
    allowed = {r['student_id'] for r in read_jsonl('data/prepared/nips_0p1pct_smoke_v1/targets.jsonl')}
    student = next(r for r in iter_sequences(source, 'question_native') if r['student_id'] in allowed and len(r['events']) >= 204)
    samples = []
    for index in (150, 151, 152, 198, 199, 200, 201, 202, 300, 301, 302):
        if index >= len(student['events']):
            continue
        protocol = 'full_prefix' if index >= 300 else 'pykt_window'
        start = 0 if protocol == 'full_prefix' else max(0, index - 199)
        events = student['events']
        item = {k: events[index][k] for k in ('question_id', 'concept_ids')}
        history = [{**{k: e[k] for k in ('question_id', 'concept_ids')}, 'correct': bool(e['response'])} for e in events[start:index]]
        samples.append({'sample_id': digest([file_hash(source), student['student_id'], index, start]),
                        'student_id': student['student_id'], 'event_index': index, 'history_start': start,
                        'protocol': protocol, 'history': history, 'current_item': item,
                        'y_true': events[index]['response']})
    record = {'source_path': str(source), 'source_sha256': file_hash(source), 'samples': samples}
    write_json(path, record)
    write_json(root / 'sample_evidence/samples.json', record)
    finish_stage(root / 'sample_evidence', 'adjacent_cache_samples', {'source_sha256': record['source_sha256']}, ['samples.json'])
    return record


def endpoint_for(root, model, tag):
    filename = 'qwen38.json' if model == 'qwen/qwen3.8-27b' else 'qwen36.json'
    data = json.loads((root / 'metadata' / filename).read_text(encoding='utf-8-sig'))['data']
    return next(e for e in data['endpoints'] if e['tag'] == tag)


async def run_plan(root, plan_path):
    plan = read_json(plan_path)
    root.mkdir(parents=True, exist_ok=True)
    source_record = prepare_samples(root)
    samples = {s['event_index']: s for s in source_record['samples']}
    ledger_path = root / 'budget.json'
    ledger = read_json(ledger_path) if ledger_path.exists() else {'limit_usd': '0.50', 'attempts': []}
    if ledger['limit_usd'] != '0.50':
        raise ValueError('Unexpected budget ceiling')
    plan_id = digest(plan)
    archive = root / 'plans' / (plan_id + '.json')
    write_json(archive, plan)
    known_jobs = {r.get('job_id'): r for r in ledger['attempts']}
    key = None
    previous_content = {}
    async with httpx.AsyncClient(timeout=45, follow_redirects=False) as client:
        for case in plan['cases']:
            endpoint = endpoint_for(root, case['model'], case['provider'])
            for position, index in enumerate(case['indices']):
                sample = samples[index]
                payload = {'model': case['model'], 'messages': messages(sample, case['layout']),
                           'reasoning': {'enabled': False}, 'temperature': 0, 'max_tokens': 128,
                           'provider': {'only': [case['provider']], 'allow_fallbacks': False, 'require_parameters': True}}
                if 'response_format' in endpoint['supported_parameters']:
                    payload['response_format'] = {'type': 'json_object'}
                session = 'kt-prefix-' + digest([root.name, case['name']])[:32]
                job_id = digest([plan_id, case['name'], position, index, payload])
                if job_id in known_jobs:
                    print('cached_attempt', case['name'], position, flush=True)
                    continue
                attempt_id = unique_id('attempt')
                ceiling = reserve_cost(payload, endpoint)
                entry = reserve(ledger, attempt_id, ceiling)
                entry.update(job_id=job_id, case=case['name'], model=case['model'], provider_tag=case['provider'],
                             layout=case['layout'], event_index=index, protocol=sample['protocol'],
                             history_start=sample['history_start'], history_length=len(sample['history']),
                             position=position, plan_id=plan_id)
                # A durable reservation exists before any paid request.
                write_json(ledger_path, ledger)
                folder = root / 'attempts' / attempt_id
                write_json(folder / 'request.json', payload)
                write_json(folder / 'context.json', {**entry, 'sample_id': sample['sample_id'], 'session_id': session,
                                                   'endpoint_snapshot_sha256': digest(endpoint), 'script_sha256': file_hash(__file__)})
                current = payload['messages'][1]['content']
                previous = previous_content.get(case['name'], '')
                prefix = 0
                for a, b in zip(previous, current):
                    if a != b:
                        break
                    prefix += 1
                previous_content[case['name']] = current
                entry['common_user_prefix_chars'] = prefix
                if key is None:
                    key = resolve_api_key({'adapter': 'openai_compatible_chat', 'api_key_env': 'OPENROUTER_API_KEY',
                                           'api_key_file': '../.secrets/openrouter.key'})
                tick = time.monotonic()
                try:
                    response = await client.post('https://openrouter.ai/api/v1/chat/completions', json=payload,
                                                 headers={'Authorization': 'Bearer ' + key, 'x-session-id': session})
                    raw = re.sub(r'sk-or-v1-[A-Za-z0-9_-]+', '[REDACTED]', response.text.replace(key, '[REDACTED]'))
                    (folder / 'response.body.txt').write_text(raw, encoding='utf-8')
                    entry['http_status'] = response.status_code
                    data = json.loads(raw)
                    write_json(folder / 'response.json', data)
                    usage = data.get('usage') or {}
                    cost = usage.get('cost')
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
                        entry['reported_cost_usd'] = cost
                    details = usage.get('prompt_tokens_details') or {}
                    entry.update(usage=usage, provider_returned=data.get('provider'), model_returned=data.get('model'),
                                 cached_tokens=details.get('cached_tokens'), prompt_tokens=usage.get('prompt_tokens'),
                                 completion_tokens=usage.get('completion_tokens'))
                    if response.status_code != 200:
                        entry['status'] = 'http_error'
                    else:
                        entry['status'] = 'response_received'
                        if data.get('provider', '').casefold() != endpoint['provider_name'].casefold() or data.get('model') != case['model']:
                            raise ValueError('Returned model/provider mismatch')
                        try:
                            entry['p_correct'] = Chat({}).probability(data)
                            entry['status'] = 'success'
                        except (ValueError, KeyError, TypeError, IndexError):
                            entry['status'] = 'invalid_prediction'
                except httpx.TransportError as exc:
                    entry.update(status='transport_error', error=type(exc).__name__)
                except (ValueError, KeyError, TypeError) as exc:
                    entry.update(status='invalid_response', error=type(exc).__name__)
                finally:
                    entry.update(ended_at=utc_now(), latency_ms=(time.monotonic() - tick) * 1000)
                    write_json(folder / 'result.json', entry)
                    files = [p.name for p in folder.iterdir() if p.is_file()]
                    finish_stage(folder, 'remote_cache_attempt', {'attempt_id': attempt_id, 'job_id': job_id}, files, [source_record['source_sha256']])
                    write_json(ledger_path, ledger)
                known, uncertain = budget_totals(ledger)
                print(canonical({k: entry.get(k) for k in ('case', 'event_index', 'status', 'prompt_tokens', 'cached_tokens', 'reported_cost_usd')}),
                      'budget_known', str(known), 'reserved_unknown', str(uncertain), flush=True)
                if entry.get('reported_cost_usd', 0) is not None and Decimal(str(entry.get('reported_cost_usd', 0))) > ceiling:
                    raise ValueError('Reported charge exceeded reservation; stopping')
                if known + uncertain > Decimal(ledger['limit_usd']):
                    raise ValueError('Budget exhausted; stopping')
                # Sequential dispatch and short spacing give upstream writes time.
                await asyncio.sleep(1)
    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'experiment.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if lock.tell() == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run_plan(root, args.plan))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
