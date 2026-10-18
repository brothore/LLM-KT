"""Read-only audit of reusable predictions across runs; no API or key access."""
from collections import defaultdict
from pathlib import Path
from llm_kt.common import read_json, read_jsonl, write_json, digest, file_hash
from llm_kt.config import load_config
from llm_kt.protocols import load_prepared
from llm_kt.providers import make_provider
from llm_kt.costs import calibration, input_characters
from llm_kt.runtime import verify_run
from llm_kt.artifacts import finish_stage


def key(model, replicate, payload):
    return digest({'endpoint':model['endpoint'], 'adapter':model['adapter'],
                   'probability_strategy':model['probability_strategy'],
                   'replicate':replicate, 'payload':payload})


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'data/reports/four_student_cache_overlap_v1'
    if output.exists():
        raise ValueError('Use a new output version instead of replacing the audit')
    index, scanned, excluded = defaultdict(list), [], []
    for manifest_path in sorted((root/'runs').rglob('manifest.json')):
        run = manifest_path.parent
        if not (run/'predictions.jsonl').exists():
            continue
        manifest = read_json(manifest_path)
        config = manifest.get('identity',{}).get('config')
        if not config or not (run/'attempts').exists():
            excluded.append({'run':str(run.relative_to(root)), 'reason':'Not a runtime prediction archive with verifiable identity'})
            continue
        model = config['model']
        if model['model_id'] != 'qwen/qwen3.8-27b':
            continue
        try:
            verification = verify_run(run)
        except (ValueError,OSError,KeyError) as exc:
            excluded.append({'run':str(run.relative_to(root)), 'reason':type(exc).__name__})
            continue
        provider = make_provider(model)
        count = 0
        for prediction in read_jsonl(run/'predictions.jsonl'):
            if prediction.get('status') != 'success' or prediction.get('provider') != 'Tenstorrent':
                continue
            attempt = prediction['local_request_id']
            payload_path = run/'requests'/f'{attempt}.json'
            response_path = run/'responses'/f'{attempt}.json'
            payload, response = read_json(payload_path), read_json(response_path)
            if provider.probability(response) != prediction['p_correct']:
                raise ValueError('Archived response probability mismatch')
            old_hash = digest({'payload':payload,'model':model,'replicate':config['execution']['replicate_id'],'adapter_version':1})
            if old_hash != prediction['request_hash']:
                raise ValueError('Archived runtime request identity mismatch')
            index[key(model,config['execution']['replicate_id'],payload)].append({
                'run':str(run.relative_to(root)), 'sample_id':prediction['sample_id'],
                'attempt_id':attempt, 'p_correct':prediction['p_correct'],
                'source_billed_usd':prediction.get('cost'),
                'request_sha256':file_hash(payload_path), 'response_sha256':file_hash(response_path),
                'manifest_sha256':file_hash(manifest_path)})
            count += 1
        scanned.append({'run':str(run.relative_to(root)),'valid_records_indexed':count,'verification':verification})
    summary, all_matches = [], []
    for name in ('baseline','round1','round2','round3'):
        config = load_config(root/f'configs/experiments/nips_{name}.yaml')
        config['dataset']['snapshot'] = str(root/'data/snapshots/nips_validation_iteration_02cny_v1')
        _, targets, inputs = load_prepared(root/'data/prepared/nips_validation_iteration_02cny_v1',config)
        provider = make_provider(config['model'])
        measured = calibration(config,root/f'runs/prompt_iterations_20261018/{name}')
        hits, conflicts, saved = 0, 0, 0.
        for target,item in zip(targets,inputs):
            payload = provider.request(item)
            fingerprint = key(config['model'],config['execution']['replicate_id'],payload)
            candidates = index.get(fingerprint,[])
            if not candidates:
                continue
            distinct = {p['p_correct'] for p in candidates}
            conflict = len(distinct)>1
            conflicts += int(conflict)
            hits += int(not conflict)
            estimate = (input_characters(payload,'openai_compatible_chat')/measured['input_characters_per_token']*1.25*.02 + max(32,measured['mean_output_tokens'])*.5)/1e6
            saved += 0 if conflict else estimate
            all_matches.append({'prompt':name,'new_sample_id':item.sample_id,'student_id':target['student_id'],
                                'event_index':target['event_index'],'request_fingerprint':fingerprint,
                                'conflicting_archived_probabilities':conflict,'sources':candidates,
                                'estimated_avoided_usd_conservative':0 if conflict else estimate})
        summary.append({'prompt':name,'new_targets':len(inputs),'exact_reusable_targets':hits,
                        'conflicting_targets':conflicts,'new_requests_needed':len(inputs)-hits,
                        'estimated_saved_cny_conservative':saved*7})
    result = {'summary':summary,'scanned_runs':scanned,'excluded_runs':excluded,
              'matches':all_matches,'paid_api_calls':0,'reuse_implemented':False,
              'usd_to_cny_planning':7,
              'matching':'Canonical full API body + endpoint + adapter + probability parser + replicate; validated archived Tenstorrent success only',
              'script_sha256':file_hash(Path(__file__))}
    write_json(output/'report.json',result)
    finish_stage(output,'offline_cache_overlap',{'report_hash':digest(result)},['report.json'])
    print({'summary':summary,'scanned_runs':len(scanned),'excluded_runs':len(excluded),'total_reusable':sum(r['exact_reusable_targets'] for r in summary),'paid_api_calls':0})


if __name__ == '__main__':
    main()
