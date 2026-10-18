"""Offline acceptance, recommended configs and uncertainty accounting. No API calls."""
import csv
from decimal import Decimal
from pathlib import Path
import yaml
from llm_kt.artifacts import finish_stage, verify_stage
from llm_kt.budget import Budget
from llm_kt.common import read_json, read_jsonl, write_json, file_hash, digest
from llm_kt.config import load_config
from llm_kt.protocols import load_prepared
from llm_kt.providers import make_provider
from llm_kt.runtime import verify_run


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'runs/r2_cost_optimization_20261018_v1'
    acceptance=output/'final_acceptance'
    if acceptance.exists():raise ValueError('Final acceptance version exists; do not overwrite')
    ledger=read_json(output/'budget.json')
    known,reserved=Budget(output/'budget.json',ledger['limit_usd']).totals()
    assert known+reserved<=Decimal(ledger['limit_usd'])
    attempts={}
    complete=[]
    for run in sorted(output.iterdir()):
        if not (run/'attempts').exists():continue
        verify_run(run)
        for f in (run/'attempts').glob('*.json'):
            attempt=read_json(f)
            assert f.stem not in attempts
            attempts[f.stem]={'run':run.name,'metadata_sha256':file_hash(f),'status':attempt['status'],
                              'error':attempt.get('error'),'http_status':attempt.get('http_status')}
        if (run/'optimization_result.json').exists():
            result=read_json(run/'optimization_result.json')
            assert result['metrics']['complete'] and result['metrics']['coverage']['n_success']==492
            predictions=read_jsonl(run/'predictions.jsonl')
            _,targets,_=load_prepared(run/'prepared')
            by_id={p['sample_id']:p for p in predictions}
            assert len(by_id)==492 and set(by_id)=={t['sample_id'] for t in targets}
            with (run/'per_student_standard_v1.csv').open(encoding='utf-8-sig') as f:
                students=list(csv.DictReader(f))
            # Independently recompute within-student rank AUC for every complete candidate.
            for student in students:
                ts=[t for t in targets if t['student_id']==student['student_id']]
                pos=[by_id[t['sample_id']]['p_correct'] for t in ts if t['y_true']==1]
                neg=[by_id[t['sample_id']]['p_correct'] for t in ts if t['y_true']==0]
                auc=sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
                assert abs(auc-float(student['auc']))<1e-12
            complete.append(result)
    assert len(complete)==10
    assert set(ledger['attempts'])==set(attempts)
    unknown=[{'attempt_id':k,**attempts[k],'reserved_usd':v['reserved'],
              'final_classification':'interrupted_inflight_unknown' if attempts[k]['status']=='running' else 'transport_failure_unknown'}
             for k,v in ledger['attempts'].items() if v.get('cost') is None]
    chosen_name='cost_r2_rows_c8_m16_decimal'
    chosen=next(r for r in complete if r['candidate']==chosen_name)
    assert chosen['passes_effect_gate'] and chosen['execution']['unknown_cost_attempts']==0
    source_model=read_json(output/chosen_name/'manifest.json')['identity']['config']['model']
    snapshot=source_model['prompt_snapshot']
    prompt={k:v for k,v in snapshot.items() if k!='prior_sha256'}
    prompt['version']='r2_cost_rows_decimal_v1'
    prompt['prior_file']='../../data/processed/prompt_priors_fold0/priors.json'
    prompt_path=root/'configs/prompts/r2_cost_rows_decimal.yaml'
    prompt_path.write_text(yaml.safe_dump(prompt,sort_keys=False),encoding='utf-8')
    original_config=yaml.safe_load((root/'configs/experiments/nips_validation_iteration_02cny_v1_round2.yaml').read_text())
    configurations=[]
    for tier in ('iteration_02cny','large_1cny'):
        config=yaml.safe_load(yaml.safe_dump(original_config))
        dataset=f'nips_validation_{tier}_v1'
        config['dataset']['snapshot']=f'../../data/snapshots/{dataset}'
        config['model']['prompt_file']='../prompts/r2_cost_rows_decimal.yaml'
        config['model']['probability_strategy']='decimal_probability'
        config['model']['parameters']['max_tokens']=16
        config['execution']['student_serial']=False
        path=root/f'configs/experiments/{dataset}_r2_cost_optimized.yaml'
        path.write_text('# Cost candidate validated on four development students; large validation not run.\n'+yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        loaded=load_config(path)
        _,_,inputs=load_prepared(root/'data/prepared'/dataset,loaded)
        if tier=='iteration_02cny':
            provider=make_provider(loaded['model'])
            source=make_provider(source_model)
            assert all(provider.request(item)==source.request(item) for item in inputs)
        configurations.append({'path':str(path.relative_to(root)),'sha256':file_hash(path)})
    control=read_json(output/'round2/execution_summary.json')
    assert control['n_success']==48 and control['n_unfinished']==444
    result={'complete_candidates':complete,'n_complete_candidates':10,'full_candidate_predictions':4920,
            'incomplete_control':control,'planned_optimized_repeat_executed':False,
            'known_usd':str(known),'unknown_reserved_usd':str(reserved),'limit_usd':ledger['limit_usd'],
            'known_cny':str(known*7),'reserved_cny':str(reserved*7),'total_with_reserve_cny':str((known+reserved)*7),
            'unknown_attempts':unknown,'all_budget_attempts_linked_to_metadata':True,
            'independent_auc_checks':40,'selected':chosen_name,'recommended_configs':configurations,
            'recommended_prompt_sha256':file_hash(prompt_path),'new_config_requests_equal_selected':True,
            'selection_reason':'Passes predeclared quality gate; zero unknown bills; lower observed cost than other normal-concurrency passing candidates. Serial option has major latency and unknown-billing penalty.',
            'no_more_paid_calls':True}
    write_json(acceptance/'summary.json',result)
    write_json(acceptance/'interruption.json',{'reason':'Connection failures and timeouts exhausted conservative budget reservations during final original-R2 control. No complete repeat claim.',
               'original_control_manifest_sha256':file_hash(output/'round2/manifest.json'),
               'original_control_execution_sha256':file_hash(output/'round2/execution_summary.json'),
               'running_attempts_preserved_as_original_evidence':sum(v['status']=='running' for v in attempts.values()),
               'runtime_budget_drain_fix_applied_after_stop':True})
    finish_stage(acceptance,'r2_cost_final_acceptance',{'summary_sha256':file_hash(acceptance/'summary.json')},['summary.json','interruption.json'])
    files=['budget.json','final_acceptance/artifact_manifest.json']
    for run in sorted(output.iterdir()):
        if (run/'manifest.json').exists():files.append(f'{run.name}/manifest.json')
        if (run/'optimization_result.json').exists():files.append(f'{run.name}/optimization_result.json')
    finish_stage(output,'r2_cost_optimization',{'limit_usd':ledger['limit_usd'],'selected':chosen_name},files)
    verify_stage(output);verify_stage(acceptance)
    print({k:v for k,v in result.items() if k in ('n_complete_candidates','known_cny','reserved_cny','total_with_reserve_cny','selected','independent_auc_checks','all_budget_attempts_linked_to_metadata')})


if __name__=='__main__':main()
