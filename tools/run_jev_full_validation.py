"""Fresh full-sequence Jev evaluation and offline comparison with frozen Qwen runs."""
import asyncio
from decimal import Decimal
from pathlib import Path
from llm_kt.artifacts import finish_stage
from llm_kt.budget import Budget
from llm_kt.common import read_json, write_json, file_hash
from llm_kt.config import load_config
from llm_kt.evaluation import evaluate, compare
from llm_kt.protocols import load_prepared
from llm_kt.runtime import infer, verify_run


async def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'runs/jev_r2_full_validation_20261018_v1'
    if output.exists():raise ValueError('Existing full run must be audited before resuming; do not overwrite')
    prepared=root/'data/prepared/nips_validation_iteration_02cny_v1'
    config=load_config(root/'configs/experiments/nips_jev_r2_full_validation.yaml')
    config['execution'].update(max_concurrency=4,requests_per_minute=120,max_attempts=2,max_requests=984)
    manifest,targets,_=load_prepared(prepared,config)
    assert len(targets)==492 and manifest['n_students']==4
    references=[root/'runs/four_student_revalidation_20261018_v1/baseline',
                root/'runs/four_student_revalidation_20261018_v1/round2',
                root/'runs/r2_cost_optimization_20261018_v1/cost_r2_rows_c8_m16_decimal']
    reference_files=[]
    for run in references:
        verify_run(run)
        old=read_json(run/'manifest.json')
        assert old['identity']['task_id']==manifest['task_id']
        metrics=read_json(run/'metrics_standard_v1.json')
        assert metrics['complete'] and metrics['coverage']['n_success']==492
        reference_files.append({'run':str(run.relative_to(root)),
                                'manifest_sha256':file_hash(run/'manifest.json'),
                                'predictions_sha256':file_hash(run/'predictions.jsonl')})
    limit=str(Decimal('0.65')/Decimal(7))
    plan={'config':config,'task_id':manifest['task_id'],'n_targets':492,'n_students':4,
          'budget_cny':.65,'usd_to_cny_planning':7,'limit_usd':limit,
          'fresh_requests':True,'reuse_smoke_predictions':False,'qwen_references':reference_files,
          'prepared_manifest_sha256':file_hash(prepared/'manifest.json'),
          'decision_template_sha256':file_hash(config['model']['decision_file']),
          'script_sha256':file_hash(Path(__file__)),
          'rate_policy':'Local 120 RPM and 4 workers, not an official provider limit'}
    write_json(output/'plan'/'plan.json',plan)
    finish_stage(output/'plan','jev_full_validation_plan',{'plan_sha256':file_hash(output/'plan'/'plan.json')},['plan.json'],[manifest['task_id']])
    budget=Budget(output/'budget.json',limit)
    write_json(budget.path,budget.data)
    print('START Jev: 492 fresh targets, CNY .65 cap, 4 workers / local 120 RPM',flush=True)
    execution=await infer(config,prepared,output/'inference',budget)
    verification=verify_run(output/'inference')
    metrics=evaluate(prepared,output/'inference')
    known,reserved=budget.totals()
    summary={'execution':execution,'metrics':metrics,'verification':verification,
             'known_usd':str(known),'reserved_usd':str(reserved),
             'known_cny_planning':str(known*7),'reserved_cny_planning':str(reserved*7),
             'complete':metrics['complete'],'limit_usd':limit}
    write_json(output/'summary.json',summary)
    if metrics['complete']:
        compare(prepared,[*references,output/'inference'],output/'qwen_comparison.json')
    finish_stage(output,'jev_full_validation',{'task_id':manifest['task_id'],'complete':metrics['complete']},
                 ['summary.json','budget.json','plan/artifact_manifest.json']+(['qwen_comparison.json'] if metrics['complete'] else []),[manifest['task_id']])
    print('RESULT',summary,flush=True)
    if not metrics['complete']:raise ValueError('Full evaluation incomplete; do not interpret partial results as a complete comparison')


if __name__=='__main__':asyncio.run(main())
