"""Complete Jev-only rounds, frozen hypotheses, per-round reflection, shared CNY3 ledger."""
import argparse
import asyncio
from decimal import Decimal
from pathlib import Path
from llm_kt.artifacts import finish_stage
from llm_kt.budget import Budget
from llm_kt.common import read_json,write_json,read_jsonl,file_hash,digest
from llm_kt.config import load_config
from llm_kt.decision_prompts import load_decision_prompt,recover_facts
from llm_kt.kt_features import evidence
from llm_kt.protocols import load_prepared
from llm_kt.providers import make_provider
from llm_kt.runtime import infer,verify_run
from llm_kt.evaluation import evaluate


HYPOTHESES={
 'jev_iter_r1':('Short atomic question with named guidance fields reduces ambiguity versus one long instruction string.','Inspect whether question structure alone improves ranking; preserve all source history and evidence.'),
 'jev_iter_r2':('Decision card plus fixed code-computed odds reference removes numerical reasoning burden.','Compare with round1; improvements include deterministic feature engineering, not pure wording.'),
 'jev_iter_r3':('Lossless table history reduces redundant tokens while preserving decision-card/reference information.','Check both quality and billed tokens; do not assume character reduction equals token savings.'),
 'jev_iter_r4':('Predicting incorrectness with explicit complement mapping tests event polarity and calibration robustness.','Invert only native error probability; retain raw response and reject any sign ambiguity.'),
 'baseline_control':('Repeat original Jev v1 under the improved runtime to measure same-payload variability.','Do not interpret service/runtime changes as a prompt gain.')}


async def main(names,repeat_of=None,repeat_name=None):
    root=Path(__file__).resolve().parents[1]
    output=root/'runs/jev_iterations_20261018_v1'
    prepared=root/'data/prepared/nips_validation_iteration_02cny_v1'
    base_run=root/'runs/jev_r2_full_validation_20261018_v1/inference'
    baseline=read_json(base_run/'metrics_standard_v1.json')
    limit=str(Decimal(3)/Decimal(7))
    budget=Budget(output/'budget.json',limit)
    if not budget.path.exists():write_json(budget.path,budget.data)
    global_plan={'budget_cny':3,'usd_to_cny_planning':7,'limit_usd':limit,
      'dataset':'nips_validation_iteration_02cny_v1','students':4,'targets':492,
      'model':'typesafe/jev-1.14','provider':'typesafe','history_protocol_unchanged':True,
      'baseline_manifest_sha256':file_hash(base_run/'manifest.json'),
      'baseline_predictions_sha256':file_hash(base_run/'predictions.jsonl'),
      'selection_rule':{'overall_auc_min_gain':.005,'student_mean_max_drop':.005,'logloss_max_increase':.02,'max_student_auc_drop':.04},
      'scope':'Development optimization, not independent-test validation. No fitting statistical-reference coefficients on current labels.'}
    if (output/'plan'/'plan.json').exists():assert read_json(output/'plan'/'plan.json')==global_plan
    else:
        write_json(output/'plan'/'plan.json',global_plan)
        finish_stage(output/'plan','jev_iteration_global_plan',{'plan_sha256':file_hash(output/'plan'/'plan.json')},['plan.json'])
        import shutil
        snapshot=output/'source_snapshot'
        copied=[]
        for source in sorted((root/'src/llm_kt').rglob('*.py')):
            relative=Path('package')/source.relative_to(root/'src/llm_kt')
            (snapshot/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,snapshot/relative);copied.append(relative.as_posix())
        shutil.copyfile(Path(__file__),snapshot/'run_jev_iteration.py');copied.append('run_jev_iteration.py')
        finish_stage(snapshot,'jev_iteration_source_snapshot',{'global_plan_sha256':file_hash(output/'plan'/'plan.json')},copied)
    if repeat_of:
        if not repeat_name:raise ValueError('Repeat requires its own explicit name')
        names=[repeat_name]
    for name in names:
        run=output/name
        if run.exists():raise ValueError('Round exists; audit before resuming, never overwrite')
        config=load_config(root/'configs/experiments/nips_jev_r2_full_validation.yaml')
        config['execution'].update(max_concurrency=4,requests_per_minute=240,total_timeout_seconds=30,max_attempts=2,max_requests=984)
        if repeat_of:
            source=read_json(output/repeat_of/'manifest.json')['identity']['config']
            config=source
            hypothesis='Independent full remote repeat of selected round '+repeat_of
            reflection_note='Same payload repeat measures within-backend variability; input cache is not assumed.'
        else:
            template=root/'configs/decisions'/('jev_r2_v1.yaml' if name=='baseline_control' else name+'.yaml')
            config['model']['decision_file']=str(template)
            config['model']['decision_snapshot']=load_decision_prompt(template)
            hypothesis,reflection_note=HYPOTHESES[name]
        task,targets,inputs=load_prepared(prepared,config)
        assert len(targets)==492 and task['n_students']==4
        provider=make_provider(config['model'])
        base_prompt=read_json(base_run/'manifest.json')['identity']['config']['model']['decision_snapshot']
        payload_hashes=[]
        for item in inputs:
            payload=provider.request(item)
            restored=recover_facts(payload['state'])
            assert restored=={'history':item.history,'current_item':item.current_item,'evidence':evidence(item,base_prompt)}
            payload_hashes.append(digest(payload))
        known,reserved=budget.totals()
        # Conservative start admission leaves room for an entire round and existing unknowns.
        from llm_kt.costs import input_characters
        chars=sum(input_characters(provider.request(item),'openrouter_decisions') for item in inputs)
        expected_upper=Decimal(str(chars/2*.042/1e6*1.35))
        if known+reserved+expected_upper>Decimal(limit):
            write_json(output/'admission_stop.json',{'round':name,'known_usd':str(known),'reserved_usd':str(reserved),'conservative_full_round_usd':str(expected_upper)})
            print('STOP: insufficient conservative balance for a complete next round',flush=True)
            return
        plan={'round':name,'hypothesis':hypothesis,'reflection_focus':reflection_note,'config':config,
              'task_id':task['task_id'],'input_hash':task['input_hash'],'source_prepared_sha256':file_hash(prepared/'manifest.json'),
              'payload_hashes':payload_hashes,'script_sha256':file_hash(Path(__file__)),'all_492_inputs_losslessly_verified':True,
              'reference_method':'odds_v1 inherited coefficients 0.7/0.3, not fitted on evaluation labels',
              'estimated_upper_usd':str(expected_upper),'repeat_of':repeat_of}
        plan_root=output/'round_plans'/name
        write_json(plan_root/'plan.json',plan)
        finish_stage(plan_root,'jev_round_plan',{'plan_sha256':file_hash(plan_root/'plan.json')},['plan.json'],[task['task_id']])
        print('START',name,'estimated_upper_cny',str(expected_upper*7),flush=True)
        execution=await infer(config,prepared,run,budget)
        verification=verify_run(run)
        metrics=evaluate(prepared,run)
        import csv
        def student_aucs(path):
            with path.open(encoding='utf-8-sig') as f:return {r['student_id']:float(r['auc']) for r in csv.DictReader(f)}
        old=student_aucs(base_run/'per_student_standard_v1.csv');new=student_aucs(run/'per_student_standard_v1.csv')
        deltas={'overall_auc':metrics['overall']['auc']-baseline['overall']['auc'],
                'student_mean_auc':metrics['student_auc']['mean']-baseline['student_auc']['mean'],
                'logloss':metrics['overall']['logloss']-baseline['overall']['logloss'],
                'max_student_drop':max(old[k]-v for k,v in new.items())}
        eligible=metrics['complete'] and deltas['overall_auc']>=.005 and deltas['student_mean_auc']>=-.005 and deltas['logloss']<=.02 and deltas['max_student_drop']<=.04
        predictions=read_jsonl(run/'predictions.jsonl')
        usage={'input_tokens':sum(p['usage']['input_tokens'] for p in predictions),'reported_output_tokens':sum(p['usage']['output_tokens'] for p in predictions)}
        known,reserved=budget.totals()
        result={'round':name,'hypothesis':hypothesis,'execution':execution,'metrics':metrics,'deltas_vs_original_jev':deltas,
                'selection_eligible':eligible,'usage':usage,'verification':verification,
                'budget_known_usd':str(known),'budget_reserved_usd':str(reserved)}
        write_json(run/'round_summary.json',result)
        lines=[f'# {name} reflection','',hypothesis,'',f"Complete targets: {metrics['coverage']['n_success']}/492; valid students: {metrics['coverage']['n_valid_auc_students']}/4.",
               f"Overall AUC {metrics['overall']['auc']:.6f} (delta {deltas['overall_auc']:+.6f}); student mean {metrics['student_auc']['mean']:.6f} (delta {deltas['student_mean_auc']:+.6f}).",
               f"Log loss delta {deltas['logloss']:+.6f}; largest student loss {deltas['max_student_drop']:.6f}; selection eligible: {eligible}.",
               f"Known cost USD {execution['known_cost_usd']}; unknown attempts {execution['unknown_cost_attempts']}; input tokens {usage['input_tokens']}.",'',
               'Interpretation: '+('Meets predeclared development gate; requires complete repeat before recommendation.' if eligible else 'Do not adopt based on this result; inspect failure or student-specific regression before the next round.'),
               'Next analysis: '+reflection_note,'','Four development students do not establish population generalization.']
        (run/'reflection.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        import shutil
        write_json(run/'round_acceptance'/'summary.json',result)
        shutil.copyfile(run/'reflection.md',run/'round_acceptance'/'reflection.md')
        finish_stage(run/'round_acceptance','jev_round_acceptance',{'summary_sha256':file_hash(run/'round_summary.json'),'reflection_sha256':file_hash(run/'reflection.md')},['summary.json','reflection.md'],[task['task_id']])
        print('RESULT',name,{'overall':metrics['overall'],'student_auc':metrics['student_auc'],'deltas':deltas,'eligible':eligible,'known_cny':str(known*7),'reserved_cny':str(reserved*7),'run_cost_cny':execution['known_cost_usd']*7},flush=True)
        if not metrics['complete']:raise ValueError('Round incomplete; stop rather than report a partial round as complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='*');p.add_argument('--repeat-of');p.add_argument('--repeat-name')
    a=p.parse_args();asyncio.run(main(a.names,a.repeat_of,a.repeat_name))
