"""R2-only cost experiments with frozen inputs and a shared CNY 1 budget."""
import argparse
import asyncio
import csv
import json
from decimal import Decimal
from pathlib import Path
from llm_kt.artifacts import finish_stage, verify_stage
from llm_kt.budget import Budget
from llm_kt.common import read_json, write_json, read_jsonl, file_hash
from llm_kt.config import load_config
from llm_kt.costs import input_characters
from llm_kt.evaluation import evaluate
from llm_kt.prompts import load_prompt, decode_visible, evidence
from llm_kt.protocols import load_prepared
from llm_kt.providers import make_provider
from llm_kt.runtime import infer, verify_run


async def main(names, concurrency, max_tokens, integer_output=False, student_serial=False, decimal_output=False):
    root = Path(__file__).resolve().parents[1]
    output = root/'runs/r2_cost_optimization_20261018_v1'
    output.mkdir(parents=True,exist_ok=True)
    limit=str(Decimal(1)/Decimal(7))
    budget=Budget(output/'budget.json',limit)
    if not budget.path.exists():write_json(budget.path,budget.data)
    prepared=root/'data/prepared/nips_validation_iteration_02cny_v1'
    baseline=root/'runs/four_student_revalidation_20261018_v1/round2'
    reference=read_json(baseline/'metrics_standard_v1.json')
    with (baseline/'per_student_standard_v1.csv').open(encoding='utf-8-sig') as f:
        old_students={r['student_id']:float(r['auc']) for r in csv.DictReader(f)}
    for name in names:
        run_name=name+(f'_c{concurrency}_m{max_tokens}' if concurrency!=8 or max_tokens!=128 else '')
        run_name += ('_integer' if integer_output else '') + ('_studentserial' if student_serial else '')
        run_name += '_decimal' if decimal_output else ''
        run=output/run_name
        if run.exists():raise ValueError('Use a fresh candidate name; do not silently rerun or overwrite')
        config=load_config(root/'configs/experiments/nips_validation_iteration_02cny_v1_round2.yaml')
        prompt=root/'configs/prompts'/f'{name}.yaml'
        config['model']['prompt_file']=str(prompt)
        config['model']['prompt_snapshot']=load_prompt(prompt)
        config['execution']['max_concurrency']=concurrency
        config['execution']['student_serial']=student_serial
        config['model']['parameters']['max_tokens']=max_tokens
        if integer_output or decimal_output:
            config['model']['probability_strategy']='decimal_probability' if decimal_output else 'scaled_integer_probability'
            original=config['model']['prompt_snapshot']['system']
            before=original.split('Output exactly')[0] if 'Output exactly' in original else original.split('Return only')[0]
            instruction = ' Output only P(correct) as a decimal with four places, e.g. 0.6237. No JSON, markdown, explanation or extra text.' if decimal_output else ' Output only an integer from 0 to 10000 equal to round(10000*P(correct)); no JSON, punctuation or explanation. This preserves four-decimal probability precision.'
            config['model']['prompt_snapshot']['system']=before+instruction
        task,targets,inputs=load_prepared(prepared,config)
        assert len(inputs)==492 and task['n_students']==4
        provider=make_provider(config['model'])
        chars=0
        for item in inputs:
            payload=provider.request(item)
            text=decode_visible(payload['messages'][1]['content'],config['model']['prompt_snapshot'])
            assert text['history']==item.history and text['current_item']==item.current_item
            assert text['evidence']==evidence(item,config['model']['prompt_snapshot'])
            chars+=input_characters(payload,'openai_compatible_chat')
        plan={'name':run_name,'config':config,'n_targets':492,'input_character_proxy':chars,
              'reference_manifest_sha256':file_hash(baseline/'manifest.json'),
              'reference_metrics_sha256':file_hash(baseline/'metrics_standard_v1.json'),
              'prompt_sha256':file_hash(prompt),'script_sha256':file_hash(Path(__file__)),
              'predeclared_gate':{'overall_auc_max_drop':.01,'student_auc_mean_max_drop':.01,
                                 'logloss_max_increase':.02,'max_single_student_auc_drop':.03},
              'budget_cny':1,'usd_to_cny_planning':7,'all_visible_events_losslessly_verified':True}
        plan_root=output/'plans'/run_name
        write_json(plan_root/'plan.json',plan)
        finish_stage(plan_root,'r2_cost_candidate_plan',{'plan_sha256':file_hash(plan_root/'plan.json')},['plan.json'],[task['task_id']])
        print('START',run_name,'characters',chars,flush=True)
        execution=await infer(config,prepared,run,budget)
        verify_run(run)
        metrics=evaluate(prepared,run)
        with (run/'per_student_standard_v1.csv').open(encoding='utf-8-sig') as f:
            new_students={r['student_id']:float(r['auc']) for r in csv.DictReader(f)}
        maximum_drop=max(old_students[k]-v for k,v in new_students.items())
        delta={'overall_auc':metrics['overall']['auc']-reference['overall']['auc'],
               'student_mean_auc':metrics['student_auc']['mean']-reference['student_auc']['mean'],
               'logloss':metrics['overall']['logloss']-reference['overall']['logloss'],
               'maximum_student_drop':maximum_drop}
        passes=(metrics['complete'] and delta['overall_auc']>=-.01 and delta['student_mean_auc']>=-.01
                and delta['logloss']<=.02 and maximum_drop<=.03)
        predictions=read_jsonl(run/'predictions.jsonl')
        usage={'input_tokens':sum(r['usage']['prompt_tokens'] for r in predictions),
               'output_tokens':sum(r['usage']['completion_tokens'] for r in predictions),
               'cached_tokens':sum(r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0) for r in predictions)}
        known,reserved=budget.totals()
        result={'candidate':run_name,'metrics':metrics,'delta_vs_r2':delta,'passes_effect_gate':passes,
                'usage':usage,'execution':execution,'shared_known_usd':str(known),'shared_reserved_usd':str(reserved)}
        write_json(run/'optimization_result.json',result)
        write_json(run/'cost_evaluation'/'result.json',result)
        finish_stage(run/'cost_evaluation','r2_cost_evaluation',{'result_sha256':file_hash(run/'optimization_result.json')},['result.json'],[task['task_id']])
        print('RESULT',run_name,{'overall':metrics['overall'],'student_auc':metrics['student_auc'],
                               'passes_gate':passes,'deltas':delta,'usage':usage,
                               'known_cny':str(known*7),'run_cost_cny':execution['known_cost_usd']*7},flush=True)
        if not metrics['complete']:raise ValueError('Incomplete evaluation; no cost or quality claim permitted')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('names',nargs='+')
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--integer-output',action='store_true')
    p.add_argument('--decimal-output',action='store_true')
    p.add_argument('--student-serial',action='store_true')
    a=p.parse_args()
    if a.integer_output and a.decimal_output:p.error('Choose one output encoding')
    asyncio.run(main(a.names,a.concurrency,a.max_tokens,a.integer_output,a.student_serial,a.decimal_output))
