"""Fresh, sequential B0/R1/R2/R3 evaluation under a shared RMB 0.753 plan."""
import asyncio
from decimal import Decimal
from pathlib import Path
from llm_kt.artifacts import finish_stage
from llm_kt.budget import Budget
from llm_kt.common import write_json, file_hash
from llm_kt.config import load_config
from llm_kt.protocols import load_prepared
from llm_kt.runtime import infer, verify_run
from llm_kt.evaluation import evaluate, compare


async def main():
    root = Path(__file__).resolve().parents[1]
    output = root/'runs/four_student_revalidation_20261018_v1'
    if output.exists():
        raise ValueError('Fresh experiment requires a new destination; inspect existing evidence before resuming')
    output.mkdir(parents=True)
    prepared = root/'data/prepared/nips_validation_iteration_02cny_v1'
    limit = str(Decimal('0.753')/Decimal(7))
    budget = Budget(output/'budget.json', limit)
    write_json(budget.path,budget.data)
    configs = {}
    for name in ('baseline','round1','round2','round3'):
        config = load_config(root/f'configs/experiments/nips_{name}.yaml')
        config['dataset']['snapshot'] = str(root/'data/snapshots/nips_validation_iteration_02cny_v1')
        config['execution']['max_requests'] = 10000
        manifest, targets, _ = load_prepared(prepared,config)
        assert manifest['n_students']==4 and len(targets)==492
        configs[name] = config
        write_json(output/'plan'/f'{name}.json', config)
    plan = {'budget_cny':.753,'usd_to_cny_planning':7,'limit_usd':limit,
            'fresh_run':True,'reuse_previous_predictions':False,'targets_each':492,
            'order':list(configs),'task_id':manifest['task_id'],
            'script_sha256':file_hash(Path(__file__)),'source_prepared_manifest_sha256':file_hash(prepared/'manifest.json'),
            'note':'Provider-side implicit prompt caching may occur; no imported local predictions. No prompt changes.'}
    write_json(output/'plan'/'plan.json',plan)
    finish_stage(output/'plan','frozen_revalidation_plan',plan,['plan.json']+[n+'.json' for n in configs],[manifest['task_id']])
    results=[]
    for name,config in configs.items():
        print('START',name,flush=True)
        try:
            execution = await infer(config,prepared,output/name,budget)
        except Exception as exc:
            known,reserved=budget.totals()
            write_json(output/'interruption.json',{'prompt':name,'exception_type':type(exc).__name__,
                       'known_usd':str(known),'reserved_usd':str(reserved),
                       'budget_exceeded':str(exc)=='Experiment budget would be exceeded'})
            raise
        verification = verify_run(output/name)
        metrics = evaluate(prepared,output/name)
        known,reserved=budget.totals()
        row={'prompt':name,'execution':execution,'metrics':metrics,'verification':verification,
             'shared_known_usd':str(known),'shared_reserved_usd':str(reserved)}
        write_json(output/name/'revalidation_summary.json',row)
        results.append(row)
        write_json(output/'results_so_far.json',results)
        print('COMPLETE',name,{'coverage':metrics['coverage'],'overall':metrics['overall'],
                             'student_auc':metrics['student_auc'],'known_usd':str(known),
                             'reserved_usd':str(reserved)},flush=True)
        if not metrics['complete']:
            raise ValueError('Incomplete prompt run; stop comparison rather than report partial data as complete')
    compare(prepared,[output/n for n in configs],output/'comparison.json')
    known,reserved=budget.totals()
    summary={'results':results,'known_usd':str(known),'reserved_usd':str(reserved),
             'known_cny_planning':str(known*7),'complete':True,'budget_usd':limit}
    write_json(output/'summary.json',summary)
    finish_stage(output,'four_student_revalidation',plan,['summary.json','comparison.json','budget.json','results_so_far.json'],[manifest['task_id']])
    print('FINISHED',{'known_usd':str(known),'known_cny':str(known*7),'reserved_usd':str(reserved)},flush=True)


if __name__=='__main__':
    asyncio.run(main())
