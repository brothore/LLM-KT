"""Two-request Decisions adaptation smoke; independent of Qwen experiments and budgets."""
import asyncio
from dataclasses import asdict
from pathlib import Path
from llm_kt.artifacts import finish_stage
from llm_kt.budget import Budget
from llm_kt.common import read_json, read_jsonl, write_json, write_jsonl, digest, file_hash
from llm_kt.config import load_config, task_settings
from llm_kt.evaluation import evaluate
from llm_kt.protocols import load_prepared
from llm_kt.providers import make_provider
from llm_kt.runtime import infer, verify_run


async def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'runs/jev_r2_smoke_20261018_v1'
    if output.exists():raise ValueError('Smoke run already exists; inspect evidence instead of blindly resending')
    config_path=root/'configs/experiments/nips_jev_r2.yaml'
    config=load_config(config_path)
    source=root/'data/prepared/nips_validation_iteration_02cny_v1'
    manifest,targets,inputs=load_prepared(source,config)
    # Refactoring shared features must not change any frozen Qwen R2 request.
    reference=root/'runs/four_student_revalidation_20261018_v1/round2'
    qwen_model=read_json(reference/'manifest.json')['identity']['config']['model']
    qwen=make_provider(qwen_model)
    old_predictions={p['sample_id']:p for p in read_jsonl(reference/'predictions.jsonl')}
    for item in inputs:
        p=old_predictions[item.sample_id]
        assert qwen.request(item)==read_json(reference/'requests'/f"{p['local_request_id']}.json")
    students=sorted({t['student_id'] for t in targets})[:2]
    selected=[next(i for i,t in enumerate(targets) if t['student_id']==uid and t['event_index']==index)
              for uid,index in zip(students,[20,80])]
    identity={'source_task_id':manifest['task_id'],'purpose':'two_target_interface_smoke_not_evaluation_dataset',
              'selection':'first two source student IDs, event indices 20 and 80; no label-based selection',
              'source_sample_ids':[targets[i]['sample_id'] for i in selected]}
    task_id=digest(identity)
    smoke_targets,smoke_inputs=[],[]
    provider=make_provider(config['model'])
    for i in selected:
        sample_id=digest([task_id,targets[i]['sample_id']])
        smoke_targets.append({**targets[i],'sample_id':sample_id,'source_sample_id':targets[i]['sample_id']})
        smoke_inputs.append({**asdict(inputs[i]),'sample_id':sample_id})
        payload=provider.request(inputs[i])
        original_state=__import__('json').loads(qwen.request(inputs[i])['messages'][1]['content'])
        for field in ('history','current_item','evidence'):assert payload['state'][field]==original_state[field]
        assert set(payload)=={'model','state','questions','provider'}
        assert 'correct' not in payload['state']['current_item']
    prepared=root/'data/prepared/jev_r2_smoke_20261018_v1'
    if prepared.exists():raise ValueError('Smoke prepared destination exists; choose a new version')
    write_jsonl(prepared/'targets.jsonl',smoke_targets)
    write_jsonl(prepared/'inputs.jsonl',smoke_inputs)
    write_jsonl(prepared/'identity_map.jsonl',[r for r in read_jsonl(source/'identity_map.jsonl') if r['student_id'] in students])
    files={n:file_hash(prepared/n) for n in ('targets.jsonl','inputs.jsonl','identity_map.jsonl')}
    smoke_manifest={**manifest,'task_id':task_id,'n_targets':2,'n_students':2,'input_hash':digest(smoke_inputs),
                    'settings':task_settings(config),'files':files,'scope':identity}
    write_json(prepared/'manifest.json',smoke_manifest)
    finish_stage(prepared,'prepare',identity,[*files,'manifest.json'],[manifest['task_id']])
    plan={'config':config,'purpose':'two-request smoke only; no KT quality claims',
          'budget_limit_usd':'0.01','max_remote_attempts':2,'source_prepared_manifest_sha256':file_hash(source/'manifest.json'),
          'decision_file_sha256':file_hash(config['model']['decision_file']),
          'script_sha256':file_hash(Path(__file__)),'qwen_full_request_regression_checks':492,
          'documents':['https://docs.typesafe.ai/concepts/how-to-build-with-system-one',
                       'https://docs.typesafe.ai/primitives/noul',
                       'https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README.md'],
          'no_chat_prompt_dependency':True,'template':config['model']['decision_snapshot']['version']}
    write_json(output/'plan'/'plan.json',plan)
    finish_stage(output/'plan','jev_smoke_plan',{'plan_sha256':file_hash(output/'plan'/'plan.json')},['plan.json'],[task_id])
    budget=Budget(output/'budget.json','0.01')
    write_json(budget.path,budget.data)
    execution=await infer(config,prepared,output/'inference',budget)
    verification=verify_run(output/'inference')
    metrics=evaluate(prepared,output/'inference')
    known,reserved=budget.totals()
    predictions=read_jsonl(output/'inference'/'predictions.jsonl')
    result={'execution':execution,'verification':verification,'metrics':metrics,
            'known_usd':str(known),'reserved_usd':str(reserved),'interface_complete':execution['n_success']==2,
            'qwen_request_regression_passed':492,'quality_evaluation':False,
            'predictions':[{'p_correct':p['p_correct'],'provider':p['provider'],'usage':p['usage']} for p in predictions]}
    write_json(output/'summary.json',result)
    finish_stage(output,'jev_r2_smoke',{'task_id':task_id,'decision_sha256':plan['decision_file_sha256']},
                 ['summary.json','budget.json','plan/artifact_manifest.json'],[task_id])
    print({'n_success':execution['n_success'],'attempts':execution['attempts_this_invocation'],
           'known_usd':str(known),'reserved_usd':str(reserved),
           'probabilities':[p['p_correct'] for p in predictions],'regression_checked_qwen_requests':492},flush=True)
    if execution['n_success']!=2:raise ValueError('Smoke incomplete; retain all evidence and do not claim success')


if __name__=='__main__':asyncio.run(main())
