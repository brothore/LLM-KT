"""Offline Jev iteration closure; no API calls."""
import csv
from decimal import Decimal
from pathlib import Path
from llm_kt.common import read_json,read_jsonl,write_json,file_hash,digest
from llm_kt.providers import make_provider
from llm_kt.config import load_config
from llm_kt.runtime import verify_run
from llm_kt.protocols import load_prepared
from llm_kt.budget import Budget
from llm_kt.artifacts import finish_stage,verify_stage

def main():
    root=Path(__file__).resolve().parents[1]; stage=root/'runs/jev_iterations_20261018_v1'
    verify_stage(stage/'plan');verify_stage(stage/'source_snapshot')
    ledger=read_json(stage/'budget.json'); known,reserved=Budget(stage/'budget.json',ledger['limit_usd']).totals()
    assert known+reserved<=Decimal(ledger['limit_usd'])
    results=[]; attempts={}; checks=0; repeat_checks=[]
    for run in sorted(stage.iterdir()):
        if not (run/'round_summary.json').exists():continue
        verify_stage(stage/'round_plans'/run.name);verify_stage(run/'round_acceptance')
        verify_run(run); result=read_json(run/'round_summary.json')
        assert result['metrics']['complete']
        _,targets,_=load_prepared(run/'prepared')
        predictions=read_jsonl(run/'predictions.jsonl'); by_id={p['sample_id']:p for p in predictions}
        assert len(by_id)==492 and set(by_id)=={t['sample_id'] for t in targets}
        with (run/'per_student_standard_v1.csv').open(encoding='utf-8-sig') as f:students=list(csv.DictReader(f))
        for s in students:
            ts=[t for t in targets if t['student_id']==s['student_id']]
            pos=[by_id[t['sample_id']]['p_correct'] for t in ts if t['y_true']==1]
            neg=[by_id[t['sample_id']]['p_correct'] for t in ts if t['y_true']==0]
            auc=sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
            assert abs(auc-float(s['auc']))<1e-12;checks+=1
        for p in (run/'attempts').glob('*.json'):
            assert p.stem not in attempts
            a=read_json(p);assert a['status']!='running'
            attempts[p.stem]={'run':run.name,'status':a['status'],'error':a.get('error'),'sha256':file_hash(p)}
        plan=read_json(stage/'round_plans'/run.name/'plan.json')
        if run.name=='baseline_control':
            original_run=root/'runs/jev_r2_full_validation_20261018_v1/inference'
            original_model=read_json(original_run/'manifest.json')['identity']['config']['model']
            _,_,original_inputs=load_prepared(run/'prepared',plan['config'])
            original_provider=make_provider(original_model)
            assert plan['payload_hashes']==[digest(original_provider.request(x)) for x in original_inputs]
        if plan.get('repeat_of'):
            original=read_json(stage/'round_plans'/plan['repeat_of']/'plan.json')
            assert plan['payload_hashes']==original['payload_hashes']
            repeat_checks.append({'repeat':run.name,'source':plan['repeat_of'],'all_492_payloads_equal':True})
        allpos=[by_id[t['sample_id']]['p_correct'] for t in targets if t['y_true']==1]
        allneg=[by_id[t['sample_id']]['p_correct'] for t in targets if t['y_true']==0]
        overall=sum((a>b)+.5*(a==b) for a in allpos for b in allneg)/(len(allpos)*len(allneg))
        assert abs(overall-result['metrics']['overall']['auc'])<1e-12
        results.append(result)
    assert set(attempts)==set(ledger['attempts'])
    unknown=[{'attempt_id':k,**attempts[k],'reserved_usd':v['reserved']} for k,v in ledger['attempts'].items() if v.get('cost') is None]
    candidates=[r for r in results if r['round'].startswith('jev_iter_') and r['selection_eligible']]
    selected=max(candidates,key=lambda r:r['metrics']['overall']['auc'])['round'] if candidates else None
    recommended=root/'configs/experiments/nips_jev_optimized_validation.yaml'
    cfg=load_config(recommended)
    _,_,items=load_prepared(root/'data/prepared/nips_validation_iteration_02cny_v1',cfg)
    provider=make_provider(cfg['model'])
    assert [digest(provider.request(x)) for x in items]==read_json(stage/'round_plans'/selected/'plan.json')['payload_hashes']
    result={'recommended_config_sha256':file_hash(recommended),'recommended_payloads_equal_selected':True,'rounds':results,'selected':selected,'independent_student_auc_checks':checks,'repeat_checks':repeat_checks,'all_attempts_linked':True,'unknown_attempts':unknown,
        'known_cny':str(known*7),'reserved_cny':str(reserved*7),'total_with_reserve_cny':str((known+reserved)*7),'limit_cny':3,
        'no_more_paid_calls':True,'qualification':'Four development students; selection is not independent test confirmation.'}
    write_json(stage/'final_acceptance/summary.json',result)
    finish_stage(stage/'final_acceptance','jev_iterations_closure',{'ledger_sha256':file_hash(stage/'budget.json')},['summary.json'])
    print({k:v for k,v in result.items() if k not in {'rounds','unknown_attempts'}})
if __name__=='__main__':main()
