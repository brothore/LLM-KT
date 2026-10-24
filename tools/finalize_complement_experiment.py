"""Offline audit and public aggregate export for complementary-event experiments."""
from pathlib import Path
from decimal import Decimal
import csv
from llm_kt.common import read_json,read_jsonl,write_json,file_hash,digest
from llm_kt.artifacts import verify_stage,finish_stage
from llm_kt.runtime import verify_run
from llm_kt.budget import Budget
from llm_kt.protocols import load_prepared
from llm_kt.providers import make_provider
from llm_kt.prompts import decode_visible
from llm_kt.decision_prompts import recover_facts
from llm_kt.kt_features import evidence
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT/'runs/complement_20261019_v1'

def main():
 verify_stage(STAGE/'plan');verify_stage(STAGE/'source_snapshot')
 ledger=read_json(STAGE/'budget.json');known,reserved=Budget(STAGE/'budget.json',str(Decimal(3)/7)).totals()
 assert known+reserved<=Decimal(ledger['limit_usd'])
 attempts={};runs=[];checks=0;analysis={};payload_checks={}; visible_payloads={}; replayed=0; remote_ids=[]
 for model in ['qwen','jev']:
  for r in [1,2]:
   for event in ['positive','negative']:
    name=f'{model}_r{r}_{event}';run=STAGE/name
    verify_run(run);verify_stage(STAGE/'round_plans'/name)
    if (run/'bounded_recovery').exists():verify_stage(run/'bounded_recovery')
    summary=read_json(run/'paired_run_summary.json');assert summary['metrics']['complete'] and summary['metrics']['coverage']['n_success']==492
    plan=read_json(STAGE/'round_plans'/name/'plan.json');cfg=plan['config'];_,targets,inputs=load_prepared(run/'prepared',cfg);provider=make_provider(cfg['model'])
    assert [digest(provider.request(x)) for x in inputs]==plan['payload_hashes']
    for item in inputs:
     payload=provider.request(item)
     facts=recover_facts(payload['state']) if model=='jev' else decode_visible(payload['messages'][1]['content'],cfg['model']['prompt_snapshot'])
     assert facts['history']==item.history and facts['current_item']==item.current_item
     assert facts['evidence']==evidence(item,cfg['model']['decision_snapshot' if model=='jev' else 'prompt_snapshot'])
    payload_checks[name]=plan['payload_hashes']
    visible_payloads[name]=[provider.request(x)['state'] if model=='jev' else provider.request(x)['messages'][1]['content'] for x in inputs]
    for pred in read_jsonl(run/'predictions.jsonl'):
     raw=read_json(run/'responses'/(pred['local_request_id']+'.json'))
     assert abs(provider.probability(raw)-pred['p_correct'])<1e-12
     if event=='negative':assert 'probability_transform' in pred
     if raw.get('id'):remote_ids.append(raw['id'])
     replayed+=1
    for p in (run/'attempts').glob('*.json'):
     a=read_json(p);assert p.stem not in attempts and a['status']!='running'
     attempts[p.stem]={'run':name,'status':a['status'],'error':a.get('error'),'sha256':file_hash(p)}
    runs.append(summary)
  for r in [1,2]:assert visible_payloads[f'{model}_r{r}_positive']==visible_payloads[f'{model}_r{r}_negative']
  for event in ['positive','negative']:assert payload_checks[f'{model}_r1_{event}']==payload_checks[f'{model}_r2_{event}']
  for r in [1,2]:
   path=STAGE/f'analysis/{model}_r{r}';verify_stage(path)
   for f in path.glob('*.json'):
    if f.name in {'summary.json','artifact_manifest.json'}:continue
    data=read_json(f);pred={p['sample_id']:p['p_correct'] for p in data['predictions']};assert len(pred)==492
    for s in data['students']:
     ts=[t for t in targets if t['student_id']==s['student_id']];pos=[pred[t['sample_id']] for t in ts if t['y_true']];neg=[pred[t['sample_id']] for t in ts if not t['y_true']]
     auc=sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg));assert abs(auc-s['auc'])<1e-12;checks+=1
    pos=[pred[t['sample_id']] for t in targets if t['y_true']];neg=[pred[t['sample_id']] for t in targets if not t['y_true']]
    auc=sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg));assert abs(auc-data['metrics']['overall']['auc'])<1e-12
   analysis[f'{model}_r{r}']=read_json(path/'summary.json')
 assert set(attempts)==set(ledger['attempts'])
 for key,val in ledger['attempts'].items():
  metadata=read_json(STAGE/attempts[key]['run']/'attempts'/(key+'.json'))
  assert val.get('cost')==metadata.get('usage',{}).get('cost')
 unknown=[{'attempt_id':key,**attempts[key],'reserved_usd':val['reserved']} for key,val in ledger['attempts'].items() if val.get('cost') is None]
 result={'n_attempts':len(attempts),'n_complete_runs':len(runs),'n_predictions':sum(r['execution']['n_success'] for r in runs),'runs':runs,'analysis':analysis,'unknown_attempts':unknown,'all_budget_attempts_linked':True,'independent_student_auc_checks':checks,'repeated_same_orientation_payloads_equal':True,'all_visible_facts_verified':True,'opposite_queries_visible_facts_equal':True,'raw_response_probability_replays':replayed,'reported_remote_ids':len(remote_ids),'unique_reported_remote_ids':len(set(remote_ids)),'known_cny':str(known*7),'reserved_cny':str(reserved*7),'total_with_reserve_cny':str((known+reserved)*7),'limit_cny':3,'no_more_paid_calls':True}
 write_json(STAGE/'final_acceptance/summary.json',result)
 finish_stage(STAGE/'final_acceptance','complement_final_acceptance',{'ledger_sha256':file_hash(STAGE/'budget.json')},['summary.json'])
 public={k:v for k,v in result.items() if k not in {'runs','unknown_attempts'}}
 public['unknown_attempt_count']=len(unknown)
 public['runs']=[{'name':r['name'],'metrics':r['metrics']['overall'],'student_auc':r['metrics']['student_auc'],'known_cost_cny':r['execution']['known_cost_usd']*7,'unknown_attempts':r['execution']['unknown_cost_attempts']} for r in runs]
 public['local_acceptance_sha256']=file_hash(STAGE/'final_acceptance/summary.json')
 write_json(ROOT/'docs/complement/验收摘要.json',public)
 print({k:result[k] for k in ['n_complete_runs','n_predictions','known_cny','reserved_cny','total_with_reserve_cny','independent_student_auc_checks']})
if __name__=='__main__':main()
