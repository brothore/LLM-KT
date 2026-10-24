"""Counterbalanced paired-event KT experiments; independent CNY3 ledger."""
import argparse,asyncio,copy,shutil,math
from pathlib import Path
from decimal import Decimal
import numpy as np
from llm_kt.common import read_json,write_json,read_jsonl,file_hash,digest
from llm_kt.artifacts import finish_stage,verify_stage
from llm_kt.budget import Budget
from llm_kt.config import load_config
from llm_kt.protocols import load_prepared
from llm_kt.providers import make_provider
from llm_kt.prompts import load_prompt,decode_visible
from llm_kt.decision_prompts import load_decision_prompt,recover_facts
from llm_kt.kt_features import evidence
from llm_kt.runtime import infer,verify_run
from llm_kt.evaluation import evaluate,calculate
from llm_kt.complement import fuse,logit
from llm_kt.costs import input_characters
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/complement_20261019_v1'
PREP=ROOT/'data/prepared/nips_validation_iteration_02cny_v1'
LIMIT=str(Decimal(3)/7)
METHODS=['arithmetic','logit','normalized_support']

def setup():
 plan={'budget_cny':3,'planning_usd_to_cny':7,'students':4,'targets':492,'rounds':2,'methods':METHODS,'primary_method':'logit','primary_comparator':'each single orientation; repeated-same-orientation ensemble at equal request count','coefficient_fitting':False,'no_ground_truth_in_fusion':True,'qwen_probability':'verbalized decimal, not token logits','jev_probability':'native noul','interpretation':'Polarity symmetrization under assumptions, not guaranteed unbiased or calibrated','paper':'https://arxiv.org/html/2505.20416v1#S4'}
 if (STAGE/'plan/plan.json').exists():assert read_json(STAGE/'plan/plan.json')==plan
 else:
  write_json(STAGE/'plan/plan.json',plan);finish_stage(STAGE/'plan','complement_plan',plan,['plan.json'])
  files=[]
  for src in list((ROOT/'src/llm_kt').rglob('*.py'))+[Path(__file__)]:
   rel=Path('package')/src.relative_to(ROOT/'src/llm_kt') if src.is_relative_to(ROOT/'src/llm_kt') else Path(src.name)
   dst=STAGE/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);files.append(rel.as_posix())
  finish_stage(STAGE/'source_snapshot','complement_source',{},files)

def config_for(model,event):
 path='configs/experiments/'+('nips_jev_optimized_validation.yaml' if model=='jev' else 'nips_validation_iteration_02cny_v1_r2_cost_optimized.yaml')
 cfg=load_config(ROOT/path);field='decision' if model=='jev' else 'prompt'
 template=ROOT/f'configs/complement/{model}_{event}.yaml'
 cfg['model'][field+'_file']=str(template)
 cfg['model'][field+'_snapshot']=(load_decision_prompt if model=='jev' else load_prompt)(template)
 cfg['execution'].update(total_timeout_seconds=30,max_attempts=2,max_requests=984)
 return cfg

async def run(round_number,resume=False):
 setup();budget=Budget(STAGE/'budget.json',LIMIT)
 if not budget.path.exists():write_json(budget.path,budget.data)
 # Reverse polarity dispatch order on repetition. Never issue both in a shared conversation.
 for model in ['qwen','jev']:
  for event in (['positive','negative'] if round_number==1 else ['negative','positive']):
   name=f'{model}_r{round_number}_{event}';dest=STAGE/name
   existing=dest.exists()
   if existing:
    if not resume:raise ValueError('Existing run; audit before any resume, do not overwrite')
    verify_run(dest)
    previous=read_json(dest/'paired_run_summary.json')
    if previous['metrics']['complete']:
     print('VERIFIED_COMPLETE',name,flush=True);continue
    recovery=dest/'bounded_recovery'
    if recovery.exists() and previous['execution']['attempts_this_invocation']!=0:raise ValueError('Only one paid bounded recovery invocation allowed per run')
    if previous['execution']['fatal_http_error'] or previous['execution']['budget_exhausted']:raise ValueError('Do not auto-recover fatal/budget stop')
    if recovery.exists() and (recovery/'artifact_manifest.json').exists() and not (recovery/'original_artifact_manifest.json').exists():
     shutil.copyfile(recovery/'artifact_manifest.json',recovery/'original_artifact_manifest.json')
    if not recovery.exists():write_json(recovery/'previous_summary.json',previous)
    write_json(recovery/('plan_amendment.json' if (recovery/'plan.json').exists() else 'plan.json'),{'reason':'Complete missing targets after transient transport or HTTP 529 failures','missing':previous['execution']['n_unfinished'],'max_additional_attempts_per_missing_target':1,'same_payload_config':True,'unknown_reservations_retained':True,'script_sha256':file_hash(Path(__file__))})
    shutil.copyfile(Path(__file__),recovery/('runner_amendment.py' if (recovery/'runner.py').exists() else 'runner.py'))
    for source in (ROOT/'src/llm_kt').rglob('*.py'):
     target=recovery/'source'/source.relative_to(ROOT/'src/llm_kt');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    finish_stage(recovery,'bounded_recovery_plan',{},[p.relative_to(recovery).as_posix() for p in recovery.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'])
   cfg=config_for(model,event);task,targets,inputs=load_prepared(PREP,cfg);assert len(targets)==492 and task['n_students']==4
   provider=make_provider(cfg['model']);hashes=[];chars=0
   for item in inputs:
    payload=provider.request(item);hashes.append(digest(payload));chars+=input_characters(payload,cfg['model']['adapter'])
    state=recover_facts(payload['state']) if model=='jev' else decode_visible(payload['messages'][1]['content'],cfg['model']['prompt_snapshot'])
    assert state['history']==item.history and state['current_item']==item.current_item
    assert state['evidence']==evidence(item,cfg['model']['decision_snapshot' if model=='jev' else 'prompt_snapshot'])
   known,reserved=budget.totals();upper=Decimal(str(chars/2*(.042 if model=='jev' else .05)/1e6*1.5 + (.005 if model=='qwen' else 0)))
   if known+reserved+upper>Decimal(LIMIT):
    write_json(STAGE/'admission_stop.json',{'next':name,'known':str(known),'reserved':str(reserved),'upper':str(upper)});print('STOP full-round admission',name,flush=True);return
   plan={'name':name,'config':cfg,'task_id':task['task_id'],'payload_hashes':hashes,'all_facts_verified':True,'source_prepared_sha256':file_hash(PREP/'manifest.json'),'upper_usd':str(upper),'script_sha256':file_hash(Path(__file__))}
   if existing:
    frozen=read_json(STAGE/'round_plans'/name/'plan.json')
    assert frozen['config']==cfg and frozen['payload_hashes']==hashes
   else:
    write_json(STAGE/'round_plans'/name/'plan.json',plan)
    shutil.copyfile(Path(__file__),STAGE/'round_plans'/name/'runner.py')
    finish_stage(STAGE/'round_plans'/name,'complement_run_plan',{},['plan.json','runner.py'])
   print('START',name,'upper_cny',str(upper*7),flush=True)
   result=await infer(cfg,PREP,dest,budget,allow_code_change=existing,retry_failed_once=existing);verify_run(dest);m=evaluate(PREP,dest)
   write_json(dest/'paired_run_summary.json',{'name':name,'execution':result,'metrics':m})
   print('RESULT',name,'auc',m['overall']['auc'],'student',m['student_auc']['mean'],'cost_cny',result['known_cost_usd']*7,flush=True)
   if not m['complete']:raise ValueError('Incomplete round; do not use a selected common subset')
  summarize(model,round_number)

def summarize(model,upto):
 _,targets,_=load_prepared(PREP);ids=[t['sample_id'] for t in targets];rows={};reports={};out=STAGE/f'analysis/{model}_r{upto}'
 if (out/'artifact_manifest.json').exists():
  verify_stage(out);print('VERIFIED_ANALYSIS',model,upto,flush=True);return
 def score(name,ps):
  predictions=[{'sample_id':sid,'status':'success','p_correct':float(p)} for sid,p in zip(ids,ps)]
  m,students=calculate(targets,predictions,{'threshold':.5,'student_std_ddof':1})
  # Independent pairwise rank checks within students.
  idx=dict(zip(ids,ps))
  for s in students:
   ts=[t for t in targets if t['student_id']==s['student_id']];pos=[idx[t['sample_id']] for t in ts if t['y_true']];neg=[idx[t['sample_id']] for t in ts if not t['y_true']]
   auc=sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg));assert abs(auc-s['auc'])<1e-12
  write_json(out/(name+'.json'),{'metrics':m,'students':students,'predictions':predictions});reports[name]=m;rows[name]=list(map(float,ps))
 diagnostics=[];sources={}
 for r in range(1,upto+1):
  loaded={}
  for e in ['positive','negative']:
   path=STAGE/f'{model}_r{r}_{e}';pred=read_jsonl(path/'predictions.jsonl');idx={p['sample_id']:p for p in pred};assert len(idx)==492 and set(idx)==set(ids)
   loaded[e]=np.array([idx[i]['p_correct'] for i in ids]);sources[path.name]=file_hash(path/'predictions.jsonl')
  a=loaded['positive'];b=1-loaded['negative']
  score(f'r{r}_positive',a);score(f'r{r}_negative_aligned',1-b)
  for method in METHODS:score(f'r{r}_paired_{method}',[fuse(x,y,method) for x,y in zip(a,b)])
  d=a+b-1;diagnostics.append({'round':r,'mean_sum_minus_one':float(np.mean(d)),'mean_abs_disagreement':float(np.mean(abs(d))),'p95_abs_disagreement':float(np.quantile(abs(d),.95)),'mean_logit_yes_bias':float(np.mean([(logit(x)+logit(y))/2 for x,y in zip(a,b)]))})
 if upto==2:
  for e in ['positive','negative_aligned']:
   score('same_orientation_repeat_mean_'+e,(np.array(rows['r1_'+e])+rows['r2_'+e])/2)
   score('same_orientation_repeat_logit_'+e,[fuse(x,1-y,'logit') for x,y in zip(rows['r1_'+e],rows['r2_'+e])])
  for method in METHODS:score('two_pairs_'+method,(np.array(rows['r1_paired_'+method])+rows['r2_paired_'+method])/2)
 write_json(out/'summary.json',{'model':model,'rounds':upto,'reports':reports,'diagnostics':diagnostics,'source_prediction_sha256':sources,'no_labels_in_pooling':True})
 finish_stage(out,'complement_analysis',{'sources':sources},[p.name for p in out.glob('*.json')])
 print('ANALYSIS',model,upto,{n:round(m['overall']['auc'],6) for n,m in reports.items()},flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--round',type=int,choices=[1,2]);p.add_argument('--analyze',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.analyze:
  for model in ['qwen','jev']:summarize(model,a.round or 2)
 else:asyncio.run(run(a.round or 1,a.resume))
