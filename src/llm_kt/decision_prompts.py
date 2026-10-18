"""Independent typed-decision templates; never import or render Chat prompts."""
from pathlib import Path
import math
import yaml
from .common import assert_keys, file_hash
from .kt_features import evidence


def load_decision_prompt(path):
    path=Path(path).resolve()
    prompt=yaml.safe_load(path.read_text(encoding='utf-8'))
    assert_keys(prompt,{'version','instructions','criteria','evidence','prior_file','state_description','state_layout','reference_mode','history_encoding','positive_class','question_key'},'decision_prompt')
    if not prompt.get('version') or not isinstance(prompt.get('instructions'),(str,dict,list)) or not prompt['instructions']:
        raise ValueError('Decision prompt requires version and instructions')
    if not isinstance(prompt.get('question_key','will_answer_correctly'),str) or not prompt.get('question_key','will_answer_correctly'):
        raise ValueError('Decision question_key must be a nonempty string')
    for key,allowed,default in [('state_layout',{'r2','decision_card'},'r2'),('reference_mode',{'none','odds_v1'},'none'),('history_encoding',{'objects','table'},'objects'),('positive_class',{'correct','incorrect'},'correct')]:
        if prompt.get(key,default) not in allowed:raise ValueError('Unknown decision '+key)
    criteria=prompt.get('criteria',{})
    if set(criteria)!={'true','false'} or not all(isinstance(x,str) and x for x in criteria.values()):
        raise ValueError('Decision criteria require explicit true and false strings')
    if prompt.get('evidence','history') not in {'history','training'}:
        raise ValueError('Unknown decision evidence mode')
    if not isinstance(prompt.get('state_description',{}),dict):
        raise ValueError('Decision state description must be an object')
    if prompt.get('prior_file'):
        prior=(path.parent/prompt['prior_file']).resolve()
        prompt['prior_file']=str(prior)
        prompt['prior_sha256']=file_hash(prior)
    if prompt.get('evidence')=='training' and not prompt.get('prior_file'):
        raise ValueError('Training decision evidence needs frozen priors')
    return prompt


def statistical_reference(facts):
    population=facts.get('training_population')
    if not population:raise ValueError('odds_v1 requires frozen training evidence')
    g=population['global']['p']
    q=population['question']['p'] if population['question'] else sum(c['p'] for c in population['concepts'])/len(population['concepts']) if population['concepts'] else g
    r=(facts['recent_10']['correct']+2*g)/(facts['recent_10']['n']+2)
    u=(facts['overall']['correct']+4*g)/(facts['overall']['n']+4)
    logit=lambda p:math.log(min(.99,max(.01,p))/(1-min(.99,max(.01,p))))
    z=logit(q)+.7*(logit(r)-logit(g))+.3*(logit(u)-logit(g))
    p=1/(1+math.exp(-z))
    return {'training_global_p':g,'item_easiness_p':q,'student_recent_smoothed_p':round(r,6),
            'student_overall_smoothed_p':round(u,6),'student_adjusted_item_p':round(p,6),
            'student_adjusted_item_incorrect_p':round(1-p,6),
            'method':'Fixed odds_v1: logit(item)+0.7*(logit(recent)-logit(global))+0.3*(logit(overall)-logit(global)); not fitted on evaluation labels'}


def render_decision(item,prompt,strategy):
    facts=evidence(item,prompt)
    history=item.history
    if prompt.get('history_encoding')=='table':
        history={'columns':['question_id','concept_ids','correct'],
                 'rows':[[h['question_id'],h['concept_ids'],int(h['correct'])] for h in history]}
    if prompt.get('state_layout')=='decision_card':
        state={'current_item':item.current_item,'history':history,
               'learner':{k:v for k,v in facts.items() if k!='training_population'},
               'training_population':facts['training_population'],'interpretation':prompt.get('state_description',{})}
    else:
        state={'history':history,'current_item':item.current_item,
               'evidence':facts,'interpretation':prompt.get('state_description',{})}
    if prompt.get('reference_mode')=='odds_v1':state['statistical_reference']=statistical_reference(facts)
    correct_key='false' if prompt.get('positive_class')=='incorrect' else 'true'
    incorrect_key='true' if correct_key=='false' else 'false'
    criteria=prompt['criteria'] if strategy=='noul' else {'correct':prompt['criteria'][correct_key],'incorrect':prompt['criteria'][incorrect_key]}
    return state,{'type':strategy,'instructions':prompt['instructions'],'criteria':criteria}


def recover_facts(state):
    history=state['history']
    if isinstance(history,dict):
        if history['columns']!=['question_id','concept_ids','correct']:raise ValueError('Unknown decision history columns')
        history=[dict(question_id=q,concept_ids=cs,correct=bool(y)) for q,cs,y in history['rows']]
    facts=state['evidence'] if 'evidence' in state else {**state['learner'],'training_population':state['training_population']}
    return {'history':history,'current_item':state['current_item'],'evidence':facts}
