"""Freeze item/concept priors from training folds only; no validation labels."""
import argparse
from collections import defaultdict
from pathlib import Path
from llm_kt.artifacts import finish_stage, verify_stage
from llm_kt.common import file_hash, read_json, write_json, digest
from llm_kt.data import iter_sequences


def build(snapshot, output, validation_fold=0, strength=20):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    manifest = read_json(snapshot / 'manifest.json')
    source = snapshot / manifest['files']['train_valid']['path']
    if file_hash(source) != manifest['files']['train_valid']['sha256']:
        raise ValueError('Training source changed')
    identity = {'snapshot_id':manifest['snapshot_id'],'source_sha256':file_hash(source),
                'excluded_validation_fold':validation_fold,'smoothing_strength':strength}
    if output.exists():
        artifact=verify_stage(output)
        if artifact['identity']!=identity:raise ValueError('Prior specification changed')
        return read_json(output/'priors.json')
    questions, concepts = defaultdict(lambda:[0,0]), defaultdict(lambda:[0,0])
    train_students, excluded_students=set(),set()
    n=k=0
    for row in iter_sequences(source,manifest['representation']):
        if row['fold']==validation_fold:
            excluded_students.add(row['student_id']);continue
        train_students.add(row['student_id'])
        for event in row['events']:
            n+=1;k+=event['response']
            questions[event['question_id']][0]+=1;questions[event['question_id']][1]+=event['response']
            for c in set(event['concept_ids']):
                concepts[c][0]+=1;concepts[c][1]+=event['response']
    if train_students & excluded_students:raise ValueError('Student leakage between training and validation')
    p=k/n
    def freeze(groups):
        return {key:{'id':key,'n':v[0],'correct':v[1],'p':round((v[1]+strength*p)/(v[0]+strength),6)} for key,v in groups.items()}
    result={'global':{'n':n,'correct':k,'p':round(p,6)},'questions':freeze(questions),'concepts':freeze(concepts),
            'smoothing_strength':strength,'provenance':identity,'n_training_students':len(train_students),
            'n_excluded_validation_students':len(excluded_students)}
    write_json(output/'priors.json',result)
    finish_stage(output,'training_only_prompt_priors',identity,['priors.json'],[manifest['snapshot_id']])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=build(a.snapshot,a.output)
    print('training_students',r['n_training_students'],'excluded_validation_students',r['n_excluded_validation_students'],'training_events',r['global']['n'])
