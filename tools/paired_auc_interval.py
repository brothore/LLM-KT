"""Paired student-cluster bootstrap for fixed, already selected predictions."""
import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from llm_kt.protocols import load_prepared
from llm_kt.common import read_jsonl, write_json


def interval(prepared, baseline, candidate, output, replicates=2000):
    _, targets, _ = load_prepared(prepared)
    b = {r['sample_id']:r['p_correct'] for r in read_jsonl(baseline)}
    c = {r['sample_id']:r['p_correct'] for r in read_jsonl(candidate)}
    if set(b)!=set(c) or set(b)!={t['sample_id'] for t in targets}:
        raise ValueError('Paired full-coverage predictions required')
    groups=defaultdict(list)
    for i,t in enumerate(targets):groups[t['student_id']].append(i)
    blocks=[np.asarray(v) for v in groups.values()]
    y=np.asarray([t['y_true'] for t in targets]);bp=np.asarray([b[t['sample_id']] for t in targets]);cp=np.asarray([c[t['sample_id']] for t in targets])
    rng=np.random.default_rng(20261018);scores=[]
    for _ in range(replicates):
        selected=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),size=len(blocks))])
        if len(set(y[selected]))<2:continue
        x=roc_auc_score(y[selected],bp[selected]);z=roc_auc_score(y[selected],cp[selected]);scores.append([x,z,z-x])
    sample=np.asarray(scores)
    result={'baseline_auc':float(roc_auc_score(y,bp)),'candidate_auc':float(roc_auc_score(y,cp)),
            'auc_difference':float(roc_auc_score(y,cp)-roc_auc_score(y,bp)),
            'baseline_95_percentile_interval':np.quantile(sample[:,0],[.025,.975]).tolist(),
            'candidate_95_percentile_interval':np.quantile(sample[:,1],[.025,.975]).tolist(),
            'paired_difference_95_percentile_interval':np.quantile(sample[:,2],[.025,.975]).tolist(),
            'bootstrap_unit':'student','n_students':len(blocks),'n_targets':len(targets),'replicates':len(scores),
            'limitations':'Exploratory percentile interval for fixed predictions; no correction for prompt selection on validation data'}
    write_json(output,result);print(result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['prepared','baseline','candidate','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();interval(a.prepared,a.baseline,a.candidate,a.output)
