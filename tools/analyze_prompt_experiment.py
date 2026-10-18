"""Offline diagnostics for prompt iteration; reports aggregate groups only."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score
from llm_kt.common import read_jsonl, write_json
from llm_kt.protocols import load_prepared
from llm_kt.prompts import evidence


def analyze(prepared, run):
    _, targets, inputs = load_prepared(prepared)
    predictions = {p['sample_id']: p for p in read_jsonl(Path(run) / 'predictions.jsonl')}
    groups, baselines = defaultdict(list), defaultdict(list)
    labels = []
    for target, item in zip(targets, inputs):
        if target['sample_id'] not in predictions:
            continue
        y, p = target['y_true'], predictions[target['sample_id']]['p_correct']
        e = evidence(item, {'evidence': 'history'})
        labels.append(y)
        for name, feature in [('overall_history', e['overall']), ('shared_concept',e['shared_concept']), ('recent_10',e['recent_10'])]:
            baselines[name].append((feature['correct'] + 1) / (feature['n'] + 2))
        categories = ['history_short' if len(item.history)<50 else 'history_medium' if len(item.history)<150 else 'history_long',
                      'same_question_seen' if e['same_question']['n'] else 'same_question_unseen',
                      'concept_seen' if e['shared_concept']['n'] else 'concept_unseen']
        for name in categories:
            groups[name].append((y,p))
    result = {'n_predictions':len(predictions), 'n_unique_probabilities':len({p['p_correct'] for p in predictions.values()}),
              'frequent_probabilities': Counter(str(p['p_correct']) for p in predictions.values()).most_common(10),
              'history_only_baseline_auc': {name:float(roc_auc_score(labels, values)) for name,values in baselines.items()},
              'groups':{name:{'n':len(rows),'auc':float(roc_auc_score([r[0] for r in rows],[r[1] for r in rows])) if len({r[0] for r in rows})==2 else None,
                              'mean_prediction':sum(r[1] for r in rows)/len(rows),'positive_rate':sum(r[0] for r in rows)/len(rows)} for name,rows in groups.items()}}
    write_json(Path(run)/'diagnostics.json',result)
    print(result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--run',required=True)
    args=p.parse_args();analyze(args.prepared,args.run)
