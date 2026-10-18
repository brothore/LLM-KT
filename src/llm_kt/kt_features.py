"""Shared KT facts derived from allowed history and frozen training priors."""
from functools import lru_cache
from .common import file_hash, read_json


def counts(events):
    n = len(events)
    k = sum(int(e['correct']) for e in events)
    return {'n': n, 'correct': k, 'rate': round(k / n, 5) if n else None}


@lru_cache(maxsize=8)
def training_priors(path, checksum):
    if file_hash(path) != checksum:
        raise ValueError('Training prior artifact changed')
    return read_json(path)


def evidence(item, prompt):
    history, current = item.history, item.current_item
    concepts = set(current['concept_ids'])
    shared = [h for h in history if concepts.intersection(h['concept_ids'])]
    same = [h for h in history if h['question_id'] == current['question_id']]
    result = {'overall': counts(history), 'recent_10': counts(history[-10:]), 'recent_20': counts(history[-20:]),
              'shared_concept': counts(shared), 'recent_shared_10': counts(shared[-10:]),
              'same_question': counts(same), 'last_same_question_correct': same[-1]['correct'] if same else None,
              'per_current_concept': {c: counts([h for h in history if c in h['concept_ids']]) for c in sorted(concepts)}}
    if prompt.get('evidence') == 'training':
        prior = training_priors(prompt['prior_file'], prompt['prior_sha256'])
        q = prior['questions'].get(current['question_id'])
        cs = [prior['concepts'][c] for c in current['concept_ids'] if c in prior['concepts']]
        result['training_population'] = {'global': prior['global'], 'question': q, 'concepts': cs,
                                         'smoothing_strength': prior['smoothing_strength']}
    return result
