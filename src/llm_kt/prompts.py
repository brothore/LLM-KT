"""Versioned KT prompts and evidence derived solely from permitted history."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import yaml
from .common import canonical, file_hash, read_json, assert_keys


def load_prompt(path):
    path = Path(path).resolve()
    prompt = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert_keys(prompt, {'version', 'system', 'evidence', 'prior_file'}, 'prompt')
    if not isinstance(prompt.get('system'), str) or not prompt['system'] or not prompt.get('version'):
        raise ValueError('Prompt requires version and system text')
    if prompt.get('evidence', 'none') not in {'none', 'history', 'training'}:
        raise ValueError('Unknown prompt evidence mode')
    if prompt.get('prior_file'):
        prior = (path.parent / prompt['prior_file']).resolve()
        prompt['prior_file'] = str(prior)
        prompt['prior_sha256'] = file_hash(prior)
    if prompt.get('evidence') == 'training' and not prompt.get('prior_file'):
        raise ValueError('Training evidence requires a frozen prior file')
    return prompt


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


def render(item, prompt):
    # Raw history comes first to preserve potential prefix reuse, without changing facts.
    text = '{"history":' + canonical(item.history) + ',"current_item":' + canonical(item.current_item)
    if prompt.get('evidence', 'none') != 'none':
        text += ',"evidence":' + canonical(evidence(item, prompt))
    text += '}'
    return [{'role': 'system', 'content': prompt['system']}, {'role': 'user', 'content': text}]
