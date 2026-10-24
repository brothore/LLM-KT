"""Versioned KT prompts and evidence derived solely from permitted history."""
from pathlib import Path
import yaml
from .common import canonical, file_hash, assert_keys
from .kt_features import counts, training_priors, evidence


def load_prompt(path):
    path = Path(path).resolve()
    prompt = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert_keys(prompt, {'version', 'system', 'evidence', 'prior_file', 'history_encoding', 'evidence_encoding', 'output_event'}, 'prompt')
    if prompt.get('output_event','correct') not in {'correct','incorrect'}:raise ValueError('Unknown output event')
    if prompt.get('history_encoding', 'objects') not in {'objects', 'tuples_v1', 'rows_v1', 'numeric_tuples_v1'}:
        raise ValueError('Unknown history encoding')
    if prompt.get('evidence_encoding', 'objects') not in {'objects', 'packed_v1'}:
        raise ValueError('Unknown evidence encoding')
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


def pack_evidence(value):
    if isinstance(value, dict):
        keys = set(value)
        if keys == {'n', 'correct', 'rate'}:
            return [value['n'], value['correct'], value['rate']]
        if keys == {'n', 'correct', 'p'}:
            return [value['n'], value['correct'], value['p']]
        if keys == {'id', 'n', 'correct', 'p'}:
            return [value['id'], value['n'], value['correct'], value['p']]
        return {k:pack_evidence(v) for k,v in value.items()}
    if isinstance(value, list):
        return [pack_evidence(v) for v in value]
    return value


def unpack_evidence(value, key=''):
    if isinstance(value, dict):
        return {k:unpack_evidence(v,k) for k,v in value.items()}
    if isinstance(value, list):
        if key == 'concepts':
            return [dict(zip(('id','n','correct','p'),v)) for v in value]
        fields = ('id','n','correct','p') if key == 'question' else ('n','correct','p') if key == 'global' else ('n','correct','rate')
        return dict(zip(fields,value))
    return value


def decode_visible(text, prompt):
    """Restore exact input facts for offline encoding acceptance, not model reasoning."""
    import json
    data = json.loads(text)
    encoding = prompt.get('history_encoding', 'objects')
    if encoding in {'tuples_v1', 'rows_v1', 'numeric_tuples_v1'}:
        history = data['history']
        if encoding == 'rows_v1':
            history = [(q, cs.split(','), int(y)) for q,cs,y in (line.split('|') for line in history.splitlines())]
        restore = lambda x: str(x) if isinstance(x,int) else x
        data['history'] = [dict(question_id=restore(q),concept_ids=[restore(c) for c in cs],correct=bool(y)) for q,cs,y in history]
        data['current_item'] = dict(question_id=restore(data['current_item'][0]),concept_ids=[restore(c) for c in data['current_item'][1]])
    if prompt.get('evidence_encoding') == 'packed_v1' and 'evidence' in data:
        data['evidence'] = unpack_evidence(data['evidence'])
    return data


def render(item, prompt):
    # Raw history comes first to preserve potential prefix reuse, without changing facts.
    encoding = prompt.get('history_encoding')
    if encoding in {'tuples_v1', 'rows_v1', 'numeric_tuples_v1'}:
        history = [[h['question_id'], h['concept_ids'], int(h['correct'])] for h in item.history]
        current = [item.current_item['question_id'], item.current_item['concept_ids']]
        if encoding == 'numeric_tuples_v1':
            def numeric(value):
                return int(value) if isinstance(value,str) and value.isascii() and value.isdigit() and str(int(value))==value else value
            history = [[numeric(q),[numeric(c) for c in cs],y] for q,cs,y in history]
            current = [numeric(current[0]),[numeric(c) for c in current[1]]]
        if encoding == 'rows_v1':
            for q,cs,_ in history:
                if any(any(c in value for c in '|,\n\r') for value in [q,*cs]):
                    raise ValueError('rows_v1 requires delimiter-free identifiers')
            history = '\n'.join(f"{q}|{','.join(cs)}|{y}" for q,cs,y in history)
    else:
        history, current = item.history, item.current_item
    text = '{"history":' + canonical(history) + ',"current_item":' + canonical(current)
    if prompt.get('evidence', 'none') != 'none':
        facts = evidence(item, prompt)
        if prompt.get('evidence_encoding') == 'packed_v1':
            facts = pack_evidence(facts)
        text += ',"evidence":' + canonical(facts)
    text += '}'
    return [{'role': 'system', 'content': prompt['system']}, {'role': 'user', 'content': text}]
