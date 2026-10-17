"""Load derived target lists; preserve original history and parent event identity."""
from pathlib import Path
from .common import file_hash, read_jsonl


def target_selection(snapshot, config):
    entry = snapshot.get('target_selection')
    if not entry:
        return None
    expected = {k: config['task'][k] for k in ('protocol', 'level', 'maxlen', 'min_seq_len')}
    if entry['protocol'] != expected:
        raise ValueError('Derived target list requires the protocol used during sampling')
    root = Path(config['dataset']['snapshot']).resolve()
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root) or file_hash(path) != entry['sha256']:
        raise ValueError('Target selection file/hash mismatch')
    records = read_jsonl(path)
    result = {}
    for row in records:
        key = (row['student_id'], row['event_index'])
        if key in result:
            raise ValueError('Duplicate derived target')
        result[key] = row
    limit = config['task'].get('target_limit')
    if limit:
        fold = config['dataset']['validation_fold']
        split = config['dataset']['split']
        eligible = [r for r in records if (r['fold'] == -1 if split == 'test' else r['fold'] >= 0 and ((r['fold'] == fold) == (split == 'validation')))]
        # Smoke selection: alternate label pools, median history first, stable tie-break.
        pools = {label: sorted([r for r in eligible if r['label'] == label],
                               key=lambda r: (abs(r['history_length'] - 100), r['source_event_id'])) for label in (0, 1)}
        chosen = []
        while len(chosen) < limit and any(pools.values()):
            for label in (0, 1):
                if pools[label] and len(chosen) < limit:
                    chosen.append(pools[label].pop(0))
        result = {(r['student_id'], r['event_index']): r for r in chosen}
    return result
