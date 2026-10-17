"""Deterministic stratified target sampling with whole-history context retention."""
import csv
import heapq
import math
from collections import Counter, defaultdict
from pathlib import Path
from .common import digest, file_hash, read_json, write_json, write_jsonl
from .data import iter_sequences, import_snapshot
from .artifacts import finish_stage, verify_stage, record_event
from .vendor.pykt.sequence import target_ranges


def history_bin(length):
    return '1-9' if length < 10 else '10-49' if length < 50 else '50-99' if length < 100 else '100-199'


def allocate(counts, fraction):
    total = sum(counts.values())
    size = min(total, max(1, math.floor(total * fraction + 0.5)))
    quota = {s: math.floor(n * size / total) for s, n in counts.items()}
    remainder = sorted(counts, key=lambda s: (-(counts[s] * size / total - quota[s]), s))
    for s in remainder[:size - sum(quota.values())]:
        quota[s] += 1
    return quota


def create_subset(parent, output, dataset_name, fraction=0.001, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError('Sampling fraction must be in (0,1]')
    parent, output = Path(parent).resolve(), Path(output).resolve()
    source = read_json(parent / 'manifest.json')
    if source.get('target_selection') or source['representation'] != 'question_native':
        raise ValueError('Sampling currently requires an original question-native snapshot')
    protocol = {'protocol': 'pykt_window', 'level': 'question', 'maxlen': 200, 'min_seq_len': 3}
    identity = {'source_snapshot_id': source['snapshot_id'], 'dataset': dataset_name, 'fraction': fraction,
                'seed': seed, 'unit': 'prediction_target', 'protocol': protocol, 'algorithm': 'stratified_sha256_bottom_k_v1'}
    if output.exists():
        cached = verify_stage(output)
        if cached['identity'] != identity:
            raise ValueError('Subset destination belongs to another sampling specification')
        record_event(output, 'subset', cached['artifact_id'], True)
        return read_json(output / 'sampling_report.json')

    def candidates():
        for split in ('train_valid', 'test'):
            entry = source['files'][split]
            path = (parent / entry['path']).resolve()
            if not path.is_relative_to(parent) or file_hash(path) != entry['sha256']:
                raise ValueError('Parent snapshot failed validation')
            for row in iter_sequences(path, source['representation']):
                events = row['events']
                for start, index in target_ranges(len(events), 'pykt_window', 200, 3):
                    event = events[index]
                    length = index - start
                    stratum = f"{event['response']}:{history_bin(length)}:{int(len(event['concept_ids']) > 1)}"
                    event_id = digest([source['snapshot_id'], row['student_id'], index])
                    yield {'student_id': row['student_id'], 'event_index': index, 'fold': row['fold'],
                           'label': event['response'], 'history_length': length, 'stratum': stratum,
                           'source_event_id': event_id, 'source_row': row['source_row'],
                           'concept_ids': event['concept_ids']}
    counts = defaultdict(Counter)
    for row in candidates():
        counts[row['fold']][row['stratum']] += 1
    quotas = {fold: allocate(group, fraction) for fold, group in counts.items()}
    heaps = defaultdict(list)
    for row in candidates():
        key = row['fold'], row['stratum']
        k = quotas[row['fold']][row['stratum']]
        if not k:
            continue
        priority = int(digest([seed, row['source_event_id']]), 16)
        heap = heaps[key]
        entry = (-priority, row['source_event_id'], row)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = sorted([entry[2] for heap in heaps.values() for entry in heap], key=lambda r: (r['fold'], r['student_id'], r['event_index']))
    uids = {r['student_id'] for r in selected}
    # Write selected students' whole histories; selection.jsonl defines scored targets.
    staging = output.with_name(output.name + '.building')
    if staging.exists():
        raise ValueError('Incomplete subset build exists; inspect it before retrying')
    staging.mkdir(parents=True)
    for split in ('train_valid', 'test'):
        with (parent / source['files'][split]['path']).open(encoding='utf-8-sig', newline='') as fin, (staging / (split + '.csv')).open('w', encoding='utf-8', newline='') as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(row for row in reader if row['uid'] in uids)
    manifest = import_snapshot(staging / 'test.csv', output, dataset_name, 'question_native', staging / 'train_valid.csv')
    write_jsonl(output / 'selection.jsonl', selected)
    manifest['lineage'] = {'parent_snapshot_id': source['snapshot_id'], 'parent_manifest_sha256': file_hash(parent / 'manifest.json'),
                           'sampling_specification': identity}
    manifest['target_selection'] = {'path': 'selection.jsonl', 'sha256': file_hash(output / 'selection.jsonl'), 'protocol': protocol}
    # Derivation identity must include the target list, not only the context CSVs.
    manifest['snapshot_id'] = digest({'context_snapshot_id': manifest['snapshot_id'], 'selection_sha256': manifest['target_selection']['sha256'], 'derivation': identity})
    write_json(output / 'manifest.json', manifest)
    reports = []
    for fold, groups in sorted(counts.items()):
        chosen = [r for r in selected if r['fold'] == fold]
        reports.append({'fold': fold, 'population_targets': sum(groups.values()), 'selected_targets': len(chosen),
                        'actual_fraction': len(chosen) / sum(groups.values()), 'selected_students': len({r['student_id'] for r in chosen}),
                        'population_strata': dict(groups), 'selected_strata': dict(Counter(r['stratum'] for r in chosen)),
                        'selected_concepts': len({c for r in chosen for c in r['concept_ids']})})
    report = {'identity': identity, 'snapshot_id': manifest['snapshot_id'], 'total_selected_targets': len(selected),
              'folds': reports, 'context_policy': 'Whole parent student histories retained; only explicit selected targets are scored',
              'limitations': ['0.1% is rounded independently per fold', 'Small proportional strata can receive zero samples',
                             'Targets, not students, are sampled; student-level AUC coverage can be low']}
    write_json(output / 'sampling_report.json', report)
    artifact = finish_stage(output, 'subset', identity, ['manifest.json', 'selection.jsonl', 'sampling_report.json', 'train_valid.csv', 'test.csv'], [source['snapshot_id']])
    record_event(output, 'subset', artifact['artifact_id'], False)
    # Keep build inputs as cached intermediate artifacts, with explicit completion metadata.
    finish_stage(staging, 'subset_context_rows', identity, ['train_valid.csv', 'test.csv'], [source['snapshot_id']])
    return report
