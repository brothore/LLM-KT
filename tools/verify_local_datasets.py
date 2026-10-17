"""Offline verification of stored dataset segments and Git exclusion rules."""
import csv
import shutil
import subprocess
import uuid
from pathlib import Path
from llm_kt.common import read_json, write_json

csv.field_size_limit(1024 * 1024 * 1024)
PROJECT = Path(__file__).resolve().parents[1]


def check_segments(original, segmented):
    students = segments = valid = 0
    with original.open(encoding="utf-8-sig", newline="") as full, segmented.open(encoding="utf-8-sig", newline="") as parts:
        sequences = iter(csv.DictReader(parts))
        for row in csv.DictReader(full):
            students += 1
            values = {k: row[k].split(',') for k in ('questions', 'concepts', 'responses') if k in row}
            n = len(values['responses'])
            for start in range(0, n, 200):
                count = min(200, n - start)
                if count < 3:
                    continue
                segment = next(sequences)
                assert segment['uid'] == row['uid'] and segment['fold'] == row['fold'], 'Student/fold mismatch'
                for key, sequence in values.items():
                    assert segment[key].split(',') == sequence[start:start + count] + ['-1'] * (200 - count), f'{key} segment mismatch'
                assert segment['selectmasks'].split(',') == ['1'] * count + ['-1'] * (200 - count), 'Mask mismatch'
                segments += 1
                valid += count
        assert next(sequences, None) is None, 'Unexpected trailing segment'
    return {'students': students, 'segments': segments, 'valid_interactions_in_segments': valid, 'verified': True}


def check_ignore():
    root = PROJECT / 'tests' / ('work_ignore_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
        shutil.copyfile(PROJECT / '.gitignore', root / '.gitignore')
        paths = ['data/raw/archives/peiyou.zip', 'data/processed/peiyou/test.csv',
                 'data/snapshots/assist2009/test.csv', 'data/prepared/demo/inputs.jsonl',
                 'data/reports/datasets_summary.json', '.secrets/openrouter.key', 'runs/example/responses/x.json']
        result = subprocess.run(['git', '-C', str(root), 'check-ignore', '--no-index', '-z', '--stdin'],
                                input=('\0'.join(paths) + '\0').encode(), capture_output=True, check=True)
        assert set(result.stdout.decode().strip('\0').split('\0')) == set(paths), 'A private path is not ignored'
        return {'all_private_paths_ignored': True, 'paths': paths}
    finally:
        resolved = root.resolve()
        assert resolved.parent == (PROJECT / 'tests').resolve() and resolved.name.startswith('work_ignore_')
        shutil.rmtree(resolved)


def main():
    result = {'git_exclusions': check_ignore(), 'datasets': {}}
    for dataset in ('peiyou', 'assist2009', 'nips_task34'):
        root = PROJECT / 'data/processed' / dataset
        read_json(root / 'processing_manifest.json')
        result['datasets'][dataset] = {split: check_segments(root / (split + '_quelevel.csv'), root / (split + '_sequences_quelevel.csv'))
                                      for split in ('train_valid', 'test')}
        print(dataset, result['datasets'][dataset], flush=True)
    write_json(PROJECT / 'data/reports/verification.json', result)


if __name__ == '__main__':
    main()
