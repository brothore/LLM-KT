"""Offline, cost-constrained whole-student subset; never calls a model or reads keys."""
import csv
from collections import defaultdict
from pathlib import Path
import yaml
from llm_kt.common import read_json, write_json, digest, file_hash
from llm_kt.data import iter_sequences, import_snapshot
from llm_kt.config import load_config
from llm_kt.costs import calibration, input_characters
from llm_kt.providers import make_provider
from llm_kt.domain import ModelInput
from llm_kt.vendor.pykt.sequence import target_ranges
from llm_kt.artifacts import finish_stage, verify_stage


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root / 'data/snapshots/nips_task34'
    output = root / 'data/snapshots/nips_complete_students_budget_v1'
    if output.exists():
        verify_stage(output)
        print('Verified existing immutable subset; use a new version to change selection.')
        return
    manifest = read_json(parent / 'manifest.json')
    models = []
    for name, run, val, test in [('baseline', 'baseline', .011493208, .014253580),
                                 ('round2', 'round2', .013202960, .016164480)]:
        config = load_config(root / f'configs/experiments/nips_{name}.yaml')
        measured = calibration(config, root / f'runs/prompt_iterations_20261018/{run}')
        models.append((name, make_provider(config['model']), measured, val, test))
    groups = defaultdict(list)
    for split, entry in manifest['files'].items():
        source = parent / entry['path']
        if file_hash(source) != entry['sha256']:
            raise ValueError('Parent hash mismatch')
        for row in iter_sequences(source, manifest['representation']):
            groups[row['fold']].append(row)

    def costs(row):
        totals = {}
        for name, provider, measured, _, _ in models:
            chars = 0
            for start, index in target_ranges(len(row['events']), 'pykt_window'):
                visible = [dict(question_id=e['question_id'], concept_ids=e['concept_ids'], correct=bool(e['response'])) for e in row['events'][start:index]]
                current = {k: row['events'][index][k] for k in ('question_id', 'concept_ids')}
                chars += input_characters(provider.request(ModelInput('', visible, current)), 'openai_compatible_chat')
            # No cache discount; 25% more input tokens than measured mean, output allowance >=32.
            totals[name] = chars / measured['input_characters_per_token'] * 1.25 * .02 / 1e6 + (len(row['events']) - 1) * max(32, measured['mean_output_tokens']) * .5 / 1e6
        return totals

    chosen, reports = [], []
    old_counts = {0:221, 1:226, 2:215, 3:219, 4:224, -1:272}
    for fold, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (len(r['events']), digest([42, r['student_id']])))
        budgets = {name: test if fold == -1 else val * old_counts[fold] / 221 for name, _, _, val, test in models}
        cache = {}
        def candidate(q):
            index = int((len(rows)-1)*q)
            if index not in cache:
                cache[index] = costs(rows[index])
            return rows[index], cache[index]
        # Predeclared length-only quantile design: lower-quartile + median student.
        # If infeasible, reduce both quantiles uniformly, without inspecting labels.
        for step in range(21):
            scale = 1-step*.025
            selected = [candidate(.25*scale), candidate(.50*scale)]
            total = {name: sum(c[name] for _, c in selected) for name in budgets}
            if all(total[n] <= budgets[n] for n in budgets):
                break
        else:
            raise ValueError('Two whole students do not fit; revise design explicitly')
        if selected[0][0]['student_id'] == selected[1][0]['student_id']:
            raise ValueError('Duplicate student selection')
        chosen.extend(r for r, _ in selected)
        students = []
        for row, cost in selected:
            ys = [e['response'] for e in row['events'][1:]]
            students.append({'student_id': row['student_id'], 'sequence_length': len(row['events']),
                             'targets': len(ys), 'positives': sum(ys), 'negatives': len(ys)-sum(ys),
                             'auc_defined': len(set(ys)) == 2, 'estimated_usd': cost})
        reports.append({'fold': fold, 'population_students': len(rows), 'selected_students': students,
                        'length_quantiles': [.25*scale, .50*scale], 'cost_caps_usd': budgets,
                        'estimated_usd': total, 'targets': sum(s['targets'] for s in students)})
    staging = output.with_name(output.name + '.building')
    staging.mkdir(parents=True)
    ids = {r['student_id'] for r in chosen}
    for split, entry in manifest['files'].items():
        with (parent / entry['path']).open(encoding='utf-8-sig', newline='') as fin, (staging / f'{split}.csv').open('w', encoding='utf-8', newline='') as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(r for r in reader if r['uid'] in ids)
    new = import_snapshot(staging/'test.csv', output, output.name, 'question_native', staging/'train_valid.csv')
    identity = {'algorithm': 'length_quantile_budget_v1', 'seed': 42, 'parent': manifest['snapshot_id'],
                'unit': 'whole_student', 'protocol': 'pykt_window', 'maxlen': 200,
                'budget_policy': 'baseline_and_round2_previous_split_cost', 'input_token_safety_factor':1.25}
    new['lineage'] = identity
    new['snapshot_id'] = digest({'copied_snapshot':new['snapshot_id'], 'design':identity})
    write_json(output/'manifest.json', new)
    report = {'identity': identity, 'folds': reports, 'limitations': [
        'Two students per fold cannot estimate population student AUC reliably.',
        'Budget intentionally underrepresents long sequences; not a representative population benchmark.',
        'All time steps after the first are scored; history window remains at most 199 events.',
        'Costs are offline estimates at the recorded Tenstorrent prices, not guaranteed bills.',
        'No labels or model predictions used in selection; label coverage checked afterwards.',
        'Round2 costing uses existing fold0 prior only as size proxy for other folds; do not run that prior on other validation folds.']}
    write_json(output/'sampling_report.json', report)
    finish_stage(output, 'whole_student_subset', identity, ['manifest.json','sampling_report.json','test.csv','train_valid.csv'], [manifest['snapshot_id']])
    finish_stage(staging, 'whole_student_source_rows', identity, ['test.csv','train_valid.csv'], [manifest['snapshot_id']])
    print([{k:v for k,v in r.items() if k != 'selected_students'} | {'lengths':[s['sequence_length'] for s in r['selected_students']]} for r in reports])


if __name__ == '__main__':
    main()
