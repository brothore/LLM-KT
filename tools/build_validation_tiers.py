"""Build disjoint whole-student validation tiers offline; no credential access."""
import csv
from pathlib import Path
import yaml
from llm_kt.common import read_json, write_json, digest, file_hash, read_jsonl
from llm_kt.artifacts import finish_stage, verify_stage
from llm_kt.config import load_config
from llm_kt.costs import calibration, input_characters
from llm_kt.data import iter_sequences
from llm_kt.domain import ModelInput
from llm_kt.providers import make_provider
from llm_kt.protocols import prepare, load_prepared
from llm_kt.vendor.pykt.sequence import target_ranges


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'data/snapshots/nips_task34'
    parent = read_json(source/'manifest.json')
    old = root/'data/snapshots/nips_complete_students_budget_v1'
    old_report = read_json(old/'sampling_report.json')
    small_ids = {s['student_id'] for f in old_report['folds'] if f['fold'] in (-1, 0)
                 for s in f['selected_students']}
    assert len(small_ids) == 4
    models = {}
    provenance = {}
    for name in ('baseline', 'round2'):
        path = root/f'configs/experiments/nips_{name}.yaml'
        config = load_config(path)
        run = root/f'runs/prompt_iterations_20261018/{name}'
        models[name] = (make_provider(config['model']), calibration(config, run))
        for p in (path, run/'manifest.json', run/'predictions.jsonl'):
            provenance[str(p.relative_to(root))] = file_hash(p)
    for p in (Path(__file__).resolve(), root/'configs/pricing/qwen3.8-27b.json', source/'manifest.json', old/'sampling_report.json'):
        provenance[str(p.relative_to(root))] = file_hash(p)
    raw_rows, rows = {}, {}
    for split, entry in parent['files'].items():
        path = source/entry['path']
        assert file_hash(path) == entry['sha256']
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            raw_rows.update({r['uid']: r for r in reader})
        rows.update({r['student_id']:r for r in iter_sequences(path, parent['representation'])})
    used = {t['student_id'] for t in read_jsonl(root/'data/prepared/nips_0p1pct_validation_fold0_v1/targets.jsonl')}
    excluded = used | small_ids
    cache_root = root/'data/reports/validation_tiers_candidate_costs_v1'
    cache_root.mkdir(parents=True, exist_ok=True)
    spec = {'seed':42, 'usd_to_cny_planning':7, 'input_token_safety_factor':1.25,
            'output_tokens_minimum':32, 'cache_discount_assumed':False,
            'prices_usd_per_million':{'input':.02, 'output':.5}, 'provenance':provenance,
            'parent_snapshot_id':parent['snapshot_id'], 'protocol':'pykt_window', 'maxlen':200}

    def cost(uid):
        row = rows[uid]
        identity = {'spec':spec, 'student_id':uid}
        file = cache_root/(digest(identity)+'.json')
        if file.exists():
            report = read_json(file)
            assert report['identity'] == identity
            return report['costs_usd']
        totals = {}
        for name, (provider, measured) in models.items():
            chars = 0
            for start, index in target_ranges(len(row['events']), 'pykt_window'):
                history = [dict(question_id=e['question_id'], concept_ids=e['concept_ids'], correct=bool(e['response']))
                           for e in row['events'][start:index]]
                item = {k:row['events'][index][k] for k in ('question_id','concept_ids')}
                chars += input_characters(provider.request(ModelInput('', history, item)), 'openai_compatible_chat')
            totals[name] = chars/measured['input_characters_per_token']*1.25*.02/1e6 + (len(row['events'])-1)*max(32, measured['mean_output_tokens'])*.5/1e6
        write_json(file, {'identity':identity, 'costs_usd':totals})
        return totals

    # Length quintiles of all original fold0 students, before exclusions.
    population = sorted((r for r in rows.values() if r['fold']==0),
                        key=lambda r:(len(r['events']),digest([42,r['student_id']])))
    strata = [[] for _ in range(5)]
    for i, row in enumerate(population):
        if row['student_id'] not in excluded:
            strata[min(4,i*5//len(population))].append(row['student_id'])
    for group in strata:
        group.sort(key=lambda uid:digest([42,uid,'large_validation_v1']))
    selected, totals, decisions = [], dict(baseline=0.,round2=0.), []
    # Whole rounds of five strata: avoid topping up with cheap short students.
    # The first candidate of each stratum is fixed by hash, never by labels.
    for round_index in range(min(map(len,strata))):
        batch = [group[round_index] for group in strata]
        batch_cost = {name:sum(cost(uid)[name] for uid in batch) for name in models}
        fits = all(totals[name]+batch_cost[name] <= 1/7 for name in models)
        decisions.append({'round':round_index,'students':batch,'costs_usd':batch_cost,'accepted':fits})
        if not fits:
            break
        selected.extend(batch)
        totals = {name:totals[name]+batch_cost[name] for name in models}
    if not selected:
        raise ValueError('One balanced round exceeds budget; revise design instead of silently dropping long students')
    assert not (set(selected)&small_ids) and not(set(selected)&used)

    for tier, ids, budget in [('iteration_02cny',sorted(small_ids),.2),('large_1cny',selected,1.)]:
        name = f'nips_validation_{tier}_v1'
        out = root/'data/snapshots'/name
        if out.exists():
            verify_stage(out)
            print(name,'existing immutable snapshot verified')
        else:
            out.mkdir()
            with (out/'train_valid.csv').open('w',encoding='utf-8',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                # Evaluation role reclassification only; event order and all labels unchanged.
                writer.writerows({**raw_rows[uid],'fold':'0'} for uid in ids)
            mapping = [{'student_id':uid,'original_fold':rows[uid]['fold'],'original_source_row':rows[uid]['source_row'],
                        'evaluation_fold':0,'sequence_length':len(rows[uid]['events']),
                        'source_row_sha256':digest(raw_rows[uid])} for uid in ids]
            students = []
            for uid in ids:
                ys = [e['response'] for e in rows[uid]['events'][1:]]
                students.append({'student_id':uid,'sequence_length':len(ys)+1,'n_targets':len(ys),
                                 'positive':sum(ys),'negative':len(ys)-sum(ys),'auc_defined':len(set(ys))==2,
                                 'estimated_usd':cost(uid)})
            total = {n:sum(s['estimated_usd'][n] for s in students) for n in models}
            identity = {'spec':spec,'tier':tier,'budget_cny':budget,'students':ids,
                        'exclusion_student_ids':sorted(excluded) if tier.startswith('large') else [],
                        'sampling':'five_length_quintiles_hash_order_complete_rounds' if tier.startswith('large') else 'reuse_existing_four_students',
                        'role':'validation','all_eligible_time_steps':True}
            snapshot = {'schema_version':1,'dataset':name,'representation':'question_native',
                        'files':{'train_valid':{'path':'train_valid.csv','sha256':file_hash(out/'train_valid.csv')}},
                        'lineage':identity}
            snapshot['snapshot_id'] = digest(snapshot)
            write_json(out/'manifest.json',snapshot)
            write_json(out/'source_mapping.json',mapping)
            report = {'identity':identity,'students':students,'n_students':len(ids),
                      'n_targets':sum(s['n_targets'] for s in students),'estimated_usd':total,
                      'estimated_cny':{n:v*7 for n,v in total.items()},
                      'auc_defined_students':sum(s['auc_defined'] for s in students),
                      'selection_rounds':decisions if tier.startswith('large') else [],
                      'limitations':['Small pilot; not a statistically stable population benchmark.',
                                     'Former test students in iteration tier are now development data.',
                                     'Large tier excludes previously used validation students; conditional population.',
                                     'Planning FX assumption, not current market quote; estimates are not bills.']}
            write_json(out/'sampling_report.json',report)
            finish_stage(out,'validation_tier',identity,['manifest.json','train_valid.csv','source_mapping.json','sampling_report.json'],[parent['snapshot_id']])
        for prompt in ('baseline','round2'):
            raw = yaml.safe_load((root/f'configs/experiments/nips_{prompt}.yaml').read_text())
            raw['dataset'] = {'snapshot':f'../../data/snapshots/{name}','split':'validation','validation_fold':0}
            raw['execution']['max_requests'] = 10000
            path = root/f'configs/experiments/{name}_{prompt}.yaml'
            path.write_text('# Validation only; all selected students and time steps. No inference performed.\n'+yaml.safe_dump(raw,sort_keys=False),encoding='utf-8')
            config = load_config(path)
            prepared = root/'data/prepared'/name
            if prompt=='baseline':
                prepare(config,prepared)
            _,targets,_ = load_prepared(prepared,config)
            for uid in ids:
                assert [t['event_index'] for t in targets if t['student_id']==uid] == list(range(1,len(rows[uid]['events'])))
            for record in iter_sequences(out/'train_valid.csv','question_native'):
                assert record['events'] == rows[record['student_id']]['events']
        print(name, {k:v for k,v in read_json(out/'sampling_report.json').items() if k in ('n_students','n_targets','estimated_cny','auc_defined_students')},flush=True)
    stage_files = [f.name for f in sorted(cache_root.glob('*.json')) if f.name != 'artifact_manifest.json']
    finish_stage(cache_root,'candidate_costs',spec,stage_files,[parent['snapshot_id']])


if __name__ == '__main__':
    main()
