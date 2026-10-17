"""Prepare the user's archives with vendored pykt preprocessing and compact storage.

Run from the project root with PYTHONPATH=src. No network/model calls.
"""
import argparse
import csv
import gc
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from llm_kt.common import file_hash, write_json, read_json
from llm_kt.data import import_snapshot, iter_sequences
from llm_kt.vendor.pykt.preprocess import split_datasets as concept
from llm_kt.vendor.pykt.preprocess import split_datasets_que as question
from llm_kt.vendor.pykt.preprocess.assist2009_preprocess import read_data_from_csv as assist_processor
from llm_kt.vendor.pykt.preprocess.nips_task34_preprocess import read_data_from_csv as nips_processor

csv.field_size_limit(1024 * 1024 * 1024)
PROJECT = Path(__file__).resolve().parents[1]


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def extract_selected(archive, destination, names):
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    with zipfile.ZipFile(archive) as zipped:
        for name in names:
            info = zipped.getinfo(name)
            path = (destination / name).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member path")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size != info.file_size:
                log(f"Extract {name}: {info.file_size:,} bytes")
                with zipped.open(info) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            inventory.append({"name": name, "bytes": info.file_size, "sha256": file_hash(path)})
    return inventory


def generate_compact(data_txt, directory, dataset):
    """Use original pykt functions, omitting only giant redundant window CSVs."""
    frame, keys = concept.read_data(str(data_txt))
    maximum = concept.get_max_concepts(frame)
    summaries = {}
    for representation in ("concept", "question"):
        log(f"{dataset}: mapping/splitting {representation}")
        current_keys = set(keys)
        if representation == "concept":
            current, current_keys = concept.extend_multi_concepts(frame, current_keys)
            current, mapping = concept.id_mapping(current)
            suffix = ""
            generator = concept.generate_sequences
        else:
            current, mapping = question.id_mapping_que(frame)
            suffix = "_quelevel"
            generator = question.generate_sequences
        mapping["max_concepts"] = maximum
        write_json(directory / f"keyid2idx_{representation}.json", mapping)
        if representation == "question":
            write_json(directory / "keyid2idx.json", mapping)
        train, test = concept.train_test_split(current, 0.2)
        train = concept.KFold_split(train, 5)
        test = test.copy()
        test["fold"] = -1
        current_keys.add("fold")
        columns = [key for key in concept.ALL_KEYS if key in current_keys]
        train[columns].to_csv(directory / f"train_valid{suffix}.csv", index=False)
        test[columns].to_csv(directory / f"test{suffix}.csv", index=False)
        log(f"{dataset}: generating {representation} non-overlapping sequences")
        train_sequences = generator(train, current_keys, 3, 200)
        train_sequences.to_csv(directory / f"train_valid_sequences{suffix}.csv", index=False)
        del train_sequences
        test_sequences = generator(test, current_keys, 3, 200)
        test_sequences.to_csv(directory / f"test_sequences{suffix}.csv", index=False)
        summaries[representation] = {"num_questions": len(mapping.get("questions", {})),
                                      "num_concepts": len(mapping.get("concepts", {})),
                                      "max_concepts": maximum}
        del current, mapping, train, test, test_sequences
        gc.collect()
    return summaries


def audit(path, representation):
    count = interactions = multi = max_length = targets_window = targets_segmented = 0
    folds = Counter()
    uids = set()
    for row in iter_sequences(path, representation):
        n = len(row["events"])
        count += 1
        interactions += n
        multi += sum(len(e["concept_ids"]) > 1 for e in row["events"])
        max_length = max(max_length, n)
        folds[str(row["fold"])] += 1
        targets_window += max(0, n - 1)
        full, rest = divmod(n, 200)
        targets_segmented += full * 199 + (rest - 1 if rest >= 3 else 0)
        uids.add(row["student_id"])
    return {"students": count, "interactions": interactions, "multi_concept_events": multi,
            "max_student_length": max_length, "fold_students": dict(sorted(folds.items())),
            "question_window_targets": targets_window, "question_segmented_targets": targets_segmented}, uids


def process(dataset, input_root):
    source_archive = input_root / f"{dataset}.zip"
    raw_root = PROJECT / "data/raw"
    archives = raw_root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / source_archive.name
    if not archive.exists():
        shutil.copyfile(source_archive, archive)
    source_hash = file_hash(source_archive)
    if file_hash(archive) != source_hash:
        raise ValueError("Copied archive differs from source")
    processed = PROJECT / "data/processed" / dataset
    processed.mkdir(parents=True, exist_ok=True)
    done = processed / "processing_manifest.json"
    if done.exists():
        previous = read_json(done)
        if previous["archive_sha256"] != source_hash:
            raise ValueError("Archive changed; choose a new dataset snapshot")
        log(f"{dataset}: processed files already complete; verifying")
        for name, checksum in previous["processed_sha256"].items():
            if file_hash(processed / name) != checksum:
                raise ValueError(f"Processed file changed: {name}")
        return previous
    if dataset == "peiyou":
        with zipfile.ZipFile(archive) as zipped:
            selected = [i.filename for i in zipped.infolist() if not i.is_dir()
                        and (i.filename.endswith('.json') or i.filename in {
                            'train_valid.csv', 'test.csv', 'train_valid_quelevel.csv', 'test_quelevel.csv',
                            'train_valid_sequences.csv', 'test_sequences.csv',
                            'train_valid_sequences_quelevel.csv', 'test_sequences_quelevel.csv'})]
            omitted = [i.filename for i in zipped.infolist() if not i.is_dir() and i.filename not in selected]
        inventory = extract_selected(archive, processed, selected)
        method = "Reuse preprocessed archive files byte-for-byte; preserve existing split/folds"
        dimensions = {}
    else:
        raw = raw_root / dataset
        names = ['skill_builder_data_corrected_collapsed.csv'] if dataset == 'assist2009' else [
            'train_task_3_4.csv', 'metadata/answer_metadata_task_3_4.csv',
            'metadata/question_metadata_task_3_4.csv', 'metadata/student_metadata_task_3_4.csv',
            'metadata/subject_metadata.csv']
        inventory = extract_selected(archive, raw, names)
        data_txt = processed / 'data.txt'
        if not data_txt.exists():
            temporary_txt = processed / 'data.txt.tmp'
            log(f"{dataset}: raw preprocessing")
            if dataset == 'assist2009':
                input_csv = raw / names[0]
                try:
                    assist_processor(str(input_csv), str(temporary_txt))
                except UnicodeDecodeError:
                    normalized = raw / 'source_utf8.csv'
                    with input_csv.open(encoding='latin1') as fin, normalized.open('w', encoding='utf-8', newline='') as fout:
                        shutil.copyfileobj(fin, fout)
                    log('ASSIST2009 input converted from latin1 to UTF-8 for original processor')
                    assist_processor(str(normalized), str(temporary_txt))
            else:
                nips_processor(str(raw / 'train_task_3_4.csv'), str(raw / 'metadata'), 'task_3_4', str(temporary_txt))
            temporary_txt.replace(data_txt)
        dimensions = generate_compact(data_txt, processed, dataset)
        omitted = ['test_window_sequences.csv', 'test_window_sequences_quelevel.csv',
                   'test_question_sequences.csv', 'test_question_window_sequences.csv']
        method = 'Vendored original pykt raw processors, read_data, ID mapping, 80/20 split, 5 folds, maxlen=200/minlen=3'
    train_stats, train_uids = audit(processed / 'train_valid_quelevel.csv', 'question_native')
    test_stats, test_uids = audit(processed / 'test_quelevel.csv', 'question_native')
    if train_uids & test_uids:
        raise ValueError(f"{dataset}: train/test student leakage")
    if set(train_stats['fold_students']) != set('01234') or set(test_stats['fold_students']) != {'-1'}:
        raise ValueError(f"{dataset}: invalid fold set")
    log(f"{dataset}: importing standalone snapshot")
    snapshot = PROJECT / 'data/snapshots' / dataset
    if not snapshot.exists():
        import_snapshot(processed / 'test_quelevel.csv', snapshot, dataset, 'question_native', processed / 'train_valid_quelevel.csv')
        shutil.copyfile(processed / 'keyid2idx.json', snapshot / 'keyid2idx.json')
    result = {'dataset': dataset, 'archive_sha256': source_hash, 'method': method,
              'source_commit': 'a0cc4151f0b8c740b60c1cb18a070c8885baa9a6',
              'split_random_state': 1024 if dataset != 'peiyou' else 'existing archive split preserved',
              'kfold': 5, 'maxlen': 200, 'min_seq_len': 3, 'dimensions': dimensions,
              'train_valid': train_stats, 'test': test_stats, 'train_test_overlap_students': 0,
              'selected_archive_files': inventory, 'not_materialized_or_extracted': omitted,
              'window_policy': 'Reconstruct exact question-native windows lazily from complete sequences; no redundant giant window CSV',
              'processed_sha256': {p.name: file_hash(p) for p in processed.iterdir() if p.is_file() and p.name != done.name}}
    write_json(done, result)
    log(f"{dataset}: COMPLETE train={train_stats['students']} test={test_stats['students']} interactions={train_stats['interactions'] + test_stats['interactions']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=PROJECT.parent)
    parser.add_argument('--datasets', nargs='+', default=['peiyou', 'assist2009', 'nips_task34'])
    args = parser.parse_args()
    results = [process(dataset, args.input_root.resolve()) for dataset in args.datasets]
    write_json(PROJECT / 'data/reports/datasets_summary.json', results)
    if set(args.datasets) == {'peiyou', 'assist2009', 'nips_task34'}:
        write_overview(results)


def write_overview(results):
    lines = ['# 三个 KT 数据集的处理与存放', '', '处理日期：2026-10-17。全部数据位于独立项目内，未调用模型接口。', '',
             '| 数据集 | 训练/验证学生 | 测试学生 | 总作答次数（题目级） | 测试窗口目标数 |',
             '|---|---:|---:|---:|---:|']
    for result in results:
        train, test = result['train_valid'], result['test']
        lines.append(f"| {result['dataset']} | {train['students']:,} | {test['students']:,} | {train['interactions'] + test['interactions']:,} | {test['question_window_targets']:,} |")
    lines += ['', '## 数据来源与处理规则', '',
              '- 培优压缩包已经包含完整 pykt 处理产物，复用原文件，保留原有学生划分、fold、ID 映射，不重新随机切分。',
              '- ASSIST2009 原始 CSV 通过项目内复制的 pykt 数据处理器排序、过滤、生成六行格式，再做概念展开/题目级映射和划分。',
              '- NIPS 仅处理 Task 3/4，关联作答时间和三级知识点；Task 1/2 不参与。本地原始压缩包仍完整保留。',
              '- 新处理数据采用 pykt 默认的 random_state=1024、测试比例 0.2、训练验证 5 折、maxlen=200、min_seq_len=3。',
              '- 概念级和题目级分别生成映射，保存为 keyid2idx_concept.json / keyid2idx_question.json；通用 keyid2idx.json 对应题目级。培优保留包内映射。',
              '- NIPS 处理器将解析元数据列表的 eval 改为 ast.literal_eval，并缓存重复时间值的原函数转换结果；排序、时间转换规则、三级知识点及响应转换沿用 pykt。', '',
              '## 项目内目录', '',
              '```text', 'data/', '  raw/archives/                    # 三个完整来源压缩包',
              '  raw/assist2009/                  # 解压的原始 ASSIST2009 CSV',
              '  raw/nips_task34/                 # Task 3/4 训练日志及相应 metadata',
              '  processed/<dataset>/            # pykt 完整序列、普通分段、映射和处理清单',
              '  snapshots/<dataset>/            # 本框架直接使用的题目级数据快照',
              '  reports/                        # 汇总、校验结果和处理日志', '```', '',
              '每个 processed 目录包含 train_valid.csv、test.csv、train_valid_quelevel.csv、test_quelevel.csv、',
              'train_valid_sequences.csv、test_sequences.csv、train_valid_sequences_quelevel.csv、test_sequences_quelevel.csv，',
              '以及 ID 映射和 processing_manifest.json。新处理数据还保存 data.txt。', '',
              'snapshots 目录保存 test.csv 和 train_valid.csv（内容为题目级表示）、keyid2idx.json 和 manifest.json。',
              'manifest 记录来源、表示、数据哈希与快照身份，运行时无需访问原 pykt 仓库。', '',
              '## 窗口文件与推理输入', '',
              '没有额外展开巨大的 test_window 或 test_question_window CSV。培优包内这些文件仍保留在原始归档中，',
              '独立项目按 pykt 题目原生窗口协议，从完整学生序列重建目标和对应历史。普通分段文件则已经物化保存。',
              '这不同于概念展开后的题目专用融合窗口，不将两种历史口径混称为相同协议。', '',
              '当前尚未为全部目标生成 inputs.jsonl 或发起推理。百万级目标的历史 JSON 会产生大量重复数据，',
              '测试应先选定一个小样本任务；现有 prepare 会展开所有选择的学生目标，不应直接拿全量数据做第一次测试。', '',
              '## 配置与验证', '',
              '三个入口配置位于 configs/datasets/peiyou.yaml、assist2009.yaml、nips_task34.yaml。',
              '默认 split=test、validation_fold=0、题目级 pykt_window。可切换 split=train/validation 读取现有折号。',
              'API key 路径指向项目外的 D:/1017KT/.secrets/openrouter.key；数据与密钥位置分开。', '',
              '完成的校验包括：字段长度、二元响应、单学生唯一完整行、训练测试 UID 无交叉、训练验证 fold 0–4、测试 fold=-1、',
              '来源归档/复制产物的 SHA-256，以及普通题目分段与原始序列逐项、mask、学生及 fold 的一致性。',
              '校验明细见 data/reports/verification.json。', '',
              'data/raw、data/processed、data/snapshots、data/prepared、data/reports、runs、密钥目录和压缩包已加入 Git 忽略规则。',
              '通过临时 Git 仓库的 check-ignore 验证了这些路径被排除；本次没有上传任何数据。', '',
              '处理脚本：tools/prepare_local_datasets.py；离线校验：tools/verify_local_datasets.py。',
              '处理源代码及哈希位于 third_party/pykt/provenance.json。']
    (PROJECT / 'docs/三个KT数据集的处理与存放.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
