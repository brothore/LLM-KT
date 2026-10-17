from pathlib import Path
from .common import digest, write_json, write_jsonl, read_json, read_jsonl, file_hash
from .config import task_settings
from .data import load_snapshot
from .domain import ModelInput
from .vendor.pykt.sequence import target_ranges


def prepare(config, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Prepared directory already exists; use a new version")
    snapshot, rows = load_snapshot(config)
    task = config["task"]
    if snapshot["representation"] == "concept_expanded" and task["protocol"] != "full_prefix":
        raise ValueError("Expanded concept CSV can only be reconstructed as full_prefix in v0.1; use original quelevel CSV for exact question windows")
    if task["level"] == "concept" and any(len(e["concept_ids"]) != 1 for row in rows for e in row["events"]):
        raise ValueError("v0.1 concept evaluation requires single-concept data")
    identity = {"snapshot_id": snapshot["snapshot_id"], "dataset": snapshot["dataset"],
                "split": config["dataset"]["split"], "validation_fold": config["dataset"]["validation_fold"],
                "task": task, "input": config["input"], "protocol_version": 1}
    task_id = digest(identity)
    targets, inputs, mapping = [], [], []
    for row in rows:
        uid, events = row["student_id"], row["events"]
        mapping.append({"student_id": uid, "source_row": row["source_row"], "fold": row["fold"]})
        for start, index in target_ranges(len(events), task["protocol"], task["maxlen"], task["min_seq_len"]):
            current = events[index]
            event_id = digest([snapshot["snapshot_id"], uid, index])
            sample_id = digest([task_id, uid, event_id, task["level"]])
            item = {k: current[k] for k in ("question_id", "concept_ids")}
            history = [{"question_id": e["question_id"], "concept_ids": e["concept_ids"], "correct": bool(e["response"])} for e in events[start:index]]
            inputs.append({"sample_id": sample_id, "history": history, "current_item": item, "input_version": config["input"]["template"]})
            targets.append({"sample_id": sample_id, "student_id": uid, "event_id": event_id,
                            "event_index": index, "history_start": start, "y_true": current["response"]})
    if not targets:
        raise ValueError("No eligible prediction targets")
    output.mkdir(parents=True)
    write_jsonl(output / "targets.jsonl", targets)
    write_jsonl(output / "inputs.jsonl", inputs)
    write_jsonl(output / "identity_map.jsonl", mapping)
    manifest = {**identity, "task_id": task_id, "settings": task_settings(config), "n_targets": len(targets),
                "n_students": len({t["student_id"] for t in targets}), "input_hash": digest(inputs),
                "files": {name: file_hash(output / name) for name in ("targets.jsonl", "inputs.jsonl", "identity_map.jsonl")}}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_prepared(directory, config=None):
    root = Path(directory)
    manifest = read_json(root / "manifest.json")
    for name, checksum in manifest["files"].items():
        if file_hash(root / name) != checksum:
            raise ValueError(f"Prepared data modified: {name}")
    if config is not None and manifest["settings"] != task_settings(config):
        raise ValueError("Configuration task does not match prepared data")
    targets, inputs = read_jsonl(root / "targets.jsonl"), read_jsonl(root / "inputs.jsonl")
    ids = [r["sample_id"] for r in targets]
    if len(set(ids)) != len(ids) or ids != [r["sample_id"] for r in inputs]:
        raise ValueError("Duplicate or misaligned target/input identities")
    return manifest, targets, [ModelInput(**row) for row in inputs]
