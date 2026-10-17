"""Import complete pykt sequence CSVs; never read labels in a provider."""
import csv
import shutil
from pathlib import Path
from .common import read_json, write_json, file_hash, digest

# Original student histories may exceed Python csv's default 128 KiB field limit.
csv.field_size_limit(1024 * 1024 * 1024)


def import_snapshot(source, output, dataset, representation="question_native", train_source=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Snapshot destination already exists: {output}")
    if representation not in {"question_native", "single_concept", "concept_expanded"}:
        raise ValueError("Unknown representation")
    sources = {"test": Path(source).resolve()}
    if train_source:
        sources["train_valid"] = Path(train_source).resolve()
    # Validate before copying so malformed input does not create a usable snapshot.
    records = {key: [{"student_id": r["student_id"], "fold": r["fold"]}
                     for r in iter_sequences(path, representation)] for key, path in sources.items()}
    test_uids = {r["student_id"] for r in records["test"]}
    if "train_valid" in records:
        if test_uids & {r["student_id"] for r in records["train_valid"]}:
            raise ValueError("Students overlap between test and train_valid")
        folds = {}
        for row in records["train_valid"]:
            if row["fold"] < 0:
                raise ValueError("train_valid contains a negative fold")
            uid = row["student_id"]
            if uid in folds and folds[uid] != row["fold"]:
                raise ValueError("Student spans multiple validation folds")
            folds[uid] = row["fold"]
    if any(r["fold"] != -1 for r in records["test"]):
        raise ValueError("Test source must have fold=-1")
    output.mkdir(parents=True)
    files = {}
    for key, source_path in sources.items():
        name = f"{key}.csv"
        shutil.copyfile(source_path, output / name)
        files[key] = {"path": name, "sha256": file_hash(output / name), "source": str(source_path)}
    identity = {"dataset": dataset, "representation": representation,
                "files": {k: v["sha256"] for k, v in files.items()}}
    manifest = {**identity, "snapshot_id": digest(identity), "files": files, "schema_version": 1}
    write_json(output / "manifest.json", manifest)
    return manifest


def read_sequences(path, representation):
    return list(iter_sequences(path, representation))


def iter_sequences(path, representation):
    count, seen = 0, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"uid", "responses", "concepts", "fold"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires uid, responses, concepts, fold")
        for line, row in enumerate(reader, 2):
            if any(k in row for k in ("selectmasks", "qidxs", "rest")):
                raise ValueError("Import complete original CSVs, not precomputed segment/window files")
            uid = row["uid"]
            if not uid or uid in seen:
                raise ValueError(f"Line {line}: requires one complete row per unique student")
            seen.add(uid)
            responses = [int(x) for x in row["responses"].split(",")]
            concepts = row["concepts"].split(",")
            questions = row["questions"].split(",") if row.get("questions") else [None] * len(responses)
            repeats = [int(x) for x in row["is_repeat"].split(",")] if row.get("is_repeat") else [0] * len(responses)
            if len({len(responses), len(concepts), len(questions), len(repeats)}) != 1:
                raise ValueError(f"Line {line}: sequence lengths differ")
            events = []
            for q, c, r, repeat in zip(questions, concepts, responses, repeats):
                if r not in (0, 1) or repeat not in (0, 1):
                    raise ValueError("Original data must contain binary responses/repeat flags without padding")
                cs = c.split("_")
                if any(not x or x == "-1" for x in cs) or q == "-1":
                    raise ValueError("Invalid or padded item")
                if representation == "single_concept" and (len(cs) != 1 or repeat):
                    raise ValueError("single_concept requires exactly one concept per event")
                if representation == "concept_expanded" and len(cs) != 1:
                    raise ValueError("concept_expanded requires one concept per expanded row")
                if repeat:
                    if representation != "concept_expanded" or not events or events[-1]["question_id"] != q or events[-1]["response"] != r:
                        raise ValueError("Invalid concept expansion group")
                    events[-1]["concept_ids"].extend(cs)
                else:
                    events.append({"event_index": len(events), "question_id": q, "concept_ids": cs, "response": r})
            count += 1
            yield {"student_id": uid, "source_row": line - 2, "fold": int(row["fold"]), "events": events}
    if not count:
        raise ValueError("Empty dataset")


def load_snapshot(config):
    manifest, iterator = stream_snapshot(config)
    return manifest, list(iterator)


def stream_snapshot(config):
    root = Path(config["dataset"]["snapshot"])
    manifest = read_json(root / "manifest.json")
    split = config["dataset"]["split"]
    key = "test" if split == "test" else "train_valid"
    if key not in manifest["files"]:
        raise ValueError(f"Snapshot lacks {key} data")
    entry = manifest["files"][key]
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != entry["sha256"]:
        raise ValueError("Snapshot path/hash validation failed")
    def selected():
        seen_folds, count = set(), 0
        fold = config["dataset"]["validation_fold"]
        for row in iter_sequences(path, manifest["representation"]):
            seen_folds.add(row["fold"])
            if split == "test" or ((row["fold"] == fold) == (split == "validation")):
                count += 1
                yield row
        if split != "test" and fold not in seen_folds:
            raise ValueError("Requested validation fold is absent")
        if not count:
            raise ValueError("Selected split is empty")
    return manifest, selected()
