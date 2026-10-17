import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from .common import read_jsonl, read_json, write_json, digest, file_hash
from .artifacts import finish_stage, verify_stage, record_event, code_identity
from .domain import checked_probability
from .protocols import load_prepared
from .vendor.pykt.student_metrics import sliding_window_collect, calculate_wealth_metrics


def metrics(y, p, threshold):
    return {"n_targets": len(y), "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "acc": float(accuracy_score(y, np.asarray(p) >= threshold)),
            "logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p))}


def index_predictions(rows, target_ids):
    result = {}
    for row in rows:
        sid = row["sample_id"]
        if sid not in target_ids or sid in result or row.get("status") != "success":
            raise ValueError("Unknown, duplicate or unsuccessful prediction record")
        checked_probability(row["p_correct"])
        result[sid] = row
    return result


def calculate(targets, predictions, config, legacy=False):
    index = index_predictions(predictions, {t["sample_id"] for t in targets})
    groups = defaultdict(list)
    for t in targets:
        groups[t["student_id"]].append(t)
    all_y, all_p, students = [], [], []
    for uid, records in sorted(groups.items()):
        records.sort(key=lambda x: x["event_index"])
        available = [r for r in records if r["sample_id"] in index]
        y = [r["y_true"] for r in available]
        p = [index[r["sample_id"]]["p_correct"] for r in available]
        if legacy and len(y) >= 200:
            y, p = sliding_window_collect(np.asarray(y), np.asarray(p), window_size=200)
        cur = metrics(y, p, config["threshold"]) if y else {"n_targets": 0, "auc": None, "acc": None, "logloss": None, "brier": None}
        if legacy and y and cur["auc"] is None:
            cur["auc"] = 0.5
        students.append({"student_id": uid, **cur, "n_expected": len(records), "n_success": len(available),
                         "n_positive": sum(y), "n_negative": len(y) - sum(y),
                         "auc_reason": None if cur["auc"] is not None else ("no_prediction" if not y else "single_class")})
        all_y.extend(y)
        all_p.extend(p)
    overall = metrics(all_y, all_p, config["threshold"]) if all_y else {"n_targets": 0, "auc": None, "acc": None, "logloss": None, "brier": None}
    if legacy and all_y and overall["auc"] is None:
        overall["auc"] = 0.5
    aucs = [s["auc"] for s in students if s["auc"] is not None]
    ddof = config["student_std_ddof"]
    stats = {"mean": float(np.mean(aucs)) if aucs else None,
             "std": float(np.std(aucs, ddof=ddof)) if len(aucs) > ddof else None,
             "median": float(np.median(aucs)) if aucs else None,
             "min": min(aucs) if aucs else None, "max": max(aucs) if aucs else None,
             "range": max(aucs) - min(aucs) if aucs else None,
             "q1": float(np.percentile(aucs, 25)) if aucs else None,
             "q3": float(np.percentile(aucs, 75)) if aucs else None,
             "iqr": float(np.percentile(aucs, 75) - np.percentile(aucs, 25)) if aucs else None,
             "ddof": ddof}
    if legacy and aucs:
        avg, gini, eawi = calculate_wealth_metrics(aucs)
        stats.update(average_auc=float(avg), gini_coefficient=float(gini), **eawi)
        stats["std"] = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        stats["ddof"] = 1
        denominator = stats["std"] * stats["range"]
        stats["epi"] = stats["mean"] / denominator if denominator else 0.0
    coverage = {"n_expected": len(targets), "n_success": len(index), "prediction_coverage": len(index) / len(targets) if targets else 0,
                "n_students": len(students), "n_valid_auc_students": len(aucs), "student_auc_coverage": len(aucs) / len(students) if students else 0}
    return {"profile": "pykt_legacy_v1" if legacy else "standard_v1", "overall": overall, "student_auc": stats, "coverage": coverage}, students


def evaluate(prepared, run_directory, profile=None):
    task, targets, _ = load_prepared(prepared)
    root = Path(run_directory)
    run = read_json(root / "manifest.json")
    if run["identity"]["task_id"] != task["task_id"] or run["identity"]["input_hash"] != task["input_hash"]:
        raise ValueError("Run/prepared task mismatch")
    config = run["identity"]["config"]["evaluation"]
    profile = profile or config["profile"]
    if profile not in {"standard_v1", "pykt_legacy_v1"}:
        raise ValueError("Unknown profile")
    config = {**config, "profile": profile}
    if profile == "pykt_legacy_v1":
        config["student_std_ddof"] = 1
    identity = {"task_id": task["task_id"], "predictions_sha256": file_hash(root / "predictions.jsonl"), "evaluation_config": config, "source_sha256": code_identity()["source_sha256"]}
    cached = root / "evaluations" / digest(identity)
    if cached.exists():
        artifact = verify_stage(cached)
        record_event(cached, "evaluate", artifact["artifact_id"], True, {})
        return read_json(cached / "metrics.json")
    report, students = calculate(targets, read_jsonl(root / "predictions.jsonl"), config, profile == "pykt_legacy_v1")
    report["task_id"] = task["task_id"]
    report["complete"] = report["coverage"]["prediction_coverage"] == 1
    report["evaluation_config"] = config
    write_json(root / f"metrics_{profile}.json", report)
    with (root / f"per_student_{profile}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(students[0]))
        writer.writeheader()
        writer.writerows(students)
    text = (f"# KT evaluation\n\nTask: `{task['task_id']}`\n\nProfile: `{profile}`\n\n"
            f"Coverage: {report['coverage']['n_success']}/{len(targets)} targets\n\n"
            f"Overall AUC: {report['overall']['auc']}\n\nStudent AUC mean/std: {report['student_auc']['mean']} / {report['student_auc']['std']}\n\n"
            f"Valid AUC students: {report['coverage']['n_valid_auc_students']}/{len(students)}\n")
    (root / f"report_{profile}.md").write_text(text, encoding="utf-8")
    import shutil
    write_json(cached / "metrics.json", report)
    shutil.copyfile(root / f"per_student_{profile}.csv", cached / "per_student.csv")
    (cached / "report.md").write_text(text, encoding="utf-8")
    finish_stage(cached, "evaluate", identity, ["metrics.json", "per_student.csv", "report.md"], [task["task_id"]])
    return report


def compare(prepared, runs, output):
    task, targets, _ = load_prepared(prepared)
    target_ids = {t["sample_id"] for t in targets}
    indexed = []
    configs = []
    for directory in runs:
        root = Path(directory)
        run = read_json(root / "manifest.json")["identity"]
        if run["task_id"] != task["task_id"] or run["input_hash"] != task["input_hash"]:
            raise ValueError("Cannot compare different tasks/inputs")
        indexed.append(index_predictions(read_jsonl(root / "predictions.jsonl"), target_ids))
        configs.append(run["config"]["evaluation"])
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("Evaluation configs differ")
    common = set.intersection(*(set(p) for p in indexed))
    if not common:
        raise ValueError("No common successful targets")
    selected = [t for t in targets if t["sample_id"] in common]
    report = {"task_id": task["task_id"], "n_expected": len(targets), "n_common": len(common),
              "comparison": "complete" if common == target_ids else "exploratory_common_subset", "runs": []}
    for directory, index in zip(runs, indexed):
        result, _ = calculate(selected, [index[s] for s in common], configs[0], configs[0]["profile"] == "pykt_legacy_v1")
        report["runs"].append({"run": str(Path(directory).resolve()), "individual_coverage": len(index) / len(targets), "metrics_on_common_targets": result})
    write_json(output, report)
    cached = Path(output).parent / (Path(output).stem + '.artifacts') / digest(report)
    if cached.exists():
        verify_stage(cached)
    else:
        write_json(cached / 'comparison.json', report)
        finish_stage(cached, 'compare', {'report_hash': digest(report), 'prediction_hashes': [file_hash(Path(r) / 'predictions.jsonl') for r in runs]}, ['comparison.json'], [task['task_id']])
    return report
