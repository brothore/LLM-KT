from pathlib import Path
import math
import yaml
from .common import assert_keys, digest
from .pricing import load_pricing

DEFAULTS = {
    "dataset": {"snapshot": "", "split": "test", "validation_fold": 0},
    "task": {"protocol": "pykt_window", "level": "question", "maxlen": 200, "min_seq_len": 3},
    "input": {"template": "history_ids_v1", "history_policy": "protocol_exact"},
    "model": {"adapter": "openrouter_decisions", "model_id": "typesafe/jev-1.14",
              "endpoint": "https://openrouter.ai/api/alpha/decisions", "api_key_env": "OPENROUTER_API_KEY",
              "api_key_file": "", "pricing_file": "", "probability_strategy": "noul", "parameters": {}},
    "execution": {"max_concurrency": 4, "max_attempts": 3, "timeout_seconds": 60,
                  "max_requests": 100, "requests_per_minute": 60, "replicate_id": "0"},
    "evaluation": {"profile": "standard_v1", "student_std_ddof": 1, "threshold": 0.5},
}


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert_keys(raw, DEFAULTS, "config")
    result = {}
    for section, defaults in DEFAULTS.items():
        part = raw.get(section, {})
        assert_keys(part, defaults, section)
        result[section] = {**defaults, **part}
    ds, task, model, ex, ev = (result[k] for k in ("dataset", "task", "model", "execution", "evaluation"))
    ds["snapshot"] = str((path.parent / ds["snapshot"]).resolve())
    if model["api_key_file"]:
        model["api_key_file"] = str((path.parent / model["api_key_file"]).resolve())
    if model["pricing_file"]:
        model["pricing_file"] = str((path.parent / model["pricing_file"]).resolve())
        model["pricing_snapshot"] = load_pricing(model["pricing_file"], model["model_id"])
    if ds["split"] not in {"test", "train", "validation"}:
        raise ValueError("dataset.split must be test, train or validation")
    if task["protocol"] not in {"pykt_window", "pykt_segmented", "full_prefix"}:
        raise ValueError("Unsupported task.protocol")
    if task["level"] not in {"question", "concept"}:
        raise ValueError("Unsupported task.level")
    for key in ("maxlen", "min_seq_len"):
        if type(task[key]) is not int or task[key] < 2:
            raise ValueError(f"task.{key} must be an integer >= 2")
    if task["min_seq_len"] > task["maxlen"]:
        raise ValueError("min_seq_len cannot exceed maxlen")
    if type(ds["validation_fold"]) is not int:
        raise ValueError("validation_fold must be an integer")
    if result["input"] != DEFAULTS["input"]:
        raise ValueError("v0.1 supports only history_ids_v1 with protocol_exact")
    choices = {"openrouter_decisions": {"noul", "choice"}, "openai_compatible_chat": {"json_probability"}, "mock": {"mock"}}
    if model["adapter"] not in choices or model["probability_strategy"] not in choices[model["adapter"]]:
        raise ValueError("Adapter/probability_strategy mismatch")
    assert_keys(model["parameters"], {"temperature", "seed", "max_tokens", "reasoning", "response_format", "provider"}, "model.parameters")
    if model["adapter"] == "openrouter_decisions" and set(model["parameters"]) - {"provider"}:
        raise ValueError("Decisions adapter supports only provider in parameters")
    if model["adapter"] != "mock" and not model["endpoint"].startswith("https://"):
        raise ValueError("Remote endpoint must use HTTPS")
    for key in ("max_concurrency", "max_attempts", "max_requests", "requests_per_minute"):
        if type(ex[key]) is not int or ex[key] < 1:
            raise ValueError(f"execution.{key} must be a positive integer")
    if not isinstance(ex["timeout_seconds"], (float, int)) or not math.isfinite(ex["timeout_seconds"]) or ex["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if ev["profile"] not in {"standard_v1", "pykt_legacy_v1"} or ev["student_std_ddof"] not in {0, 1}:
        raise ValueError("Invalid evaluation profile or ddof")
    if not isinstance(ev["threshold"], (int, float)) or not 0 <= ev["threshold"] <= 1:
        raise ValueError("threshold must be in [0,1]")
    return result


def task_settings(config):
    return {k: config[k] for k in ("dataset", "task", "input")}
