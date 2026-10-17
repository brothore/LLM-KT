"""Offline request-volume and token-cost estimates; never loads credentials."""
import math
from pathlib import Path
from .common import canonical, digest, read_json, read_jsonl, write_json
from .data import stream_snapshot
from .domain import ModelInput
from .providers import make_provider
from .vendor.pykt.sequence import target_ranges


def input_characters(payload, adapter):
    """Count a documented text proxy, not HTTP JSON bytes or exact billed tokens."""
    if adapter == "openai_compatible_chat":
        total = sum(len(message["content"]) for message in payload["messages"])
        if "response_format" in payload:
            total += len(canonical(payload["response_format"]))
        return total
    if adapter == "openrouter_decisions":
        return len(canonical({"state": payload["state"], "questions": payload["questions"]}))
    return len(canonical(payload["state"]))


def request_character_totals(config):
    """Exact character proxy for all requests in O(events+targets), without expanding histories."""
    snapshot, rows = stream_snapshot(config)
    task, model = config["task"], config["model"]
    if snapshot["representation"] == "concept_expanded" and task["protocol"] != "full_prefix":
        raise ValueError("Expanded concept input requires full_prefix in v0.1")
    provider = make_provider(model)
    empty = ModelInput("", [], {"question_id": None, "concept_ids": []})
    constant = input_characters(provider.request(empty), model["adapter"]) - len(canonical(empty.current_item))
    requests = characters = history_events = students = students_with_targets = max_history = 0
    max_chars = 0
    for row in rows:
        students += 1
        events = row["events"]
        if task["level"] == "concept" and any(len(e["concept_ids"]) != 1 for e in events):
            raise ValueError("Concept mode requires single-concept input")
        prefix = [0]
        item_lengths = []
        for event in events:
            item = {k: event[k] for k in ("question_id", "concept_ids")}
            item_lengths.append(len(canonical(item)))
            prefix.append(prefix[-1] + len(canonical({**item, "correct": bool(event["response"])})))
        contributed = False
        for start, index in target_ranges(len(events), task["protocol"], task["maxlen"], task["min_seq_len"]):
            length = index - start
            chars = constant + item_lengths[index] + prefix[index] - prefix[start] + max(0, length - 1)
            requests += 1
            characters += chars
            history_events += length
            max_history = max(max_history, length)
            max_chars = max(max_chars, chars)
            contributed = True
        students_with_targets += int(contributed)
    if not requests:
        raise ValueError("No eligible targets to estimate")
    return {"dataset": snapshot["dataset"], "snapshot_id": snapshot["snapshot_id"],
            "split": config["dataset"]["split"], "validation_fold": config["dataset"]["validation_fold"],
            "protocol": task, "n_students": students, "n_students_with_targets": students_with_targets,
            "n_requests": requests, "input_character_proxy": characters, "max_request_characters": max_chars,
            "history_events_total": history_events, "mean_history_events": history_events / requests,
            "max_history_events": max_history}


def calibration(config, run_directory):
    root = Path(run_directory).resolve()
    manifest = read_json(root / "manifest.json")["identity"]
    recorded = manifest["config"]["model"]
    for field in ("adapter", "model_id", "endpoint", "probability_strategy", "parameters"):
        if recorded[field] != config["model"][field]:
            raise ValueError(f"Calibration model differs: {field}")
    characters = tokens = count = 0
    output_tokens, outputs_complete = 0, True
    for prediction in read_jsonl(root / "predictions.jsonl"):
        usage = prediction.get("usage") or {}
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(prompt, bool) or not isinstance(prompt, (float, int)) or not math.isfinite(prompt) or prompt <= 0:
            continue
        path = (root / "requests" / (prediction["local_request_id"] + ".json")).resolve()
        if path.parent != root / "requests":
            raise ValueError("Unsafe calibration request path")
        characters += input_characters(read_json(path), config["model"]["adapter"])
        tokens += prompt
        count += 1
        if isinstance(completion, (int, float)) and not isinstance(completion, bool) and math.isfinite(completion) and completion >= 0:
            output_tokens += completion
        else:
            outputs_complete = False
    if not count or not characters:
        raise ValueError("Calibration run has no usable billed input-token samples")
    return {"run": str(root), "samples": count, "input_characters_per_token": characters / tokens,
            "mean_output_tokens": output_tokens / count if outputs_complete else None}


def cost_for_endpoint(endpoint, input_tokens, output_tokens, read_fraction=0, write_fraction=0):
    read_price = endpoint.get("cache_read_per_million_usd")
    write_price = endpoint.get("cache_write_per_million_usd")
    if read_fraction and (not endpoint.get("supports_implicit_caching") or read_price is None):
        raise ValueError(f"Cache-read scenario unsupported/unpriced for {endpoint['provider']}")
    if write_fraction and write_price is None:
        raise ValueError(f"Cache-write scenario unpriced for {endpoint['provider']}")
    ordinary = input_tokens * (1 - read_fraction - write_fraction) * endpoint["input_per_million_usd"] / 1e6
    reads = input_tokens * read_fraction * (read_price or 0) / 1e6
    writes = input_tokens * write_fraction * (write_price or 0) / 1e6
    output = output_tokens * endpoint["output_per_million_usd"] / 1e6
    return {"input_usd": ordinary + reads + writes, "output_usd": output, "total_usd": ordinary + reads + writes + output}


def estimate(config, *, provider=None, chars_per_token=3, chars_per_token_low=2,
             chars_per_token_high=4, output_tokens=None, cache_read_fraction=0, cache_write_fraction=0,
             local_hit_fraction=0, extra_attempt_fraction=0, calibration_run=None):
    for value in (chars_per_token, chars_per_token_low, chars_per_token_high):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Characters-per-token assumptions must be positive and finite")
    if not chars_per_token_low <= chars_per_token <= chars_per_token_high:
        raise ValueError("Expected low <= central <= high characters-per-token")
    for name, value in (("cache read", cache_read_fraction), ("cache write", cache_write_fraction), ("local hit", local_hit_fraction)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} fraction must be in [0,1]")
    if cache_read_fraction + cache_write_fraction > 1:
        raise ValueError("Cache-read and cache-write token fractions cannot exceed 1 together")
    if not math.isfinite(extra_attempt_fraction) or extra_attempt_fraction < 0:
        raise ValueError("Extra attempt fraction must be finite and nonnegative")
    model = config["model"]
    pricing = model.get("pricing_snapshot")
    if not pricing:
        raise ValueError("Cost estimation needs model.pricing_file with validated rates")
    endpoints = pricing["endpoints"]
    route = model["parameters"].get("provider", {})
    selectors = [provider] if provider else route.get("only", [])
    if not selectors and route.get("allow_fallbacks") is False:
        selectors = route.get("order", [])
    if selectors:
        if not isinstance(selectors, list) or not all(isinstance(s, str) and s for s in selectors):
            raise ValueError("Provider selectors must be a list of nonempty names/tags")
        for selector in selectors:
            if not any(selector.casefold() in {e["provider"].casefold(), e["tag"].casefold(), e["tag"].split('/')[0].casefold()} for e in endpoints):
                raise ValueError(f"No price recorded for requested provider: {selector}")
        endpoints = [e for e in endpoints if any(s.casefold() in {e["provider"].casefold(), e["tag"].casefold(), e["tag"].split('/')[0].casefold()} for s in selectors)]
        if not endpoints:
            raise ValueError("No priced endpoints match the requested provider")
    measured = calibration(config, calibration_run) if calibration_run else None
    if measured:
        chars_per_token = measured["input_characters_per_token"]
        # Sensitivity bounds around measured average, not statistical confidence bounds.
        chars_per_token_low, chars_per_token_high = chars_per_token * 0.8, chars_per_token * 1.2
    if output_tokens is None:
        output_tokens = measured["mean_output_tokens"] if measured and measured["mean_output_tokens"] is not None else (0 if model["adapter"] == "openrouter_decisions" else 32)
    if not math.isfinite(output_tokens) or output_tokens < 0:
        raise ValueError("Expected output tokens must be nonnegative and finite")
    volume = request_character_totals(config)
    dispatched = volume["n_requests"] * (1 - local_hit_fraction)
    attempt_factor = 1 + extra_attempt_fraction
    input_central = volume["input_character_proxy"] / chars_per_token * (1 - local_hit_fraction) * attempt_factor
    lower = volume["input_character_proxy"] / chars_per_token_high * (1 - local_hit_fraction) * attempt_factor
    upper = volume["input_character_proxy"] / chars_per_token_low * (1 - local_hit_fraction) * attempt_factor
    outputs = dispatched * output_tokens * attempt_factor
    rows = []
    for endpoint in endpoints:
        rows.append({"provider": endpoint["provider"], "tag": endpoint["tag"], "rates": endpoint,
                     "central": cost_for_endpoint(endpoint, input_central, outputs, cache_read_fraction, cache_write_fraction),
                     "input_token_sensitivity_low": cost_for_endpoint(endpoint, lower, outputs, cache_read_fraction, cache_write_fraction),
                     "input_token_sensitivity_high": cost_for_endpoint(endpoint, upper, outputs, cache_read_fraction, cache_write_fraction)})
    warnings = ["Character-based token proxy; not exact provider tokenization or a statistical confidence interval.",
                "Provider framing and internal Decisions serialization may differ from this proxy.",
                "Expected output tokens include any billed reasoning tokens; the default is an assumption, not a measurement.",
                "Full selected split is estimated; execution.max_requests limits one invocation, not the complete dataset cost.",
                "No automatic cache benefit is assumed. Cache/local-hit fractions are hypothetical homogeneous averages.",
                "Extra attempts are modeled at full per-request cost; actual failed-request billing may differ.",
                "Prices are dated snapshots; excludes tax, payment fees, batch discounts and unlisted surcharges."]
    return {"schema_version": 1, "model_id": model["model_id"], "volume": volume, "pricing": pricing,
            "pricing_hash": digest(pricing), "currency": "USD", "calibration": measured,
            "assumptions": {"chars_per_token": chars_per_token, "chars_per_token_low": chars_per_token_low,
                            "chars_per_token_high": chars_per_token_high, "expected_output_tokens_per_request": output_tokens,
                            "cache_read_token_fraction": cache_read_fraction, "cache_write_token_fraction": cache_write_fraction,
                            "local_result_hit_fraction": local_hit_fraction, "extra_attempt_fraction": extra_attempt_fraction,
                            "provider_override_is_estimate_only": bool(provider)},
            "estimated_tokens": {"input": input_central, "input_sensitivity_low": lower, "input_sensitivity_high": upper, "output": outputs},
            "estimated_remote_requests": dispatched, "estimated_attempts": dispatched * attempt_factor,
            "execution_request_cap": config["execution"]["max_requests"], "endpoint_estimates": rows,
            "central_cost_range_usd": {"min": min(r["central"]["total_usd"] for r in rows), "max": max(r["central"]["total_usd"] for r in rows)},
            "warnings": warnings}


def save_estimate(path, result):
    path = Path(path)
    write_json(path, result)
    volume = result["volume"]
    lines = ["# KT inference cost estimate", "", f"Dataset: {volume['dataset']} / {volume['split']}",
             f"Model: `{result['model_id']}`", f"Requests: {volume['n_requests']:,}",
             f"Price snapshot: {result['pricing']['verified_at']}", "",
             "| Provider | Input USD | Output USD | Total USD | Input-token sensitivity USD |",
             "|---|---:|---:|---:|---:|"]
    for row in result["endpoint_estimates"]:
        cur = row["central"]
        lines.append(f"| {row['provider']} | {cur['input_usd']:.4f} | {cur['output_usd']:.4f} | {cur['total_usd']:.4f} | {row['input_token_sensitivity_low']['total_usd']:.4f}–{row['input_token_sensitivity_high']['total_usd']:.4f} |")
    lines += ["", "These are conditional estimates, not quotes or spending limits.", "", "Assumptions:", "", "```json", canonical(result["assumptions"]), "```", "", *["- " + w for w in result["warnings"]]]
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
