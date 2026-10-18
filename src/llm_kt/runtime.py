"""Bounded remote execution with durable SQLite state and deterministic resume."""
import asyncio
import json
import os
import random
import sqlite3
import csv
import time
import re
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
import httpx
from .common import canonical, digest, read_json, read_jsonl, write_json, write_jsonl, file_hash
from .artifacts import finish_stage, record_event, code_identity, verify_stage
from .protocols import load_prepared
from .providers import make_provider
from .credentials import resolve_api_key


class RateGate:
    def __init__(self, rpm):
        self.interval = 60 / rpm
        self.next_time = 0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            delay = max(0, self.next_time - time.monotonic())
            if delay:
                await asyncio.sleep(delay)
            self.next_time = time.monotonic() + self.interval


def preview(config, prepared, count=2):
    manifest, _, inputs = load_prepared(prepared, config)
    provider = make_provider(config["model"])
    result = {"task_id": manifest["task_id"], "n_targets": len(inputs),
            "max_http_attempts_this_invocation": config["execution"]["max_requests"],
            "sample_requests": [provider.request(item) for item in inputs[:count]]}
    folder = Path(prepared) / "previews" / digest(result)
    if (folder / "artifact_manifest.json").exists():
        verify_stage(folder)
    else:
        write_json(folder / "preview.json", result)
        finish_stage(folder, "preview", {"content_hash": digest(result)}, ["preview.json"], [manifest["task_id"]])
    return result


async def infer(config, prepared, run_directory, budget=None, allow_code_change=False):
    task, _, inputs = load_prepared(prepared, config)
    root = Path(run_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep a self-contained, hashed link to the exact prepared inputs and labels.
    import shutil
    prepared_copy = root / "prepared"
    if prepared_copy.exists():
        verify_stage(prepared_copy)
    else:
        prepared_copy.mkdir()
        for name in ("manifest.json", "inputs.jsonl", "targets.jsonl", "identity_map.jsonl", "artifact_manifest.json"):
            source = Path(prepared) / name
            if source.exists():
                shutil.copyfile(source, prepared_copy / name)
    model, ex = config["model"], config["execution"]
    provider = make_provider(model)
    run_identity = {"schema_version": 2, "config": config, "task_id": task["task_id"], "input_hash": task["input_hash"], "source_sha256": code_identity()["source_sha256"]}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["identity"] != run_identity:
            before = {k:v for k,v in old['identity'].items() if k!='source_sha256'}
            after = {k:v for k,v in run_identity.items() if k!='source_sha256'}
            if not allow_code_change or before != after:
                raise ValueError("Run configuration/data changed; create a new run directory")
            write_json(root / 'code_transitions' / (uuid4().hex + '.json'), {'original_manifest_sha256':file_hash(manifest_path),
                       'original_source_sha256':old['identity']['source_sha256'], 'resumed_source_sha256':run_identity['source_sha256'],
                       'at':datetime.now(timezone.utc).isoformat(), 'reason':'Explicit recovery after infrastructure fix; config and inputs unchanged'})
    else:
        write_json(manifest_path, {"identity": run_identity, "run_id": "run_" + uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(), "code": code_identity(), "package_version": "0.1.0"})
    # A process lock prevents concurrent writers/runners; OS releases it on crash.
    lock = (root / "run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await _execute(config, task, inputs, provider, root, budget)
    finally:
        lock.close()


async def _execute(config, task, inputs, provider, root, budget=None):
    model, ex = config["model"], config["execution"]
    key = None
    db = sqlite3.connect(root / "state.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS jobs(sample_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS attempts(request_id TEXT PRIMARY KEY, sample_id TEXT, status TEXT, latency_ms REAL, usage TEXT, error TEXT)")
    for item in inputs:
        request_hash = digest({"payload": provider.request(item), "model": model, "replicate": ex["replicate_id"], "adapter_version": 1})
        existing = db.execute("SELECT request_hash FROM jobs WHERE sample_id=?", (item.sample_id,)).fetchone()
        if existing and existing[0] != request_hash:
            raise ValueError("Cached request hash mismatch")
        db.execute("INSERT OR IGNORE INTO jobs(sample_id,request_hash,status) VALUES(?,?, 'pending')", (item.sample_id, request_hash))
    db.execute("UPDATE jobs SET status='pending' WHERE status IN ('running','retryable')")
    db.commit()
    gate = RateGate(ex["requests_per_minute"])
    queue = asyncio.Queue()
    pending = []
    for item in inputs:
        state = db.execute("SELECT status,attempts FROM jobs WHERE sample_id=?", (item.sample_id,)).fetchone()
        if state[0] == "pending" and state[1] < ex["max_attempts"]:
            pending.append(item)
        elif state[0] == "success":
            cached = json.loads(db.execute("SELECT result FROM jobs WHERE sample_id=?", (item.sample_id,)).fetchone()[0])
            evidence = read_json(root / "attempts" / (cached["local_request_id"] + ".json"))
            for name, expected in evidence["files"].items():
                if file_hash(root / name) != expected:
                    db.close()
                    raise ValueError("Cached attempt artifact hash mismatch")
            if evidence.get("result_hash") != digest(cached):
                raise ValueError("Cached prediction differs from attempt evidence")
    successors = {}
    if ex.get('student_serial'):
        from collections import defaultdict
        by_student = defaultdict(list)
        mapping = {t['sample_id']:t for t in read_jsonl(root/'prepared'/'targets.jsonl')}
        for item in pending:
            by_student[mapping[item.sample_id]['student_id']].append(item)
        for items in by_student.values():
            items.sort(key=lambda item:mapping[item.sample_id]['event_index'])
            queue.put_nowait(items[0])
            successors.update({a.sample_id:b for a,b in zip(items,items[1:])})
    else:
        for item in pending:
            queue.put_nowait(item)
    if not queue.empty():
        try:
            key = resolve_api_key(model)
        except Exception:
            db.close()
            raise
    invocation_id = "inv_" + uuid4().hex
    invocation_root = root / "invocations" / invocation_id
    write_json(invocation_root / "invocation.json", {"invocation_id": invocation_id, "run_id": read_json(root / "manifest.json")["run_id"], "started_at": datetime.now(timezone.utc).isoformat(), "code": code_identity()})
    dispatched, stop = 0, asyncio.Event()
    budget_exhausted = False
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"} if key else {}
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=ex["timeout_seconds"], follow_redirects=False) as client:
        async def worker():
            nonlocal dispatched, budget_exhausted
            while not queue.empty() and not stop.is_set():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                payload = provider.request(item)
                while not stop.is_set():
                    await gate.wait()
                    if stop.is_set() or dispatched >= ex["max_requests"]:
                        break
                    dispatched += 1
                    state = db.execute("SELECT attempts,request_hash FROM jobs WHERE sample_id=?", (item.sample_id,)).fetchone()
                    attempt = state[0] + 1
                    request_id = "attempt_" + uuid4().hex
                    if budget is not None:
                        try:
                            budget.reserve(request_id, payload, model)
                        except ValueError as exc:
                            if str(exc) != 'Experiment budget would be exceeded':
                                raise
                            dispatched -= 1
                            budget_exhausted = True
                            stop.set()
                            record_event(root, 'budget_stop', task['task_id'], False,
                                         {'reason':'Reservation limit reached; drain existing HTTP requests'})
                            break
                    db.execute("UPDATE jobs SET status='running',attempts=? WHERE sample_id=?", (attempt, item.sample_id))
                    db.commit()
                    write_json(root / "requests" / f"{request_id}.json", payload)
                    body_path = root / "requests" / f"{request_id}.body.txt"
                    body_path.write_bytes(httpx.Request("POST", model["endpoint"], json=payload).content)
                    evidence = {"attempt_id": request_id, "invocation_id": invocation_id, "run_id": read_json(root / "manifest.json")["run_id"], "task_id": task["task_id"], "sample_id": item.sample_id, "request_hash": state[1], "attempt_number": attempt, "started_at": datetime.now(timezone.utc).isoformat(), "endpoint": model["endpoint"], "model_requested": model["model_id"], "status": "running", "files": {}}
                    write_json(root / "attempts" / f"{request_id}.json", evidence)
                    tick = time.monotonic()
                    retryable, fatal, response, error, wait_seconds = False, False, None, None, 0
                    try:
                        if model["adapter"] == "mock":
                            history = item.history
                            response = {"p_correct": (sum(h["correct"] for h in history) + 1) / (len(history) + 2), "model": "mock-history-frequency-v1"}
                        else:
                            http = await asyncio.wait_for(client.post(model["endpoint"], headers=headers, json=payload), timeout=ex.get('total_timeout_seconds'))
                            raw = http.text
                            if key:
                                raw = raw.replace(key, "[REDACTED]")
                            raw = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED]", raw)
                            raw_path = root / "responses" / f"{request_id}.body.txt"
                            raw_path.parent.mkdir(parents=True, exist_ok=True)
                            raw_path.write_text(raw, encoding="utf-8")
                            evidence["credential_redaction_applied"] = raw != http.text
                            evidence["http_status"] = http.status_code
                            evidence["response_headers"] = {k: v for k, v in http.headers.items() if k.lower() in {"content-type", "date", "x-request-id", "request-id", "retry-after"}}
                            if http.status_code >= 400:
                                retryable = http.status_code == 429 or http.status_code in {500, 502, 503, 504}
                                fatal = http.status_code in {400, 401, 403, 404, 422}
                                try:
                                    wait_seconds = min(60, max(0, float(http.headers.get("Retry-After", 0))))
                                except ValueError:
                                    wait_seconds = 0
                                write_json(root / "responses" / f"{request_id}.json", {"http_status": http.status_code})
                                raise ValueError(f"HTTP {http.status_code}")
                            response = json.loads(raw)
                        write_json(root / "responses" / f"{request_id}.json", response)
                        probability = provider.probability(response)
                        usage = response.get("usage") or {}
                        result = {"sample_id": item.sample_id, "p_correct": probability, "status": "success",
                                  "probability_source": model["probability_strategy"], "model_requested": model["model_id"],
                                  "model_returned": response.get("model"), "provider": response.get("provider"),
                                  "request_id": response.get("id", request_id), "local_request_id": request_id,
                                  "attempt_count": attempt, "request_hash": state[1], "usage": usage,
                                  "cost": usage.get("cost"), "latency_ms": (time.monotonic() - tick) * 1000}
                        if model.get('decision_snapshot',{}).get('positive_class')=='incorrect' and model['probability_strategy']=='noul':
                            result['probability_transform']='p_correct = 1 - raw_noul_incorrect'
                        db.execute("UPDATE jobs SET status='success',result=?,error=NULL WHERE sample_id=?", (canonical(result), item.sample_id))
                    except (TimeoutError, httpx.TransportError, ValueError, KeyError, TypeError, IndexError) as exc:
                        if isinstance(exc, TimeoutError):
                            retryable = True
                            error = 'RequestDeadlineExceeded'
                        elif isinstance(exc, httpx.TransportError):
                            retryable = True
                            error = type(exc).__name__
                        else:
                            # Never persist exception text that could contain request credentials.
                            error = str(exc) if str(exc).startswith("HTTP ") else f"InvalidResponse:{type(exc).__name__}"
                        status = "retryable" if retryable and attempt < ex["max_attempts"] else "failed"
                        db.execute("UPDATE jobs SET status=?,error=? WHERE sample_id=?", (status, error, item.sample_id))
                        if fatal:
                            stop.set()
                    usage = response.get("usage", {}) if isinstance(response, dict) else {}
                    if budget is not None:
                        budget.settle(request_id, usage.get('cost'))
                    db.execute("INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?)", (request_id, item.sample_id,
                               "failed" if error else "success", (time.monotonic() - tick) * 1000, canonical(usage), error))
                    db.commit()
                    evidence.update(status="failed" if error else "success", ended_at=datetime.now(timezone.utc).isoformat(), latency_ms=(time.monotonic() - tick) * 1000, error=error, usage=usage)
                    if not error:
                        evidence["result_hash"] = digest(result)
                    for folder in ("requests", "responses"):
                        for path in (root / folder).glob(request_id + ".*"):
                            evidence["files"][path.relative_to(root).as_posix()] = file_hash(path)
                    write_json(root / "attempts" / f"{request_id}.json", evidence)
                    if error and retryable and attempt < ex["max_attempts"] and not stop.is_set():
                        await asyncio.sleep(max(wait_seconds, min(30, 2 ** (attempt - 1) + random.random())))
                        continue
                    break
                if item.sample_id in successors and not stop.is_set():
                    queue.put_nowait(successors[item.sample_id])
                queue.task_done()

        try:
            await asyncio.gather(*(worker() for _ in range(ex["max_concurrency"])))
        finally:
            predictions, failures = [], []
            for sid, status, attempts, result, error in db.execute("SELECT sample_id,status,attempts,result,error FROM jobs ORDER BY sample_id"):
                if status == "success":
                    predictions.append(json.loads(result))
                else:
                    failures.append({"sample_id": sid, "status": status, "attempt_count": attempts, "error": error})
            write_jsonl(root / "predictions.jsonl", predictions)
            write_jsonl(root / "failures.jsonl", failures)
            with (root / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                fields = ["sample_id", "p_correct", "status", "model_requested", "model_returned", "attempt_count", "latency_ms", "cost"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(predictions)
            usage_rows = [json.loads(row[0]) for row in db.execute("SELECT usage FROM attempts")]
            costs = [u.get("cost") for u in usage_rows if isinstance(u, dict)]
            summary = {"task_id": task["task_id"], "n_targets": len(inputs), "n_success": len(predictions),
                       "n_unfinished": len(failures), "attempts_this_invocation": dispatched,
                       "elapsed_seconds": time.monotonic() - started, "fatal_http_error": stop.is_set() and not budget_exhausted,
                       "budget_exhausted": budget_exhausted,
                       "known_cost_usd": sum(c for c in costs if isinstance(c, (int, float))),
                       "unknown_cost_attempts": sum(not isinstance(c, (int, float)) for c in costs),
                       "request_limit": ex["max_requests"]}
            write_json(root / "execution_summary.json", summary)
            for name, value in (("predictions.jsonl", predictions), ("failures.jsonl", failures)):
                write_jsonl(invocation_root / name, value)
            write_json(invocation_root / "execution_summary.json", summary)
            write_json(invocation_root / "attempt_index.json", {p.name: file_hash(p) for p in (root / "attempts").glob("*.json")})
            finish_stage(invocation_root, "infer", {"invocation_id": invocation_id, "task_id": task["task_id"]}, ["invocation.json", "predictions.jsonl", "failures.jsonl", "execution_summary.json", "attempt_index.json"], [task["task_id"]])
            db.close()
    return summary


def verify_run(directory):
    root = Path(directory).resolve()
    manifest = read_json(root / "manifest.json")
    verify_stage(root / "prepared")
    count = 0
    for folder in (root / "invocations").iterdir():
        verify_stage(folder)
        for name, expected in read_json(folder / "attempt_index.json").items():
            path = root / "attempts" / name
            if path.parent.resolve() != (root / "attempts").resolve() or file_hash(path) != expected:
                raise ValueError("Attempt metadata changed")
    for path in (root / "attempts").glob("*.json"):
        attempt = read_json(path)
        for name, expected in attempt["files"].items():
            target = (root / name).resolve()
            if not target.is_relative_to(root) or file_hash(target) != expected:
                raise ValueError("Attempt payload changed")
        count += 1
    for prediction in __import__('llm_kt.common', fromlist=['read_jsonl']).read_jsonl(root / 'predictions.jsonl'):
        evidence = read_json(root / 'attempts' / (prediction['local_request_id'] + '.json'))
        if evidence.get('result_hash') != digest(prediction):
            raise ValueError('Prediction evidence mismatch')
    return {"run_id": manifest["run_id"], "verified_attempts": count}
