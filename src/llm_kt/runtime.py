"""Bounded remote execution with durable SQLite state and deterministic resume."""
import asyncio
import json
import os
import random
import sqlite3
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
from .common import canonical, digest, read_json, write_json, write_jsonl
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
    return {"task_id": manifest["task_id"], "n_targets": len(inputs),
            "max_http_attempts_this_invocation": config["execution"]["max_requests"],
            "sample_requests": [provider.request(item) for item in inputs[:count]]}


async def infer(config, prepared, run_directory):
    task, _, inputs = load_prepared(prepared, config)
    root = Path(run_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model, ex = config["model"], config["execution"]
    provider = make_provider(model)
    run_identity = {"schema_version": 1, "config": config, "task_id": task["task_id"], "input_hash": task["input_hash"]}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["identity"] != run_identity:
            raise ValueError("Run configuration/data changed; create a new run directory")
    else:
        write_json(manifest_path, {"identity": run_identity, "created_at": datetime.now(timezone.utc).isoformat(), "package_version": "0.1.0"})
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
        return await _execute(config, task, inputs, provider, root)
    finally:
        lock.close()


async def _execute(config, task, inputs, provider, root):
    model, ex = config["model"], config["execution"]
    key = resolve_api_key(model)
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
    for item in inputs:
        state = db.execute("SELECT status,attempts FROM jobs WHERE sample_id=?", (item.sample_id,)).fetchone()
        if state[0] == "pending" and state[1] < ex["max_attempts"]:
            queue.put_nowait(item)
    dispatched, stop = 0, asyncio.Event()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"} if key else {}
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=ex["timeout_seconds"], follow_redirects=False) as client:
        async def worker():
            nonlocal dispatched
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
                    request_id = f"{item.sample_id}_{attempt}"
                    db.execute("UPDATE jobs SET status='running',attempts=? WHERE sample_id=?", (attempt, item.sample_id))
                    db.commit()
                    write_json(root / "requests" / f"{request_id}.json", payload)
                    tick = time.monotonic()
                    retryable, fatal, response, error, wait_seconds = False, False, None, None, 0
                    try:
                        if model["adapter"] == "mock":
                            history = item.history
                            response = {"p_correct": (sum(h["correct"] for h in history) + 1) / (len(history) + 2), "model": "mock-history-frequency-v1"}
                        else:
                            http = await client.post(model["endpoint"], headers=headers, json=payload)
                            if http.status_code >= 400:
                                retryable = http.status_code == 429 or http.status_code in {500, 502, 503, 504}
                                fatal = http.status_code in {400, 401, 403, 404, 422}
                                try:
                                    wait_seconds = min(60, max(0, float(http.headers.get("Retry-After", 0))))
                                except ValueError:
                                    wait_seconds = 0
                                write_json(root / "responses" / f"{request_id}.json", {"http_status": http.status_code})
                                raise ValueError(f"HTTP {http.status_code}")
                            response = http.json()
                        write_json(root / "responses" / f"{request_id}.json", response)
                        probability = provider.probability(response)
                        usage = response.get("usage") or {}
                        result = {"sample_id": item.sample_id, "p_correct": probability, "status": "success",
                                  "probability_source": model["probability_strategy"], "model_requested": model["model_id"],
                                  "model_returned": response.get("model"), "provider": response.get("provider"),
                                  "request_id": response.get("id", request_id), "local_request_id": request_id,
                                  "attempt_count": attempt, "request_hash": state[1], "usage": usage,
                                  "cost": usage.get("cost"), "latency_ms": (time.monotonic() - tick) * 1000}
                        db.execute("UPDATE jobs SET status='success',result=?,error=NULL WHERE sample_id=?", (canonical(result), item.sample_id))
                    except (httpx.TransportError, ValueError, KeyError, TypeError, IndexError) as exc:
                        if isinstance(exc, httpx.TransportError):
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
                    db.execute("INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?)", (request_id, item.sample_id,
                               "failed" if error else "success", (time.monotonic() - tick) * 1000, canonical(usage), error))
                    db.commit()
                    if error and retryable and attempt < ex["max_attempts"] and not stop.is_set():
                        await asyncio.sleep(max(wait_seconds, min(30, 2 ** (attempt - 1) + random.random())))
                        continue
                    break
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
                       "elapsed_seconds": time.monotonic() - started, "fatal_http_error": stop.is_set(),
                       "known_cost_usd": sum(c for c in costs if isinstance(c, (int, float))),
                       "unknown_cost_attempts": sum(not isinstance(c, (int, float)) for c in costs),
                       "request_limit": ex["max_requests"]}
            write_json(root / "execution_summary.json", summary)
            db.close()
    return summary
