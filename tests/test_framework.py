import asyncio
import ast
import csv
import json
import os
import tempfile
import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from llm_kt.common import write_json, read_json, read_jsonl, digest
from llm_kt.config import load_config
from llm_kt.data import import_snapshot, read_sequences
from llm_kt.protocols import prepare, load_prepared
from llm_kt.domain import ModelInput, checked_probability
from llm_kt.providers import Chat, Decisions
from llm_kt.runtime import infer
from llm_kt.evaluation import calculate, evaluate, compare
from llm_kt.vendor.pykt.sequence import target_ranges
from llm_kt.credentials import resolve_api_key

FIXTURE = Path(__file__).parent / "fixtures/test_quelevel.csv"


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / f"work_{uuid.uuid4().hex}"
        self.root.mkdir()
        import_snapshot(FIXTURE, self.root / "snapshot", "demo")
        self.config_path = self.root / "config.json"
        self.raw = {"dataset": {"snapshot": "snapshot"}, "task": {"maxlen": 4},
                    "model": {"adapter": "mock", "model_id": "mock", "probability_strategy": "mock"},
                    "execution": {"requests_per_minute": 600000, "max_requests": 100}}
        write_json(self.config_path, self.raw)
        self.config = load_config(self.config_path)

    def tearDown(self):
        resolved = self.root.resolve()
        if resolved.parent != Path(__file__).parent.resolve() or not resolved.name.startswith("work_"):
            raise RuntimeError("Unsafe test cleanup path")
        shutil.rmtree(resolved)

    def prepared(self):
        path = self.root / "prepared"
        prepare(self.config, path)
        return path

    def test_protocol_boundaries(self):
        for n in (2, 3, 199, 200, 201, 400):
            windows = list(target_ranges(n, "pykt_window"))
            self.assertEqual(len(windows), n - 1)
            self.assertEqual(windows[-1], (max(0, n - 200), n - 1))
            segmented = list(target_ranges(n, "pykt_segmented"))
            expected = [(s, t) for s in range(0, n, 200) if min(200, n - s) >= 3 for t in range(s + 1, min(s + 200, n))]
            self.assertEqual(segmented, expected)

    def test_protocol_matches_copied_pykt_generators(self):
        import numpy as np
        # Execute only the copied pure sequence generation functions. No pykt imports.
        class FakeFrame:
            def __init__(self, data):
                self.data = data
            def iterrows(self):
                yield 0, self.data
        class FakePandas:
            @staticmethod
            def DataFrame(data):
                return data
        source = (Path(__file__).parents[1] / "third_party/pykt/reference/split_datasets_que.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"generate_sequences", "generate_window_sequences"}]
        namespace = {"np": np, "pd": FakePandas, "ALL_KEYS": ["uid", "responses", "selectmasks"],
                     "ONE_KEYS": ["uid"], "save_dcur": lambda row, keys: {k: row[k] if k == "uid" else row[k].split(",") for k in keys}}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "copied_pykt", "exec"), namespace)
        for n in (2, 3, 199, 200, 201, 400):
            frame = FakeFrame({"uid": "s", "responses": ",".join(map(str, range(n)))})
            for protocol, function in (("pykt_window", "generate_window_sequences"), ("pykt_segmented", "generate_sequences")):
                if n == 2 and protocol == "pykt_segmented":
                    # Upstream assumes preprocessing already filtered all short students.
                    with self.assertRaises(KeyError):
                        namespace[function](frame, ["uid", "responses"])
                    self.assertEqual(list(target_ranges(n, protocol)), [])
                    continue
                output = namespace[function](frame, ["uid", "responses"])
                actual = []
                for values, masks in zip(output["responses"], output["selectmasks"]):
                    events = [int(v) for v in values.split(",")]
                    selected = [int(v) for v in masks.split(",")]
                    for index in range(1, len(events)):
                        if selected[index] != -1:
                            actual.append((events[0], events[index]))
                self.assertEqual(actual, list(target_ranges(n, protocol)))

    def test_no_current_or_future_labels_and_history_sensitive(self):
        prepared = self.prepared()
        _, targets, items = load_prepared(prepared)
        original = items[0].state()
        source = self.root / "changed.csv"
        text = FIXTURE.read_text().replace('"1,0,1,0,1,0"', '"1,1,0,1,0,1"')
        source.write_text(text)
        import_snapshot(source, self.root / "snapshot2", "demo")
        self.config["dataset"]["snapshot"] = str(self.root / "snapshot2")
        prepare(self.config, self.root / "p2")
        _, _, changed = load_prepared(self.root / "p2")
        self.assertEqual(original, changed[0].state())
        self.assertNotEqual(items[1].state(), changed[1].state())
        self.assertNotIn("y_true", json.dumps(original))
        self.assertEqual(len({t["event_id"] for t in targets}), len(targets))

    def test_end_to_end_and_resume_no_duplicates(self):
        prepared = self.prepared()
        run = self.root / "run"
        first = asyncio.run(infer(self.config, prepared, run))
        self.assertEqual(first["n_success"], 10)
        second = asyncio.run(infer(self.config, prepared, run))
        self.assertEqual(second["attempts_this_invocation"], 0)
        report = evaluate(prepared, run)
        self.assertTrue(report["complete"])
        self.assertEqual(report["coverage"]["n_valid_auc_students"], 2)
        self.assertEqual(len(read_jsonl(run / "predictions.jsonl")), 10)

    def test_request_cap_and_resume(self):
        prepared = self.prepared()
        self.config["execution"]["max_requests"] = 2
        run = self.root / "run"
        first = asyncio.run(infer(self.config, prepared, run))
        self.assertEqual(first["n_success"], 2)
        second = asyncio.run(infer(self.config, prepared, run))
        self.assertEqual(second["n_success"], 4)
        self.assertFalse(evaluate(prepared, run)["complete"])

    def test_strict_config_and_snapshot_hash(self):
        self.raw["execution"]["typo"] = 1
        write_json(self.config_path, self.raw)
        with self.assertRaises(ValueError):
            load_config(self.config_path)
        (self.root / "snapshot/test.csv").write_text("changed")
        with self.assertRaises(ValueError):
            self.prepared()

    def test_train_validation_folds_and_student_isolation(self):
        from llm_kt.data import load_snapshot
        train = self.root / "train.csv"
        train.write_text('fold,uid,questions,concepts,responses\n0,t0,"1,2,3","1,2,3","0,1,0"\n1,t1,"1,2,3","1,2,3","1,0,1"\n')
        import_snapshot(FIXTURE, self.root / "with_train", "demo", train_source=train)
        self.config["dataset"].update(snapshot=str(self.root / "with_train"), split="validation", validation_fold=0)
        _, rows = load_snapshot(self.config)
        self.assertEqual([r["student_id"] for r in rows], ["t0"])
        self.config["dataset"]["split"] = "train"
        _, rows = load_snapshot(self.config)
        self.assertEqual([r["student_id"] for r in rows], ["t1"])
        train.write_text(train.read_text().replace("t0", "s1"))
        with self.assertRaises(ValueError):
            import_snapshot(FIXTURE, self.root / "overlap", "demo", train_source=train)

    def test_probability_parsing(self):
        model = {"model_id": "test", "probability_strategy": "noul", "parameters": {}}
        decisions = Decisions(model)
        self.assertEqual(decisions.probability({"answers": {"will_answer_correctly": {"type": "noul", "noul": 0.7}}}), 0.7)
        chat = Chat(model)
        self.assertEqual(chat.probability({"choices": [{"finish_reason": "stop", "message": {"content": '{"p_correct":0.7}'}}]}), 0.7)
        for bad in (True, "0.7", float("nan"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                checked_probability(bad)
        with self.assertRaises(ValueError):
            chat.probability({"choices": [{"finish_reason": "length", "message": {"content": '{"p_correct":0.7}'}}]})

    def test_known_auc_and_single_class(self):
        targets = [{"sample_id": str(i), "student_id": "s", "event_index": i, "y_true": y} for i, y in enumerate([0, 1, 0, 1])]
        predictions = [{"sample_id": str(i), "status": "success", "p_correct": p} for i, p in enumerate([0.1, 0.8, 0.6, 0.4])]
        cfg = self.config["evaluation"]
        report, _ = calculate(targets, predictions, cfg)
        self.assertEqual(report["overall"]["auc"], 0.75)
        self.assertIsNone(report["student_auc"]["std"])
        for row in targets:
            row["y_true"] = 1
        standard, _ = calculate(targets, predictions, cfg)
        legacy, _ = calculate(targets, predictions, cfg, True)
        self.assertIsNone(standard["overall"]["auc"])
        self.assertEqual(legacy["overall"]["auc"], 0.5)

    def test_concept_grouping_and_duplicate_student_rejection(self):
        path = self.root / "expanded.csv"
        path.write_text('fold,uid,questions,concepts,responses,is_repeat\n-1,s,"1,1,2","2,7,3","1,1,0","0,1,0"\n')
        rows = read_sequences(path, "concept_expanded")
        self.assertEqual(rows[0]["events"][0]["concept_ids"], ["2", "7"])
        self.assertEqual(len(rows[0]["events"]), 2)
        path.write_text(path.read_text() + '-1,s,3,3,0,0\n')
        with self.assertRaises(ValueError):
            read_sequences(path, "concept_expanded")

    def test_mock_http_retry_and_headers(self):
        prepared = self.prepared()
        self.config["model"] = {"adapter": "openrouter_decisions", "model_id": "test", "endpoint": "https://example.invalid/decisions", "api_key_env": "TEST_KT_KEY", "probability_strategy": "noul", "parameters": {}}
        key_file = self.root / "test.key"
        key_file.write_text("secret", encoding="utf-8")
        self.config["model"]["api_key_file"] = str(key_file)
        self.config["execution"].update(max_concurrency=1, max_requests=2)
        calls = []
        async def post(client, url, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={"answers": {"will_answer_correctly": {"type": "noul", "noul": 0.7}}, "usage": {"cost": 0.01}})
        with patch.dict(os.environ, {"TEST_KT_KEY": ""}), patch("httpx.AsyncClient.post", post), patch("llm_kt.runtime.random.random", return_value=0):
            result = asyncio.run(infer(self.config, prepared, self.root / "remote"))
        self.assertEqual(result["n_success"], 1)
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer secret")
        self.assertNotIn("decisionsRequest", calls[0]["json"])
        self.assertNotIn("secret", (self.root / "remote/manifest.json").read_text())
        for directory in ("requests", "responses"):
            for path in (self.root / "remote" / directory).glob("*.json"):
                self.assertNotIn("secret", path.read_text())

    def test_credential_file_and_environment_priority(self):
        path = self.root / "test.key"
        path.write_text("\ufefffake-test-key\n", encoding="utf-8")
        model = {"adapter": "openrouter_decisions", "api_key_env": "TEST_KT_KEY", "api_key_file": str(path)}
        with patch.dict(os.environ, {"TEST_KT_KEY": ""}):
            self.assertEqual(resolve_api_key(model), "fake-test-key")
            path.write_text("")
            with self.assertRaises(ValueError):
                resolve_api_key(model)
        with patch.dict(os.environ, {"TEST_KT_KEY": "environment-key"}):
            self.assertEqual(resolve_api_key(model), "environment-key")

    def test_compare_task_and_partial_coverage(self):
        prepared = self.prepared()
        self.config["execution"]["max_requests"] = 2
        runs = [self.root / "r1", self.root / "r2"]
        for run in runs:
            asyncio.run(infer(self.config, prepared, run))
        report = compare(prepared, runs, self.root / "comparison.json")
        self.assertEqual(report["comparison"], "exploratory_common_subset")
        self.assertEqual(report["n_common"], 2)


if __name__ == "__main__":
    unittest.main()
