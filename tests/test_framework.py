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

    def test_subset_selection_matches_cost_and_context(self):
        from llm_kt.subsets import create_subset
        from llm_kt.costs import request_character_totals
        train = self.root / 'train.csv'
        train.write_text('fold,uid,questions,concepts,responses\n0,t0,"1,2,3,4","1,2,3,4","0,1,0,1"\n1,t1,"1,2,3,4","1,2,3,4","1,0,1,0"\n')
        parent = self.root / 'parent'
        import_snapshot(FIXTURE, parent, 'demo', train_source=train)
        subset = self.root / 'subset'
        report = create_subset(parent, subset, 'sample', 0.5, 42)
        self.assertEqual(report, create_subset(parent, subset, 'sample', 0.5, 42))
        self.config['dataset']['snapshot'] = str(subset)
        self.config['task'].update(maxlen=200, target_limit=2)
        prepared = self.root / 'selected'
        manifest = prepare(self.config, prepared)
        self.assertEqual(manifest['n_targets'], 2)
        self.assertEqual(request_character_totals(self.config)['n_requests'], 2)
        _, targets, inputs = load_prepared(prepared)
        original = {r['student_id']: r for r in read_sequences(FIXTURE, 'question_native')}
        for target, item in zip(targets, inputs):
            expected = original[target['student_id']]['events'][target['history_start']:target['event_index']]
            self.assertEqual([h['correct'] for h in item.history], [bool(e['response']) for e in expected])
            self.assertNotIn('response', item.current_item)

    def test_trace_integrity_and_zero_request_resume(self):
        from llm_kt.runtime import verify_run
        prepared = self.prepared()
        run = self.root / 'trace'
        first = asyncio.run(infer(self.config, prepared, run))
        self.assertGreater(first['attempts_this_invocation'], 0)
        with patch('llm_kt.runtime.resolve_api_key', side_effect=AssertionError('Cache must not read credentials')):
            second = asyncio.run(infer(self.config, prepared, run))
        self.assertEqual(second['attempts_this_invocation'], 0)
        self.assertEqual(verify_run(run)['verified_attempts'], first['n_success'])
        prediction = read_jsonl(run / 'predictions.jsonl')[0]
        response = run / 'responses' / (prediction['local_request_id'] + '.json')
        response.write_text('{}')
        with self.assertRaises(ValueError):
            verify_run(run)
        with self.assertRaises(ValueError):
            asyncio.run(infer(self.config, prepared, run))

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

    def test_student_serial_runs_students_concurrently_without_internal_overlap(self):
        prepared=self.prepared()
        self.config['model'].update(adapter='openai_compatible_chat',model_id='synthetic',probability_strategy='json_probability')
        self.config['execution'].update(student_serial=True,max_concurrency=8)
        async def fake_post(client,url,**kwargs):
            await asyncio.sleep(.01)
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{"p_correct":0.6}'}}]})
        run=self.root/'serial'
        with patch('llm_kt.runtime.resolve_api_key',return_value=''), patch('httpx.AsyncClient.post',fake_post):
            result=asyncio.run(infer(self.config,prepared,run))
        self.assertEqual(result['n_success'],10)
        _,targets,_=load_prepared(prepared)
        mapping={t['sample_id']:t for t in targets}
        groups={}
        for path in (run/'attempts').glob('*.json'):
            a=read_json(path);t=mapping[a['sample_id']]
            groups.setdefault(t['student_id'],[]).append((t['event_index'],a))
        for group in groups.values():
            group.sort()
            for (_,a),(_,b) in zip(group,group[1:]):
                self.assertLessEqual(a['ended_at'],b['started_at'])
        spans=[(min(a['started_at'] for _,a in g),max(a['ended_at'] for _,a in g)) for g in groups.values()]
        self.assertTrue(any(a[0]<b[1] and b[0]<a[1] for i,a in enumerate(spans) for b in spans[i+1:]))

    def test_budget_stop_drains_inflight_requests_and_finishes_evidence(self):
        prepared=self.prepared()
        self.config['model'].update(adapter='openai_compatible_chat',model_id='synthetic',probability_strategy='json_probability')
        class OneRequestBudget:
            used=False
            def reserve(self,*args):
                if self.used:raise ValueError('Experiment budget would be exceeded')
                self.used=True
            def settle(self,*args):pass
        async def fake_post(client,url,**kwargs):
            await asyncio.sleep(.01)
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{"p_correct":0.6}'}}]})
        run=self.root/'budget_stop'
        with patch('llm_kt.runtime.resolve_api_key',return_value=''),patch('httpx.AsyncClient.post',fake_post):
            result=asyncio.run(infer(self.config,prepared,run,OneRequestBudget()))
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(result['n_success'],1)
        self.assertEqual(result['attempts_this_invocation'],1)
        from llm_kt.runtime import verify_run
        self.assertEqual(verify_run(run)['verified_attempts'],1)
        self.assertTrue(all(read_json(p)['status']=='success' for p in (run/'attempts').glob('*.json')))

    def test_total_deadline_records_timeout_without_losing_evidence(self):
        prepared=self.prepared()
        self.config['model'].update(adapter='openai_compatible_chat',model_id='synthetic',probability_strategy='json_probability')
        self.config['execution'].update(total_timeout_seconds=.005,max_concurrency=1,max_requests=1,max_attempts=1)
        async def slow_post(*args,**kwargs):await asyncio.sleep(.2)
        run=self.root/'deadline'
        with patch('llm_kt.runtime.resolve_api_key',return_value=''),patch('httpx.AsyncClient.post',slow_post):
            result=asyncio.run(infer(self.config,prepared,run))
        self.assertEqual(result['n_success'],0)
        attempts=[read_json(p) for p in (run/'attempts').glob('*.json')]
        self.assertEqual(len(attempts),1)
        self.assertEqual(attempts[0]['error'],'RequestDeadlineExceeded')
        self.assertEqual(attempts[0]['status'],'failed')
        from llm_kt.runtime import verify_run
        self.assertEqual(verify_run(run)['verified_attempts'],1)

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

    def test_explicit_transport_recovery_preserves_successes_and_is_bounded(self):
        prepared=self.prepared()
        self.config['model']={'adapter':'openrouter_decisions','model_id':'test','endpoint':'https://example.invalid','api_key_env':'TEST_KT_KEY','probability_strategy':'noul','parameters':{}}
        self.config['execution'].update(max_concurrency=1,max_attempts=1)
        calls=[]
        async def post(client,url,**kwargs):
            calls.append(kwargs['json'])
            if len(calls)==1:raise httpx.ReadTimeout('synthetic')
            return httpx.Response(200,json={'answers':{'will_answer_correctly':{'type':'noul','noul':.7}}})
        run=self.root/'recovery'
        with patch.dict(os.environ,{'TEST_KT_KEY':'synthetic'}),patch('httpx.AsyncClient.post',post):
            first=asyncio.run(infer(self.config,prepared,run))
            old={p.name:p.read_bytes() for p in (run/'attempts').glob('*.json')}
            n=len(calls)
            normal=asyncio.run(infer(self.config,prepared,run))
            self.assertEqual(len(calls),n)
            recovered=asyncio.run(infer(self.config,prepared,run,retry_failed_once=True))
            self.assertEqual(len(calls),n+1)
            self.assertEqual(recovered['n_success'],first['n_success']+1)
            for name,data in old.items():self.assertEqual((run/'attempts'/name).read_bytes(),data)
            asyncio.run(infer(self.config,prepared,run,retry_failed_once=True))
            self.assertEqual(len(calls),n+1)

    def test_failed_recovery_cannot_exceed_one_extra_attempt(self):
        prepared=self.prepared()
        self.config['model']={'adapter':'openrouter_decisions','model_id':'test','endpoint':'https://example.invalid','api_key_env':'TEST_KT_KEY','probability_strategy':'noul','parameters':{}}
        self.config['execution'].update(max_concurrency=1,max_attempts=1)
        calls=[]
        async def post(client,url,**kwargs):
            calls.append(1)
            raise httpx.ReadTimeout('synthetic')
        run=self.root/'failed_recovery'
        with patch.dict(os.environ,{'TEST_KT_KEY':'synthetic'}),patch('httpx.AsyncClient.post',post):
            asyncio.run(infer(self.config,prepared,run));n=len(calls)
            asyncio.run(infer(self.config,prepared,run,retry_failed_once=True))
            self.assertEqual(len(calls),2*n)
            asyncio.run(infer(self.config,prepared,run,retry_failed_once=True))
            self.assertEqual(len(calls),2*n)

    def test_http_529_retries_within_attempt_limit(self):
        prepared=self.prepared()
        self.config['model']={'adapter':'openrouter_decisions','model_id':'test','endpoint':'https://example.invalid','api_key_env':'TEST_KT_KEY','probability_strategy':'noul','parameters':{}}
        self.config['execution'].update(max_concurrency=1,max_attempts=2,max_requests=2)
        calls=[]
        async def post(client,url,**kwargs):
            calls.append(1)
            if len(calls)==1:return httpx.Response(529)
            return httpx.Response(200,json={'answers':{'will_answer_correctly':{'type':'noul','noul':.7}}})
        with patch.dict(os.environ,{'TEST_KT_KEY':'synthetic'}),patch('httpx.AsyncClient.post',post),patch('llm_kt.runtime.random.random',return_value=0):
            result=asyncio.run(infer(self.config,prepared,self.root/'http529'))
        self.assertEqual(len(calls),2)
        self.assertEqual(result['n_success'],1)

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
