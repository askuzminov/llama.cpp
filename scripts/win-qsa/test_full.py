"""Run with python scripts/win-qsa/test_full.py; native checks also accept two environment paths."""

import io
import contextlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import _full as full
import _run as bench


class FullContextTests(unittest.TestCase):
    def test_probes_reach_the_native_end(self):
        positions = full.probe_positions(262144)
        self.assertEqual(positions[-1], 262143)
        self.assertEqual(positions, sorted(set(positions)))
        for depth in (4096, 32768, 65536, 131072, 200000, 262144):
            self.assertIn(depth - 1, positions)
        self.assertLess(len(positions), 1024)

    def test_short_context_and_dense(self):
        self.assertEqual(full.probe_positions(64), list(range(64)))
        self.assertEqual(full.probe_positions(513, dense=True), list(range(513)))
        with self.assertRaises(ValueError):
            full.probe_positions(0)

    def samples(self, delta, count=256):
        return [{"position": i, "delta_nll": delta, "max_logit_difference": abs(delta),
                 "nll": 2 + delta, "kl": abs(delta), "tv": abs(delta), "top1_equal": True} for i in range(count)]

    def test_nll_regression_is_rejected(self):
        self.assertEqual(full.quality_gate(self.samples(.02))["status"], "REGRESSION")

    def test_passing_average_does_not_hide_bad_tail(self):
        samples = self.samples(-.1)
        samples += [dict(s, position=s["position"] + 1000) for s in self.samples(.02, 128)]
        summary = full.quality_summary({"samples": samples, "processed": 1128, "context": 1128}, 0)
        self.assertEqual(summary["status"], "REGRESSION_AT_DEPTH")

    def test_dense_scoring_keeps_a_separate_tail_check(self):
        samples = self.samples(-.1, 1024)
        samples[-128:] = [dict(s, position=s["position"] + 896) for s in self.samples(.02, 128)]
        summary = full.quality_summary({"samples": samples, "processed": 1024, "context": 1024}, 0)
        self.assertEqual(summary["status"], "REGRESSION_AT_DEPTH")

    def test_insufficient_evidence_is_inconclusive(self):
        self.assertEqual(full.quality_gate(self.samples(-.01, 2))["status"], "INCONCLUSIVE")
        self.assertEqual(full.quality_gate(self.samples(0))["status"], "IDENTICAL_AT_PROBES")
        self.assertEqual(full.quality_gate(self.samples(math.nan))["status"], "INVALID")

    def test_nested_tool_json_and_reasoning(self):
        text = '<think>{"wrong":1}</think> <tool_call>{"name":"lookup","arguments":{"key":"B"}}</tool_call>'
        self.assertEqual(full.answer_json(text), {"name": "lookup", "arguments": {"key": "B"}})
        self.assertIsNone(full.answer_json('<think>{"answer":1}'))

    def test_repeated_generation_tail(self):
        self.assertEqual(full.repeated_tail(list(range(8)) * 6), 8)
        self.assertEqual(full.repeated_tail([7] * 40), 1)
        self.assertIsNone(full.repeated_tail([1, 2, 3, 4]))
        self.assertIsNone(full.repeated_tail(list(range(200))))

    def sse(self, events):
        return io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events))

    def test_sse_bursts_and_terminal_tokens_not_counted_twice(self):
        events = [{"tokens": [1, 2], "content": "ab"}, {"tokens": [3], "content": "c"},
                  {"stop": True, "tokens": [1, 2, 3], "tokens_predicted": 3, "tokens_evaluated": 200}]
        with patch.object(full.time, "perf_counter", side_effect=[1., 3., 4., 4.]):
            row = full.read_sse(self.sse(events), 0.)
        self.assertEqual(row["n"], 3)
        self.assertEqual(row["tokens"], [1, 2, 3])
        self.assertEqual(row["content"], "abc")
        self.assertEqual(row["token_gaps_s"], [0., 2.])
        self.assertEqual(row["max_burst"], 2)

    def test_sse_truncation_or_missing_terminal_fails(self):
        for events in ([{"tokens": [1], "content": "a"}],
                       [{"tokens": [1]}, {"stop": True, "truncated": True}]):
            with self.assertRaises(RuntimeError):
                full.read_sse(self.sse(events), 0.)

    def test_stream_token_mismatch_fails(self):
        events = [{"tokens": [1]}, {"stop": True, "tokens_predicted": 4, "tokens_evaluated": 200}]
        with self.assertRaises(RuntimeError):
            full.read_sse(self.sse(events), 0.)

    def test_immediate_eos_is_an_empty_answer(self):
        events = [{"stop": True, "tokens_predicted": 1, "tokens_evaluated": 200, "stop_type": "eos"}]
        row = full.read_sse(self.sse(events), 0.)
        self.assertEqual(row["tokens"], [])
        self.assertIsNone(row["ttft_s"])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="llama-resume-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.object(bench, "CACHE", self.root / "cache"))
        self.enterContext(patch.dict(os.environ, {"FORCE": "", "FULL_ARMS": "none --spec-type none",
                                                 "FULL_TASKS": "retrieve", "SPECDRAFT": ""}))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.calls = []
        self.opens = []
        self.stopped = threading.Event()

    def suite(self):
        suite = full.Suite.__new__(full.Suite)
        suite.b = bench
        suite.proc = None
        suite.context, suite.ngen, suite.timeout = 512, 16, 30
        suite.pars, suite.seeds, suite.repeats = [1], [1], 1
        suite.results = []
        folder = self.root / f"run-{len(list(self.root.glob('run-*')))}"
        folder.mkdir()

        class Run:
            dir = folder
            failed = False
            def add(self, *lines):
                pass
            def measure(self, label, identity, function):
                record = bench.cache_get(identity, label)
                if record:
                    return record["result"], True
                result = function(folder / "run.log")
                bench.cache_put(identity, result, folder / "run.log")
                return result, False
        suite.run = Run()
        suite.server_identity = lambda arm: {"fixture": "resume", "arm": arm.text}
        suite.task = lambda url, filler, kind, index, length: ([index], {"answer": index})
        suite.stop_requests = self.stopped.set

        @contextlib.contextmanager
        def server(name, arm, par, cache_dir=None, stride=None, profile=False):
            self.opens.append(name)
            log = folder / (name + ".log")
            log.write_text("fixture log\n")
            if profile:
                (folder / "profile-timeline.json").write_text("[\n" + '{"name":"event"},\n' * 20)
            yield "http://fixture", log
        suite.server = server
        suite.request = self.response
        return suite

    def response(self, url, prompt, seed=1, cache=False, first=None, n=None, greedy=False, fixed=True):
        index = prompt[0] if isinstance(prompt, list) else 0
        self.calls.append((index, cache))
        started = time.perf_counter()
        if first:
            first.set()
        time.sleep(.002)
        finished = time.perf_counter()
        return {"content": json.dumps({"answer": index}), "tokens": [index + 10, index + 11], "n": 2,
                "prompt_n": len(prompt), "cached_n": 0, "timings": {"cache_n": 0}, "stop_type": "eos",
                "ttft_s": .001, "event_gaps_s": [.001], "token_gaps_s": [.001],
                "first_at": started + .001, "finished_at": finished, "wall_s": finished - started}

    def interrupt_after_one_response(self, functional):
        suite = self.suite()
        progress = full.Checkpoints(suite, "batch", {"fixture": "batch"})
        saved = threading.Event()
        original_put = progress.put
        def put(key, result):
            original_put(key, result)
            if key == "pass-request-0":
                saved.set()
        progress.put = put
        def request(url, prompt, *args, **kwargs):
            if prompt[0] == 0:
                return self.response(url, prompt, *args, **kwargs)
            if prompt[0] == 1:
                self.assertTrue(saved.wait(5))
                raise KeyboardInterrupt()
            self.assertTrue(self.stopped.wait(5))
            raise RuntimeError("cancelled")
        suite.request = request
        prompts = [([i], {"answer": i}) for i in range(4)]
        with self.assertRaises(KeyboardInterrupt):
            suite.batch("http://fixture", prompts, 1, functional=functional, progress=progress, key="pass")
        self.assertIsNotNone(progress.get("pass-request-0"))
        self.assertIsNone(progress.get("pass"))
        self.calls.clear()
        self.stopped.clear()
        resumed = self.suite()
        progress = full.Checkpoints(resumed, "batch", {"fixture": "batch"})
        return resumed.batch("http://fixture", prompts, 1, functional=functional, progress=progress, key="pass")

    def test_completed_quality_requests_survive_interruption(self):
        result = self.interrupt_after_one_response(functional=True)
        self.assertEqual(sorted(i for i, _ in self.calls), [1, 2, 3])
        self.assertEqual(result["task_success"], 1)
        self.assertTrue(result["resumed_partial"])
        self.assertIsNone(result["aggregate_tps"])
        self.assertIsNone(result["ttft_p95_s"])

    def test_incomplete_parallel_performance_repeats_only_its_group(self):
        result = self.interrupt_after_one_response(functional=False)
        self.assertEqual(sorted(i for i, _ in self.calls), [0, 1, 2, 3])
        self.assertFalse(result["resumed_partial"])
        self.assertGreater(result["aggregate_tps"], 0)

    def test_complete_responses_recover_a_lost_group_commit(self):
        suite = self.suite()
        progress = full.Checkpoints(suite, "batch", {"fixture": "commit"})
        original_put = progress.put
        def put(key, result):
            if key == "pass":
                raise KeyboardInterrupt()
            original_put(key, result)
        progress.put = put
        prompts = [([i], {"answer": i}) for i in range(4)]
        with self.assertRaises(KeyboardInterrupt):
            suite.batch("http://fixture", prompts, 1, progress=progress, key="pass")
        self.calls.clear()
        resumed = self.suite()
        progress = full.Checkpoints(resumed, "batch", {"fixture": "commit"})
        result = resumed.batch("http://fixture", prompts, 1, progress=progress, key="pass")
        self.assertEqual(self.calls, [])
        self.assertFalse(result["resumed_partial"])
        self.assertGreater(result["aggregate_tps"], 0)

    def test_warm_resume_rebuilds_cache_without_replacing_cold_measurement(self):
        suite = self.suite()
        def interrupted(url, prompt, seed=1, cache=False, **kwargs):
            if cache:
                raise KeyboardInterrupt()
            return self.response(url, prompt, seed, cache, **kwargs)
        suite.request = interrupted
        with self.assertRaises(KeyboardInterrupt):
            suite.server_tests([100] * 512)
        progress = full.Checkpoints(suite, "full-server-none-p1", {"fixture": "resume", "arm": "none --spec-type none", "par": 1})
        cold = progress.get("r0-t0-retrieve-s0-1-cold")
        self.calls.clear()
        resumed = self.suite()
        resumed.server_tests([100] * 512)
        self.assertEqual([cache for _, cache in self.calls], [False, True, False])
        self.assertEqual(resumed.results[0]["runs"][0]["requests"], cold["requests"])
        self.calls.clear()
        self.opens.clear()
        self.suite().server_tests([100] * 512)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.opens, [])

    def test_broken_checkpoint_is_recomputed(self):
        suite = self.suite()
        progress = full.Checkpoints(suite, "batch", {"fixture": "broken"})
        progress.put("valid", {"answer": 1})
        identity = dict(progress.identity, checkpoint="broken")
        path = bench.CACHE / "results" / (bench.identity_key(identity) + ".json")
        path.write_text('{"identity":')
        self.assertIsNone(progress.get("broken"))
        self.assertEqual(progress.measure("broken", lambda: {"answer": 2}), {"answer": 2})
        self.assertEqual(progress.get("valid"), {"answer": 1})

    def test_checkpoint_write_failure_stops_the_run(self):
        suite = self.suite()
        progress = full.Checkpoints(suite, "batch", {"fixture": "full-disk"})
        with patch.object(bench.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(bench.Fail, "progress could not be saved"):
                suite.attempt("server", "case", lambda: progress.put("pass", {"done": True}))
        self.assertIsNone(progress.get("pass"))

    def test_atomic_report_keeps_the_previous_file_if_flush_fails(self):
        path = self.root / "report.json"
        full.write_json(path, {"completed": 1})
        with patch.object(full.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                full.write_json(path, {"completed": 2})
        self.assertEqual(json.loads(path.read_text()), {"completed": 1})

    def test_mixed_resume_keeps_finished_repeats(self):
        with patch.dict(os.environ, {"FULL_BUDGETS": "64"}):
            suite = self.suite()
            suite.mixed([100] * 512)
            first = suite.results[0]["runs"][0]
            interrupted = self.suite()
            interrupted.repeats = 2
            original_put = full.Checkpoints.put
            def put(progress, key, result):
                original_put(progress, key, result)
                if key == "repeat-1-request-0":
                    raise KeyboardInterrupt()
            with patch.object(full.Checkpoints, "put", put):
                with self.assertRaises(KeyboardInterrupt):
                    interrupted.mixed([100] * 512)
            self.calls.clear()
            resumed = self.suite()
            resumed.repeats = 2
            resumed.mixed([100] * 512)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(resumed.results[0]["runs"][0], first)

    def test_profile_is_reused_and_missing_timeline_is_recaptured(self):
        suite = self.suite()
        suite.tool_task = lambda *args, **kwargs: {"chat": {}, "prompt_tokens": 1}
        with patch.object(full.time, "sleep"):
            suite.profile([100] * 512)
            kept = suite.results[0]
            self.opens.clear()
            self.suite().profile([100] * 512)
            self.assertEqual(self.opens, [])
            Path(kept["timeline"]).unlink()
            resumed = self.suite()
            resumed.tool_task = suite.tool_task
            resumed.profile([100] * 512)
        self.assertEqual(self.opens, ["profile"])

    def test_phase_selection_does_not_invalidate_request_progress(self):
        suite = self.suite()
        del suite.server_identity
        suite.identity = {"fixture": "identity"}
        suite.model = self.root / "model.gguf"
        suite.model.write_bytes(b"model")
        corpus = self.root / "corpus.txt"
        corpus.write_text("natural text")
        suite.corpora = [("text", corpus)]
        binary = self.root / "llama-server"
        binary.write_bytes(b"binary")
        arm = bench.Arm("none --spec-type none")
        arm.binary = lambda name: binary
        before = suite.server_identity(arm)
        with patch.dict(os.environ, {"FULL_STAGES": "mixed", "FULL_PAR": "1 4", "FULL_SEEDS": "1 2 3"}):
            self.assertEqual(suite.server_identity(arm), before)
        with patch.dict(os.environ, {"FULL_SAMP": "0.8 40 0.95"}):
            self.assertNotEqual(suite.server_identity(arm), before)


@unittest.skipUnless(os.environ.get("LLAMA_QUALITY_TEST_BIN") and os.environ.get("LLAMA_QUALITY_TEST_MODEL"),
                     "set LLAMA_QUALITY_TEST_BIN and LLAMA_QUALITY_TEST_MODEL for native integration")
class NativeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="llama-quality-")
        cls.folder = Path(cls.temp.name)
        cls.exe = str(Path(os.environ["LLAMA_QUALITY_TEST_BIN"]).resolve())
        cls.model = str(Path(os.environ["LLAMA_QUALITY_TEST_MODEL"]).resolve())
        cls.call({"action": "info"})
        cls.context = min(128, cls.report()["native_context"])
        corpus = cls.folder / "corpus.txt"
        corpus.write_text("Once upon a time a little girl lived in a village. She liked to read books and play outside.\n" * 100)
        cls.tokens = cls.folder / "tokens.bin"
        cls.call({"action": "tokens", "corpus": str(corpus), "count": cls.context + 1, "tokens": str(cls.tokens)})
        cls.reference = cls.folder / "reference.bin"
        cls.plan = {"context": cls.context, "tokens": str(cls.tokens), "reference": str(cls.reference),
                    "positions": list(range(cls.context - 16, cls.context)), "max_outputs": 4}
        cls.call(dict(cls.plan, action="capture"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def call(cls, plan, ok=True):
        path = cls.folder / "plan.json"
        full.write_json(path, dict(plan, report=str(cls.folder / "report.json")))
        result = subprocess.run([cls.exe, "-m", cls.model, "-ngl", "0", "-b", "64", "-ub", "32",
                                 "--no-warmup", "--probe-plan", str(path)], capture_output=True)
        if (result.returncode == 0) != ok:
            raise AssertionError(result.stderr.decode(errors="replace")[-4000:])

    @classmethod
    def report(cls):
        return json.loads((cls.folder / "report.json").read_text())

    def test_exact_reference_and_full_length(self):
        self.call(dict(self.plan, action="compare"))
        result = self.report()
        self.assertEqual(result["processed"], self.context)
        self.assertEqual(result["samples"][-1]["position"], self.context - 1)
        self.assertTrue(all(s["kl"] == s["max_logit_difference"] == 0 for s in result["samples"]))

    def test_wrong_input_rejected(self):
        path = self.folder / "wrong-tokens.bin"
        data = bytearray(self.tokens.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        self.call(dict(self.plan, action="compare", tokens=str(path)), ok=False)

    def test_short_corpus_rejected(self):
        path = self.folder / "short.txt"
        path.write_text("hello")
        self.call({"action": "tokens", "corpus": str(path), "count": 1000,
                   "tokens": str(self.folder / "short.bin")}, ok=False)

    def test_truncated_and_nan_reference_rejected(self):
        path = self.folder / "broken.bin"
        original = self.reference.read_bytes()
        path.write_bytes(original[:-1])
        self.call(dict(self.plan, action="compare", reference=str(path)), ok=False)
        path.write_bytes(original[:-4] + struct.pack("<f", math.nan))
        self.call(dict(self.plan, action="compare", reference=str(path)), ok=False)


if __name__ == "__main__":
    unittest.main()
