"""Run with python scripts/win-qsa/test_full.py; native checks also accept two environment paths."""

import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import _full as full


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
