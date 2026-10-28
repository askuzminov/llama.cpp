"""Full-context checks for 09-full-context.bat. Uses only the Python standard library."""

import array
import concurrent.futures
import contextlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import statistics
import threading
import time
import urllib.request


REFERENCE = {"LLAMA_HC_F16": "0", "LLAMA_QSA_POOL_F32": "1", "GGML_VK_F16ACC": "0"}
DEFAULT = {"LLAMA_HC_F16": "1", "LLAMA_QSA_POOL_F32": "0", "GGML_VK_F16ACC": "0"}
STAGES = {"quality", "server", "mixed", "cache", "profile"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def probe_positions(context, window=128, dense=False):
    if context < 2 or window < 1:
        raise ValueError("context >= 2 and window >= 1 required")
    if dense:
        return list(range(context))
    depths = sorted({min(context, d) for d in (4096, 32768, 65536, 131072, 200000, context)})
    return sorted({p for d in depths for p in range(max(0, d - window), d)})


def quality_gate(samples, margin=0.0, block=32):
    """Paired block bootstrap. Adjacent tokens in a probe window are not independent."""
    if not samples or any(not math.isfinite(s["delta_nll"]) for s in samples):
        return {"status": "INVALID"}
    mean = statistics.mean(s["delta_nll"] for s in samples)
    if all(s["max_logit_difference"] == 0 for s in samples):
        return {"status": "IDENTICAL_AT_PROBES", "delta_nll": mean, "ci95": [mean, mean]}
    blocks, current = [], []
    for s in samples:
        if current and (len(current) == block or s["position"] != current[-1]["position"] + 1):
            blocks.append(current)
            current = []
        current.append(s)
    if current:
        blocks.append(current)
    if len(blocks) < 8:
        return {"status": "INCONCLUSIVE", "delta_nll": mean, "blocks": len(blocks)}
    rng = random.Random(1759)
    means = []
    totals = [(sum(s["delta_nll"] for s in g), len(g)) for g in blocks]
    for _ in range(2000):
        selected = [rng.choice(totals) for _ in totals]
        means.append(sum(s for s, _ in selected) / sum(n for _, n in selected))
    means.sort()
    low, high = means[49], means[1949]
    status = "PASS_SAMPLED_NLL" if high <= margin else "REGRESSION" if low > margin else "INCONCLUSIVE"
    return {"status": status, "delta_nll": mean, "ci95": [low, high], "margin": margin, "blocks": len(blocks)}


def quality_summary(result, margin):
    samples = result["samples"]
    windows, group = [], []
    for s in samples:
        if group and s["position"] != group[-1]["position"] + 1:
            windows.append(group)
            group = []
        group.append(s)
    if group:
        windows.append(group)
    if len(samples) == result["context"]:
        depths = sorted({min(result["context"], d) for d in (4096, 32768, 65536, 131072, 200000, result["context"])})
        windows = [[s for s in samples if max(0, depth - 128) <= s["position"] < depth] for depth in depths]
    bands = []
    for group in windows:
        row = {"depth": group[-1]["position"] + 1, "probes": len(group),
               "sampled_nll": statistics.mean(s["nll"] for s in group)}
        row["sampled_ppl"] = math.exp(min(row["sampled_nll"], 700))
        if "delta_nll" in group[0]:
            row.update(quality_gate(group, margin, block=8))
            row["kl"] = statistics.mean(s["kl"] for s in group)
            row["tv"] = statistics.mean(s["tv"] for s in group)
            row["top1_agreement"] = statistics.mean(s["top1_equal"] for s in group)
        bands.append(row)
    nll = statistics.mean(s["nll"] for s in samples)
    out = {"processed": result["processed"], "context": result["context"], "bands": bands,
           "sampled_nll": nll, "sampled_ppl": math.exp(min(nll, 700))}
    if "delta_nll" in samples[0]:
        out.update(quality_gate(samples, margin))
        if any(b.get("status") == "REGRESSION" for b in bands):
            out["status"] = "REGRESSION_AT_DEPTH"
        elif any(b.get("status") == "INCONCLUSIVE" for b in bands):
            out["status"] = "INCONCLUSIVE_AT_DEPTH"
        out["kl"] = statistics.mean(s["kl"] for s in samples)
        out["tv"] = statistics.mean(s["tv"] for s in samples)
        out["max_logit_difference"] = max(s["max_logit_difference"] for s in samples)
    return out


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def read_sse(response, started, first=None):
    content, arrivals, tokens, terminal = [], [], [], None
    for raw in response:
        if not raw.startswith(b"data:"):
            continue
        payload = raw[5:].strip()
        if payload == b"[DONE]":
            break
        event = json.loads(payload)
        if "error" in event:
            raise RuntimeError(str(event["error"]))
        if event.get("stop"):
            terminal = event
            continue
        count = len(event.get("tokens", []))
        tokens.extend(event.get("tokens", []))
        if count:
            now = time.perf_counter()
            arrivals.append((now, count))
            if first:
                first.set()
        content.append(event.get("content", ""))
    if terminal is None:
        raise RuntimeError("incomplete SSE response")
    if terminal.get("truncated"):
        raise RuntimeError("server truncated the prompt")
    generated = terminal["tokens_predicted"]
    emitted = sum(n for _, n in arrivals)
    if emitted != generated and not (terminal.get("stop_type") == "eos" and generated == emitted + 1):
        raise RuntimeError("stream token count differs from final tokens_predicted")
    timings = terminal.get("timings", {})
    # A speculative burst may contain several tokens at one timestamp. Do not invent their individual latency.
    gaps = [b[0] - a[0] for a, b in zip(arrivals, arrivals[1:])]
    token_gaps = [0.0] * (arrivals[0][1] - 1) if arrivals else []
    for gap, (_, count) in zip(gaps, arrivals[1:]):
        token_gaps += [gap] + [0.0] * (count - 1)
    return {"content": "".join(content), "tokens": tokens, "n": generated, "emitted_n": emitted,
            "prompt_n": terminal["tokens_evaluated"], "cached_n": timings.get("cache_n", 0),
            "ttft_s": arrivals[0][0] - started if arrivals else None, "wall_s": time.perf_counter() - started,
            "itl_event_p95_s": percentile(gaps, .95), "event_gaps_s": gaps,
            "itl_p95_s": percentile(token_gaps, .95), "token_gaps_s": token_gaps,
            "max_burst": max((n for _, n in arrivals), default=0), "timings": timings, "stop_type": terminal.get("stop_type"),
            "first_at": arrivals[0][0] if arrivals else None, "finished_at": time.perf_counter()}


def answer_json(text):
    # Ignore JSON in the reasoning section; only the final answer is scored.
    text = text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        return None
    objects = []
    decoder = json.JSONDecoder()
    end = 0
    for m in re.finditer(r"\{", text):
        if m.start() < end:
            continue
        try:
            value, size = decoder.raw_decode(text[m.start():])
            if isinstance(value, dict):
                objects.append(value)
                end = m.start() + size
        except ValueError:
            pass
    return objects[-1] if objects else None


def repeated_tail(tokens):
    for period in range(1, min(128, len(tokens) // 4) + 1):
        repeats = max(4, math.ceil(32 / period))
        if len(tokens) >= repeats * period and tokens[-period:] * repeats == tokens[-period * repeats:]:
            return period
    return None


def server_arms(b):
    specs = ["none --spec-type none", "auto @draft --spec-auto"]
    for cap in (1, 2, 3):
        specs.append(f"n{cap} LLAMA_SPEC_DRAFT_BY_SLOTS=0 @draft --spec-type draft-mtp --spec-draft-n-max {cap} --spec-draft-sampling probabilistic")
    for window in (8192, 16384, 32768):
        specs.append(f"w{window} LLAMA_MTP_WINDOW={window} @draft --spec-auto")
    return parse_arms(b, "FULL_ARMS", specs)


def parse_arms(b, variable, defaults):
    arms = [b.Arm(s) for s in b.env(variable, ";".join(defaults)).split(";") if s.strip()]
    if not arms or len({a.name for a in arms}) != len(arms) or any(not re.fullmatch(r"[\w.-]+", a.name) for a in arms):
        raise b.Fail(f"{variable} needs unique arm names without path separators")
    return arms


class Suite:
    def __init__(self, b):
        self.b = b
        self.run = b.Run("full-context")
        self.run.dir = self.run.dir.resolve()
        self.model = b.model_path()
        self.base = b.Arm("current")
        self.exe = self.base.binary("llama-quality-fork")
        self.qargs = b.split_args(b.env("FULL_QARGS", "-b 16384 -ub " + ("8192" if b.env("BACKEND") == "cuda" else "4096")))
        self.results = []
        self.timeout = b.env_int("FULL_TIMEOUT", 7200)
        self.port = b.env_int("FULL_PORT", 8098)
        self.identity = {"driver": b.digest(Path(__file__)), "model": b.model_identity(self.model),
                         "probe_binary": b.digest(self.exe), "machine": b.machine(),
                         "os": b.platform.platform(), "processor": b.platform.processor(),
                         "runtime": {p.name: b.digest(p) for p in sorted(self.exe.parent.iterdir())
                                     if p.is_file() and p.suffix.lower() in {".dll", ".so", ".dylib"}}}
        if os.name == "nt":
            try:
                command = "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"
                result = b.subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                          capture_output=True, text=True, errors="replace", timeout=20, check=True)
                self.identity["display_driver"] = json.loads(result.stdout)
            except (OSError, ValueError, b.subprocess.SubprocessError):
                self.run.add("Display driver identity unavailable: use FORCE=1 after driver changes.")

    def native(self, name, plan, arm=None, metadata=False):
        b = self.b
        plan = dict(plan, report=str(self.run.dir / (name + ".json")))
        path = self.run.dir / (name + "-plan.json")
        write_json(path, plan)
        arm = arm or self.base
        args = [str(self.exe), *b.model_args(self.model), *self.qargs, *arm.args, "--probe-plan", str(path)]
        b.execute(args, self.run.dir / (name + ".log"), b.arm_env(arm), model=not metadata, timed=False)
        return json.loads(Path(plan["report"]).read_text(encoding="utf-8"))

    def prepare(self):
        b = self.b
        meta = self.native("metadata", {"action": "info"}, metadata=True)
        self.native_context = meta["native_context"]
        self.context = b.env_int("FULL_CTX", self.native_context)
        if not 2 <= self.context <= self.native_context:
            raise b.Fail("FULL_CTX must be in 2..native_context; extrapolation is a separate experiment")
        self.positions = probe_positions(self.context, b.env_int("FULL_WINDOW", 128), b.env("FULL_DENSE") == "1")
        self.stages = set(b.env("FULL_STAGES", "quality server mixed cache profile").split())
        if not self.stages or self.stages - STAGES:
            raise b.Fail(f"invalid FULL_STAGES: {self.stages}")
        self.ngen = b.env_int("FULL_NGEN", 512)
        if self.ngen < 1 or (self.stages - {"quality"} and self.ngen + 128 >= self.context):
            raise b.Fail("FULL_NGEN leaves too little context for tasks; use a smaller value for smoke tests")
        if not 1 <= b.env_int("FULL_ROWS", 32) <= 128 or not 1 <= self.port <= 65535 or self.timeout < 1:
            raise b.Fail("FULL_ROWS must be 1..128, FULL_PORT 1..65535 and FULL_TIMEOUT positive")
        self.pars = b.env_ints("FULL_PAR", "1 4")
        if not self.pars or any(n < 1 for n in self.pars):
            raise b.Fail("FULL_PAR must contain positive slot counts")
        self.repeats = b.env_int("FULL_REPEATS", 1)
        self.seeds = b.env_ints("FULL_SEEDS", "1 2 3")
        if self.repeats < 1 or not self.seeds:
            raise b.Fail("positive FULL_REPEATS and at least one FULL_SEEDS required")
        self.corpora = []
        for entry in b.env("FULL_CORPORA", b.env("PPLFILE")).split(";"):
            name, sep, source = entry.partition("=")
            path = Path(source if sep else entry).expanduser().resolve()
            if not path.is_file():
                raise b.Fail(f"corpus not found: {path}; FULL_CORPORA accepts name=path;name=path")
            label = name if sep else path.stem
            if not re.fullmatch(r"[\w.-]+", label) or any(c[0] == label for c in self.corpora):
                raise b.Fail("corpus labels must be unique file names (letters, digits, dot, dash)")
            self.corpora.append((label, path))
        arms = server_arms(b)
        manifest = {"native_context": self.native_context, "context": self.context,
                    "identity": self.identity, "quality_args": [*b.model_args(self.model), *self.qargs],
                    "server_extra_args": b.env("DECARGS"), "environment": b.env_identity(os.environ),
                    "full_native_context": self.context == self.native_context, "probe_positions": self.positions,
                    "reference_bytes_per_corpus": 24 + (self.context + 1) * 4 + len(self.positions) * 4 * (meta["vocab"] + 1),
                    "max_logits_rows_in_ram": b.env_int("FULL_ROWS", 32), "server_slots": self.pars,
                    "shared_kv_capacity": self.context, "four_full_contexts_may_queue": True,
                    "corpora": [(n, str(p)) for n, p in self.corpora], "stages": sorted(self.stages),
                    "server_arms": [a.text for a in arms], "repeats": self.repeats, "seeds": self.seeds,
                    "sampling": b.env("FULL_SAMP", "1.0 20 0.95"),
                    "note": "Long runs. Sampled NLL is not all-token perplexity. No setting is changed automatically."}
        write_json(self.run.dir / "manifest.json", manifest)
        self.run.add(f"Native context {self.native_context}; tested {self.context}; full={self.context == self.native_context}.",
                     f"{len(self.positions)} probes including final token; reference {manifest['reference_bytes_per_corpus'] / 2**30:.2f} GiB/corpus.",
                     f"Slots {self.pars}, shared KV {self.context}: full requests may queue, lengths are never reduced.",
                     "Sampled NLL / KL plus task checks; no finite test certifies zero quality loss.")
        print("\n".join(self.run.lines), flush=True)
        return manifest

    def record(self, stage, name, result):
        self.results.append({"stage": stage, "name": name, **result})
        write_json(self.run.dir / "results.json", self.results)
        self.run.add(f"{stage}/{name}: {result.get('status', 'MEASURED')}" + (": " + result["error"] if "error" in result else ""))

    def attempt(self, stage, name, function, identity=None):
        print(f"{stage}/{name}: running", flush=True)
        try:
            if identity:
                result, kept = self.run.measure("full-" + stage + "-" + name, identity(), lambda log: function())
                result = dict(result, kept=kept)
            else:
                result = function()
            self.record(stage, name, result)
        except self.b.Skip as error:
            self.record(stage, name, {"status": "NOT_TESTED", "error": str(error)})
        except (self.b.Fail, OSError, ValueError, RuntimeError) as error:
            self.run.failed = True
            self.record(stage, name, {"status": "ERROR", "error": str(error)})

    def tokens(self, label, corpus):
        b = self.b
        ident = {**self.identity, "kind": "full-tokens", "corpus": b.digest(corpus), "count": self.context + 1,
                 "args": [*b.model_args(self.model), *self.qargs], "env": b.env_identity(b.arm_env(self.base))}
        path = b.CACHE / "full" / (b.identity_key(ident) + ".tokens")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = path.with_suffix(".json")
        if not (path.exists() and path.stat().st_size == (self.context + 1) * 4 and meta.exists()
                and json.loads(meta.read_text(encoding="utf-8")).get("sha256") == b.digest(path)):
            self.native("tokens-" + label, {"action": "tokens", "corpus": str(corpus), "count": self.context + 1,
                                          "tokens": str(path)}, metadata=True)
            write_json(meta, {"sha256": b.digest(path), "identity": ident})
        return path

    def quality(self):
        b = self.b
        specs = ["repeat " + " ".join(k + "=" + v for k, v in REFERENCE.items()),
                 "default " + " ".join(k + "=" + v for k, v in DEFAULT.items()),
                 "hc32 LLAMA_HC_F16=0 LLAMA_QSA_POOL_F32=0 GGML_VK_F16ACC=0",
                 "pool32 LLAMA_HC_F16=1 LLAMA_QSA_POOL_F32=1 GGML_VK_F16ACC=0"]
        if b.env("FULL_F16ACC") == "1":
            specs.append("f16acc LLAMA_HC_F16=1 LLAMA_QSA_POOL_F32=0 GGML_VK_F16ACC=1")
        arms = parse_arms(b, "FULL_QARMS", specs)
        ref_arm = b.Arm("reference " + " ".join(k + "=" + v for k, v in REFERENCE.items()))
        margin = float(b.env("FULL_NLL_MARGIN", "0"))
        if margin < 0 or not math.isfinite(margin):
            raise b.Fail("FULL_NLL_MARGIN must be finite and nonnegative")
        for label, corpus in self.corpora:
            def check_corpus():
                tokens = self.tokens(label, corpus)
                base = {**self.identity, "kind": "full-quality", "tokens": b.digest(tokens),
                        "positions": self.positions, "context": self.context,
                        "args": [*b.model_args(self.model), *self.qargs], "rows": b.env_int("FULL_ROWS", 32)}
                ref_id = {**base, "env": b.env_identity(b.arm_env(ref_arm))}
                reference = b.CACHE / "full" / (b.identity_key(ref_id) + ".ref")
                sidecar = reference.with_suffix(".ref.json")
                plan = {"context": self.context, "positions": self.positions, "tokens": str(tokens),
                        "reference": str(reference), "max_outputs": base["rows"]}
                valid = (reference.exists() and sidecar.exists()
                         and json.loads(sidecar.read_text(encoding="utf-8")).get("sha256") == b.digest(reference))
                if not valid or b.forced("full-ref-" + label):
                    reference.unlink(missing_ok=True)
                    result = self.native("ref-" + label, dict(plan, action="capture"), ref_arm)
                    write_json(sidecar, {"sha256": b.digest(reference), "result": result})
                else:
                    result = json.loads(sidecar.read_text(encoding="utf-8"))["result"]
                self.record("quality", label + "/reference", {"status": "REFERENCE", **quality_summary(result, margin)})
                repeat_varies = False
                for arm in arms:
                    def compare(arm=arm):
                        nonlocal repeat_varies
                        if arm.build:
                            raise b.Fail("FULL_QARMS uses the current probe binary; alternate builds are not supported")
                        identity = {**base, "env": b.env_identity(b.arm_env(arm)), "arm_args": arm.args,
                                    "reference": b.digest(reference)}
                        def go(log):
                            return self.native(label + "-" + arm.name, dict(plan, action="compare"), arm)
                        result, _ = self.run.measure("full-q-" + label + "-" + arm.name, identity, go)
                        write_json(self.run.dir / (label + "-" + arm.name + ".json"), result)
                        if result["processed"] != self.context or [s["position"] for s in result["samples"]] != self.positions:
                            raise b.Fail("quality result does not cover the exact requested context and probe positions")
                        summary = quality_summary(result, margin)
                        if arm.name == "repeat" and summary["max_logit_difference"] != 0:
                            repeat_varies = True
                            summary["statistical_status"] = summary["status"]
                            summary["status"] = "REPEAT_VARIATION"
                        elif repeat_varies and summary["status"] == "PASS_SAMPLED_NLL":
                            summary["status"] = "INCONCLUSIVE_CONTROL"
                        if summary.get("status", "").startswith("REGRESSION"):
                            self.run.failed = True
                        return summary
                    self.attempt("quality", label + "/" + arm.name, compare)
                return {"status": "COMPLETED"}
            self.attempt("corpus", label, check_corpus)

    def request(self, url, prompt, seed=1, cache=False, first=None, n=None, greedy=False, fixed=True):
        b = self.b
        temp, top_k, top_p = map(float, b.env("FULL_SAMP", "1.0 20 0.95").split())
        if isinstance(prompt, dict):
            started = time.perf_counter()
            body = dict(prompt["chat"], seed=seed, cache_prompt=cache, stream=False, max_tokens=n or self.ngen,
                        temperature=0 if greedy else temp, top_k=int(top_k), top_p=top_p, ignore_eos=fixed)
            _, raw = b.http(url + "/v1/chat/completions", body, self.timeout)
            answer = json.loads(raw)
            usage = answer["usage"]
            if fixed and usage["completion_tokens"] != (n or self.ngen):
                raise b.Fail("profile chat stopped before the requested generation length")
            if usage["prompt_tokens"] != prompt["prompt_tokens"]:
                raise b.Fail("tool chat length differs from the exact prepared token budget")
            message = answer["choices"][0]["message"]
            calls = message.get("tool_calls", [])
            parsed = None
            if len(calls) == 1:
                function = calls[0]["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except (ValueError, TypeError):
                    arguments = None
                parsed = {"name": function["name"], "arguments": arguments}
            return {"answer": parsed, "content": message.get("content") or "", "raw": answer,
                    "stop_type": answer["choices"][0]["finish_reason"],
                    "n": usage["completion_tokens"], "prompt_n": usage["prompt_tokens"],
                    "cached_n": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    "wall_s": time.perf_counter() - started, "ttft_s": None,
                    "event_gaps_s": [], "token_gaps_s": [], "timings": answer.get("timings", {})}
        body = {"prompt": prompt, "n_predict": n or self.ngen, "seed": seed,
                "temperature": 0 if greedy else temp, "top_k": int(top_k), "top_p": top_p,
                "cache_prompt": cache, "stream": True, "return_tokens": True, "ignore_eos": fixed}
        req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        started = time.perf_counter()
        with b.OPENER.open(req, timeout=self.timeout) as response:
            result = read_sse(response, started, first)
        if fixed and result["n"] != (n or self.ngen):
            raise b.Fail(f"generation ended at {result['n']} instead of {n or self.ngen}")
        if result["prompt_n"] != len(prompt):
            raise b.Fail(f"server evaluated {result['prompt_n']} tokens, expected exactly {len(prompt)}")
        return result

    def tokenize(self, url, text, special=False):
        _, raw = self.b.http(url + "/tokenize", {"content": text, "add_special": special, "parse_special": True}, self.timeout)
        return json.loads(raw)["tokens"]

    def task(self, url, filler, kind, index, length):
        filler = filler[1:]
        facts = {"alpha": f"A{index}48317", "beta": f"B{index}72941", "gamma": f"C{index}56083"}
        expected = facts
        question = 'Return ONLY a JSON object with the values of alpha, beta, gamma from the records.'
        if kind == "ru":
            question = "Верни только JSON с точными значениями alpha, beta, gamma из записей."
        elif kind == "code":
            expected = {"result": 27 + index}
            question = f'Read the function in the records. What does calc({index + 4}) return? Return JSON {{"result": number}}.'
        elif kind == "tool":
            expected = {"name": "lookup", "arguments": {"key": facts["beta"]}}
            return self.tool_task(url, filler, facts, index, length), expected
        marker = f"FULL_CONTEXT_RECORDS_{index}_d7c4"
        _, raw = self.b.http(url + "/apply-template", {"messages": [{"role": "user", "content": marker + "\n" + question}],
                            "add_generation_prompt": True, "reasoning_effort": "medium"}, self.timeout)
        rendered = json.loads(raw)["prompt"]
        if rendered.count(marker) != 1:
            raise self.b.Fail("chat template does not preserve the task marker exactly once")
        head, tail = rendered.split(marker)
        prefix = self.tokenize(url, head, True)
        suffix = self.tokenize(url, tail)
        records = [f'\nRecord alpha = "{facts["alpha"]}".\n',
                   f'\nRecord beta = "{facts["beta"]}".\n',
                   f'\nRecord gamma = "{facts["gamma"]}".\n']
        if kind == "code":
            records[1] += f"\ndef calc(x):\n    return x + {23}\n"
        needles = [self.tokenize(url, text) for text in records]
        room = length - len(prefix) - len(suffix) - sum(map(len, needles))
        if room < 3 or len(filler) < room:
            raise self.b.Fail("not enough room or natural corpus tokens for the task")
        # Each slot starts with a different record id, preventing an accidental cache hit between cold requests.
        nonce = self.tokenize(url, f"\nDocument {index}.\n")
        room -= len(nonce)
        if room < 3:
            raise self.b.Fail("task exceeds requested context")
        cuts = [room // 10, room // 2, room * 9 // 10]
        prompt = prefix + nonce
        previous = 0
        for cut, needle in zip(cuts, needles):
            prompt += filler[previous:cut] + needle
            previous = cut
        prompt += filler[previous:room] + suffix
        if len(prompt) != length:
            raise self.b.Fail("task length accounting failed")
        return prompt, expected

    def tool_task(self, url, filler, facts, index, length, with_tools=True):
        tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up a record by its key.",
                  "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]
        question = "\nCall lookup once with the exact value of beta from the records." if with_tools else "\nExplain the text in detail."
        room = length - 256
        for _ in range(12):
            if not 3 <= room <= len(filler):
                raise self.b.Fail("not enough context for the actual tool-call template")
            record_tokens = self.tokenize(url, f'\nRecord beta = "{facts["beta"]}".\n')
            tokens = filler[:room // 2] + record_tokens + filler[room // 2:room]
            _, raw = self.b.http(url + "/detokenize", {"tokens": tokens}, self.timeout)
            text = json.loads(raw)["content"]
            chat = {"messages": [{"role": "user", "content": f"Document {index}.\n" + text + question}], "reasoning_effort": "medium"}
            if with_tools:
                chat["tools"] = tools
            _, raw = self.b.http(url + "/apply-template", chat, self.timeout)
            count = len(self.tokenize(url, json.loads(raw)["prompt"], True))
            if count == length:
                return {"chat": chat, "prompt_tokens": count}
            room += length - count
        raise self.b.Fail("could not prepare a tool-call prompt of the exact full length")

    def server_identity(self, arm):
        b = self.b
        draft = b.arm_draft(arm, b.arm_drafts([arm]))
        exe = arm.binary("llama-server")
        return {**self.identity, "kind": "full-server", "arm": arm.text,
                "server_binary": b.digest(exe), "args": b.model_args(self.model),
                "server_runtime": {p.name: b.digest(p) for p in sorted(exe.parent.iterdir())
                                   if p.is_file() and p.suffix.lower() in {".dll", ".so", ".dylib"}},
                "decargs": b.env("DECARGS"), "env": b.env_identity(b.arm_env(arm, DEFAULT)),
                "knobs": {k: v for k, v in os.environ.items() if k.startswith("FULL_")},
                "context": self.context, "draft": b.model_identity(draft) if draft else None,
                "corpus": b.digest(self.corpora[0][1])}

    @contextlib.contextmanager
    def server(self, name, arm, par, cache_dir=None, stride=None, profile=False):
        b = self.b
        draft = b.arm_draft(arm, b.arm_drafts([arm]))
        exe = arm.binary("llama-server")
        args = [*b.model_args(self.model), *b.split_args(b.env("DECARGS")), *arm.args]
        if draft:
            args += ["-md", str(draft)]
        # Last options own the test's memory budget and cache path, regardless of production DECARGS.
        args += ["-fit", "off", "-c", str(self.context), "-np", str(par), "--kv-unified", "--no-context-shift",
                 "--cache-ram", "-1", "--cache-disk", b.env("FULL_CACHE_MIB", "50000"),
                 "--cache-spill-dir", str(cache_dir) if cache_dir else "", "--cache-min-tokens", str(stride if stride is not None else 32768),
                 "--host", "127.0.0.1", "--port", str(self.port)]
        if cache_dir:
            args += ["--cache-idle-slots"]
        run_env = b.arm_env(arm, DEFAULT)
        if profile:
            run_env.update(LLAMA_INPUT_TIMING="1", GGML_TIMELINE=str(self.run.dir / (name + "-timeline.json")),
                           LLAMA_TRACE_REQUESTS="1:1", LLAMA_MOE_CACHE_STATS="1")
            run_env.update({"GGML_VK_PERF_LOGGER": "1"} if b.env("BACKEND") == "vulkan" else {"GGML_CUDA_OP_TIMING": "idle"})
        write_json(self.run.dir / (name + "-server.json"), {"args": [str(exe), *args], "env": b.env_identity(run_env),
                   "binary": b.digest(exe), "identity": self.identity})
        with socket.socket() as check:
            check.settimeout(.5)
            if check.connect_ex(("127.0.0.1", self.port)) == 0:
                raise b.Fail(f"port {self.port} is already in use; choose FULL_PORT")
        proc, log = b.start_server(exe, args, run_env, self.run.dir / (name + "-server.log"))
        try:
            yield f"http://127.0.0.1:{self.port}", log
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            b.mark_end()

    def batch(self, url, prompts, seed, cache=False, functional=False):
        before = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(self.request, url, prompt, seed + i, cache, fixed=not functional) for i, (prompt, _) in enumerate(prompts)]
            rows = [f.result() for f in futures]
        wall = time.perf_counter() - before
        events = sorted([(r["first_at"], 1) for r in rows if r.get("first_at") is not None]
                        + [(r["finished_at"], -1) for r in rows if r.get("first_at") is not None])
        active, max_active = 0, 0
        for _, delta in events:
            active += delta
            max_active = max(max_active, active)
        for row, (_, expected) in zip(rows, prompts):
            row["expected"] = expected
            row["answer"] = row.get("answer", answer_json(row["content"]))
            row["loop_period_tokens"] = repeated_tail(row.get("tokens", []))
            row["answer_complete"] = row.get("stop_type") not in {"limit", "length"}
            row["task_ok"] = row["answer"] == expected and row["answer_complete"] and not row["loop_period_tokens"]
        return {"wall_s": wall, "aggregate_tps": sum(r["n"] for r in rows) / wall,
                "max_overlapping_generation_streams": max_active,
                "ttft_p95_s": percentile([r["ttft_s"] for r in rows if r["ttft_s"] is not None], .95),
                "itl_event_p95_s": percentile([gap for r in rows for gap in r["event_gaps_s"]], .95),
                "itl_p95_s": percentile([gap for r in rows for gap in r["token_gaps_s"]], .95),
                "task_success": statistics.mean(r["task_ok"] for r in rows), "requests": rows}

    def server_tests(self, filler):
        b = self.b
        tasks = b.env("FULL_TASKS", "retrieve ru code tool").split()
        if not tasks or set(tasks) - {"retrieve", "ru", "code", "tool"}:
            raise b.Fail("FULL_TASKS accepts retrieve ru code tool")
        for arm in server_arms(b):
            for par in self.pars:
                name = f"{arm.name}-p{par}"
                def go(arm=arm, par=par, name=name):
                    rows = []
                    with self.server(name, arm, par) as (url, log):
                        for rep in range(self.repeats):
                            for ti, kind in enumerate(tasks):
                                prompts = [self.task(url, filler, kind, 1000 * rep + 100 * ti + i,
                                                    self.context - self.ngen - 1) for i in range(par)]
                                for seed in self.seeds:
                                    for warm in (False, True):
                                        result = self.batch(url, prompts, seed, cache=warm, functional=True)
                                        rows.append({"repeat": rep, "task": kind, "seed": seed, "warm_requested": warm, **result})
                                        write_json(self.run.dir / (name + "-requests.json"), rows)
                                        print(f"{name} {kind} rep={rep + 1} seed={seed} warm={warm}: "
                                              f"{result['aggregate_tps']:.2f} t/s, tasks {result['task_success']:.0%}", flush=True)
                        prompts = [self.task(url, filler, "retrieve", 90000 + i, self.context - self.ngen - 1) for i in range(par)]
                        performance = [self.batch(url, prompts, seed, cache=False) for seed in self.seeds]
                    return {"status": "MEASURED", "arm": arm.text, "par": par, "runs": rows,
                            "performance": performance,
                            "task_success": statistics.mean(r["task_success"] for r in rows)}
                self.attempt("server", name, go, lambda arm=arm, par=par: dict(self.server_identity(arm), par=par))

    def mixed(self, filler):
        b = self.b
        for budget in b.env_ints("FULL_BUDGETS", "256 1024 2048 4096 0"):
            name = f"budget-{budget}"
            arm = b.Arm(f"{name} LLAMA_PREFILL_BUDGET={budget} LLAMA_PREFILL_EXCLUSIVE=0 "
                        + ("@draft --spec-auto" if b.env("SPECDRAFT") else "--spec-type none"))
            def go(arm=arm, name=name):
                runs = []
                with self.server(name, arm, 4) as (url, log):
                    for rep in range(self.repeats):
                        n = min(b.env_int("FULL_MIX_NGEN", 1024), self.context // 32)
                        if n < 1:
                            raise b.Fail("context too short for mixed prefill test")
                        events = [threading.Event() for _ in range(3)]
                        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                            fs = [pool.submit(self.request, url, filler[:self.context // 8 - i], rep + i,
                                              False, events[i], n) for i in range(3)]
                            deadline = time.monotonic() + self.timeout
                            for event, future in zip(events, fs):
                                while not event.wait(1):
                                    if future.done():
                                        future.result()
                                        raise b.Fail("mixed test: a generation ended before its first token")
                                    if time.monotonic() >= deadline:
                                        raise b.Fail("mixed test: a generation never started")
                            fourth_start = time.perf_counter()
                            fourth = pool.submit(self.request, url, filler[:self.context // 2], rep + 4, False, None, n)
                            rows = [f.result() for f in fs] + [fourth.result()]
                        overlap = all(r["finished_at"] > fourth_start for r in rows[:3])
                        runs.append({"overlap_observed": overlap, "fourth_ttft_s": rows[3]["ttft_s"],
                                     "generating_itl_event_p95_s": percentile([g for r in rows[:3] for g in r["event_gaps_s"]], .95),
                                     "requests": rows})
                        write_json(self.run.dir / (name + "-requests.json"), runs)
                return {"status": "MEASURED" if all(r["overlap_observed"] for r in runs) else "NO_OVERLAP", "runs": runs}
            self.attempt("mixed", name, go, lambda arm=arm: self.server_identity(arm))

    def cache(self, filler):
        b = self.b
        def wait_disk(folder):
            deadline = time.monotonic() + self.timeout
            previous, stable = None, time.monotonic()
            while time.monotonic() < deadline:
                files = list(folder.glob("*.bin"))
                try:
                    snapshot = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files)
                except FileNotFoundError:
                    continue
                if snapshot != previous or list(folder.glob("*.tmp")):
                    previous, stable = snapshot, time.monotonic()
                if files and time.monotonic() - stable >= 2:
                    return sum(p.stat().st_size for p in files)
                time.sleep(.2)
            raise b.Fail("disk cache did not reach a completed, stable state before timeout")

        for stride in b.env_ints("FULL_CACHE_STEPS", "0 8192 32768"):
            name = f"cache-{stride}"
            folder = self.run.dir / (name + "-data")
            folder.mkdir()
            arm = b.Arm("cache --spec-type none")
            def go(stride=stride, name=name, folder=folder):
                if stride > self.context - self.ngen - 1:
                    return {"status": "NOT_EXERCISED", "reason": "minimum exceeds prompt length"}
                prompt = filler[:self.context - self.ngen - 1]
                with self.server(name + "-cold", arm, 1) as (url, log):
                    cold = self.request(url, prompt, cache=False, greedy=True)
                turns = b.env_int("FULL_CACHE_TURNS", 4)
                step = b.env_int("FULL_CACHE_GROW", min(2048, self.context // 16))
                if turns < 1 or step < 1 or len(prompt) - turns * step <= max(16, stride):
                    raise b.Fail("cache growth needs positive turns/step and an initial prompt above the minimum")
                growth = []
                with self.server(name + "-grow", arm, 1, folder, stride) as (url, log):
                    for turn in range(turns + 1):
                        length = len(prompt) - (turns - turn) * step
                        offset = log.stat().st_size
                        began = time.perf_counter()
                        response = self.request(url, prompt[:length], cache=True, greedy=True)
                        size = wait_disk(folder)
                        with log.open("rb") as file:
                            file.seek(offset)
                            text = file.read().decode("utf-8", errors="replace")
                        writes = re.findall(r"wrote (\d+) state\(s\) to disk while idle, ([\d.]+) MiB in ([\d.]+) ms", text)
                        growth.append({"prompt_n": length, "request": response, "disk_bytes": size,
                                       "wall_including_idle_s": time.perf_counter() - began,
                                       "written_mib": sum(float(w[1]) for w in writes),
                                       "write_ms": sum(float(w[2]) for w in writes)})
                        write_json(self.run.dir / (name + "-growth.json"), growth)
                with self.server(name + "-restore", arm, 1, folder, stride) as (url, log):
                    restored = self.request(url, prompt, cache=True, greedy=True)
                restore_same = restored["tokens"] == growth[-1]["request"]["tokens"] and bool(restored["tokens"])
                cold_same = cold["tokens"] == restored["tokens"] and bool(cold["tokens"])
                hit = restored["timings"].get("cache_n", 0) > 0
                budget = b.env_int("FULL_CACHE_MIB", 50000) * 2**20
                limit_ok = budget == 0 or size <= budget
                status = "FAIL" if not restore_same or not hit or not limit_ok else "PASS" if cold_same else "INCONCLUSIVE_PREFILL_PATH"
                return {"status": status, "disk_bytes": size,
                        "disk_limit_ok": limit_ok,
                        "restore_same_tokens": restore_same, "cold_same_tokens": cold_same, "restore_hit": hit,
                        "cold": cold, "restored": restored, "growth": growth,
                        "written_mib": sum(r["written_mib"] for r in growth)}
            self.attempt("cache", name, go)

    def profile(self, filler):
        b = self.b
        arm = b.Arm("profile @draft --spec-auto" if b.env("SPECDRAFT") else "profile --spec-type none")
        def go():
            with self.server("profile", arm, 1, profile=True) as (url, log):
                prompt = self.tool_task(url, filler[1:], {"beta": "profile"}, 0, self.context - self.ngen - 1, with_tools=False)
                result = self.request(url, prompt, n=self.ngen)
                time.sleep(2)
                text = b.read_log(log)
                ops = b.sum_ops(b.vulkan_blocks(text)) if b.env("BACKEND") == "vulkan" else []
                timeline = self.run.dir / "profile-timeline.json"
                if not timeline.is_file() or timeline.stat().st_size < 100:
                    raise b.Fail("profile wrote no timeline; the traced non-streaming chat was not captured")
            return {"status": "PROFILE_ONLY", "request": result, "top_vulkan_ops": ops[:40],
                    "log": str(log), "timeline": str(timeline),
                    "note": "Timing instrumentation changes speed; use server runs for performance comparisons."}
        self.attempt("profile", arm.name, go)

    def finish(self):
        # A failed baseline task makes the corresponding candidate comparison inconclusive.
        baselines = {r["par"]: r for r in self.results if r["stage"] == "server" and r.get("arm", "").startswith("none ")}
        comparisons = []
        for r in self.results:
            if r["stage"] == "server" and "par" in r and r["par"] in baselines:
                base = baselines[r["par"]]
                lost = []
                for baseline_run, candidate_run in zip(base["runs"], r["runs"]):
                    for slot, (old, new) in enumerate(zip(baseline_run["requests"], candidate_run["requests"])):
                        if old["task_ok"] and not new["task_ok"]:
                            lost.append({"task": candidate_run["task"], "repeat": candidate_run["repeat"],
                                         "seed": candidate_run["seed"], "slot": slot, "warm": candidate_run["warm_requested"]})
                status = "TASK_CHECKS_PASS" if r["task_success"] == base["task_success"] == 1 else "INCONCLUSIVE"
                if lost:
                    status = "TASK_REGRESSION"
                comparisons.append({"arm": r["name"], "status": status, "baseline_success": base["task_success"],
                                    "candidate_success": r["task_success"],
                                    "lost_cases": lost,
                                    "median_aggregate_tps": statistics.median(x["aggregate_tps"] for x in r["performance"]),
                                    "ttft_p95_s": percentile([x["ttft_p95_s"] for x in r["performance"]], .95),
                                    "itl_p95_s": percentile([x["itl_p95_s"] for x in r["performance"] if x["itl_p95_s"] is not None], .95),
                                    "itl_event_p95_s": percentile([x["itl_event_p95_s"] for x in r["performance"] if x["itl_event_p95_s"] is not None], .95)})
        write_json(self.run.dir / "comparisons.json", comparisons)
        for c in comparisons:
            self.run.add(f"{c['arm']}: {c['status']}, median aggregate {c['median_aggregate_tps']:.2f} t/s, "
                         f"TTFT p95 {c['ttft_p95_s']:.3f} s, ITL p95 {c['itl_p95_s']} s, event p95 {c['itl_event_p95_s']} s")
        if any(r.get("status") in {"FAIL", "ERROR", "TASK_REGRESSION"} for r in self.results + comparisons):
            self.run.failed = True
        self.run.add("No automatic winner: sampled NLL, functional checks and performance must be assessed together.",
                     "Missing corpora, omitted stages and NOT_TESTED/INCONCLUSIVE results do not validate quality.")
        return self.run.finish()


def run(b):
    suite = Suite(b)
    suite.prepare()
    if b.env("FULL_DRY_RUN") == "1":
        suite.run.add("DRY RUN: metadata/manifest only, no inference was tested.")
        return suite.run.finish()
    if "quality" in suite.stages:
        suite.quality()
    if suite.stages - {"quality"}:
        try:
            tokens = suite.tokens(*suite.corpora[0])
            filler = array.array("i")
            filler.frombytes(tokens.read_bytes())
            if filler.itemsize != 4:
                raise b.Fail("32-bit native token representation required")
            filler = list(filler)
        except (b.Fail, OSError, ValueError) as error:
            suite.record("server-input", "corpus", {"status": "ERROR", "error": str(error)})
            suite.run.failed = True
            return suite.finish()
        for stage in ("server", "mixed", "cache", "profile"):
            if stage in suite.stages:
                getattr(suite, "server_tests" if stage == "server" else stage)(filler)
    return suite.finish()
