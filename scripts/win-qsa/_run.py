#!/usr/bin/env python3
"""Checks and measurements of this fork on the 395 (Vulkan) and the 3090 (CUDA), see README.md.

Each result is kept in cache/ with the identity of what made it: the source of the build (00-build.bat records it),
the arguments, the GGML_/LLAMA_/CUDA_ environment, the model files, the inputs and the machine. When that identity
did not change, the script prints the kept result and does not run the measurement again. FORCE=1 runs it again.
Each new result also goes to history.csv in the logs folder, one row per number (Run.record)."""

import argparse
import bisect
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / "cache"
VERSION = 1
# the version of the kept llama-bench results of 02 (02 is gone since 13.10), for history.csv from the cache
BENCH_VERSION = 3
BUILD_RECORD = "qsa-build.json"
ENV_PREFIXES = ("GGML_", "LLAMA_", "CUDA_", "HIP_", "OMP_")
ENV_IGNORED = ("LLAMA_SPEC_TRACE",)
# the 395 runs at the edge of its memory at -c 262144: right after another model run these come and go
MEMORY_ERRORS = ("ErrorOutOfDeviceMemory", "ErrorMemoryMapFailed", "Device memory allocation", "failed to allocate",
                 "unable to allocate", "out of memory", "cudaMalloc failed", "bad allocation", "row cache: read error",
                 "failed to create context")
DEVICES = {"vulkan": "Vulkan0", "cuda": "CUDA0"}
EXE = ".exe" if os.name == "nt" else ""

PROMPTS = {
    "os": "Explain in detail how a modern operating system schedules threads on a multi core processor. Cover run "
          "queues, load balancing, priority inheritance and cache affinity.",
    "db": "Explain in detail how a relational database executes a join between two large tables. Cover hash join, "
          "merge join, nested loops, spilling to disk and cardinality estimation.",
    "code": "Write a C++17 header-only thread-safe LRU cache class template with get, put and erase, then a short "
            "usage example and unit tests with assert.",
    "repeat": "Write the following block of text back to me, exactly as it is. Output only the block, no commentary "
              "and no code fences.\n\n" + "\n".join(
                  f"    const int row_{i:03d} = accumulate(buffer, stride * {i}, offset + {i}, mask_{i:03d});"
                  for i in range(1, 81)),
}
WARMUP = "Describe your favourite season in a short paragraph."


class Fail(Exception):
    pass


class Skip(Exception):
    pass


# ---------------------------------------------------------------- settings

def env(name, default=""):
    value = os.environ.get(name, "").strip()
    return value if value else default


def env_int(name, default):
    try:
        return int(env(name, str(default)))
    except ValueError:
        raise Fail(f"{name} must be a number, got {env(name)!r}")


def env_ints(name, default):
    try:
        return [int(x) for x in re.split(r"[,\s]+", env(name, default)) if x]
    except ValueError:
        raise Fail(f"{name} must be numbers, got {env(name)!r}")


def forced(label):
    """FORCE=1 runs everything again, any other value is a regex of the labels to run again."""
    force = env("FORCE")
    return force == "1" or (force != "" and re.search(force, label) is not None)


def split_args(text):
    # posix=False keeps the backslashes of windows paths, the quotes of a quoted argument go
    return [s[1:-1] if len(s) >= 2 and s[0] == s[-1] == '"' else s for s in shlex.split(text, posix=False)]


def now():
    return datetime.now().strftime("%m-%d %H:%M")


def build_folder(build):
    build = Path(build)
    return build / "bin" / "Release" if (build / "bin" / "Release").is_dir() else build / "bin"


def stable_folder():
    return build_folder(Path(env("STABLEBUILD", str(ROOT / "build-stable"))))


class Arm:
    """One variant of a measurement: "name [VAR=value ...] [@stable | @<bin folder>] [@draft] [@d=depth,...] [tool
    arguments ...]". VAR=value goes to the environment of the tool, @stable runs the build of 00-build.bat stable,
    @draft loads SPECDRAFT with -md, @d= limits the arm to these depths of 02 and 05, @p= to these prefixes of 04
    (DECPREFIX), @c= and @ub= set the -c and the -ub of 02 and 05 for this arm, @pp= the prompt of 02, @o=OP,... limits
    it to these ops of 01 and 06, the rest goes to the tool."""

    def __init__(self, text):
        tokens = split_args(text)
        if not tokens:
            raise Fail("empty arm")
        self.name = tokens[0]
        self.text = text.strip()
        self.env = {}
        self.args = []
        self.build = None
        self.stable = False
        self.draft = False
        self.depths = None
        self.prefixes = None
        self.ctx = None
        self.ub = None
        self.pp = None
        self.ops = None
        for t in tokens[1:]:
            if t == "@draft":
                self.draft = True
            elif t.startswith("@o="):
                self.ops = {o for o in t[3:].split(",") if o}
            elif t.startswith("@ub="):
                try:
                    self.ub = int(t[4:])
                except ValueError:
                    raise Fail(f"arm {self.name}: {t} must be a ubatch size, as @ub=2048")
            elif t.startswith("@pp="):
                try:
                    self.pp = int(t[4:])
                except ValueError:
                    raise Fail(f"arm {self.name}: {t} must be a prompt size, as @pp=8192")
            elif t.startswith("@c="):
                try:
                    self.ctx = int(t[3:])
                except ValueError:
                    raise Fail(f"arm {self.name}: {t} must be a context size, as @c=131072")
            elif t.startswith("@d="):
                try:
                    self.depths = {int(d) for d in t[3:].split(",") if d}
                except ValueError:
                    raise Fail(f"arm {self.name}: {t} must list depths, as @d=122880")
            elif t.startswith("@p="):
                try:
                    self.prefixes = {int(p) for p in t[3:].split(",") if p}
                except ValueError:
                    raise Fail(f"arm {self.name}: {t} must list prefixes, as @p=0,500000")
            elif t == "@stable":
                self.build = stable_folder()
                self.stable = True
            elif t.startswith("@"):
                self.build = Path(t[1:])
            elif re.fullmatch(r"[A-Z][A-Z0-9_]*=.*", t):
                key, value = t.split("=", 1)
                self.env[key] = value
            else:
                self.args.append(t)

    def binary(self, name):
        folder = self.build or Path(env("BIN"))
        path = folder / (name + EXE)
        if not path.is_file():
            raise Skip(f"no {path}" + (", run 00-build.bat stable" if self.stable else ", run 00-build.bat"))
        return path.resolve()


def parse_arms(name, default=""):
    arms = [Arm(a) for a in env(name, default).split(";") if a.strip()]
    if not arms:
        raise Fail(f"{name} has no arms")
    return arms


def model_path(name="MODEL"):
    path = Path(env(name))
    if not env(name) or not path.is_file():
        raise Fail(f"{name} not found: {env(name)!r}, set it in _local.bat")
    return path.resolve()


def warmup_args():
    # the warmup run pages in mmap weights and pays first-use costs before the measurement. -lm dio reads the
    # weights at load, the depth fill and the warmup request of 04 warm the rest, and the median of 02 drops a
    # cold first rep: WARMUP=1 only adds load time and memory pressure at start
    return [] if env("WARMUP") == "1" else ["--no-warmup"]


def model_args(model):
    return ["-m", str(model), "-fa", "on", *split_args(env("LOADMODE")), *split_args(env("EXTRA")), *warmup_args()]


# ---------------------------------------------------------------- identity

_digests = {}


def digest(path):
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _digests:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _digests[key] = h.hexdigest()
    return _digests[key]


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args])


def model_identity(model):
    split = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})(\.gguf)", model.name, re.IGNORECASE)
    files = [model] if not split else [model.with_name(f"{split[1]}-{i:05d}-of-{split[3]}{split[4]}")
                                      for i in range(1, int(split[3]) + 1)]
    return [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]


def machine():
    return {"host": platform.node(), "backend": env("BACKEND")}


def env_identity(run_env):
    # a secret ends in one of these words. a substring test dropped GGML_VK_HEADSUM_MIN_TOKENS on 06.10, so the arm
    # that set it and the default shared one result
    return {k: v for k, v in sorted(run_env.items())
            if k.upper().startswith(ENV_PREFIXES) and k.upper() not in ENV_IGNORED
            and not re.search(r"(^|_)(TOKEN|SECRET|PASSWORD|PASS|KEY)$", k.upper())}


def arm_env(arm, extra=None):
    run_env = os.environ.copy()
    run_env.update(extra or {})
    run_env.update(arm.env)
    return run_env


def source_id(backend):
    """Hash of the sources of this backend's build: HEAD, the uncommitted changes and the untracked files. The
    directories of the other backends are left out, so a change there keeps the results of this machine. Returns the
    id and whether there were uncommitted changes or untracked files in those sources."""
    entries = ["CMakeLists.txt", "cmake", "common", "include", "src", "tools", "vendor", "tests/test-backend-ops.cpp",
               "ggml/CMakeLists.txt", "ggml/cmake", "ggml/include"]
    own = {"ggml-cpu", "ggml-" + backend}
    for line in git("ls-tree", "HEAD", "ggml/src/").decode().splitlines():
        meta, path = line.split("\t", 1)
        name = path.rsplit("/", 1)[-1]
        if meta.split()[1] == "tree" and name.startswith("ggml-") and name not in own:
            continue
        entries.append(path)
    h = hashlib.sha256(f"{backend}\n{env('CMAKE_BACKEND')}\n".encode())
    h.update(git("ls-tree", "HEAD", "--", *entries))
    diff = git("diff", "HEAD", "--binary", "--", *entries)
    h.update(diff)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *entries).decode().splitlines()
    for path in untracked:
        h.update(f"{path} {digest(ROOT / path)}\n".encode())
    return "src-" + h.hexdigest()[:16], bool(diff or untracked)


def is_binary(path):
    if not path.is_file():
        return False
    if os.name == "nt":
        return path.suffix.lower() in (".exe", ".dll")
    return path.suffix in (".so", ".dylib") or os.access(path, os.X_OK)


_build_ids = {}


def build_id(folder):
    """The source id that 00-build.bat recorded, or the hash of the binaries when there is no record or a binary
    is newer than it (built in another way)."""
    folder = Path(folder).resolve()
    if folder not in _build_ids:
        files = sorted(p for p in folder.iterdir() if is_binary(p))
        record = folder / BUILD_RECORD
        result = None
        try:
            if max(p.stat().st_mtime for p in files) <= record.stat().st_mtime + 2:
                result = json.loads(record.read_text(encoding="utf-8"))["id"]
        except (OSError, ValueError, KeyError):
            pass
        if result is None:
            h = hashlib.sha256()
            for p in files:
                h.update(f"{p.name} {digest(p)}\n".encode())
            result = "bin-" + h.hexdigest()[:16]
        _build_ids[folder] = result
    return _build_ids[folder]


def build_head(folder):
    """The commit of the build and whether it had uncommitted changes, as 00-build.bat recorded them; empty when
    unknown (an older record, a build made another way)."""
    try:
        record = json.loads((Path(folder).resolve() / BUILD_RECORD).read_text(encoding="utf-8"))
        return record.get("head", ""), record.get("dirty", "")
    except (OSError, ValueError):
        return "", ""


def current_build():
    try:
        return build_id(Path(env("BIN")))
    except OSError:
        return "not built"


def identity_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def cache_get(identity, label):
    if forced(label):
        return None
    try:
        record = json.loads((CACHE / "results" / (identity_key(identity) + ".json")).read_text(encoding="utf-8"))
        return record if record["identity"] == identity else None
    except (OSError, ValueError, KeyError):
        return None


def cache_put(identity, result, log):
    folder = CACHE / "results"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identity_key(identity) + ".json")
    part = path.with_suffix(".part")
    record = {"identity": identity, "result": result, "log": str(log), "time": now()}
    part.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    part.replace(path)


# ---------------------------------------------------------------- running

LAST_RUN = CACHE / "last-model-run.txt"
# the end of the last model run or build, for COOL: a build loads every core for minutes and heats the 395 as a run does
LAST_HEAT = CACHE / "last-heat.txt"


DEVICE_LINE = re.compile(r"^\s*(\S+): .*\((\d+) MiB, (\d+) MiB free\)\s*$")
IDLE_USED = CACHE / "idle-used.json"
# the GPU memory counters of Windows (Task Manager shows them): the memory of all processes on all adapters, also the
# memory the driver has not released yet. The Vulkan memory budget the 395 reports does not see it, it stays at
# 108782 MiB free also in the runs that failed on memory
GPU_COUNTERS = ("Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | "
                "ForEach-Object { [string]$_.DedicatedUsage + ' ' + [string]$_.SharedUsage }")
SHOWN_USED = []


def device_used():
    """MiB of GPU memory in use, None without a reading: on Windows the dedicated and shared usage of all adapters,
    elsewhere total - free of the device of the runs (llama-server --list-devices)."""
    try:
        if os.name == "nt":
            out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", GPU_COUNTERS],
                                 capture_output=True, text=True, timeout=60).stdout
            values = [int(x) for x in out.split() if x.isdigit()]
            return sum(values) >> 20 if values else None
        exe = Path(env("BIN")) / ("llama-server" + EXE)
        dev = env("CHECKDEVICE", DEVICES.get(env("BACKEND"), ""))
        if not exe.is_file() or not dev:
            return None
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        m = DEVICE_LINE.match(line)
        if m and m[1] == dev:
            return int(m[2]) - int(m[3])
    return None


def gib(mib):
    return f"{mib / 1024:.1f}"


def settle():
    """Before a model run: wait until the GPU memory of the last one is free again. The driver releases the memory of
    a finished process lazily: on the 395 a run right after another failed on memory or hung while loading
    (08.10-11.10). Done at once when the memory in use is within SETTLE_MARGIN MiB (2048) of the least seen on this
    machine (cache/idle-used.json), else the memory is read every SETTLE_STEP seconds (3) until it stops going down,
    SETTLE_MAX seconds (180) at most. Without a reading the old rule: SETTLE seconds (30) after the last model run."""
    try:
        since = time.time() - float(LAST_RUN.read_text())
    except (OSError, ValueError):
        since = None
    # COOL seconds (0): at least that long after the last model run or build, so that the arms of a run start from about
    # the same temperature (the 395 reached 95 C in an hour of chats to 213K on 14.10, the ops of the last profile depth
    # ran 2 percent slower than the night before; the first arm after the build of 14.10 ran 3 to 5 percent slower in
    # every phase than the second, also in the steps that read nothing from the drive)
    cool = env_int("COOL", 0)
    try:
        heat = time.time() - float(LAST_HEAT.read_text())
    except (OSError, ValueError):
        heat = since
    if cool > 0 and heat is not None and heat < cool:
        print(f"cooling {cool - heat:.0f} s after the last model run or build (COOL)", flush=True)
        time.sleep(cool - heat)
        if since is not None:
            since += cool - heat
    max_wait, step = env_int("SETTLE_MAX", 180), env_int("SETTLE_STEP", 3)
    if since is not None and since > max_wait:
        return
    used = device_used()
    if used is None:
        wait = float(env("SETTLE", "30")) - (since if since is not None else 1e9)
        if wait > 0:
            print(f"waiting {wait:.0f} s for the driver to release the memory of the last model run", flush=True)
            time.sleep(wait)
        return
    key = platform.node()
    try:
        idle = json.loads(IDLE_USED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        idle = {}
    margin = env_int("SETTLE_MARGIN", 2048)
    known = key in idle
    start, seen, flat = time.time(), [used], 0
    while True:
        if used < idle.get(key, used + 1):
            idle[key] = used
            CACHE.mkdir(parents=True, exist_ok=True)
            IDLE_USED.write_text(json.dumps(idle) + "\n", encoding="utf-8")
        if known and used <= idle[key] + margin or flat >= 2 or time.time() - start > max_wait:
            break
        time.sleep(step)
        prev, used = used, device_used()
        if used is None:
            used = prev
            break
        seen.append(used)
        flat = 0 if used < prev - 256 else flat + 1
    if not SHOWN_USED:
        SHOWN_USED.append(used)
        print(f"GPU memory in use: {gib(used)} GiB, least seen {gib(idle[key])} GiB", flush=True)
    if len(seen) > 1:
        trail = " -> ".join(gib(x) for x in seen[-8:])
        note = "" if used <= idle[key] + margin else f", {gib(idle[key])} GiB when idle: something else holds memory"
        print(f"GPU memory in use {trail} GiB in {time.time() - start:.0f} s{note}", flush=True)


def retry_wait(attempt):
    """Seconds before another try of a model run that failed on memory: none when the GPU memory can be read (settle()
    waits until it stops going down), else SETTLE x attempt."""
    return 0 if device_used() is not None else float(env("SETTLE", "30")) * attempt


def mark_end():
    CACHE.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(f"{time.time()}\n")
    LAST_HEAT.write_text(f"{time.time()}\n")


def exit_text(code):
    return f"exit {code}" if 0 <= code < 256 else f"exit {code & 0xffffffff:#010x}"


def last_error(text):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    errors = [s for s in lines if re.search(r"error|exception|failed|abort", s, re.IGNORECASE)]
    if errors:
        return errors[-1][:200]
    return "no error message, the log ends with: " + (lines or [""])[-1][:160]


def load_watch(log):
    """The step a stopped model load stopped in, from the watchdog of load_all_data (src/fork/llama-load-watch-fork.h)."""
    try:
        hits = [s for s in read_log(log).splitlines() if "load_all_data: no progress" in s]
    except OSError:
        return ""
    return ", " + hits[-1].split("load_all_data: ", 1)[1].strip()[:160] if hits else ""


def read_log(path):
    # a windows error text comes in the ansi code page, the rest is utf-8
    out = []
    for line in Path(path).read_bytes().splitlines():
        try:
            out.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(line.decode("mbcs" if os.name == "nt" else "latin-1", errors="replace"))
    return "\n".join(out)


def retryable(code, text):
    # an exception exit code without a message: the process died before it could print one (a load or an abort).
    # 75: the load watchdog ended a load that stopped (LLAMA_LOAD_WATCH_EXIT, src/fork/llama-load-watch-fork.h)
    if code == 75 and "load_all_data: no progress" in text:
        return True
    return any(s in text for s in MEMORY_ERRORS) or (code & 0xffffffff >= 0xC0000000 and "error" not in text.lower())


def execute(cmd, log, run_env, stdout=None, model=True, check=True):
    """Run cmd with its output in log, its stdout in a separate file when given. A model run that fails on memory gets
    two more tries after longer waits: the driver frees the memory of the previous process lazily."""
    cmd = [str(c) for c in cmd]
    if model:
        no_server_running()
    first = log
    for attempt in (1, 2, 3):
        if model:
            settle()
        with log.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"command": cmd, "env": env_identity(run_env)}) + "\n")
            f.flush()
            out = stdout.open("w", encoding="utf-8") if stdout else f
            try:
                code = subprocess.run(cmd, stdout=out, stderr=f, env=run_env).returncode
            finally:
                if stdout:
                    out.close()
        if model:
            mark_end()
        text = read_log(log)
        if code == 0 or not check:
            return code, text
        if attempt < 3 and model and retryable(code, text):
            wait = retry_wait(attempt)
            print(f"{log.name}: {exit_text(code)}, {last_error(text)}; one more try" + (f" in {wait:.0f} s" if wait else ""), flush=True)
            time.sleep(wait)
            log = first.with_name(f"{first.stem}-try{attempt + 1}.log")
            continue
        raise Fail(f"{exit_text(code)}: {last_error(text)} ({log.name}, try {attempt})")


# history.csv: one row per number of a new result
HISTORY_FIELDS = ["time", "host", "backend", "test", "arm", "point", "metric", "value", "build", "head", "dirty", "spec"]


def decode_metrics(rs):
    """TG, the accepted share of the drafts and PP of the short (<= 512 tokens) and long prompts of a decode result."""
    n, ms = sum(x["n"] for x in rs), sum(x["ms"] for x in rs)
    dn, da = sum(x["draft_n"] for x in rs), sum(x["draft_acc"] for x in rs)
    pp = []
    for part in ([x for x in rs if x["prompt_n"] <= 512], [x for x in rs if x["prompt_n"] > 512]):
        pp.append(rate(sum(x["prompt_n"] for x in part), sum(x["prompt_ms"] for x in part)) if part else None)
    return {"tg": rate(n, ms), "accept": da / dn if dn else None, "pp_short": pp[0], "pp_long": pp[1]}


def history_from_cache(path):
    """Starts history.csv with the kept results of cache/results: the arm and the point come from the log name of the
    run that measured them, the head, dirty flag and spec of those builds are unknown."""
    rows = []
    for f in sorted((CACHE / "results").glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            ident, res = rec["identity"], rec["result"]
        except (OSError, ValueError, KeyError):
            continue
        kind = ident.get("kind")
        # the speed results of 02 from before BENCH_VERSION were measured another way
        if kind in ("bench", "bench-small") and ident.get("v") != BENCH_VERSION:
            continue
        stem = re.split(r"[\\/]", rec.get("log", ""))[-1]
        stem = stem[:-4] if stem.endswith(".log") else stem
        arm, point, metrics = None, None, {}
        try:
            if kind == "bench" and (m := re.fullmatch(r"(.+)-d(\d+)", stem)):
                arm, point = m[1], f"d{m[2]}"
                metrics["replans"] = res.get("replans")
                for k in ("pp", "tg"):
                    if k in res:
                        metrics.update({k: res[k][2], k + "_min": res[k][3], k + "_max": res[k][4]})
            elif kind == "bench-small" and stem.endswith("-small"):
                arm, point = stem[:-6], "small"
                metrics = {f"pp{n}": v[0] for n, v in res.items()}
            elif kind == "quality" and "_vs_" in stem:
                arm, ref = stem.split("_vs_", 1)
                point = f"vs {ref}"
                metrics = {"ppl_ratio": res["ratio"][0], "ppl_ratio_err": res["ratio"][1], "kld": res["kld"],
                           "kld999": res["kld999"], "same_top": res["top"]}
            elif kind == "decode" and (m := re.fullmatch(r"(.+)-p(\d+)", stem)):
                arm, point = m[1], f"p{m[2]}"
                metrics = decode_metrics(res["rows"])
            elif kind == "profile" and (m := re.fullmatch(r"(pp\d*|tg)-d(\d+)-(.+)", stem)):
                arm, point = m[3], f"{m[1]}@{m[2]}"
                host = res.get("host")
                if host:
                    metrics = {"host_graph": host[0], "host_inputs": host[1], "host_compute": host[2]}
                if "ops" in res:
                    metrics["gpu_ms"] = sum(o[2] for o in res["ops"])
            elif kind == "check" and "-o" in ident.get("args", []):
                args = ident["args"]
                op = args[args.index("-o") + 1] + (" " + args[args.index("-p") + 1] if "-p" in args else "")
                prefix = re.sub(r"[^\w.-]+", "_", op + " ")
                arm, point = (stem[len(prefix):] if stem.startswith(prefix) else stem), op
                metrics = {"passed": res["passed"], "total": res["total"]}
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if arm is None:
            continue
        mach = ident.get("machine") or {}
        # the time of a kept result has no year (now())
        t = rec.get("time", "")
        if re.fullmatch(r"\d\d-\d\d \d\d:\d\d", t):
            t = f"{datetime.now().year}-{t}:00"
        for metric, value in metrics.items():
            if value is not None:
                rows.append([t, mach.get("host", ""), mach.get("backend", ""), kind.split("-")[0], arm, point,
                             metric, f"{value:.6g}" if isinstance(value, float) else value, ident.get("build", ""), "", "", ""])
    rows.sort(key=lambda r: r[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HISTORY_FIELDS)
        w.writerows(rows)
    print(f"{path.name}: started with {len(rows)} numbers of the kept results", flush=True)


class Tee:
    """A console stream that also writes into the console.txt of the current run: why a try was repeated (a stall
    while loading, an exit code) is printed only, and the run folder kept no trace of it."""

    def __init__(self, stream):
        self.stream = stream
        self.file = None

    def write(self, text):
        self.stream.write(text)
        if self.file:
            self.file.write(text)
            self.file.flush()
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def console_to(path):
    out = path.open("a", encoding="utf-8", errors="backslashreplace")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if not isinstance(stream, Tee):
            stream = Tee(stream)
            setattr(sys, name, stream)
        stream.file = out


class Run:
    """One call of a script: its log folder, its summary and the counts of new and kept results."""

    def __init__(self, name):
        self.start = datetime.now()
        self.dir = Path(env("LOGS", str(HERE / "logs"))) / f"{name}-{self.start:%Y%m%d-%H%M%S}"
        self.dir.mkdir(parents=True, exist_ok=True)
        console_to(self.dir / "console.txt")
        if not (self.dir.parent / "history.csv").is_file():
            history_from_cache(self.dir.parent / "history.csv")
        self.lines = []
        self.failed = False
        self.new = 0
        self.kept = 0
        print(f"logs in {self.dir}", flush=True)

    def add(self, *lines):
        self.lines.extend(lines)
        (self.dir / "summary.txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")

    def measure(self, label, identity, run):
        """The kept result of this identity, or a new one from run(log). Returns (result, kept)."""
        record = cache_get(identity, label)
        if record:
            self.kept += 1
            print(f"{label}: kept result of {record['time']}", flush=True)
            return record["result"], True
        log = self.dir / (re.sub(r"[^\w.-]+", "_", label) + ".log")
        print(f"{label}: running, {log.name}", flush=True)
        result = run(log)
        cache_put(identity, result, log)
        self.new += 1
        return result, False

    def record(self, test, arm, point, metrics, kept, folder):
        """A new result into history.csv next to the run folders, one row per metric (None is left out). A kept
        result is there from the run that measured it."""
        if kept:
            return
        path = self.dir.parent / "history.csv"
        head, dirty = build_head(folder)
        bid = build_id(folder)
        new_file = not path.is_file()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(HISTORY_FIELDS)
            for metric, value in metrics.items():
                if value is None:
                    continue
                w.writerow([f"{self.start:%Y-%m-%d %H:%M:%S}", platform.node(), env("BACKEND"), test, arm.name, point, metric,
                            f"{value:.6g}" if isinstance(value, float) else value, bid, head, dirty, arm.text])

    def finish(self):
        if self.new or self.kept:
            self.add("", f"{self.new} new, {self.kept} kept (k): the same build measured them before. FORCE=1 runs "
                         f"them again, FORCE=<regex> the matching ones (as default-d32768)")
        self.add(f"logs in {self.dir}")
        print("\n" + "\n".join(self.lines), flush=True)
        return 1 if self.failed else 0


def rate(n, ms):
    return 1000 * n / ms if ms > 0 else 0.0


def table(header, rows):
    rows = [list(r) + [""] * (len(header) - len(r)) for r in rows]
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    return ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [header, *rows]]


# ---------------------------------------------------------------- check

def cmd_check():
    run = Run("check")
    backend = env("BACKEND")
    device = env("CHECKDEVICE", DEVICES.get(backend, "CPU"))
    ops = [split_args(o) for o in env("CHECKOPS", "MUL_MAT").split(";") if o.strip()]
    arms = parse_arms("CHECKARMS", "default")
    rows, fails = [], []
    for arm in arms:
        try:
            exe = arm.binary("test-backend-ops")
        except Skip as e:
            rows.append(["", arm.name, f"skip: {e}"])
            continue
        bid = build_id(exe.parent)
        for op in ops:
            if arm.ops and op[0] not in arm.ops:
                continue
            cmd = [exe, "test", "-b", device, "-o", op[0], *(["-p", op[1]] if len(op) > 1 else []), *arm.args]
            run_env = arm_env(arm)
            identity = {"v": VERSION, "kind": "check", "build": bid, "args": [str(c) for c in cmd[1:]],
                        "env": env_identity(run_env), "machine": machine()}

            def go(log):
                code, text = execute(cmd, log, run_env, model=False, check=False)
                counts = re.findall(r"(\d+)/(\d+) tests passed", text)
                if not counts:
                    raise Fail(f"{exit_text(code)}, no test result: {last_error(text)} ({log.name})")
                return {"passed": sum(int(a) for a, _ in counts), "total": sum(int(b) for _, b in counts),
                        "fails": [s.strip() for s in text.splitlines() if "FAIL" in s][:8]}

            label = " ".join(op) + " " + arm.name
            try:
                r, kept = run.measure(label, identity, go)
            except Fail as e:
                run.failed = True
                rows.append([" ".join(op), arm.name, f"FAIL {e}"])
                continue
            run.record("check", arm, " ".join(op), {"passed": r["passed"], "total": r["total"]}, kept, exe.parent)
            ok = r["passed"] == r["total"]
            run.failed |= not ok
            # 0/0: the backend supports none of the cases
            rows.append([" ".join(op), arm.name, f"{r['passed']}/{r['total']}" + ("" if ok else " FAIL") + (" k" if kept else "")])
            fails.extend(f"  {op[0]} {arm.name}: {s}" for s in r["fails"])
    run.add(f"check: test-backend-ops on {device} against the CPU, build {current_build()}", "",
            *table(["op", "arm", "passed"], rows))
    if fails:
        run.add("", "failed cases:", *fails)
    return run.finish()


# ---------------------------------------------------------------- bench

def logits_info(data, ctx, chunks):
    with data.open("rb") as f:
        header = f.read(20)
    if len(header) != 20 or header[:8] != b"_logits_":
        raise Fail("bad header of the reference logits")
    nctx, vocab, nchunk = struct.unpack("<Iii", header[8:])
    expected = 20 + nctx*nchunk*4 + nchunk*(nctx - nctx//2 - 1)*(2*((vocab + 1)//2) + 4)*2
    if (nctx, nchunk) != (ctx, chunks) or vocab < 1 or data.stat().st_size != expected:
        raise Fail("reference logits incomplete, or the corpus is too short for QCTX x QCHUNKS")
    return vocab


def number(pattern, text, label):
    m = re.search(pattern, text)
    if not m:
        raise Fail(f"no {label} in the output")
    return [float(g) for g in m.groups()]


def cmd_quality():
    run = Run("quality")
    model = model_path()
    corpus = Path(env("PPLFILE"))
    if not corpus.is_file():
        raise Fail(f"no corpus {corpus}, run get-wikitext.bat")
    ctx, chunks, ub = env_int("QCTX", 16384), env_int("QCHUNKS", 2), env_int("QUB", 2048)
    keep = max(1, env_int("QREFKEEP", 2))
    spec = env("QREF", "stable @stable" if (stable_folder() / ("llama-perplexity" + EXE)).is_file() else "default")
    ref, repeat = Arm(spec), Arm(spec)
    repeat.name = "repeat"
    arms = [repeat, *parse_arms("QARMS", "default")]
    params = [*model_args(model), "-f", corpus.resolve(), "-c", ctx, "--chunks", chunks, "-b", ub, "-ub", ub,
              *split_args(env("QARGS"))]
    inputs = {"model": model_identity(model), "corpus": digest(corpus)}

    exe = ref.binary("llama-perplexity")
    ref_env = arm_env(ref)
    ref_identity = {"v": VERSION, "kind": "quality-ref", "build": build_id(exe.parent),
                    "args": [str(c) for c in [*params, *ref.args]], "env": env_identity(ref_env), **inputs,
                    "machine": machine()}
    key = identity_key(ref_identity)
    folder = CACHE / "ref"
    folder.mkdir(parents=True, exist_ok=True)
    data, meta = folder / f"{key[:24]}.dat", folder / f"{key[:24]}.json"
    record = None
    if not forced(f"reference {ref.name}"):
        try:
            record = json.loads(meta.read_text(encoding="utf-8"))
            logits_info(data, ctx, chunks)
            if record["identity"] != ref_identity or record["size"] != data.stat().st_size:
                record = None
        except (OSError, ValueError, KeyError, Fail):
            record = None
    ref_kept = record is not None
    if record:
        os.utime(data)
        print(f"reference {ref.name}: kept from {record['time']}", flush=True)
    else:
        # the reference files are large (about QCTX/2 x vocab x 2 bytes per chunk): keep the newest ones only
        for old in sorted(folder.glob("*.dat"), key=lambda p: p.stat().st_mtime, reverse=True)[keep - 1:]:
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)
        part = data.with_suffix(".part")
        log = run.dir / f"reference-{ref.name}.log"
        print(f"reference {ref.name}: running, {log.name}; about {ctx // 2 * chunks * 248324 * 2 / 2**30:.1f} GiB, "
              f"{shutil.disk_usage(folder).free / 2**30:.0f} GiB free", flush=True)
        try:
            _, text = execute([exe, *params, *ref.args, "--kl-divergence-base", part], log, ref_env)
            ppl = number(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", text, "final PPL")
            logits_info(part, ctx, chunks)
            part.replace(data)
        except Fail as e:
            part.unlink(missing_ok=True)
            run.failed = True
            run.add(f"quality: reference {ref.name} FAIL {e}")
            return run.finish()
        finally:
            part.unlink(missing_ok=True)
        record = {"identity": ref_identity, "size": data.stat().st_size, "ppl": ppl, "time": now(), "log": str(log)}
        meta.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        run.new += 1

    rows, notes = [], []
    for arm in arms:
        try:
            exe = arm.binary("llama-perplexity")
        except Skip as e:
            rows.append([arm.name, f"skip: {e}"])
            continue
        run_env = arm_env(arm)
        cmd = [exe, *params, *arm.args, "--kl-divergence-base", data, "--kl-divergence"]
        identity = {"v": VERSION, "kind": "quality", "build": build_id(exe.parent), "reference": key,
                    "args": [str(c) for c in [*params, *arm.args]], "env": env_identity(run_env), **inputs,
                    "machine": machine()}

        def go(log):
            _, text = execute(cmd, log, run_env)
            return {"ratio": number(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([-\d.e+]+)[^\d]+([\d.e+-]+)", text, "PPL ratio"),
                    "kld": number(r"Mean\s+KLD:\s*([-\d.e+]+)", text, "mean KLD")[0],
                    "kld999": number(r"99\.9%\s+KLD:\s*([-\d.e+]+)", text, "99.9% KLD")[0],
                    "top": number(r"Same top p:\s*([\d.]+)", text, "same top p")[0]}

        try:
            r, kept = run.measure(f"{arm.name} vs {ref.name}", identity, go)
        except Fail as e:
            run.failed = True
            rows.append([arm.name, f"FAIL {e}"])
            continue
        (ratio, err) = r["ratio"]
        run.record("quality", arm, f"vs {ref.name}", {"ppl_ratio": ratio, "ppl_ratio_err": err, "kld": r["kld"],
                   "kld999": r["kld999"], "same_top": r["top"]}, kept, exe.parent)
        far = abs(ratio - 1) > 2 * err
        rows.append([arm.name, f"{ratio:.6f} +-{err:.6f}" + (" *" if far else ""), f"{r['kld']:.6f}",
                     f"{r['kld999']:.5f}", f"{r['top']:.3f}%" + (" k" if kept else "")])
    run.add(f"quality: llama-perplexity -c {ctx} --chunks {chunks} -ub {ub} on {corpus.name}, {model.name}",
            f"reference {ref.name}: PPL {record['ppl'][0]:.4f} +-{record['ppl'][1]:.4f}" + (" k" if ref_kept else ""),
            "", *table(["arm", "PPL / reference", "mean KLD", "99.9% KLD", "same top"], rows), "",
            "* the PPL ratio is more than 2 sigma from 1. repeat runs the reference again: not 0 means the build does",
            "not give the same result twice. a change with the same math but other rounding (an arm without fusion)",
            "shows how far rounding alone moves this model: QSA top-k picks other blocks.")
    return run.finish()


# ---------------------------------------------------------------- decode

# the server log lines a result keeps; DECGREP picks from them for the summary
KEPT_LINES = r"moe_cache:|moe_stats|phase_mem:|sched_prof|buffer size|QSA KV"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with OPENER.open(req, timeout=timeout) as r:
        return r.status, r.read()


def wait_no_server(timeout=180):
    """Until no llama-server.exe is listed: a killed server with 74 GB of pinned host memory leaves the list a while
    after its parent (nsys) ended, and the next arm refused to start (14.10)"""
    if os.name != "nt":
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(["tasklist", "/fi", "imagename eq llama-server.exe"], capture_output=True).stdout
        if b"llama-server.exe" not in out:
            return
        time.sleep(2)


def no_server_running():
    if os.name != "nt":
        return
    # bytes: the text of tasklist is in the console code page
    out = subprocess.run(["tasklist", "/fi", "imagename eq llama-server.exe"], capture_output=True).stdout
    if b"llama-server.exe" in out:
        raise Fail("a llama-server.exe is running: it holds gpu memory and spoils the numbers, stop it first")


def prompt(name):
    if name in PROMPTS:
        return name, PROMPTS[name]
    path = Path(name)
    if path.is_file():
        return path.stem, path.read_text(encoding="utf-8")
    raise Fail(f"unknown prompt {name}: one of {', '.join(PROMPTS)} or a text file")


def start_server(exe, args, run_env, log):
    """Start llama-server and wait for /health, with the retries of a load that fails on memory or stalls; returns the
    process and the log it writes (a retry writes log-tryN)"""
    wait = env_int("DECWAIT", 900)
    port = args[args.index("--port") + 1]
    url = f"http://127.0.0.1:{port}"
    no_server_running()
    first = log
    # a server that fails on memory while loading gets two more tries after longer waits, as in execute(): the driver
    # frees the memory of the previous server lazily (09.10, 395: the arm right after none failed in two runs)
    for attempt in (1, 2, 3):
        settle()
        with log.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"command": [str(exe), *args], "env": env_identity(run_env)}) + "\n")
            f.flush()
            proc = subprocess.Popen([str(exe), *args], stdout=f, stderr=f, env=run_env)
        start = time.time()
        # a load that stops writing its log hangs: 11.10, 395: mtp3 right after auto stopped in the load of the MTP head
        # (as on 08.10) and the arm failed after DECWAIT; a normal load is silent for 20 s at most
        stall = env_int("DECSTALL", 240)
        last_size, last_growth, stalled = -1, time.time(), False
        while proc.poll() is None:
            time.sleep(3)
            try:
                if http(url + "/health")[0] == 200:
                    break
            except (urllib.error.URLError, OSError):
                pass
            size = log.stat().st_size if log.exists() else 0
            if size != last_size:
                last_size, last_growth = size, time.time()
            stalled = time.time() - last_growth > stall
            if stalled or time.time() - start > wait:
                proc.kill()
                proc.wait()
                mark_end()
                if stalled and attempt < 3:
                    break
                raise Fail(f"no answer from /health in {int(time.time() - start)} s" + (f", no log output for {stall} s" if stalled else "")
                           + f"{load_watch(log)} ({log.name}, try {attempt})")
        if proc.poll() is None:
            break
        mark_end()
        text = read_log(log)
        if stalled:
            pause = retry_wait(attempt + 1)
            print(f"{log.name}: no log output for {stall} s while loading{load_watch(log)}; one more try" + (f" in {pause:.0f} s" if pause else ""), flush=True)
            time.sleep(pause)
            log = first.with_name(f"{first.stem}-try{attempt + 1}.log")
            continue
        if attempt < 3 and retryable(proc.returncode, text):
            pause = retry_wait(attempt)
            print(f"{log.name}: {exit_text(proc.returncode)} while loading, {last_error(text)}; one more try" + (f" in {pause:.0f} s" if pause else ""), flush=True)
            time.sleep(pause)
            log = first.with_name(f"{first.stem}-try{attempt + 1}.log")
            continue
        raise Fail(f"llama-server {exit_text(proc.returncode)} while loading: {last_error(text)} ({log.name}, try {attempt})")
    return proc, log


def progress(text):
    """The server's own speed lines of one request: the prompt progress at every batch, [[n_tokens, seconds], ...]
    (n_tokens counts the cached part too, the seconds run from the start of this prompt), and the generation speed
    over the last 3 s, printed every 3 s"""
    pp = [[int(m[1]), float(m[2])] for m in re.finditer(r"prompt processing, n_tokens =\s*(\d+), progress = [\d.]+, t =\s*([\d.]+) s", text)]
    tg = [float(m[1]) for m in re.finditer(r"n_gen =\s*\d+, tg =\s*[\d.]+ t/s, tg_3s =\s*([\d.]+) t/s", text)]
    return pp, tg


def ubatches(text, least=1025):
    """LLAMA_INPUT_TIMING of the prompt ubatches of the model, [[tokens, graph ms, inputs ms, submit ms, first], ...],
    first when the ubatch starts a batch (-b): after a progress line or the ubatches of the MTP head, which computes
    the prompt in ubatches of 1024 (LLAMA_MTP_UBATCH) and a decode step in 1-4, left out by least"""
    out, first = [], True
    for m in re.finditer(r"ubatch timing: (\d+) tokens, graph ([\d.]+) ms, inputs ([\d.]+) ms, submit ([\d.]+) ms"
                         r"|prompt processing, n_tokens =", text):
        if m[1] is None or int(m[1]) < least:
            first = True
            continue
        out.append([int(m[1]), float(m[2]), float(m[3]), float(m[4]), first])
        first = False
    return out


def ubatch_gpu(rows):
    """The compute time of each prompt ubatch, ms, None where the log does not show it. A backend that returns before
    the GPU is done (Vulkan) shows the first ubatch of a batch with a short submit, and each next ubatch waits in its
    submit for the one before: the compute of a ubatch is then the step to the next one of the same batch (graph,
    inputs, submit), the last of a batch ends outside the ubatches. Else the submit is the compute"""
    rows = [list(r) + [False] * (5 - len(r)) for r in rows]  # a cached result from before the first flag
    firsts = [r[3] for r in rows if r[4]]
    others = [r[3] for r in rows if not r[4]]
    lagged = bool(firsts and others) and statistics.median(firsts) < 0.3 * statistics.median(others)
    if not lagged:
        return [r[3] for r in rows], False
    return [sum(nxt[1:4]) if nxt is not None and not nxt[4] else None
            for nxt in rows[1:] + [None]], True


def segments(curve, start=0, least=2048):
    """The speed of each stretch between progress lines, [[from, to, t/s], ...]; a stretch shorter than least tokens
    (the tail of a prompt, around a context checkpoint) joins the next one"""
    out, n0, t0 = [], start, 0.0
    for n, t in curve:
        if n - n0 >= least and t > t0:
            out.append([n0, n, (n - n0) / (t - t0)])
            n0, t0 = n, t
    return out


def serve(exe, args, run_env, log, requests, sampling, port, rows_text="", chats=()):
    """Start llama-server, send the warmup and the requests one after another, stop it. The answers stay next to
    the log. rows_text, one token long and not measured, reads the PLE rows of the prompts into the page cache first:
    otherwise the arm that runs first reads them from the drive. chats: conversations as a
    client keeps them, (name, [(user text, tokens), ...]), each turn sent with the messages before it and the prompt
    cache on, so a turn after the first computes only its new part at the depth of the conversation"""
    timeout = env_int("DECTIMEOUT", 1800)
    url = f"http://127.0.0.1:{port}"
    run_env = dict(run_env, LLAMA_SPEC_TRACE=str(log.with_suffix(".trace.csv")))
    proc, log = start_server(exe, args, run_env, log)
    rows = []
    try:
        temp, top_k, top_p = sampling
        warm = [("warmup", WARMUP, 0, 256)] + ([("rows", rows_text, 0, 1)] if rows_text else [])
        for name, text, seed, n in [*warm, *requests]:
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": temp,
                    "top_k": top_k, "top_p": top_p, "seed": seed, "cache_prompt": False, "stream": False}
            before = log.stat().st_size
            try:
                _, raw = http(url + "/v1/chat/completions", body, timeout)
            except urllib.error.HTTPError as e:
                raise Fail(f"request {name} seed {seed}: HTTP {e.code} {e.read()[:300]!r}")
            except (urllib.error.URLError, OSError) as e:
                raise Fail(f"request {name} seed {seed}: {e} ({log.name})")
            answer = json.loads(raw)
            log.with_name(f"{log.stem}-{name}-s{seed}.json").write_text(
                json.dumps(answer, ensure_ascii=False, indent=1), encoding="utf-8")
            if "timings" not in answer:
                raise Fail(f"request {name} seed {seed}: no timings in the answer {raw[:300]!r}")
            if name in ("warmup", "rows"):
                continue
            t = answer["timings"]
            message = answer["choices"][0]["message"]
            time.sleep(0.5)  # the last speed lines of the slot
            with log.open("rb") as f:
                f.seek(before)
                part_text = f.read().decode("utf-8", errors="replace")
            pp_curve, tg3 = progress(part_text)
            rows.append({"name": f"{name}-s{seed}", "pp_curve": pp_curve, "tg3": tg3, "ubatches": ubatches(part_text),
                         "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"],
                         "n": t["predicted_n"], "ms": t["predicted_ms"], "draft_n": t.get("draft_n", 0),
                         "draft_acc": t.get("draft_n_accepted", 0),
                         "answer": text_digest(f"{message.get('content')}\0{message.get('reasoning_content')}")[:16]})
        chat_rows = []
        for conv, turns in chats:
            messages = []
            for ti, (text, n) in enumerate(turns):
                messages.append({"role": "user", "content": text})
                body = {"messages": messages, "max_tokens": n, "temperature": temp, "top_k": top_k, "top_p": top_p,
                        "seed": 1, "cache_prompt": True, "stream": False}
                before = log.stat().st_size
                try:
                    _, raw = http(url + "/v1/chat/completions", body, timeout)
                except urllib.error.HTTPError as e:
                    raise Fail(f"chat {conv} turn {ti}: HTTP {e.code} {e.read()[:300]!r}")
                except (urllib.error.URLError, OSError) as e:
                    raise Fail(f"chat {conv} turn {ti}: {e} ({log.name})")
                answer = json.loads(raw)
                log.with_name(f"{log.stem}-chat-{conv}-t{ti}.json").write_text(
                    json.dumps(answer, ensure_ascii=False, indent=1), encoding="utf-8")
                if "timings" not in answer:
                    raise Fail(f"chat {conv} turn {ti}: no timings in the answer {raw[:300]!r}")
                t = answer["timings"]
                message = answer["choices"][0]["message"]
                messages.append({"role": "assistant", "content": message.get("content") or ""})
                time.sleep(0.5)
                with log.open("rb") as f:
                    f.seek(before)
                    pp_curve, tg3 = progress(f.read().decode("utf-8", errors="replace"))
                chat_rows.append({"conv": conv, "turn": ti, "pp_curve": pp_curve, "tg3": tg3,
                                  "cache_n": t.get("cache_n", 0), "prompt_n": t["prompt_n"],
                                  "prompt_ms": t["prompt_ms"], "n": t["predicted_n"], "ms": t["predicted_ms"],
                                  "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0)})
        # the slot prints its timings as it is released
        time.sleep(2)
    finally:
        proc.kill()
        proc.wait()
        mark_end()
    lines = read_log(log).splitlines()
    # the speculative statistics are printed after each request and add up, keep the last block
    spec = [s.strip() for s in lines if "auto:" in s or "statistics " in s]
    starts = [i for i, s in enumerate(spec) if re.search(r"print: auto: (on|trace only),", s)]
    extra = [s.strip() for s in lines if re.search(KEPT_LINES, s)]
    return {"rows": rows, "spec": spec[starts[-1]:] if starts else [], "extra": extra[-60:],
            "graph_timing": graph_timing(lines), **({"chat": chat_rows} if chats else {})}


def graph_timing(lines):
    """The medians over the windows of GGML_VK_GRAPH_TIMING (Vulkan): the time from one graph to the next, the gpu
    time of a graph, the host time in graph_compute and the gpu idle time. A window with a prompt ubatch is an
    outlier, the median drops it."""
    windows = [[float(v) for v in m.groups()] for s in lines for m in [re.search(
        r"graph timing over \d+ graphs: one every ([\d.]+) ms, gpu ([\d.]+) ms, in graph_compute ([\d.]+) ms, "
        r"gpu idle (-?[\d.]+) ms", s)] if m]
    if not windows:
        return None
    med = [statistics.median(w[i] for w in windows) for i in range(4)]
    return {"windows": len(windows), "period": med[0], "gpu": med[1], "host": med[2], "idle": med[3]}


def host_timing(lines, n_tokens):
    """Medians of the LLAMA_INPUT_TIMING lines of the ubatches of n_tokens tokens: graph build and allocation, input
    fills, the compute call. None without such lines."""
    rows = [[float(m[2]), float(m[3]), float(m[4])] for m in
            (re.search(r"ubatch timing: (\d+) tokens, graph ([\d.]+) ms, inputs ([\d.]+) ms, submit ([\d.]+) ms", x) for x in lines)
            if m and int(m[1]) == n_tokens]
    if not rows:
        return None
    med = [statistics.median(r[i] for r in rows) for i in range(3)]
    return {"n": len(rows), "graph": med[0], "inputs": med[1], "submit": med[2]}


def graph_timing_text(gt):
    return (f"graph timing, median of {gt['windows']} windows: one graph every {gt['period']:.2f} ms, gpu {gt['gpu']:.2f} ms, "
            f"host in graph_compute {gt['host']:.2f} ms, gpu idle {gt['idle']:.2f} ms ({100 * gt['idle'] / gt['period']:.0f}%)")


def cmd_decode():
    run = Run("decode")
    model = model_path()
    arms = parse_arms("DECARMS", "none")
    draft = model_path("SPECDRAFT") if env("SPECDRAFT") and any(a.draft for a in arms) else None
    ctx, ngen, port = env_int("DECCTX", 262144), env_int("DECNGEN", 1024), env_int("DECPORT", 8097)
    seeds = env_ints("DECSEEDS", "1 2")
    try:
        temp, top_k, top_p = split_args(env("DECSAMP", "1.0 20 0.95"))
        sampling = [float(temp), int(top_k), float(top_p)]
    except ValueError:
        raise Fail(f"DECSAMP must be \"temperature top_k top_p\", got {env('DECSAMP')!r}")
    prompts = [prompt(p) for p in split_args(env("DECPROMPTS", "os db code repeat"))]
    # DECPREFIX=none: only the conversations of DECCHAT
    prefixes = [] if env("DECPREFIX") == "none" else env_ints("DECPREFIX", "0")
    # DECCHAT: conversations as a client keeps them (the prompt cache on), "start[:turns[:add]]" each, in chars of
    # PPLFILE: the first turn is start chars and a question, then turns (DECCHATTURNS) turns of add (DECCHATADD) more
    # chars and a question each, DECCHATGEN tokens per answer. what a chat sees: the new part of a turn at the depth so
    # far; "0:30:16000" is a document read in parts of about 4K tokens up to about 120K, the speed against the length
    chat_specs = []
    for item in split_args(env("DECCHAT", "")):
        parts = [int(x) for x in item.split(":")]
        chat_specs.append((parts[0], parts[1] if len(parts) > 1 else env_int("DECCHATTURNS", 3),
                           parts[2] if len(parts) > 2 else env_int("DECCHATADD", 2000)))
    chat_starts = [c[0] for c in chat_specs]
    chat_gen = env_int("DECCHATGEN", 256)
    corpus = (Path(env("PPLFILE")).read_text(encoding="utf-8", errors="replace")
              if max(prefixes, default=0) > 0 or chat_specs else "")
    results = {}
    folders = {}
    for prefix in prefixes:
        head = corpus[:prefix] + "\n\n" if prefix else ""
        requests = [(name, head + text, seed, ngen) for name, text in prompts for seed in seeds]
        rows_text = head + "\n\n".join(text for _, text in prompts)
        for arm in arms:
            if arm.prefixes is not None and prefix not in arm.prefixes:
                continue
            label = f"{arm.name}-p{prefix}"
            try:
                if arm.draft and not draft:
                    raise Skip("SPECDRAFT is not set, it is the draft model of @draft")
                exe = arm.binary("llama-server")
            except Skip as e:
                results[arm.name, prefix] = f"skip: {e}"
                continue
            folders[arm.name] = exe.parent
            args = [str(a) for a in ["-m", model, *(["-md", draft] if arm.draft else []), "-c", ctx, "-np", 1,
                    "-fa", "on", "--repeat-penalty", "1.0", *split_args(env("LOADMODE")), *split_args(env("EXTRA")),
                    *warmup_args(), *split_args(env("DECARGS")), *arm.args, "--host", "127.0.0.1", "--port", port, "-lv", 4]]
            # the timing of every ubatch (one stderr line each) for the table of the prompt ubatches
            run_env = arm_env(arm, {"LLAMA_INPUT_TIMING": "1"})
            identity = {"v": VERSION, "kind": "decode", "build": build_id(exe.parent), "args": args,
                        "env": env_identity(run_env), "model": model_identity(model),
                        "draft": model_identity(draft) if arm.draft else None, "sampling": sampling,
                        "requests": [[name, text_digest(text), seed, n] for name, text, seed, n in requests],
                        "rows": text_digest(rows_text), "machine": machine()}
            try:
                results[arm.name, prefix] = run.measure(label, identity,
                                                        lambda log: serve(exe, args, run_env, log, requests, sampling, port, rows_text))
            except Fail as e:
                run.failed = True
                results[arm.name, prefix] = f"FAIL {e}"
    chats = []
    for start, n_turns, add_chars in chat_specs:
        turns = [((corpus[:start] + "\n\n" if start else "") + "Summarize the text above in two sentences.", chat_gen)]
        for ti in range(n_turns):
            add = corpus[start + ti*add_chars:start + (ti + 1)*add_chars]
            turns.append((add + "\n\nAnd this part, in two sentences?", chat_gen))
        chats.append((f"c{start}-{add_chars}", turns))  # two conversations from one start differ by their turns
    if chats:
        for arm in arms:
            label = f"{arm.name}-chat"
            try:
                if arm.draft and not draft:
                    raise Skip("SPECDRAFT is not set, it is the draft model of @draft")
                exe = arm.binary("llama-server")
            except Skip as e:
                results[arm.name, "chat"] = f"skip: {e}"
                continue
            folders[arm.name] = exe.parent
            args = [str(a) for a in ["-m", model, *(["-md", draft] if arm.draft else []), "-c", ctx, "-np", 1,
                    "-fa", "on", "--repeat-penalty", "1.0", *split_args(env("LOADMODE")), *split_args(env("EXTRA")),
                    *warmup_args(), *split_args(env("DECARGS")), *arm.args, "--host", "127.0.0.1", "--port", port, "-lv", 4]]
            # the timing of every ubatch (one stderr line each), for the host share of the verify and draft steps at
            # depth (not in the summary, the logs keep it)
            run_env = arm_env(arm, {"LLAMA_INPUT_TIMING": "1"})
            identity = {"v": VERSION, "kind": "decode-chat", "build": build_id(exe.parent), "args": args,
                        "env": env_identity(run_env), "model": model_identity(model),
                        "draft": model_identity(draft) if arm.draft else None, "sampling": sampling,
                        "chats": [[c, [[text_digest(t), n] for t, n in turns]] for c, turns in chats], "machine": machine()}
            try:
                results[arm.name, "chat"] = run.measure(label, identity,
                                                        lambda log: serve(exe, args, run_env, log, [], sampling, port, "", chats))
            except Fail as e:
                run.failed = True
                results[arm.name, "chat"] = f"FAIL {e}"
    run.add(f"decode: llama-server -c {ctx}, prompts {' '.join(n for n, _ in prompts)} x seeds "
            f"{' '.join(map(str, seeds))}, {ngen} tokens, sampling {temp} {top_k} {top_p}, {model.name}, "
            f"build {current_build()}")
    details = []
    for prefix in prefixes:
        rows = []
        for arm in arms:
            if arm.prefixes is not None and prefix not in arm.prefixes:
                continue
            entry = results.get((arm.name, prefix))
            if not isinstance(entry, tuple):
                rows.append([arm.name, entry or "skip"])
                continue
            r, kept = entry
            rs = r["rows"]
            n, ms = sum(x["n"] for x in rs), sum(x["ms"] for x in rs)
            dn, da = sum(x["draft_n"] for x in rs), sum(x["draft_acc"] for x in rs)
            # short prompts and long ones apart: a short one can take the generation layout (LLAMA_PHASE_SMALL_PROMPT)
            pp = []
            for part in ([x for x in rs if x["prompt_n"] <= 512], [x for x in rs if x["prompt_n"] > 512]):
                pn, pms = sum(x["prompt_n"] for x in part), sum(x["prompt_ms"] for x in part)
                pp.append(f"{rate(pn, pms):.1f} ({pn})" if part else "-")
            run.record("decode", arm, f"p{prefix}", decode_metrics(rs), kept, folders[arm.name])
            rows.append([arm.name, f"{rate(n, ms):.2f}", f"{da / dn:.3f} ({da}/{dn})" if dn else "-", *pp]
                        + (["k"] if kept else []))
            details += ["", f"{arm.name}, prefix {prefix} chars:"]
            details += [f"  {x['name']}: {x['n']} tokens, {rate(x['n'], x['ms']):.2f} t/s, draft {x['draft_acc']}/"
                        f"{x['draft_n']}, prompt {x['prompt_n']} tokens {rate(x['prompt_n'], x['prompt_ms']):.1f} t/s"
                        for x in rs]
            if r.get("graph_timing"):
                details.append(f"  {graph_timing_text(r['graph_timing'])}")
            grep = env("DECGREP")
            details += [f"  {s}" for s in r["spec"] + ([s for s in r["extra"] if re.search(grep, s)][-20:] if grep else [])]
        run.add("", f"prefix {prefix} chars of {Path(env('PPLFILE')).name}" if prefix else "short prompts",
                *table(["arm", "TG t/s", "draft accepted", "PP t/s <= 512 (tokens)", "PP t/s > 512 (tokens)"], rows))
        # the speed along the prompt, from the server's progress line at every batch (-b): one long prompt gives the
        # curve against the depth; the mean over the requests of that prefix, stretch by stretch
        curves = []
        for arm in arms:
            entry = results.get((arm.name, prefix))
            if not isinstance(entry, tuple):
                continue
            segs = [segments(x.get("pp_curve", [])) for x in entry[0]["rows"] if x["prompt_n"] > 8192]
            segs = [sg for sg in segs if sg]
            if not segs:
                continue
            n_seg = min(len(sg) for sg in segs)
            for i in range(n_seg):
                lo, hi = segs[0][i][0], segs[0][i][1]
                curves.append([arm.name, f"{lo // 1024}K-{hi // 1024}K", f"{statistics.mean(sg[i][2] for sg in segs):.1f}"])
            tg3 = [v for x in entry[0]["rows"] if x["prompt_n"] > 8192 for v in x.get("tg3", [])]
            if tg3:
                curves.append([arm.name, "TG (tg_3s)", f"{statistics.mean(tg3):.2f} ({min(tg3):.2f}-{max(tg3):.2f})"])
        if curves:
            run.add("", "PP along the prompt (the server's progress lines), then TG after it",
                    *table(["arm", "stretch", "t/s"], curves))
            run.add("  the server stamps a batch when its last ubatch is submitted: on Vulkan that ubatch still computes"
                    " and its time lands in the next stretch, so the first stretch reads high")
        # every prompt ubatch of the first long request of each arm (LLAMA_INPUT_TIMING): the curve at ubatch steps,
        # the host share (graph build, inputs) and the compute of each ubatch (ubatch_gpu); "outside" is the time of
        # the prompt no ubatch line of the model covers: the MTP head, the server between batches and, on Vulkan,
        # the compute of the last ubatch of each batch
        for arm in arms:
            entry = results.get((arm.name, prefix))
            if not isinstance(entry, tuple):
                continue
            long = [x for x in entry[0]["rows"] if x["prompt_n"] > 8192 and x.get("ubatches")]
            if not long:
                continue
            x, ub_rows, depth = long[0], [], 0
            gpu, lagged = ubatch_gpu(x["ubatches"])
            for (n, g, inp, sub, *_), comp in zip(x["ubatches"], gpu):
                ub_rows.append([f"{depth // 1024}K", n, f"{g:.0f}", f"{inp:.0f}", f"{sub:.0f}",
                                f"{comp:.0f}" if comp else "-", f"{n / comp * 1000:.1f}" if comp else "-"])
                depth += n
            ub_time = sum(sum(r[1:4]) for r in x["ubatches"]) / 1000
            pp_time = x["pp_curve"][-1][1] if x.get("pp_curve") else 0
            note = (" compute = the step to the next ubatch of the batch (the backend returns before the GPU is done),"
                    " the last of a batch computes outside" if lagged else "")
            run.add("", f"{arm.name}, {x['name']}: prompt ubatches (ms), {ub_time:.1f} s in them of {pp_time:.1f} s of prompt"
                    f" ({pp_time - ub_time:.1f} s outside);" + note,
                    *table(["depth", "tokens", "graph", "inputs", "submit", "compute", "t/s"], ub_rows))
        for pair in split_args(env("DECSAME")):
            a, _, b = pair.partition(":")
            ra, rb = results.get((a, prefix)), results.get((b, prefix))
            if not (isinstance(ra, tuple) and isinstance(rb, tuple)):
                run.add(f"same answers {pair}: no result for both arms")
                continue
            other = [x["name"] for x, y in zip(ra[0]["rows"], rb[0]["rows"]) if x["answer"] != y["answer"]]
            run.failed |= bool(other)
            run.add(f"same answers {pair}: " + ("all" if not other else "differ in " + " ".join(other)))
    if chats:
        # one row per turn: the depth is what the cache held plus the new part
        rows = []
        for arm in arms:
            entry = results.get((arm.name, "chat"))
            if not isinstance(entry, tuple):
                rows.append([arm.name, "", "", "", "", "", entry or "skip"])
                continue
            r, kept = entry
            for x in r.get("chat", []):
                rows.append([arm.name, x["conv"][1:], x["turn"], x["cache_n"] + x["prompt_n"], x["prompt_n"],
                             f"{rate(x['prompt_n'], x['prompt_ms']):.1f}", f"{rate(x['n'], x['ms']):.2f}",
                             f"{x['draft_acc']}/{x['draft_n']}" if x["draft_n"] else "-"] + (["k"] if kept else []))
        run.add("", f"chat: conversations (start:turns:chars) {' '.join(f'{a}:{b}:{c}' for a, b, c in chat_specs)} of "
                f"{Path(env('PPLFILE')).name}, {chat_gen} tokens per answer, prompt cache on",
                *table(["arm", "start-chars", "turn", "depth", "new", "PP t/s", "TG t/s", "draft"], rows))
    run.add(*details)
    return run.finish()


# ---------------------------------------------------------------- profile

def op_group(name):
    # the op without its shape; a fused group of GGML_CUDA_OP_TIMING keeps its ops ("FUSED SCALE+SILU")
    parts = re.split(r"[ (]", name)
    return " ".join(parts[:2]) if parts[0] == "FUSED" and len(parts) > 1 else parts[0]


def vulkan_blocks(text):
    """The GGML_VK_PERF_LOGGER blocks of a log part, one per graph compute: [[op, count, ms], ...]"""
    blocks, cur = [], None
    for line in text.splitlines():
        if line.startswith("Vulkan Timings:"):
            cur = []
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"(.+): (\d+) x [\d.e+-]+ us = ([\d.e+-]+) us", line)
        if m:
            cur.append([m[1], int(m[2]), float(m[3]) / 1000])
        elif line.startswith("Total time:"):
            cur = None
    return blocks


def cuda_report(text):
    """The first GGML_CUDA_OP_TIMING report of a log part: (compute calls, [[op, count, ms], ...]) or None"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"cuda_op_timing \S+: (\d+) compute calls", line)
        if not m:
            continue
        ops = []
        for x in lines[i + 1:]:
            r = re.match(r"cuda_op_timing \S+:\s+([\d.]+) ms\s+(\d+) x\s+[\d.]+ us\s+(.+)", x)
            if not r:
                break
            ops.append([r[3].strip(), int(r[2]), float(r[1])])
        return int(m[1]), ops
    return None


def block_kind(block):
    """("target" | "head", tokens) of a GGML_VK_PERF_LOGGER block: the target has an expert matmul per layer, the MTP
    head one layer; the tokens from the dst of its attention, None without one (the head's catch-up attends for no row)"""
    n_mmid = sum(c for op, c, _ in block if op.startswith("MUL_MAT_ID"))
    tokens = None
    for op, _, _ in block:
        m = re.match(r"FLASH_ATTN_\w+ dst\(\d+,\d+,(\d+),", op)
        if m:
            tokens = int(m[1])
            break
    return ("target" if n_mmid >= 8 else "head"), tokens


def sum_ops(blocks):
    total = {}
    for block in blocks:
        for op, count, ms in block:
            c, t = total.get(op, (0, 0.0))
            total[op] = (c + count, t + ms)
    return sorted([[op, c, t] for op, (c, t) in total.items()], key=lambda o: -o[2])


def cmd_profile():
    """Where the time goes, on the server: a conversation reaches each depth of PROFPREFIX (chars of PPLFILE) with the
    prompt cache on, then a part of PROFPART chars (pp: the prompt as a chat turn computes it at that depth) and a
    reply of PROFTG tokens (tg: one decode step, from the steps of that reply). Vulkan: GGML_VK_PERF_LOGGER, a block
    per graph compute; CUDA: GGML_CUDA_OP_TIMING=idle, a report when the GPU has been idle, so the report of a request
    comes at the start of the next one. llama-bench is not used (13.10: it has none of the server's state)"""
    run = Run("profile")
    model = model_path()
    vulkan = env("BACKEND") == "vulkan"
    arms = parse_arms("PROFARMS", "default")
    draft = model_path("SPECDRAFT") if env("SPECDRAFT") and any(a.draft for a in arms) else None
    prefixes = env_ints("PROFPREFIX", "0 500000")
    part, ntg = env_int("PROFPART", 16000), env_int("PROFTG", 16)
    ctx, port = env_int("DECCTX", 262144), env_int("DECPORT", 8097)
    corpus = Path(env("PPLFILE")).read_text(encoding="utf-8", errors="replace")
    run.add(f"profile: llama-server, {'GGML_VK_PERF_LOGGER per op' if vulkan else 'GGML_CUDA_OP_TIMING per op'}, a chat at "
            f"each depth of {' '.join(map(str, prefixes))} chars, then a part of {part} chars (pp) and {ntg} tokens (tg), "
            f"{model.name}, build {current_build()}")
    results = {}
    for arm in arms:
        try:
            if arm.draft and not draft:
                raise Skip("SPECDRAFT is not set, it is the draft model of @draft")
            exe = arm.binary("llama-server")
        except Skip as e:
            run.add("", f"{arm.name}: skip: {e}")
            continue
        # @d= limits the arm to these depths of PROFPREFIX
        arm_prefixes = [p for p in prefixes if not arm.depths or p in arm.depths]
        if not arm_prefixes:
            run.add("", f"{arm.name}: skip: none of its depths {sorted(arm.depths)} is in PROFPREFIX")
            continue
        args = [str(a) for a in ["-m", model, *(["-md", draft] if arm.draft else []), "-c", ctx, "-np", 1, "-fa", "on",
                "--repeat-penalty", "1.0", *split_args(env("LOADMODE")), *split_args(env("EXTRA")), *warmup_args(),
                *split_args(env("DECARGS")), *arm.args, "--host", "127.0.0.1", "--port", port, "-lv", 4]]
        run_env = arm_env(arm, {"LLAMA_INPUT_TIMING": "1",
                                **({"GGML_VK_PERF_LOGGER": "1"} if vulkan else {"GGML_CUDA_OP_TIMING": "idle"})})
        identity = {"v": VERSION, "kind": "profile-server", "build": build_id(exe.parent), "args": args,
                    "env": env_identity(run_env), "model": model_identity(model), "prefixes": arm_prefixes, "part": part,
                    "tg": ntg, "corpus": text_digest(corpus[:max(arm_prefixes) + part]), "machine": machine(),
                    "draft": model_identity(draft) if arm.draft else None}

        def go(log):
            proc, log = start_server(exe, args, run_env, log)
            url = f"http://127.0.0.1:{port}"
            slices = {}
            try:
                def ask(name, messages, n):
                    before = log.stat().st_size
                    body = {"messages": messages, "max_tokens": n, "temperature": 0.0, "seed": 1,
                            "cache_prompt": True, "stream": False}
                    try:
                        _, raw = http(url + "/v1/chat/completions", body, env_int("DECTIMEOUT", 1800))
                    except (urllib.error.URLError, OSError) as e:
                        raise Fail(f"{name}: {e} ({log.name})")
                    time.sleep(1.5)  # the slot prints its timings as it is released, and CUDA needs the idle gap
                    with log.open("rb") as f:
                        f.seek(before)
                        slices[name] = f.read().decode("utf-8", errors="replace")
                    return json.loads(raw)["choices"][0]["message"].get("content") or ""
                for prefix in arm_prefixes:
                    msgs = [{"role": "user", "content": (corpus[:prefix] + "\n\n" if prefix else "") + "What is this text about? One word."}]
                    msgs.append({"role": "assistant", "content": ask(f"fill{prefix}", msgs, 1)})
                    msgs.append({"role": "user", "content": corpus[prefix:prefix + part] + "\n\nAnd this part? One word."})
                    msgs.append({"role": "assistant", "content": ask(f"pp{prefix}", msgs, 1)})
                    msgs.append({"role": "user", "content": "Write a long paragraph about it."})
                    ask(f"tg{prefix}", msgs, ntg)
                    ask(f"end{prefix}", [{"role": "user", "content": "ok"}], 1)
            finally:
                proc.kill()
                proc.wait()
                mark_end()
            out = {}
            for prefix in arm_prefixes:
                pp_text, tg_text, end_text = slices[f"pp{prefix}"], slices[f"tg{prefix}"], slices[f"end{prefix}"]
                host = [[float(m[2]), float(m[3]), float(m[4])] for m in
                        (re.match(r".*ubatch timing: (\d+) tokens, graph ([\d.]+) ms, inputs ([\d.]+) ms, submit ([\d.]+) ms", x)
                         for x in pp_text.splitlines()) if m]
                if vulkan and arm.draft:
                    # with the MTP head the reply interleaves the target's verify graphs (2-4 tokens) and the head's
                    # graphs (its draft steps and the catch-up of the accepted rows): apart by the expert matmuls
                    pp_blocks = [b for b in vulkan_blocks(pp_text) if block_kind(b)[0] == "target"]
                    groups = {}
                    for b in vulkan_blocks(tg_text):
                        kind, n = block_kind(b)
                        label = (f"verify of {n} tokens" if kind == "target" else
                                 "head draft step" if n else "head catch-up of accepted rows")
                        if kind == "target" and (n is None or n > 8):
                            continue  # the prompt of the reply
                        groups.setdefault(label, []).append(b)
                    tg_groups = [[label, len(bl), [[o, c / len(bl), t / len(bl)] for o, c, t in sum_ops(bl)]]
                                 for label, bl in sorted(groups.items(), key=lambda g: -len(g[1]))]
                    verify = [g for g in tg_groups if g[0].startswith("verify")]
                    out[prefix] = {"pp": sum_ops(pp_blocks), "pp_graphs": len(pp_blocks),
                                   "tg": verify[0][2] if verify else [], "tg_groups": tg_groups,
                                   "host": [sum(h[i] for h in host) for i in range(3)] if host else None}
                elif vulkan:
                    pp_blocks, tg_blocks = vulkan_blocks(pp_text), vulkan_blocks(tg_text)
                    steps = tg_blocks[-max(1, ntg - 2):]
                    out[prefix] = {"pp": sum_ops(pp_blocks), "pp_graphs": len(pp_blocks),
                                   "tg": [[o, c / len(steps), t / len(steps)] for o, c, t in sum_ops(steps)] if steps else [],
                                   "host": [sum(h[i] for h in host) for i in range(3)] if host else None}
                else:
                    # the report of the prompt ubatch comes at the next compute call after an idle gap: inside the same
                    # request when its tail (the last tokens, kept apart for a checkpoint) follows the gap, else at the
                    # start of the next request. the first report after the last large ubatch line
                    big = [m.end() for m in re.finditer(r"ubatch timing: (\d+) tokens", pp_text) if int(m[1]) > 1024]
                    pp_r = cuda_report((pp_text + tg_text)[big[-1]:] if big else tg_text)
                    tg_r = cuda_report(end_text)
                    out[prefix] = {"pp": pp_r[1] if pp_r else [], "pp_graphs": pp_r[0] if pp_r else 0,
                                   "tg": [[o, c / ntg, t / ntg] for o, c, t in tg_r[1]] if tg_r else [],
                                   "host": [sum(h[i] for h in host) for i in range(3)] if host else None}
            return out

        try:
            r, kept = run.measure(f"{arm.name}", identity, go)
        except Fail as e:
            run.failed = True
            run.add("", f"{arm.name}: FAIL {e}")
            continue
        results[arm.name] = r
        for prefix in arm_prefixes:
            x = r.get(prefix) or r.get(str(prefix))
            if not x:
                continue
            sections = [("pp", f"prompt part of {part} chars ({x['pp_graphs']} {'graphs' if vulkan else 'compute calls'})", x["pp"])]
            if x.get("tg_groups"):
                sections += [("tg", f"{label} (mean of {n})", ops) for label, n, ops in x["tg_groups"]]
            else:
                sections.append(("tg", "decode step (mean)", x["tg"]))
            for kind, what, ops in sections:
                total = sum(o[2] for o in ops)
                head = [f"", f"{kind}-p{prefix}-{arm.name}: {what} at {prefix} chars, {total:.2f} ms of gpu time" + (" k" if kept else "")]
                if kind == "pp" and x.get("host"):
                    h = x["host"]
                    head.append(f"  host: graph build {h[0]:.2f} ms, inputs {h[1]:.2f} ms, submit {h[2]:.2f} ms")
                run.add(*head, *table(["  ms", "count", "op"], [[f"  {ms:.2f}", f"{c:g}", name] for name, c, ms in ops[:25]]))
            run.record("profile", arm, f"p{prefix}", {"gpu_pp": sum(o[2] for o in x["pp"]),
                                                      "gpu_tg": sum(o[2] for o in x["tg"])}, kept, exe.parent)
    if len(results) > 1:
        for prefix in prefixes:
            for kind in ("pp", "tg"):
                groups = {}
                for name, r in results.items():
                    x = r.get(prefix) or r.get(str(prefix)) or {}
                    for op, _, ms in x.get(kind, []):
                        groups.setdefault(op_group(op), {}).setdefault(name, 0.0)
                        groups[op_group(op)][name] += ms
                top = sorted(groups, key=lambda g: -max(groups[g].values()))[:20]
                rows = [[g, *[f"{groups[g].get(n, 0):.2f}" for n in results]] for g in top]
                rows.append(["total", *[f"{sum(groups[g].get(n, 0) for g in groups):.2f}" for n in results]])
                run.add("", f"{kind} at {prefix} chars: gpu ms per kind of op", *table(["op", *results], rows))
    return run.finish()


# ---------------------------------------------------------------- trace

def nsys_path():
    """nsys of Nsight Systems: NSYS, else nsys on PATH, else the newest one under Program Files (the CUDA toolkit
    installs one)"""
    given = env("NSYS")
    if given:
        if not Path(given).is_file():
            raise Fail(f"NSYS={given} is not a file")
        return Path(given)
    found = shutil.which("nsys")
    if found:
        return Path(found)
    root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation"
    found = sorted(root.glob("Nsight Systems */target-windows-x64/nsys.exe"), key=lambda q: q.parent.parent.name) if root.is_dir() else []
    if not found:
        raise Skip("no nsys: install Nsight Systems (the CUDA toolkit has it) or set NSYS to nsys.exe")
    return found[-1]


def merge(intervals, gap=0):
    """The union of [start, end] intervals, joining the ones less than gap apart"""
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(union, a=None, b=None):
    """The length of a merged union, within [a, b] when given"""
    total = 0
    for x, y in union:
        if a is not None:
            x, y = max(x, a), min(y, b)
        if y > x:
            total += y - x
    return total


def overlap(u, v):
    """The length of the intersection of two merged unions"""
    i = j = total = 0
    while i < len(u) and j < len(v):
        a, b = max(u[i][0], v[j][0]), min(u[i][1], v[j][1])
        if b > a:
            total += b - a
        if u[i][1] < v[j][1]:
            i += 1
        else:
            j += 1
    return total


def clip(union, starts, a, b):
    """The part of a merged union inside [a, b]; starts are the starts of its intervals"""
    i = max(0, bisect.bisect_right(starts, a) - 1)
    j = bisect.bisect_right(starts, b)
    return [[max(x, a), min(y, b)] for x, y in union[i:j] if y > a and x < b]


def trace_summary(db, gap_ms):
    """What the GPU did in the trace (the SQLite export of nsys): the kernels, the copies host to device (H2D) and
    back (D2H), and the time it did nothing, in the whole window and in each stretch of activity that a gap of
    gap_ms or more of a fully idle GPU separates (the host work between prompt ubatches); a stretch that copies a GiB
    or more to the device is a prompt ubatch with the experts uploaded, the others are decode steps and small batches.
    In a decode stretch a D2H copy is about one step (the logits), the gaps are the GPU waiting for the host: for the
    CPU experts, the graph launch, the sampling"""
    import sqlite3
    con = sqlite3.connect(str(db))
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise Fail(f"no kernels in {db.name}: the profiler range caught nothing (LLAMA_TRACE_REQUESTS, a CUDA build?)")
    names = dict(con.execute("select id, value from StringIds")) if "StringIds" in tables else {}
    kern = sorted((a, b, names.get(n, str(n)), st) for a, b, n, st in
                  con.execute("select start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL"))
    # --cuda-graph-trace=graph: a launched CUDA graph is one activity in a table of its own, not kernels (the decode
    # graphs looked idle without it, 14.10); any table of graph activities with start, end and a stream counts as kernels
    for table_name in sorted(t for t in tables if "GRAPH" in t.upper()):
        cols = {r[1] for r in con.execute(f"pragma table_info({table_name})")}
        if {"start", "end", "streamId"} <= cols:
            kern += [(a, b, "cuda graph", st) for a, b, st in con.execute(f"select start, end, streamId from {table_name}")]
    kern.sort()
    mem = sorted(con.execute("select start, end, bytes, copyKind, streamId from CUPTI_ACTIVITY_KIND_MEMCPY")) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    mset = list(con.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET")) if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else []
    con.close()
    # copyKind of CUPTI: 1 host to device, 2 device to host, 8 device to device
    h2d = [m for m in mem if m[3] == 1]
    d2h = [m for m in mem if m[3] == 2]
    u_k = merge([(a, b) for a, b, _, _ in kern])
    u_h = merge([(a, b) for a, b, _, _, _ in h2d])
    u_d = merge([(a, b) for a, b, _, _, _ in d2h])
    u_all = merge([(a, b) for a, b, _, _ in kern] + [(m[0], m[1]) for m in mem] + mset)
    s_k, s_h, s_all = [x[0] for x in u_k], [x[0] for x in u_h], [x[0] for x in u_all]
    h_starts, d_starts, k_starts = [m[0] for m in h2d], [m[0] for m in d2h], [k[0] for k in kern]
    h_bytes_sum = [0]
    for m in h2d:
        h_bytes_sum.append(h_bytes_sum[-1] + m[2])
    t0, t1 = u_all[0][0], u_all[-1][1]
    ms = 1e-6
    out = {"window_ms": (t1 - t0) * ms, "kernel_ms": covered(u_k) * ms, "h2d_ms": covered(u_h) * ms,
           "h2d_gib": h_bytes_sum[-1] / 2**30, "d2h_ms": covered(u_d) * ms, "both_ms": overlap(u_k, u_h) * ms,
           "idle_ms": (t1 - t0 - covered(u_all)) * ms, "n_kernels": len(kern),
           "kernel_streams": len({k[3] for k in kern}), "h2d_streams": len({m[4] for m in h2d})}
    rows = []
    for a, b in merge([tuple(x) for x in u_all], gap=gap_ms * 1e6):
        ck, ch, call = clip(u_k, s_k, a, b), clip(u_h, s_h, a, b), clip(u_all, s_all, a, b)
        hi, hj = bisect.bisect_left(h_starts, a), bisect.bisect_right(h_starts, b)
        ki, kj = bisect.bisect_left(k_starts, a), bisect.bisect_right(k_starts, b)
        tops = {}
        for k in kern[ki:kj]:
            tops[k[2]] = tops.get(k[2], 0) + k[1] - k[0]
        # the idle time inside the stretch by the length of the gap: launches (< 0.1 ms), short waits, the host
        gaps = [0.0, 0.0, 0.0]
        for x, y in zip(call, call[1:]):
            d = (y[0] - x[1]) * ms
            gaps[0 if d < 0.1 else 1 if d < 1 else 2] += d
        h_bytes = h_bytes_sum[hj] - h_bytes_sum[hi]
        rows.append({"start_s": (a - t0) * 1e-9, "ms": (b - a) * ms, "kernel_ms": covered(ck) * ms,
                     "h2d_ms": covered(ch) * ms, "h2d_gib": h_bytes / 2**30, "both_ms": overlap(ck, ch) * ms,
                     "idle_ms": ((b - a) - covered(call)) * ms, "gaps_ms": gaps, "n_kernels": kj - ki,
                     "n_d2h": bisect.bisect_right(d_starts, b) - bisect.bisect_left(d_starts, a),
                     "prompt": h_bytes >= 2**30,
                     "top": [[n, t * ms] for n, t in sorted(tops.items(), key=lambda kv: -kv[1])[:3]]})
    out["stretches"] = rows
    total = {}
    for a, b, n, _ in kern:
        t = total.setdefault(n, [0, 0])
        t[0] += 1
        t[1] += b - a
    out["top"] = [[n, c, t * ms] for n, (c, t) in sorted(total.items(), key=lambda kv: -kv[1][1])[:20]]
    return out


def timeline_summary(path, gap_ms):
    """The timeline of GGML_TIMELINE (Vulkan: the gpu span of each command buffer; llama: graph, inputs and submit of each
    ubatch, "llama mtp" for the MTP head; host: the syncs; server: the requests, each decode call of the target and the
    speculative process after it): how long the GPU ran and stood still in
    the requests, the stretches of gpu work apart by gap_ms or more of idle, and the idle stretches between them, each
    with the host time of every phase that started in it. A stretch with a ubatch of 1025 tokens or more is a prompt"""
    text = path.read_text(encoding="utf-8").rstrip().rstrip(",")
    events = json.loads(text if text.endswith("]") else text + "]")
    tracks = {e["tid"]: e["args"]["name"] for e in events if e.get("ph") == "M"}
    spans = [(tracks.get(e["tid"], "?"), e["name"], e["ts"], e["ts"] + e["dur"]) for e in events if e.get("ph") == "X"]
    gpu = merge([(a, b) for t, n, a, b in spans if t.startswith("gpu")])
    if not gpu:
        raise Fail(f"no gpu spans in {path.name} (a Vulkan build, LLAMA_TRACE_REQUESTS?)")
    requests = [(a, b) for t, n, a, b in spans if t == "server"]
    w0 = min(a for a, _ in requests) if requests else gpu[0][0]
    w1 = max(b for _, b in requests) if requests else gpu[-1][1]
    starts = [x[0] for x in gpu]
    # the server: the decode call of the target and the speculative process after it (the MTP catch-up), the requests
    # are the window
    host = [(t, "spec" if n.startswith("spec") else n.split()[0], int(n.split()[-1]) if n.split()[-1].isdigit() else 0, a, b)
            for t, n, a, b in spans if t in ("llama", "llama mtp", "host", "sched wait") or (t == "server" and not n.startswith("request"))]
    # the CPU splits of the scheduler: the experts in RAM of a decode step (-ncmoe)
    cpu = merge([(a, b) for t, n, a, b in spans if t == "sched cpu"])
    c_starts = [x[0] for x in cpu]
    host.sort(key=lambda h: h[3])
    h_starts = [h[3] for h in host]
    longest = max((h[4] - h[3] for h in host), default=0)

    def phases(a, b):
        """the host time of each phase inside [a, b], the part of a span that overlaps it"""
        out, i = {}, bisect.bisect_left(h_starts, a - longest)
        while i < len(host) and host[i][3] < b:
            t, phase, n, x, y = host[i]
            d = min(y, b) - max(x, a)
            if d > 0:
                key = ("mtp " if t == "llama mtp" else "") + phase
                out[key] = out.get(key, 0.0) + d / 1000
            i += 1
        return out

    def ubatches(a, b):
        i, tokens, n = bisect.bisect_left(h_starts, a), 0, 0
        while i < len(host) and host[i][3] < b:
            if host[i][0] == "llama" and host[i][1] == "submit":
                n += 1
                tokens = max(tokens, host[i][2])
            i += 1
        return n, tokens

    rows = []
    prev_end = None
    for a, b in merge([tuple(x) for x in gpu], gap=gap_ms * 1000):
        if prev_end is not None and a > prev_end:
            c_ms = covered(clip(cpu, c_starts, prev_end, a))
            rows.append({"kind": "idle", "start_s": (prev_end - w0) / 1e6, "ms": (a - prev_end) / 1000, "gpu_ms": 0.0,
                         "idle_ms": (a - prev_end) / 1000, "gaps_ms": [0.0, 0.0, 0.0], "ubatches": 0, "tokens": 0,
                         "phases": phases(prev_end, a), "cpu_ms": c_ms / 1000, "cpu_only_ms": c_ms / 1000})
        part = clip(gpu, starts, a, b)
        gaps = [0.0, 0.0, 0.0]
        for x, y in zip(part, part[1:]):
            d = (y[0] - x[1]) / 1000
            gaps[0 if d < 0.1 else 1 if d < 1 else 2] += d
        n_ub, tokens = ubatches(a, b)
        busy = covered(part)
        c_part = clip(cpu, c_starts, a, b)
        c_ms = covered(c_part)
        rows.append({"kind": "prompt" if tokens >= 1025 else "decode", "start_s": (a - w0) / 1e6, "ms": (b - a) / 1000,
                     "gpu_ms": busy / 1000, "idle_ms": ((b - a) - busy) / 1000, "gaps_ms": gaps, "ubatches": n_ub,
                     "tokens": tokens, "phases": phases(a, b), "cpu_ms": c_ms / 1000,
                     "cpu_only_ms": (c_ms - overlap(c_part, part)) / 1000})
        prev_end = b
    def layer_of(n):
        return int(n.rsplit("-", 1)[1]) if "-" in n and n.rsplit("-", 1)[1].isdigit() else None

    # each CPU split by the layer in the name of its first node ("ffn_moe_gate_up-12"): its time and the part of it with
    # the gpu idle, the wait for the experts of that layer; then the expert matmuls of the layer (MUL_MAT_ID nodes)
    layers = {}
    for t, n, a, b in spans:
        if t != "sched cpu" or layer_of(n) is None:
            continue
        g = clip(gpu, starts, a, b)
        x = layers.setdefault(layer_of(n), [0, 0.0, 0.0, 0.0, 0])
        x[0] += 1
        x[1] += (b - a) / 1000
        x[2] += ((b - a) - covered(g)) / 1000
    # the nodes of the CPU graphs (ggml-cpu/fork/timeline-fork.inc): thread 0 marks each node after the barrier that
    # waits for the slowest thread, with the weight bytes of a matmul (the experts a MUL_MAT_ID computes, once each);
    # "cpu wake": from the compute call to the start of the last thread
    nodes, wake = {}, [0, 0.0]
    for e in events:
        if e.get("ph") != "X":
            continue
        t = tracks.get(e["tid"])
        if t == "cpu nodes":
            a = e.get("args", {})
            il = layer_of(e["name"])
            key = (a.get("op", "?"), e["name"].rsplit("-", 1)[0] if il is not None else e["name"])
            x = nodes.setdefault(key, [0, 0.0, 0])
            x[0] += 1
            x[1] += e["dur"] / 1000
            x[2] += a.get("bytes", 0)
            if a.get("op") == "MUL_MAT_ID" and il in layers:
                layers[il][3] += e["dur"] / 1000
                layers[il][4] += a.get("bytes", 0)
        elif t == "cpu wake":
            wake[0] += 1
            wake[1] += e["dur"] / 1000
    total = phases(w0, w1)
    g_all = clip(gpu, starts, w0, w1)
    c_all = clip(cpu, c_starts, w0, w1)
    busy = covered(g_all)
    return {"window_ms": (w1 - w0) / 1000, "gpu_ms": busy / 1000, "idle_ms": ((w1 - w0) - busy) / 1000,
            "cpu_ms": covered(c_all) / 1000, "cpu_only_ms": (covered(c_all) - overlap(c_all, g_all)) / 1000,
            "n_cpu_splits": sum(1 for t, *_ in spans if t == "sched cpu"),
            "n_gpu_spans": sum(1 for t, *_ in spans if t.startswith("gpu")), "phases": total, "stretches": rows,
            "layers": [[k, *layers[k]] for k in sorted(layers)],
            "nodes": sorted([[op, n, *x] for (op, n), x in nodes.items()], key=lambda r: -r[3]), "wake": wake}


def timeline_lines(name, r, kept=False):
    """The summary of one timeline (timeline_summary)"""
    w = r["window_ms"]
    order = ["decode", "spec", "graph", "inputs", "submit", "sync", "wait", "mtp graph", "mtp inputs", "mtp submit"]
    ph = r["phases"]
    cpu = r.get("n_cpu_splits", 0) > 0
    lines = ["", f"{name}: {r.get('report', '')} (Perfetto, chrome://tracing)" + (" k" if kept else ""),
             f"  requests {w / 1000:.2f} s: gpu {r['gpu_ms'] / 1000:.2f} s ({100 * r['gpu_ms'] / w:.0f}%), idle "
             f"{r['idle_ms'] / 1000:.2f} s; {r['n_gpu_spans']} gpu spans" +
             (f"; cpu splits {r['cpu_ms'] / 1000:.2f} s ({r['n_cpu_splits']}), of it with the gpu idle "
              f"{r['cpu_only_ms'] / 1000:.2f} s" if cpu else "") + "; host: " +
             ", ".join(f"{k} {ph[k] / 1000:.2f} s" for k in order if k in ph)]
    rows = []
    for x in r["stretches"]:
        if x["ms"] < 1:
            continue
        p = x["phases"]
        rows.append([f"  {x['start_s']:.2f}", x["kind"], f"{x['ms']:.0f}", f"{x['gpu_ms']:.0f}", f"{x['idle_ms']:.0f}",
                     *([f"{x.get('cpu_ms', 0):.0f}", f"{x.get('cpu_only_ms', 0):.0f}"] if cpu else []),
                     "/".join(f"{g:.0f}" for g in x["gaps_ms"]), x["ubatches"] or "", x["tokens"] or "",
                     *[f"{p[k]:.0f}" if k in p else "" for k in order]])
    lines += table(["  s", "kind", "ms", "gpu", "idle", *(["cpu", "cpu only"] if cpu else []), "gaps", "ub", "tokens", *order], rows)
    if r.get("nodes"):
        n_ms = sum(x[3] for x in r["nodes"])
        wk = r.get("wake", [0, 0.0])
        lines += [f"  CPU graph nodes (thread 0, each up to the barrier after it): {n_ms:.0f} ms in nodes, wake {wk[1]:.0f} ms "
                  f"in {wk[0]} graphs ({1000 * wk[1] / max(wk[0], 1):.0f} us each, the compute call to the start of the last "
                  f"thread), the rest of the CPU splits {r['cpu_ms'] - n_ms - wk[1]:.0f} ms; GB/s of the weights read"]
        lines += table(["    op", "node", "calls", "ms", "us/call", "MB/call", "GB/s"],
                       [[f"    {op}", n, c, f"{ms:.0f}", f"{1000 * ms / c:.0f}", f"{b / c / 1e6:.2f}" if b else "",
                         f"{b / (ms * 1e6):.1f}" if b and ms > 0 else ""] for op, n, c, ms, b in r["nodes"] if ms >= 1])
    if r.get("layers"):
        mm = any(len(x) > 4 and x[5] for x in r["layers"])
        lines += ["  CPU splits by layer (the experts in RAM): calls, ms, of it with the gpu idle, us per call" +
                  ("; its MUL_MAT_ID nodes: MB of experts per split, GB/s" if mm else "")]
        lines += table(["    layer", "calls", "ms", "gpu idle", "us/call", "idle us/call", *(["mm MB", "mm GB/s"] if mm else [])],
                       [[f"    {x[0]}", x[1], f"{x[2]:.0f}", f"{x[3]:.0f}", f"{1000 * x[2] / x[1]:.0f}", f"{1000 * x[3] / x[1]:.0f}",
                         *([f"{x[5] / x[1] / 1e6:.1f}", f"{x[5] / (x[4] * 1e6):.1f}" if x[4] > 0 else ""] if mm else [])]
                        for x in r["layers"]])
    if r.get("log_lines"):
        lines += ["  from the log:", *[f"    {x}" for x in r["log_lines"]]]
    return lines


def trace_lines(name, r, kept=False):
    """The summary of one trace (trace_summary)"""
    w = r["window_ms"]
    lines = ["", f"{name}: {r.get('report', '')} (Nsight Systems)" + (" k" if kept else ""),
             f"  window {w / 1000:.2f} s: kernels {r['kernel_ms'] / 1000:.2f} s ({100 * r['kernel_ms'] / w:.0f}%), "
             f"H2D {r['h2d_ms'] / 1000:.2f} s {r['h2d_gib']:.1f} GiB, D2H {r['d2h_ms']:.0f} ms, kernels and H2D at once "
             f"{r['both_ms'] / 1000:.2f} s, GPU idle {r['idle_ms'] / 1000:.2f} s; {r['n_kernels']} kernels on "
             f"{r['kernel_streams']} streams, H2D on {r['h2d_streams']}"]
    rows = []
    for x in r["stretches"]:
        if x["ms"] < 1:
            continue
        rows.append([f"  {x['start_s']:.2f}", "prompt" if x["prompt"] else "decode", f"{x['ms']:.0f}", f"{x['kernel_ms']:.0f}",
                     f"{x['h2d_ms']:.0f}", f"{x['h2d_gib']:.1f}", f"{x['both_ms']:.0f}", f"{x['idle_ms']:.0f}",
                     "/".join(f"{g:.0f}" for g in x["gaps_ms"]), x["n_kernels"], x["n_d2h"],
                     f"{x['top'][0][1]:.0f} {x['top'][0][0][:48]}" if x["top"] else ""])
    lines += table(["  s", "kind", "ms", "kernels", "H2D", "GiB", "both", "idle", "gaps", "n_kern", "d2h", "top kernel (ms)"], rows)
    lines += ["  kernels by time:", *table(["    ms", "count", "mean us", "kernel"],
                                          [[f"    {t:.1f}", c, f"{1000 * t / c:.1f}", n[:90]] for n, c, t in r["top"]])]
    return lines


def trace_conversation(url, turns, gen, log):
    """The warmup (request 1, not traced) and the turns of the conversation with the prompt cache on"""
    messages = []
    for i, text in enumerate([WARMUP, *turns]):
        msgs = [{"role": "user", "content": text}] if i == 0 else messages + [{"role": "user", "content": text}]
        body = {"messages": msgs, "max_tokens": 16 if i == 0 else gen, "temperature": 0.0, "seed": 1,
                "cache_prompt": i > 0, "stream": False}
        try:
            _, raw = http(url + "/v1/chat/completions", body, env_int("DECTIMEOUT", 1800))
        except (urllib.error.URLError, OSError) as e:
            raise Fail(f"request {i}: {e} ({log.name})")
        if i > 0:
            messages = msgs + [{"role": "assistant", "content": json.loads(raw)["choices"][0]["message"].get("content") or ""}]


def cmd_trace():
    """A timeline of one conversation on the server, the requests of the conversation only, after the warmup
    (LLAMA_TRACE_REQUESTS, tools/server/fork/server-trace-fork.cpp). CUDA: Nsight Systems, nsys profile
    --capture-range=cudaProfilerApi: where the GPU runs kernels, copies (the experts of -ncmoe uploaded for a prompt
    ubatch) or waits (for the CPU experts of a decode step, the host); the report stays next to the log for the Nsight
    Systems window, its SQLite export is read into the summary and removed (TRACEKEEP=1 keeps it). Vulkan: GGML_TIMELINE,
    the gpu span of each command buffer next to the host phases of llama (graph, inputs, submit, sync), for Perfetto or
    chrome://tracing, read into the summary. TRACECHAT: the conversation, start:turns:chars of PPLFILE as in DECCHAT;
    TRACEGEN: tokens per answer; TRACEARMS: the arms (on CUDA an arm with TRACE_TIMELINE=1 takes the timeline: the gpu
    span of each compute call, the CPU splits and the waits of the scheduler); TRACEGAP: the idle gap in ms that ends a
    stretch (20)"""
    run = Run("trace")
    backend = env("BACKEND")
    if backend not in ("cuda", "vulkan"):
        run.add(f"trace: no timeline for {backend}")
        return run.finish()
    cuda = backend == "cuda"
    nsys = nsys_path() if cuda else None
    model = model_path()
    arms = parse_arms("TRACEARMS", "default")
    draft = model_path("SPECDRAFT") if env("SPECDRAFT") and any(a.draft for a in arms) else None
    ctx, port, gen = env_int("DECCTX", 262144), env_int("DECPORT", 8097), env_int("TRACEGEN", 64)
    gap = float(env("TRACEGAP", "20"))
    corpus = Path(env("PPLFILE")).read_text(encoding="utf-8", errors="replace")

    def chat_turns(spec):
        """TRACECHAT start:turns:chars: a prompt of start chars of PPLFILE, then turns of the next chars"""
        parts = [int(x) for x in spec.split(":")]
        start, n_turns, add_chars = parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 2000
        out = [(corpus[:start] + "\n\n" if start else "") + "Summarize the text above in two sentences."]
        out += [corpus[start + i*add_chars:start + (i + 1)*add_chars] + "\n\nAnd this part, in two sentences?" for i in range(n_turns)]
        return out, (start, n_turns, add_chars)

    turns, (start, n_turns, add_chars) = chat_turns(env("TRACECHAT", "120000:1:2000"))
    what = (f"Nsight Systems ({nsys.parent.parent.name}), cuda trace with whole CUDA graphs (an arm with TRACE_TIMELINE=1: "
            f"GGML_TIMELINE)" if cuda else
            "GGML_TIMELINE, the gpu span of each Vulkan command buffer and the host phases of llama")
    run.add(f"trace: {what}, a conversation of {start} chars and {n_turns} turns of {add_chars}, {gen} tokens per answer, "
            f"{model.name}, build {current_build()}",
            f"stretches: gpu work apart by {gap:g} ms of idle; " +
            ("prompt = a GiB or more uploaded" if cuda else "prompt = a ubatch of 1025 tokens or more; idle = the gaps "
             "between stretches with the host phases in them") +
            "; gaps inside a stretch under 0.1 ms / 0.1-1 ms / over 1 ms (the GPU waits for the host)")
    base_turns = turns
    for arm in arms:
        try:
            if arm.draft and not draft:
                raise Skip("SPECDRAFT is not set, it is the draft model of @draft")
            exe = arm.binary("llama-server")
        except Skip as e:
            run.add("", f"{arm.name}: skip: {e}")
            continue
        # an arm can take its own conversation: TRACECHAT=start:turns:chars among its variables (a turn at depth)
        turns = base_turns
        if arm.env.get("TRACECHAT"):
            turns, spec = chat_turns(arm.env["TRACECHAT"])
            run.add("", f"{arm.name}: a conversation of {spec[0]} chars and {spec[1]} turns of {spec[2]}")
        args = [str(a) for a in ["-m", model, *(["-md", draft] if arm.draft else []), "-c", ctx, "-np", 1, "-fa", "on",
                "--repeat-penalty", "1.0", *split_args(env("LOADMODE")), *split_args(env("EXTRA")), *warmup_args(),
                *split_args(env("DECARGS")), *arm.args, "--host", "127.0.0.1", "--port", port, "-lv", 4]]
        run_env = arm_env(arm, {"LLAMA_TRACE_REQUESTS": f"2:{1 + len(turns)}", "LLAMA_INPUT_TIMING": "1"})
        # CUDA: an arm with TRACE_TIMELINE=1 takes the timeline too (the gpu span of each compute call next to the CPU
        # splits of the scheduler and its waits: where a decode step waits for the experts in RAM). the tool is part of
        # the identity: TRACE_TIMELINE is no GGML_/LLAMA_ variable, and the tl arm took the nsys result of auto (14.10)
        nsys_arm = cuda and arm.env.get("TRACE_TIMELINE") != "1"
        identity = {"v": VERSION, "kind": "trace", "build": build_id(exe.parent), "args": args,
                    "env": env_identity(run_env), "model": model_identity(model),
                    "draft": model_identity(draft) if arm.draft else None, "turns": [text_digest(t) for t in turns],
                    "gen": gen, "gap": gap, "tool": nsys.parent.parent.name if nsys_arm else "timeline",
                    "analysis": 3, "machine": machine()}  # 2: CUDA graphs and the CPU splits counted (14.10), 3: CPU nodes
        base = run.dir / arm.name
        url = f"http://127.0.0.1:{port}"

        def go_nsys(log):
            # capture-range-end stop: the collection ends with the last traced request and the server answers it
            # (stop-shutdown ended the server before the answer of the last request went out, 14.10). nsys writes the
            # report when the server exits; the output of the server goes to its own log, nsys does not pass it on
            server_log = base.with_name(f"{arm.name}-server.log")
            # whole CUDA graphs: the nodes of a graph traced one by one slowed a decode step by half (14.10)
            cmd = ["profile", "--trace=cuda,nvtx", "--cuda-graph-trace=graph", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
                   "--output", str(base), str(exe), *args, "--log-file", str(server_log)]
            proc, log = start_server(nsys, cmd, run_env, log)
            try:
                trace_conversation(url, turns, gen, log)
            finally:
                # the server only, by name (no other llama-server runs, see no_server_running): nsys then writes the
                # report and exits. killing nsys too would lose the report
                if os.name == "nt":
                    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True)
                else:
                    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)
                wait = env_int("TRACEWAIT", 1800)
                try:
                    proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/f", "/t", "/pid", str(proc.pid)], capture_output=True)
                    else:
                        proc.kill()
                    proc.wait()
                    raise Fail(f"nsys did not end in {wait} s after the server ({log.name})")
                finally:
                    wait_no_server()
                    mark_end()
            report = base.with_suffix(".nsys-rep")
            if not report.is_file():
                raise Fail(f"nsys wrote no {report.name}: {last_error(read_log(log))} ({log.name})")
            db = base.with_suffix(".sqlite")
            execute([nsys, "export", "--type=sqlite", "--force-overwrite=true", "--output", db, report],
                    log.with_name(f"{arm.name}-export.log"), os.environ.copy(), model=False)
            try:
                result = trace_summary(db, gap)
            finally:
                if env("TRACEKEEP") != "1":
                    db.unlink(missing_ok=True)
            result["report"] = report.name
            return result

        def go_timeline(log):
            # the file is not part of the identity: it is in the folder of the run
            path = base.with_suffix(".trace.json")
            wait_no_server()
            proc, log = start_server(exe, args, dict(run_env, GGML_TIMELINE=str(path)), log)
            try:
                trace_conversation(url, turns, gen, log)
                time.sleep(1)
            finally:
                proc.kill()
                proc.wait()
                mark_end()
            if not path.is_file():
                raise Fail(f"no {path.name}: {last_error(read_log(log))} ({log.name})")
            result = timeline_summary(path, gap)
            result["report"] = path.name
            # LLAMA_MOE_CACHE_STATS: the last report of the cache and of the prediction (the counts add up)
            text = read_log(log).splitlines()
            keep = []
            for pattern in (r"moe_cache: \d+ layers, .*hit", r"moe_cache: host time per step"):
                hits = [x for x in text if re.search(pattern, x)]
                keep += hits[-1:]
            first = max((i for i, x in enumerate(text) if "moe_cache_stats: prefetch by prediction" in x), default=None)
            if first is not None:
                keep += [x for x in text[first:] if re.search(r"moe_cache_stats: (prefetch|each width|pred )", x)][:8]
            result["log_lines"] = [re.sub(r"^.*?(moe_cache)", r"\1", x) for x in keep]
            return result

        try:
            r, kept = run.measure(f"{arm.name}", identity, go_nsys if nsys_arm else go_timeline)
        except Fail as e:
            run.failed = True
            run.add("", f"{arm.name}: FAIL {e}")
            continue
        run.add(*(trace_lines(arm.name, r, kept) if "phases" not in r else timeline_lines(arm.name, r, kept)))
    return run.finish()


# ---------------------------------------------------------------- starts

def cmd_starts():
    """How often the server starts, for a load that stalls sometimes (the 395: the upload into the MTP head buffer,
    past the 96 GiB carve-out): STARTS starts of each arm of STARTARMS with the arguments of 04, one try each (no
    retries), each until /health answers or the process ends; the load watchdog ends a stalled load after
    LLAMA_LOAD_WATCH_EXIT seconds (45 here, 180 in the server). A start that answered is stopped at once"""
    run = Run("starts")
    model = model_path()
    arms = parse_arms("STARTARMS", "default @draft --spec-auto")
    draft = model_path("SPECDRAFT") if env("SPECDRAFT") and any(a.draft for a in arms) else None
    n_starts, ctx, port = env_int("STARTS", 10), env_int("DECCTX", 262144), env_int("DECPORT", 8097)
    run.add(f"starts: {n_starts} starts of llama-server per arm, one try each, {model.name}, build {current_build()}", "")
    rows = []
    for arm in arms:
        try:
            if arm.draft and not draft:
                raise Skip("SPECDRAFT is not set, it is the draft model of @draft")
            exe = arm.binary("llama-server")
        except Skip as e:
            run.add(f"{arm.name}: skip: {e}")
            continue
        args = [str(a) for a in ["-m", model, *(["-md", draft] if arm.draft else []), "-c", ctx, "-np", 1, "-fa", "on",
                "--repeat-penalty", "1.0", *split_args(env("LOADMODE")), *split_args(env("EXTRA")), *warmup_args(),
                *split_args(env("DECARGS")), *arm.args, "--host", "127.0.0.1", "--port", port, "-lv", 4]]
        run_env = arm_env(arm, {"LLAMA_LOAD_WATCH_EXIT": env("STARTWATCH", "45")})
        identity = {"v": VERSION, "kind": "starts", "build": build_id(exe.parent), "args": args, "env": env_identity(run_env),
                    "model": model_identity(model), "draft": model_identity(draft) if arm.draft else None,
                    "starts": n_starts, "machine": machine()}
        url = f"http://127.0.0.1:{port}"

        def go(log):
            outcomes = []
            for i in range(n_starts):
                wait_no_server()
                no_server_running()
                settle()
                one = log.with_name(f"{log.stem}-{i + 1}.log")
                with one.open("w", encoding="utf-8") as f:
                    f.write(json.dumps({"command": [str(exe), *args], "env": env_identity(run_env)}) + "\n")
                    f.flush()
                    proc = subprocess.Popen([str(exe), *args], stdout=f, stderr=f, env=run_env)
                t0, ok = time.time(), False
                while proc.poll() is None and time.time() - t0 < env_int("DECWAIT", 900):
                    time.sleep(1)
                    try:
                        if http(url + "/health")[0] == 200:
                            ok = True
                            break
                    except (urllib.error.URLError, OSError):
                        pass
                ready = time.time() - t0
                code = proc.poll()
                if code is None:
                    proc.kill()
                    proc.wait()
                mark_end()
                text = read_log(one)
                kind = "ok" if ok else "stall" if "load_all_data: no progress" in text else f"fail {exit_text(proc.returncode)}"
                outcomes.append({"kind": kind, "s": round(ready, 1), "note": "" if ok else (load_watch(one) or last_error(text))[:160]})
                print(f"{arm.name} start {i + 1}: {kind} in {ready:.0f} s", flush=True)
            return {"outcomes": outcomes}

        try:
            r, kept = run.measure(arm.name, identity, go)
        except Fail as e:
            run.failed = True
            run.add(f"{arm.name}: FAIL {e}")
            continue
        o = r["outcomes"]
        oks = [x["s"] for x in o if x["kind"] == "ok"]
        rows.append([arm.name, len(o), len(oks), sum(1 for x in o if x["kind"] == "stall"),
                     sum(1 for x in o if x["kind"].startswith("fail")),
                     f"{sorted(oks)[len(oks) // 2]:.0f}" if oks else "", "k" if kept else ""])
        for x in o:
            if x["kind"] != "ok":
                run.add(f"  {arm.name}: {x['kind']} after {x['s']:.0f} s: {x['note']}")
    run.add("", *table(["arm", "starts", "ok", "stalled", "failed", "median s to ready", ""], rows))
    return run.finish()


# ---------------------------------------------------------------- kernels

def case_name(op, params):
    # the parameters that tell the shapes of a model apart
    keep = [m[0] for m in re.finditer(r"\b(type_a=\w+|type=\w+|n_mats=\d+|n_used=\d+|m=\d+|n=\d+|k=\d+)", params)]
    return " ".join([op, *(re.sub(r"^type(_a)?=", "", k) for k in keep)]) if keep else f"{op}({params})"


def cmd_kernels():
    run = Run("kernels")
    device = env("CHECKDEVICE", DEVICES.get(env("BACKEND"), "CPU"))
    ops = [split_args(o) for o in env("KERNOPS", "MUL_MAT").split(";") if o.strip()]
    arms = parse_arms("KERNARMS", "default")
    results, cases, fails = {}, [], []
    for arm in arms:
        try:
            exe = arm.binary("test-backend-ops")
        except Skip as e:
            fails.append(f"  {arm.name}: skip: {e}")
            continue
        for op in ops:
            if arm.ops and op[0] not in arm.ops:
                continue
            cmd = [exe, "perf", "-b", device, "-o", op[0], *(["-p", op[1]] if len(op) > 1 else []), *arm.args]
            run_env = arm_env(arm)
            identity = {"v": VERSION, "kind": "kernels", "build": build_id(exe.parent), "args": [str(c) for c in cmd[1:]],
                        "env": env_identity(run_env), "machine": machine()}

            def go(log):
                code, text = execute(cmd, log, run_env, model=False, check=False)
                text = re.sub(r"\x1b\[[0-9;]*m", "", text)
                rows = []
                for m in re.finditer(r"^\s+(\w+)\((.*?)\):\s+\d+ runs\s+-\s+([\d.]+) us/run\s+-\s+[\d.]+ \w+/run\s+-\s+([\d.]+) (\S+)",
                                     text, re.MULTILINE):
                    value, unit = float(m[4]), m[5]
                    scale = {"MFLOPS": 1e-6, "GFLOPS": 1e-3, "TFLOPS": 1.0}.get(unit)
                    rows.append([case_name(m[1], m[2]), float(m[3]), value * scale if scale else value,
                                 "TFLOPS" if scale else unit])
                if not rows:
                    raise Fail(f"{exit_text(code)}, no perf result: {last_error(text)} ({log.name})")
                return {"rows": rows}

            try:
                r, kept = run.measure(f"{' '.join(op)} {arm.name}", identity, go)
            except Fail as e:
                run.failed = True
                fails.append(f"  {' '.join(op)} {arm.name}: FAIL {e}")
                continue
            for case, us, value, unit in r["rows"]:
                if case not in cases:
                    cases.append(case)
                results[arm.name, case] = (us, value, unit, kept)
    rows = []
    for case in cases:
        values = [results[a.name, case][1] for a in arms if (a.name, case) in results]
        best = max(values) if values else None
        row = [case]
        for a in arms:
            r = results.get((a.name, case))
            row.append("" if r is None else f"{r[1]:.2f}" + (" *" if len(values) > 1 and r[1] == best else "")
                       + (" k" if r[3] else ""))
        rows.append(row)
    unit = next((r[2] for r in results.values()), "")
    run.add(f"kernels: test-backend-ops perf on {device}, {unit} per case, * the best arm, build {current_build()}", "",
            *table(["case", *[a.name for a in arms]], rows))
    if fails:
        run.add("", "notes:", *fails)
    return run.finish()


# ---------------------------------------------------------------- builds

def cmd_build_begin():
    build = Path(env("BUILD")).resolve()
    build.mkdir(parents=True, exist_ok=True)
    for folder in (build / "bin" / "Release", build / "bin"):
        (folder / BUILD_RECORD).unlink(missing_ok=True)
    try:
        sid, dirty = source_id(env("BACKEND", "cpu"))
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"no source id ({e}): the results of this build are kept under the hash of its binaries")
        return 0
    head = git("rev-parse", "--short", "HEAD").decode().strip()
    (build / "qsa-build.pending.json").write_text(json.dumps({"id": sid, "head": head, "dirty": dirty}) + "\n", encoding="utf-8")
    print(f"source id {sid}")
    return 0


def cmd_build_end():
    CACHE.mkdir(parents=True, exist_ok=True)
    LAST_HEAT.write_text(f"{time.time()}\n")
    build = Path(env("BUILD")).resolve()
    pending = build / "qsa-build.pending.json"
    if pending.is_file():
        # written anew, not moved: the record must be newer than the binaries it describes
        record = dict(json.loads(pending.read_text(encoding="utf-8")), time=now())
        (build_folder(build) / BUILD_RECORD).write_text(json.dumps(record) + "\n", encoding="utf-8")
        pending.unlink()
    return 0


def baseline_perplexity(commit):
    # the evaluated half of the context only: the stable perplexity reserves the logits of all of it, 45 GiB at
    # -c 32768 with a vocab of 248320. the computation and the logits format stay
    source = git("show", f"{commit}:tools/perplexity/perplexity.cpp")
    for old, new, count in (
            (b"logits.reserve(size_t(n_ctx) * n_vocab);", b"logits.reserve(size_t(n_ctx - n_ctx/2) * n_vocab);", 2),
            (b"log_probs.resize(size_t(n_ctx) * nv);", b"log_probs.resize(size_t(n_ctx - 1 - n_ctx/2) * nv);", 1)):
        if source.count(old) != count:
            raise Fail("the buffers of the stable perplexity changed, review baseline_perplexity")
        source = source.replace(old, new)
    return source


def cmd_build_stable():
    """Build the tag stable in its own worktree, with the smaller perplexity buffers."""
    run = Run("build-stable")
    source = Path(env("STABLESRC", str(ROOT / "build-stable-src"))).resolve()
    build = Path(env("STABLEBUILD", str(ROOT / "build-stable"))).resolve()
    commit = git("rev-parse", "stable^{commit}").decode().strip()
    ppl = Path("tools/perplexity/perplexity.cpp")
    marker = source / ".qsa-stable-source"
    if source.exists():
        if not marker.is_file():
            raise Fail(f"{source} exists and is not a worktree of this script, set STABLESRC to another folder")
        subprocess.run(["git", "-C", str(source), "checkout", "--force", "--detach", commit], check=True)
    else:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(source), commit], check=True)
    marker.write_text(commit + "\n")
    (source / ppl).write_bytes(baseline_perplexity(commit))
    log = run.dir / "build.log"
    if os.name == "nt":
        scripts = source / "scripts" / "win-qsa"
        scripts.mkdir(parents=True, exist_ok=True)
        for name in ("00-build.bat", "_config.bat"):
            shutil.copy2(HERE / name, scripts / name)
        settings = {"BUILD": build, "CMAKE_BACKEND": env("CMAKE_BACKEND"), "GENERATOR": env("GENERATOR", "auto"),
                    "QSA_UNATTENDED": "1"}
        (scripts / "_local.bat").write_text("@echo off\r\n" + "".join(f'set "{k}={v}"\r\n' for k, v in settings.items()))
        execute(["cmd.exe", "/d", "/c", "call", scripts / "00-build.bat"], log, os.environ.copy(), model=False)
    else:
        execute(["cmake", "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=ON",
                 "-DLLAMA_CURL=OFF", *split_args(env("CMAKE_BACKEND"))], log, os.environ.copy(), model=False)
        execute(["cmake", "--build", build, "--config", "Release", "-j", os.cpu_count() or 4],
                log.with_name("build2.log"), os.environ.copy(), model=False)
    folder = build_folder(build)
    patch = hashlib.sha256((source / ppl).read_bytes()).hexdigest()[:8]
    sid = f"stable-{commit[:12]}-{patch}"
    (folder / BUILD_RECORD).write_text(json.dumps({"id": sid, "head": commit[:12], "time": now()}) + "\n",
                                       encoding="utf-8")
    run.add(f"stable {commit[:12]} built in {folder}, id {sid}. arms with @stable run it")
    return run.finish()


COMMANDS = {"starts": cmd_starts, "check": cmd_check, "quality": cmd_quality, "decode": cmd_decode,
            "profile": cmd_profile, "kernels": cmd_kernels, "trace": cmd_trace, "build-begin": cmd_build_begin,
            "build-end": cmd_build_end, "build-stable": cmd_build_stable}


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    command = parser.parse_args().command
    try:
        return COMMANDS[command]()
    except (Fail, Skip) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
