#!/usr/bin/env python3
"""Checks of this fork on the 395 (Vulkan) and the 3090 (CUDA), see README.md.

check    the ops of the model on the device and the accuracy of the logits at a few depths, each depth through the
         prompt path and through the decode path
long     the whole native context: the same accuracy at depths to the end, one conversation to the end with facts to
         recall on the way, four conversations at once, the speed of every turn
profile  where the time goes at depths to the end: GPU time per kind of op, the host steps, the MTP head

A result is kept in cache/ under the identity of what made it: the build (00-build.bat records its sources), the
arguments, the GGML_/LLAMA_/CUDA_ environment, the model files and the inputs. When the identity did not change, the
kept result is printed and not measured again. FORCE=1 runs everything again, FORCE=<regex> the matching parts.
Uses only the Python standard library."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / "cache"
VERSION = 2
BUILD_RECORD = "qsa-build.json"
CUDA_HEADERS_RECORD = "qsa-cuda-headers.json"
ENV_PREFIXES = ("GGML_", "LLAMA_", "CUDA_", "HIP_", "OMP_")
ENV_IGNORED = ("LLAMA_SPEC_TRACE",)
# the 395 runs at the edge of its memory at -c 262144: right after another model run these come and go
MEMORY_ERRORS = ("ErrorOutOfDeviceMemory", "ErrorMemoryMapFailed", "Device memory allocation", "failed to allocate",
                 "unable to allocate", "out of memory", "cudaMalloc failed", "bad allocation", "row cache: read error",
                 "failed to create context")
DEVICES = {"vulkan": "Vulkan0", "cuda": "CUDA0"}
EXE = ".exe" if os.name == "nt" else ""

LAST_RUN = CACHE / "last-model-run.txt"
# the end of the last model run or build, for COOL: a build loads every core for minutes and heats the 395 as a run does
LAST_HEAT = CACHE / "last-heat.txt"
DEVICE_LINE = re.compile(r"^\s*(\S+): .*\((\d+) MiB, (\d+) MiB free\)\s*$")
IDLE_USED = CACHE / "idle-used.json"
# the GPU memory counters of Windows (Task Manager shows them): the memory of all processes on all adapters, also the
# memory the driver has not released yet. The Vulkan memory budget the 395 reports does not see it
GPU_COUNTERS = ("Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | "
                "ForEach-Object { [string]$_.DedicatedUsage + ' ' + [string]$_.SharedUsage }")
SHOWN_USED = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
HISTORY_FIELDS = ["time", "host", "backend", "test", "point", "metric", "value", "build", "head", "dirty"]
_digests = {}
_build_ids = {}


class Fail(Exception):
    pass


class Skip(Exception):
    pass
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

def source_id(backend):
    """Hash of the sources of this backend's build: HEAD, the uncommitted changes and the untracked files. The
    directories of the other backends are left out, so a change there keeps the results of this machine. Returns the
    id and whether there were uncommitted changes or untracked files in those sources."""
    entries = ["CMakeLists.txt", "cmake", "common", "include", "src", "tools", "vendor", "tests/test-backend-ops.cpp",
               "tests/test-backend-ops-fork.inc", "ggml/CMakeLists.txt", "ggml/cmake", "ggml/include"]
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

def cache_put(identity, result, log, durable=False):
    folder = CACHE / "results"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identity_key(identity) + ".json")
    part = path.with_suffix(".part")
    record = {"identity": identity, "result": result, "log": str(log), "time": now()}
    with part.open("w", encoding="utf-8") as file:
        file.write(json.dumps(record, indent=1) + "\n")
        if durable:
            file.flush()
            os.fsync(file.fileno())
    part.replace(path)

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

def cool_down():
    """COOL seconds (0): at least that long after the last model run or build, so that the arms of a run start from about
    the same temperature (the 395 reached 95 C in an hour of chats to 213K on 14.10, the ops of the last profile depth
    ran 2 percent slower than the night before; the first arm after the build of 14.10 ran 3 to 5 percent slower in
    every phase than the second, also in the steps that read nothing from the drive). Returns the seconds it waited."""
    cool = env_int("COOL", 0)
    heat = None
    for path in (LAST_HEAT, LAST_RUN):
        try:
            heat = time.time() - float(path.read_text())
            break
        except (OSError, ValueError):
            continue
    if cool > 0 and heat is not None and heat < cool:
        print(f"cooling {cool - heat:.0f} s after the last model run or build (COOL)", flush=True)
        time.sleep(cool - heat)
        return cool - heat
    return 0

def settle(timed=True):
    """Before a model run: wait until the GPU memory of the last one is free again. The driver releases the memory of
    a finished process lazily: on the 395 a run right after another failed on memory or hung while loading
    (08.10-11.10). Done at once when the memory in use is within SETTLE_MARGIN MiB (2048) of the least seen on this
    machine (cache/idle-used.json), else the memory is read every SETTLE_STEP seconds (3) until it stops going down,
    SETTLE_MAX seconds (180) at most. Without a reading the old rule: SETTLE seconds (30) after the last model run.
    timed=False: the run measures no speed (03, 08), so it does not wait for COOL either."""
    try:
        since = time.time() - float(LAST_RUN.read_text())
    except (OSError, ValueError):
        since = None
    waited = cool_down() if timed else 0
    if since is not None:
        since += waited
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

def execute(cmd, log, run_env, stdout=None, model=True, check=True, cwd=None, timed=True):
    """Run cmd with its output in log, its stdout in a separate file when given, in cwd when given. A model run that
    fails on memory gets two more tries after longer waits: the driver frees the memory of the previous process lazily.
    timed=False: a model run that measures no speed, see settle()."""
    cmd = [str(c) for c in cmd]
    if model:
        no_server_running()
    first = log
    for attempt in (1, 2, 3):
        if model:
            settle(timed)
        with log.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"command": cmd, "env": env_identity(run_env)}) + "\n")
            f.flush()
            out = stdout.open("w", encoding="utf-8") if stdout else f
            try:
                code = subprocess.run(cmd, stdout=out, stderr=f, env=run_env, cwd=cwd).returncode
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

def rate(n, ms):
    return 1000 * n / ms if ms > 0 else 0.0

def table(header, rows):
    rows = [list(r) + [""] * (len(header) - len(r)) for r in rows]
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    return ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [header, *rows]]

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

def cuda_headers_digest():
    """the headers a .cu file of ggml-cuda can include: its own (with the fork's .inc), ggml/src and ggml/include"""
    h = hashlib.sha256()
    files = [p for p in (ROOT / "ggml" / "src" / "ggml-cuda").rglob("*") if p.suffix in (".cuh", ".h", ".inc")]
    files += [p for d in ("ggml/src", "ggml/include") for p in (ROOT / d).glob("*.h")]
    for p in sorted(f for f in files if f.is_file()):
        h.update(f"{p.relative_to(ROOT).as_posix()} {digest(p)}\n".encode())
    return h.hexdigest()[:16]

def cuda_objects_check(build):
    """removes the msbuild objects of ggml-cuda when the headers differ from the last good build (or it has no record)"""
    objs = build / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.dir"
    if env("BACKEND") != "cuda" or not objs.is_dir():
        return
    record = build / CUDA_HEADERS_RECORD
    try:
        before = json.loads(record.read_text(encoding="utf-8"))["headers"]
    except (OSError, ValueError, KeyError):
        before = None
    if before != cuda_headers_digest():
        print(f"the headers of ggml-cuda changed since the last good build ({before or 'no record'}): removing {objs}, "
              f"msbuild would keep objects built against the old ones")
        shutil.rmtree(objs, ignore_errors=True)

def cmd_build_begin():
    build = Path(env("BUILD")).resolve()
    build.mkdir(parents=True, exist_ok=True)
    cuda_objects_check(build)
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
    if env("BACKEND") == "cuda":
        (build / CUDA_HEADERS_RECORD).write_text(json.dumps({"headers": cuda_headers_digest(), "time": now()}) + "\n",
                                                 encoding="utf-8")
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


# ---------------------------------------------------------------- runs

class Run:
    """One call of a script: its log folder, its summary and the counts of new and kept results."""

    def __init__(self, name):
        self.start = datetime.now()
        self.dir = Path(env("LOGS", str(HERE / "logs"))) / f"{name}-{self.start:%Y%m%d-%H%M%S}"
        self.dir.mkdir(parents=True, exist_ok=True)
        console_to(self.dir / "console.txt")
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

    def record(self, test, point, metrics, kept, folder):
        """A new result into history-qa.csv next to the run folders, one row per metric (None is left out). The
        history.csv of the scripts before 28.10 has other columns."""
        if kept:
            return
        path = self.dir.parent / "history-qa.csv"
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
                w.writerow([f"{self.start:%Y-%m-%d %H:%M:%S}", platform.node(), env("BACKEND"), test, point, metric,
                            f"{value:.6g}" if isinstance(value, float) else value, bid, head, dirty])

    def finish(self):
        if self.new or self.kept:
            self.add("", f"{self.new} new, {self.kept} kept (k): the same build measured them before. FORCE=1 runs "
                         f"them again, FORCE=<regex> the matching parts")
        self.add(f"logs in {self.dir}")
        print("\n" + "\n".join(self.lines), flush=True)
        return 1 if self.failed else 0


def env_pairs(text, where):
    """VAR=value VAR=value ... as a dict"""
    out = {}
    for t in split_args(text):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", t):
            raise Fail(f"{where}: {t!r} is not VAR=value")
        key, value = t.split("=", 1)
        out[key] = value
    return out


def variants(name, default):
    """name VAR=value ...;name VAR=value ...: the variants of a part, each a name and its environment"""
    out = []
    for part in env(name, default).split(";"):
        tokens = split_args(part)
        if tokens:
            out.append((tokens[0], env_pairs(" ".join(tokens[1:]), name)))
    return out


def binary(name):
    path = Path(env("BIN")) / (name + EXE)
    if not path.is_file():
        raise Skip(f"no {path}, run 00-build.bat")
    return path.resolve()


def stats(values):
    v = sorted(values)
    if not v:
        return None
    return {"median": statistics.median(v), "p90": v[min(len(v) - 1, int(0.9 * len(v)))], "mean": statistics.mean(v),
            "max": v[-1]}


# ---------------------------------------------------------------- check: ops

class CheckArm:
    """A variant of the op checks: "name [VAR=value ...] [@o=OP,...] [test-backend-ops arguments ...]"."""

    def __init__(self, text):
        tokens = split_args(text)
        if not tokens:
            raise Fail("empty CHECKARMS entry")
        self.name = tokens[0]
        self.env, self.args, self.ops = {}, [], None
        for t in tokens[1:]:
            if t.startswith("@o="):
                self.ops = {o for o in t[3:].split(",") if o}
            elif re.fullmatch(r"[A-Z][A-Z0-9_]*=.*", t):
                key, value = t.split("=", 1)
                self.env[key] = value
            else:
                self.args.append(t)


def check_ops(run):
    """test-backend-ops of the ops of the model on the device against the CPU, and the fork's unit tests."""
    device = env("CHECKDEVICE", DEVICES.get(env("BACKEND"), "CPU"))
    ops = [split_args(o) for o in env("CHECKOPS", "MUL_MAT").split(";") if o.strip()]
    arms = [CheckArm(a) for a in env("CHECKARMS", "default").split(";") if a.strip()]
    rows, fails = [], []
    exe = binary("test-backend-ops")
    bid = build_id(exe.parent)
    for arm in arms:
        for op in ops:
            if arm.ops and op[0] not in arm.ops:
                continue
            cmd = [exe, "test", "-b", device, "-o", op[0], *(["-p", op[1]] if len(op) > 1 else []), *arm.args]
            run_env = dict(os.environ, **arm.env)
            identity = {"v": VERSION, "kind": "ops", "build": bid, "args": [str(c) for c in cmd[1:]],
                        "env": env_identity(run_env), "machine": machine()}

            def go(log, cmd=cmd, run_env=run_env):
                code, text = execute(cmd, log, run_env, model=False, check=False)
                counts = re.findall(r"(\d+)/(\d+) tests passed", text)
                if not counts:
                    raise Fail(f"{exit_text(code)}, no test result: {last_error(text)} ({log.name})")
                return {"passed": sum(int(a) for a, _ in counts), "total": sum(int(b) for _, b in counts),
                        "fails": [s.strip() for s in text.splitlines() if "FAIL" in s][:8]}

            label = "ops " + " ".join(op) + " " + arm.name
            try:
                r, kept = run.measure(label, identity, go)
            except Fail as e:
                run.failed = True
                rows.append([" ".join(op), arm.name, f"FAIL {e}"])
                continue
            ok = r["passed"] == r["total"]
            run.failed |= not ok
            run.record("ops", f"{op[0]} {arm.name}", {"passed": r["passed"], "total": r["total"]}, kept, exe.parent)
            # 0/0: the device supports none of the cases
            rows.append([" ".join(op), arm.name, f"{r['passed']}/{r['total']}" + ("" if ok else " FAIL") + (" k" if kept else "")])
            fails.extend(f"  {op[0]} {arm.name}: {s}" for s in r["fails"])
    # the fork's unit tests that build on this machine: exit code 0 passes. CHECKUNITS=none: none
    units = [] if env("CHECKUNITS") == "none" else split_args(env("CHECKUNITS", "test-row-cache test-prompt-cache-spill "
             "test-mtp-window-fork test-spec-auto-fork test-env-fork test-elastic-buffer-fork"))
    for name in units:
        try:
            uexe = binary(name)
        except Skip as e:
            rows.append([name, "unit", f"skip: {e}"])
            continue
        work = run.dir / name
        src = ROOT / "tests" / f"{name}.cpp"
        identity = {"v": VERSION, "kind": "unit", "build": build_id(uexe.parent), "test": name, "machine": machine(),
                    "src": digest(src) if src.is_file() else None,
                    "env": {k: v for k, v in sorted(os.environ.items()) if k.upper().startswith("TEST_")}}

        def go_unit(log, uexe=uexe, work=work):
            work.mkdir(parents=True, exist_ok=True)
            code, text = execute([uexe], log, os.environ.copy(), model=False, check=False, cwd=work)
            return {"code": code, "error": "" if code == 0 else f"{exit_text(code)}: {last_error(text)}"}

        try:
            r, kept = run.measure("unit " + name, identity, go_unit)
        except Fail as e:
            run.failed = True
            rows.append([name, "unit", f"FAIL {e}"])
            continue
        ok = r["code"] == 0
        run.failed |= not ok
        rows.append([name, "unit", ("passed" if ok else f"FAIL {r['error']}") + (" k" if kept else "")])
    run.add(f"ops: test-backend-ops on {device} against the CPU, unit tests of the fork", "",
            *table(["op", "variant", "passed"], rows))
    if fails:
        run.add("", "failed cases:", *fails)


# ---------------------------------------------------------------- accuracy at depth

# the reference of the accuracy probes: the same build and model through the upstream copy of qwen4exp, its graph and
# its QSA memory - another implementation of the block selection, the pooled block keys and the mask of the selected
# cells, on the same kernels. not bit for bit: the fork takes one more whole block at the positions without a tail
# (KLD 3.8e-5 at a budget of 2048 on the test model, 30.10), so a fault of the fork shows as a jump above that
REF_ENV = "LLAMA_UPSTREAM=qwen4exp"


def quality_args():
    return split_args(env("QA_QARGS", "-b 16384 -ub " + ("8192" if env("BACKEND") == "cuda" else "4096")))


def tool_run(run, exe, model, name, plan, run_env, model_run=True, timed=False, qargs=None):
    """llama-quality-fork with a probe plan; returns its report"""
    plan = dict(plan, report=str(run.dir / (name + ".json")))
    path = run.dir / (name + "-plan.json")
    path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    cmd = [exe, *model_args(model), *(quality_args() if qargs is None else qargs), "--probe-plan", path]
    execute(cmd, run.dir / (name + ".log"), run_env, model=model_run, timed=timed)
    return json.loads(Path(plan["report"]).read_text(encoding="utf-8"))


def native_context(run, exe, model):
    meta = tool_run(run, exe, model, "metadata", {"action": "info"}, os.environ.copy(), model_run=False)
    return meta["native_context"], meta["vocab"]


def corpus_tokens(run, exe, model, count):
    """count tokens of the corpus (QA_CORPUS, else PPLFILE) as the model tokenizes it, in a file kept in cache/qa"""
    corpus = Path(env("QA_CORPUS", env("PPLFILE")))
    if not corpus.is_file():
        raise Fail(f"no corpus {corpus}: set QA_CORPUS to a long text file (or run get-wikitext.bat for PPLFILE)")
    ident = {"v": VERSION, "kind": "tokens", "model": model_identity(model), "corpus": digest(corpus), "count": count}
    folder = CACHE / "qa"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identity_key(ident)[:24] + ".tokens")
    if not path.is_file() or path.stat().st_size != 4 * count:
        part = path.with_suffix(".part")
        try:
            tool_run(run, exe, model, f"tokens-{count}", {"action": "tokens", "corpus": str(corpus), "count": count,
                                                         "tokens": str(part)}, os.environ.copy(), model_run=False)
        except Fail as e:
            raise Fail(f"{e}: the corpus {corpus.name} needs at least {count} tokens, give a longer text in QA_CORPUS")
        part.replace(path)
    return path


def probe_plan(ctx, depths, window, decode_batch):
    """the probe positions: the last window positions before each depth, the second half of each window through the
    decode path (batches of decode_batch tokens), the first half in the prompt batch; the context end is a depth"""
    depths = sorted({min(ctx, d) for d in depths if d >= window} | {ctx})
    positions = sorted({p for d in depths for p in range(d - window, d)})
    ranges = [[d - window // 2, d] for d in depths] if decode_batch > 0 else []
    return depths, positions, ranges


def probe_rows(samples, depths, window):
    """per depth and path: the KL of the reference to the variant (median, 90th percentile), the share of tokens above
    0.01 and above 0.5, the top-1 agreement and the mean NLL difference"""
    out = []
    for d in depths:
        for decode in (False, True):
            s = [x for x in samples if d - window <= x["position"] < d and bool(x.get("decode")) == decode]
            if not s:
                continue
            kl = stats([x["kl"] for x in s])
            out.append({"depth": d, "path": "decode" if decode else "prompt", "n": len(s), "kl_median": kl["median"],
                        "kl_p90": kl["p90"], "kl_max": kl["max"],
                        "over_001": sum(x["kl"] > 0.01 for x in s) / len(s), "over_05": sum(x["kl"] > 0.5 for x in s),
                        "top1": sum(bool(x["top1_equal"]) for x in s) / len(s),
                        "dnll": statistics.mean(x["delta_nll"] for x in s)})
    return out


def probe_flags(rows):
    """a depth where the decode path differs from the prompt path much more than rounding does: the 90th percentile
    of the KL ten times that of the prompt probes at that depth and above 0.002, or 10 points more tokens above 0.01.
    The decode and prompt probes of one depth share the context, the reference ran the same batches"""
    flags = {}
    by_depth = {}
    for r in rows:
        by_depth.setdefault(r["depth"], {})[r["path"]] = r
    for d, p in by_depth.items():
        if "decode" in p and "prompt" in p:
            dec, pro = p["decode"], p["prompt"]
            if (dec["kl_p90"] > 10 * pro["kl_p90"] and dec["kl_p90"] > 0.002) or dec["over_001"] > pro["over_001"] + 0.10:
                flags[d] = "DECODE DIFFERS"
        for r in p.values():
            if r["top1"] < 0.9:
                flags[d] = (flags.get(d, "") + " LOW TOP1").strip()
    return flags


def quality(run, label, ctx, depths, qargs=None):
    """The logits at the probe positions against the reference (REF_ENV), for the default settings and the variants
    of QA_QARMS; the reference and the variants run the same batches (qargs, else QA_QARGS). Returns True when nothing
    was flagged."""
    qargs = quality_args() if qargs is None else qargs
    model = model_path()
    exe = binary("llama-quality-fork")
    native, vocab = native_context(run, exe, model)
    ctx = native if ctx <= 0 else ctx
    if ctx > native:
        raise Fail(f"the context {ctx} is past the native context of the model, {native}")
    window, dbatch = env_int("QA_WINDOW", 64), env_int("QA_DECODE_BATCH", 3)
    depths, positions, ranges = probe_plan(ctx, depths, window, dbatch)
    tokens = corpus_tokens(run, exe, model, ctx + 1)
    plan = {"context": ctx, "positions": positions, "tokens": str(tokens), "max_outputs": 32,
            "decode_batch": dbatch, "decode_ranges": ranges}
    base = {"v": VERSION, "kind": "quality", "build": build_id(exe.parent), "model": model_identity(model),
            "tokens": digest(tokens), "plan": {k: v for k, v in plan.items() if k != "tokens"},
            "args": [str(a) for a in [*model_args(model), *qargs]], "machine": machine()}
    ref_env = dict(os.environ, **env_pairs(env("QA_REF_ENV", REF_ENV), "QA_REF_ENV"))
    ref_id = {**base, "env": env_identity(ref_env)}
    folder = CACHE / "qa" / label
    folder.mkdir(parents=True, exist_ok=True)
    ref = folder / (identity_key(ref_id)[:24] + ".ref")
    size_gib = len(positions) * vocab * 4 / 2**30
    if not ref.is_file() or forced(f"{label}-reference"):
        # the old references of this cache folder go: they are large and belong to other builds
        for old in folder.glob("*.ref"):
            if old != ref:
                old.unlink(missing_ok=True)
        print(f"{label}: reference, {len(positions)} probes to {ctx}, {size_gib:.2f} GiB", flush=True)
        ref.unlink(missing_ok=True)
        tool_run(run, exe, model, f"{label}-reference", dict(plan, action="capture", reference=str(ref)), ref_env,
                 qargs=qargs)
    run.add(f"{label}: logits at {len(positions)} positions before the depths {' '.join(map(str, depths))} of one "
            f"context of {ctx} tokens against the reference ({env('QA_REF_ENV', REF_ENV)}); in each window of "
            f"{window} the first half through the prompt batch, the second in batches of {dbatch} (the decode path)")
    clean = True
    for name, venv in variants("QA_QARMS", "default"):
        run_env = dict(os.environ, **venv)
        identity = {**base, "env": env_identity(run_env), "reference": digest(ref)}

        def go(log, name=name, run_env=run_env):
            r = tool_run(run, exe, model, f"{label}-{name}", dict(plan, action="compare", reference=str(ref)), run_env,
                         qargs=qargs)
            if r["processed"] != ctx or len(r["samples"]) != len(positions):
                raise Fail(f"the probe run stopped at {r['processed']} of {ctx} tokens")
            return probe_rows(r["samples"], depths, window)

        try:
            rows, kept = run.measure(f"{label}-{name}", identity, go)
        except Fail as e:
            run.failed = True
            clean = False
            run.add("", f"{label} {name}: FAIL {e}")
            continue
        flags = probe_flags(rows)
        clean &= not flags
        table_rows = [[r["depth"], r["path"], r["n"], f"{r['kl_median']:.6f}", f"{r['kl_p90']:.5f}", f"{r['kl_max']:.3f}",
                       f"{100 * r['over_001']:.1f}%", r["over_05"], f"{100 * r['top1']:.1f}%", f"{r['dnll']:+.4f}",
                       flags.get(r["depth"], "") if r["path"] == "decode" else ""] for r in rows]
        run.add("", f"{label} {name}" + (" k" if kept else ""),
                *table(["depth", "path", "n", "KL median", "KL p90", "KL max", ">0.01", ">0.5", "top-1", "dNLL", ""],
                       table_rows))
        for r in rows:
            run.record(label, f"{name} d{r['depth']} {r['path']}", {"kl_median": r["kl_median"], "kl_p90": r["kl_p90"],
                       "top1": r["top1"], "dnll": r["dnll"]}, kept, exe.parent)
    if not clean:
        run.failed = True
    return clean


def cmd_check():
    run = Run("check")
    check_ops(run)
    if env("QA_CHECK_QUALITY", "1") != "0":
        try:
            quality(run, "quality", env_int("QA_CHECK_CTX", 32768), env_ints("QA_CHECK_DEPTHS", "4096 16384"))
        except (Fail, Skip) as e:
            run.failed = True
            run.add("", f"quality: FAIL {e}")
    return run.finish()


# ---------------------------------------------------------------- the server

ADJECTIVES = ["amber", "silver", "crimson", "golden", "violet", "copper", "ivory", "scarlet", "azure", "emerald",
              "coral", "bronze", "indigo", "olive", "pearl", "ruby", "sable", "teal", "umber", "jade"]
NOUNS = ["falcon", "otter", "lynx", "heron", "badger", "walrus", "gecko", "marten", "osprey", "tapir", "ibex", "kestrel",
         "panda", "raven", "salmon", "toucan", "vulture", "weasel", "yak", "zebra"]


def make_facts(n, seed):
    """n facts to recall: a unique file name and a six digit key"""
    rng = random.Random(seed)
    names = [f"{a} {b}" for a in ADJECTIVES for b in NOUNS]
    rng.shuffle(names)
    names += [f"{x} {i}" for i in range(2, 2 + n // len(names) + 1) for x in names][:max(0, n - len(names))]
    return [{"name": names[i], "key": str(rng.randrange(100000, 1000000))} for i in range(n)]


def place_fact(text, fact, k, rng):
    """the fact as a sentence of its own, put at a sentence end in the middle half of the text"""
    line = f" Archive record {k}: the key of the {fact['name']} file is {fact['key']}. "
    lo, hi = len(text) // 4, 3 * len(text) // 4
    at = text.find(". ", rng.randrange(lo, max(lo + 1, hi)))
    at = at + 1 if at >= 0 else len(text) // 2
    return text[:at] + line + text[at:]


def question(fact):
    return (f"Somewhere in the document above there is an archive record with the key of the {fact['name']} file. "
            f"What is that key? Reply with the number only.")


def repetition(text):
    """the share of repeated word 4-grams of a text, 0 for none"""
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def sampling():
    parts = [s for s in re.split(r"[\s:]+", env("QA_SAMP", "1.0 20 0.95")) if s]
    try:
        t, k, p = parts
        return float(t), int(k), float(p)
    except ValueError:
        raise Fail(f"QA_SAMP must be temperature, top_k and top_p, as \"1.0 20 0.95\", got {env('QA_SAMP')!r}")


def server_args(model, ctx, slots, port, extra=()):
    """the server as serve.bat runs it: the model, LOADMODE, EXTRA, DECARGS, the MTP head of SPECDRAFT with QA_SPEC"""
    draft = env("SPECDRAFT") if env("QA_SPEC", "--spec-auto") != "none" else ""
    if draft and not Path(draft).is_file():
        raise Fail(f"SPECDRAFT not found: {draft!r}")
    spec = split_args(env("QA_SPEC", "--spec-auto")) if draft else []
    return [str(a) for a in ["-m", model, *(["-md", draft] if draft else []), "-c", ctx, "-np", slots, "-fa", "on",
            *split_args(env("LOADMODE")), *split_args(env("EXTRA")), *warmup_args(), *split_args(env("DECARGS")),
            *spec, *(["--kv-unified"] if slots > 1 else []), *extra, "--host", "127.0.0.1", "--port", port]]


class Chat:
    """A client of one conversation: the messages it sends, the answers, the timings of every turn"""

    def __init__(self, url, log, name):
        self.url, self.log, self.name = url, log, name
        self.messages, self.turns = [], []
        self.temp, self.top_k, self.top_p = sampling()

    def ask(self, text, max_tokens, kind):
        self.messages.append({"role": "user", "content": text})
        body = {"messages": self.messages, "max_tokens": max_tokens, "temperature": self.temp, "top_k": self.top_k,
                "top_p": self.top_p, "seed": 1, "cache_prompt": True, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            _, raw = http(self.url + "/v1/chat/completions", body, env_int("QA_TIMEOUT", 3600))
        except urllib.error.HTTPError as e:
            raise Fail(f"{self.name} turn {len(self.turns)}: HTTP {e.code} {e.read()[:300]!r}")
        except (urllib.error.URLError, OSError) as e:
            raise Fail(f"{self.name} turn {len(self.turns)}: {e} ({self.log.name})")
        answer = json.loads(raw)
        if "timings" not in answer:
            raise Fail(f"{self.name} turn {len(self.turns)}: no timings in the answer {raw[:300]!r}")
        msg = answer["choices"][0]["message"]
        content = msg.get("content") or ""
        self.messages.append({"role": "assistant", "content": content})
        t = answer["timings"]
        turn = {"kind": kind, "depth": t.get("cache_n", 0) + t["prompt_n"], "prompt_n": t["prompt_n"],
                "prompt_ms": t["prompt_ms"], "n": t["predicted_n"], "ms": t["predicted_ms"],
                "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0),
                "finish": answer["choices"][0].get("finish_reason"), "content": content[:2000],
                "reasoning": (msg.get("reasoning_content") or "")[:500]}
        self.turns.append(turn)
        self.log.with_name(f"{self.log.stem}-{self.name}.json").write_text(
            json.dumps(self.turns, ensure_ascii=False, indent=1), encoding="utf-8")
        return turn


def token_ratio(url, text):
    """tokens per character of the corpus, from the server's tokenizer on a part of it"""
    sample = text[:40000]
    _, raw = http(url + "/tokenize", {"content": sample}, 120)
    n = len(json.loads(raw)["tokens"])
    return max(n, 1) / max(len(sample), 1)


def corpus_text():
    corpus = Path(env("QA_CORPUS", env("PPLFILE")))
    if not corpus.is_file():
        raise Fail(f"no corpus {corpus}: set QA_CORPUS to a long text file (or run get-wikitext.bat for PPLFILE)")
    return corpus.read_text(encoding="utf-8", errors="replace")


def converse(chat, text, offset, ratio, target, chunk, facts, rng, asks, reserve, ctx):
    """Grows the conversation of chat with parts of text of about chunk tokens, each with one fact, up to target
    tokens; at each depth of asks it asks for an early, a middle and the latest fact. Returns (recall rows, offset)."""
    recall, placed, depth = [], [], 0
    asks = sorted(asks)
    ask_tokens = env_int("QA_ASK_TOKENS", 64)

    def ask_facts():
        nonlocal depth
        picks = {placed[0][0]["name"]: placed[0], placed[len(placed) // 2][0]["name"]: placed[len(placed) // 2],
                 placed[-1][0]["name"]: placed[-1]}
        for fact_q, at in picks.values():
            if depth + ask_tokens + 256 > ctx:
                break
            t = chat.ask(question(fact_q), ask_tokens, "question")
            depth = t["depth"]
            recall.append({"depth": depth, "fact_depth": at, "name": fact_q["name"], "key": fact_q["key"],
                           "answer": t["content"].strip()[:80], "ok": fact_q["key"] in t["content"],
                           "thinking": not t["content"].strip() and bool(t["reasoning"]),
                           "in_reasoning": fact_q["key"] in t["reasoning"]})

    while True:
        # the last part is shorter: the conversation ends reserve tokens before the context (or the target) ends
        n_tok = min(chunk, min(target, ctx) - reserve - depth - 512)
        n_chars = int(n_tok / ratio)
        if n_tok < min(2048, chunk) or offset + n_chars > len(text):
            break
        fact = facts[len(placed)]
        part = place_fact(text[offset:offset + n_chars], fact, len(placed) + 1, rng)
        offset += n_chars
        turn = chat.ask(part + f"\n\n(Part {len(placed) + 1} of a long document. Reply only: OK)", 8, "part")
        placed.append((fact, turn["depth"]))
        depth = turn["depth"]
        while asks and depth >= asks[0]:
            asks.pop(0)
            ask_facts()
    if asks and placed:
        ask_facts()
    return recall, offset, depth


def turn_rows(turns):
    rows = []
    for t in turns:
        pp = rate(t["prompt_n"], t["prompt_ms"])
        tg = rate(t["n"], t["ms"]) if t["n"] > 1 else None
        rows.append([t["kind"], t["depth"], t["prompt_n"], f"{pp:.1f}", f"{tg:.2f}" if tg else "",
                     f"{t['draft_acc'] / t['draft_n']:.2f}" if t["draft_n"] else ""])
    return rows


def long_chat(run, model, ctx, port):
    """one conversation to the end of the context, facts to recall on the way, a long answer at the end"""
    exe = binary("llama-server")
    chunk, reserve = env_int("QA_LONG_CHUNK", 24576), env_int("QA_LONG_RESERVE", 2048)
    asks = env_ints("QA_LONG_ASK", "65536 131072 196608")
    final_tokens = env_int("QA_LONG_FINAL", 600)
    args = server_args(model, ctx, 1, port)
    text = corpus_text()
    identity = {"v": VERSION, "kind": "long-chat", "build": build_id(exe.parent), "args": args, "model": model_identity(model),
                "env": env_identity(os.environ), "corpus": text_digest(text), "chunk": chunk, "asks": asks,
                "reserve": reserve, "final": final_tokens, "samp": env("QA_SAMP", "1.0 20 0.95"), "machine": machine(),
                "draft": model_identity(Path(env("SPECDRAFT"))) if "-md" in args else None}

    def go(log):
        proc, log = start_server(exe, args, os.environ.copy(), log)
        try:
            url = f"http://127.0.0.1:{port}"
            ratio = token_ratio(url, text)
            chat = Chat(url, log, "chat")
            facts = make_facts(ctx // max(chunk // 2, 1) + 8, 1)
            recall, _, depth = converse(chat, text, 0, ratio, ctx, chunk, facts, random.Random(7), asks, reserve, ctx)
            room = ctx - depth - 64
            final = None
            if room > 128:
                t = chat.ask("Now write a detailed summary of the whole document, about 300 words.",
                             min(final_tokens, room), "summary")
                final = {"depth": t["depth"], "n": t["n"], "finish": t["finish"], "repetition": repetition(t["content"]),
                         "head": t["content"][:300], "tail": t["content"][-300:]}
            return {"turns": chat.turns, "recall": recall, "final": final}
        finally:
            proc.kill()
            proc.wait()
            mark_end()

    r, kept = run.measure("long-chat", identity, go)
    k = " k" if kept else ""
    turns = r["turns"]
    run.add("", f"chat: one conversation to {turns[-1]['depth'] if turns else 0} of {ctx} tokens, parts of {chunk} tokens "
                f"with a fact each, questions at {' '.join(map(str, asks))}" + k,
            *table(["turn", "depth", "new tokens", "PP t/s", "TG t/s", "accepted"], turn_rows(turns)))
    ok = sum(x["ok"] for x in r["recall"])
    run.add("", f"recall: {ok} of {len(r['recall'])} keys right",
            *table(["asked at", "fact at", "file", "key", "answer", ""],
                   [[x["depth"], x["fact_depth"], x["name"], x["key"], x["answer"],
                     "ok" if x["ok"] else ("thinking, no answer" if x.get("thinking") else "in reasoning" if x["in_reasoning"]
                                           else "WRONG")] for x in r["recall"]]))
    f = r["final"]
    bad = []
    if f:
        if f["repetition"] > 0.35:
            bad.append(f"repeats ({100 * f['repetition']:.0f}% of 4-grams)")
        if f["n"] < 50:
            bad.append(f"only {f['n']} tokens")
        run.add("", f"answer at {f['depth']}: {f['n']} tokens, finish {f['finish']}, repeated 4-grams "
                    f"{100 * f['repetition']:.0f}%" + (": " + ", ".join(bad) if bad else ""),
                "  begins: " + f["head"].replace("\n", " ")[:200], "  ends:   " + f["tail"].replace("\n", " ")[-200:])
    if ok < len(r["recall"]) or bad:
        run.failed = True
    deep = [t for t in turns if t["kind"] == "part"]
    if deep:
        run.record("long", "chat", {"depth": deep[-1]["depth"], "recall": ok / max(1, len(r["recall"])),
                   "pp_last": rate(deep[-1]["prompt_n"], deep[-1]["prompt_ms"])}, kept, exe.parent)


def long_par(run, model, ctx, port):
    """several conversations at once in the shared KV of the slots, each with facts, asked at its end"""
    exe = binary("llama-server")
    par, total = env_int("QA_PAR", 4), env_int("QA_PAR_TOKENS", 49152)
    chunk, reserve = env_int("QA_PAR_CHUNK", 12288), env_int("QA_LONG_RESERVE", 2048)
    if par < 2:
        return
    args = server_args(model, ctx, par, port)
    text = corpus_text()
    identity = {"v": VERSION, "kind": "long-par", "build": build_id(exe.parent), "args": args, "model": model_identity(model),
                "env": env_identity(os.environ), "corpus": text_digest(text), "par": par, "total": total, "chunk": chunk,
                "samp": env("QA_SAMP", "1.0 20 0.95"), "machine": machine(),
                "draft": model_identity(Path(env("SPECDRAFT"))) if "-md" in args else None}

    def go(log):
        proc, log = start_server(exe, args, os.environ.copy(), log)
        try:
            url = f"http://127.0.0.1:{port}"
            ratio = token_ratio(url, text)
            span = max(1, (len(text) - int(total / ratio)) // par)
            results, errors = [None] * par, []

            def one(i):
                try:
                    chat = Chat(url, log, f"par{i}")
                    facts = make_facts(total // max(chunk // 2, 1) + 8, 100 + i)
                    _, _, depth = converse(chat, text, i * span, ratio, total, chunk, facts, random.Random(i), [], reserve, ctx)
                    placed = [t for t in chat.turns if t["kind"] == "part"]
                    rec = []
                    for fact in (facts[0], facts[len(placed) - 1]):
                        t = chat.ask(question(fact), env_int("QA_ASK_TOKENS", 64), "question")
                        rec.append({"name": fact["name"], "key": fact["key"], "answer": t["content"].strip()[:80],
                                    "ok": fact["key"] in t["content"]})
                    results[i] = {"turns": chat.turns, "recall": rec, "depth": chat.turns[-1]["depth"]}
                except Exception as e:  # noqa: BLE001 - reported below, the others go on
                    errors.append(f"par{i}: {e}")

            start = time.time()
            threads = [threading.Thread(target=one, args=(i,)) for i in range(par)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            return {"results": results, "errors": errors, "wall": time.time() - start}
        finally:
            proc.kill()
            proc.wait()
            mark_end()

    r, kept = run.measure("long-par", identity, go)
    rows, all_ok = [], not r["errors"]
    for i, x in enumerate(r["results"]):
        if not x:
            rows.append([f"par{i}", "", "", "FAIL"])
            continue
        ok = sum(y["ok"] for y in x["recall"])
        all_ok &= ok == len(x["recall"])
        n = sum(t["n"] for t in x["turns"])
        rows.append([f"par{i}", x["depth"], n, f"{ok}/{len(x['recall'])}" + ("" if ok == len(x["recall"]) else " WRONG")])
    run.add("", f"{par} conversations at once (-np {par} --kv-unified, -c {ctx}), {total} tokens each in parts of {chunk}, "
                f"{r['wall']:.0f} s" + (" k" if kept else ""), *table(["slot", "depth", "generated", "recall"], rows))
    for e in r["errors"]:
        run.add(f"  {e}")
    if not all_ok:
        run.failed = True


def cmd_long():
    run = Run("long")
    stages = set(env("QA_LONG_STAGES", "quality chat par").split())
    model = model_path()
    ctx = env_int("QA_LONG_CTX", 0)
    port = env_int("QA_PORT", 8097)
    try:
        native, _ = native_context(run, binary("llama-quality-fork"), model)
    except (Fail, Skip) as e:
        run.failed = True
        run.add(f"long: FAIL {e}")
        return run.finish()
    ctx = native if ctx <= 0 else min(ctx, native)
    run.add(f"long: the native context {native} of {model.name}, tested {ctx}, build {current_build()}")
    # the upstream reference builds a mask of the selected cells as large as the KV for every token of a ubatch: at
    # 262144 cells a ubatch of 2048 keeps it at 1 GiB, which fits next to the model on the 395
    long_qargs = split_args(env("QA_LONG_QARGS", "-b 4096 -ub 2048"))
    for stage, fn in (("quality", lambda: quality(run, "quality", ctx, env_ints("QA_LONG_DEPTHS", "4096 32768 65536 131072 196608"),
                                                   long_qargs)),
                      ("chat", lambda: long_chat(run, model, ctx, port)),
                      ("par", lambda: long_par(run, model, ctx, port))):
        if stage not in stages:
            continue
        try:
            fn()
        except (Fail, Skip) as e:
            run.failed = True
            run.add("", f"{stage}: FAIL {e}")
    return run.finish()


# ---------------------------------------------------------------- profile

# the kinds of op of the profile tables, by the start of the op name (the first match counts)
OP_KINDS = [("attention", ("FLASH_ATTN", "SOFT_MAX")),
            ("qsa select", ("LIGHTNING_INDEXER", "MUL_MAT_HEADSUM", "MUL_MAT_RELU", "TOP_K", "TOPK", "ARGSORT")),
            ("moe", ("MUL_MAT_ID", "ADD_ID", "MULTI_ADD")),
            ("matmul", ("MUL_MAT",)),
            ("delta net", ("GATED_DELTA_NET", "SSM_CONV", "CONCAT_SSM_CONV")),
            ("hyper conn", ("DSV4_HC", "HC_POST")),
            ("norm", ("RMS_NORM", "NORM", "L2_NORM")),
            ("rows, copies", ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT", "REPEAT", "DUP")),
            ("elementwise", ("ADD", "SUB", "MUL", "DIV", "SCALE", "SILU", "SIGMOID", "GELU", "RELU", "GLU", "SWIGLU",
                             "EXP", "LOG", "SOFTPLUS", "STEP", "ABS", "SQR", "SUM", "ROPE", "CLAMP", "FILL"))]


def op_kind(name):
    op = op_group(name)
    if op.startswith("FUSED "):
        op = op[6:].split("+")[0]
    for kind, prefixes in OP_KINDS:
        if op.startswith(prefixes):
            return kind
    return "other"


def by_kind(ops):
    out = {}
    for name, _, ms in ops:
        out[op_kind(name)] = out.get(op_kind(name), 0.0) + ms
    return out


def cmd_profile():
    """Where the time goes on the server at each depth of QA_PROF_DEPTHS (tokens): a conversation reaches the depth
    with the prompt cache, then a part of QA_PROF_PART tokens (pp: one prompt ubatch at that depth) and a reply of
    QA_PROF_TG tokens (tg: the decode steps). Vulkan: GGML_VK_PERF_LOGGER, a block per graph compute; CUDA:
    GGML_CUDA_OP_TIMING=idle, a report when the GPU has been idle. LLAMA_INPUT_TIMING gives the host steps.
    QA_PARMS: variants to compare (name VAR=value ...;...), each a run of its own; the tables are those of the first,
    then the totals of every variant side by side."""
    run = Run("profile")
    model = model_path()
    exe = binary("llama-server")
    vulkan = env("BACKEND") == "vulkan"
    ctx, port = env_int("QA_PROF_CTX", 262144), env_int("QA_PORT", 8097)
    # a part of 4000 tokens plus the chat template stays within one prompt ubatch of 4096
    part, ntg = env_int("QA_PROF_PART", 4000), env_int("QA_PROF_TG", 32)
    depths = [d for d in env_ints("QA_PROF_DEPTHS", f"0 65536 131072 {ctx - 3 * part - 2048}") if d + 2 * part + 1024 < ctx]
    args = server_args(model, ctx, 1, port, ["-lv", "4"])
    text = corpus_text()
    arms = variants("QA_PARMS", "default")
    run.add(f"profile: llama-server, {'GGML_VK_PERF_LOGGER' if vulkan else 'GGML_CUDA_OP_TIMING'} per op, depths "
            f"{' '.join(map(str, depths))} tokens, a part of {part} tokens (pp) and a reply of {ntg} tokens (tg), "
            f"{model.name}, build {current_build()}" +
            (", variants " + "; ".join(f"{name} {' '.join(f'{k}={v}' for k, v in e.items())}".strip() for name, e in arms)
             if len(arms) > 1 else ""))

    def go(log, run_env):
        proc, log = start_server(exe, args, run_env, log)
        url = f"http://127.0.0.1:{port}"
        slices = {}
        try:
            ratio = token_ratio(url, text)
            msgs = []

            def ask(name, n, messages=None):
                before = log.stat().st_size
                body = {"messages": msgs if messages is None else messages, "max_tokens": n, "temperature": 0.0,
                        "seed": 1, "cache_prompt": True, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
                try:
                    _, raw = http(url + "/v1/chat/completions", body, env_int("QA_TIMEOUT", 3600))
                except (urllib.error.URLError, OSError) as e:
                    raise Fail(f"{name}: {e} ({log.name})")
                time.sleep(1.5)  # the slot prints its timings as it is released, and CUDA needs the idle gap
                with log.open("rb") as f:
                    f.seek(before)
                    slices[name] = f.read().decode("utf-8", errors="replace")
                answer = json.loads(raw)
                if messages is None:
                    msgs.append({"role": "assistant", "content": answer["choices"][0]["message"].get("content") or ""})
                return answer["timings"]

            offset, depth = 0, 0
            for d in depths:
                # up to the depth in parts of at most 32768 tokens, each a turn of its own
                while depth + 512 < d:
                    n = int(min(32768, d - depth) / ratio)
                    msgs.append({"role": "user", "content": text[offset:offset + n] + "\n\nReply only: OK"})
                    offset += n
                    t = ask(f"fill{d}-{offset}", 1)
                    depth = t.get("cache_n", 0) + t["prompt_n"]
                n = int(part / ratio)
                msgs.append({"role": "user", "content": text[offset:offset + n] + "\n\nWhat is this part about? One word."})
                offset += n
                t = ask(f"pp{d}", 1)
                depth = t.get("cache_n", 0) + t["prompt_n"]
                msgs.append({"role": "user", "content": "Write a long paragraph about it."})
                t = ask(f"tg{d}", ntg)
                depth = t.get("cache_n", 0) + t["prompt_n"] + t["predicted_n"]
                slices[f"depth{d}"] = depth
                # CUDA prints the op report of the reply after an idle gap, at the next compute: a short request
                ask(f"end{d}", 1, [{"role": "user", "content": "ok"}])
        finally:
            proc.kill()
            proc.wait()
            mark_end()
        out = {}
        for d in depths:
            pp_text, tg_text, end_text = slices[f"pp{d}"], slices[f"tg{d}"], slices[f"end{d}"]
            host = [[float(m[2]), float(m[3]), float(m[4])] for m in
                    (re.match(r".*ubatch timing: (\d+) tokens, graph ([\d.]+) ms, inputs ([\d.]+) ms, submit ([\d.]+) ms", x)
                     for x in pp_text.splitlines()) if m and int(m[1]) > 64]
            entry = {"depth": slices[f"depth{d}"], "host": [sum(h[i] for h in host) for i in range(3)] if host else None}
            if vulkan:
                pp_all, tg_all = vulkan_blocks(pp_text), vulkan_blocks(tg_text)
                # with the MTP head the target's graphs have an expert matmul per layer, the head's one layer: the
                # graphs with the most of them are the target's. without a head every graph is the target's
                n_mmid = lambda b: sum(c for op, c, _ in b if op.startswith("MUL_MAT_ID"))
                top = max([n_mmid(b) for b in pp_all + tg_all] or [0])
                is_target = lambda b: "-md" not in args or n_mmid(b) >= top
                pp_blocks = [b for b in pp_all if is_target(b)]
                groups = {}
                def tokens_of(b):
                    # the dst of the attention, else the n of the first plain matmul (the attention can run elsewhere)
                    n = block_kind(b)[1]
                    if n is None:
                        m = next((re.search(r"\bn=(\d+)", op) for op, _, _ in b
                                  if op.startswith(("MUL_MAT ", "MUL_MAT_VEC "))), None)
                        n = int(m[1]) if m else None
                    return n

                for b in tg_all:
                    n = tokens_of(b)
                    if is_target(b) and (n is None or n > 8):
                        continue  # the prompt of the reply
                    label = (("verify of %d tokens" % n if "-md" in args else "decode step") if is_target(b)
                             else "head draft step" if n else "head catch-up")
                    groups.setdefault(label, []).append(b)
                entry["pp"] = sum_ops(pp_blocks)
                entry["pp_graphs"] = len(pp_blocks)
                entry["tg_groups"] = [[label, len(bl), [[o, c / len(bl), t / len(bl)] for o, c, t in sum_ops(bl)]]
                                      for label, bl in sorted(groups.items(), key=lambda g: -len(g[1]))]
            else:
                big = [m.end() for m in re.finditer(r"ubatch timing: (\d+) tokens", pp_text) if int(m[1]) > 1024]
                pp_r = cuda_report((pp_text + tg_text)[big[-1]:] if big else tg_text)
                tg_r = cuda_report(end_text)
                entry["pp"] = pp_r[1] if pp_r else []
                entry["pp_graphs"] = pp_r[0] if pp_r else 0
                entry["tg_groups"] = [["decode, per token", ntg, [[o, c / ntg, t / ntg] for o, c, t in tg_r[1]]]] if tg_r else []
            out[str(d)] = entry
        return out

    results = []
    for name, arm_env in arms:
        run_env = dict(os.environ, LLAMA_INPUT_TIMING="1", **({"GGML_VK_PERF_LOGGER": "1"} if vulkan else {"GGML_CUDA_OP_TIMING": "idle"}),
                       **arm_env)
        identity = {"v": VERSION, "kind": "profile", "build": build_id(exe.parent), "args": args, "env": env_identity(run_env),
                    "model": model_identity(model), "depths": depths, "part": part, "tg": ntg, "corpus": text_digest(text),
                    "machine": machine(), "draft": model_identity(Path(env("SPECDRAFT"))) if "-md" in args else None}
        label = "profile" if name == "default" else f"profile {name}"
        try:
            r_arm, kept_arm = run.measure(label, identity, lambda log, e=run_env: go(log, e))
        except Fail as e:
            run.failed = True
            run.add("", f"{label}: FAIL {e}")
            continue
        results.append((name, r_arm, kept_arm))
    if not results:
        return run.finish()
    name0, r, kept = results[0]
    k = " k" if kept else ""
    kinds = [kind for kind, _ in OP_KINDS] + ["other"]
    pp_cols, tg_cols, heads = {}, {}, []
    for d in depths:
        x = r.get(str(d))
        if not x:
            continue
        heads.append(str(x["depth"]))
        pp_cols[d] = by_kind(x["pp"])
        main = next((g for g in x["tg_groups"] if g[0].startswith(("verify", "decode"))), None)
        tg_cols[d] = by_kind(main[2]) if main else {}
        pp_total = sum(o[2] for o in x["pp"])
        lines = [f"", f"at {x['depth']} tokens: prompt ubatch of {part} tokens {pp_total:.1f} ms of GPU time over "
                      f"{x['pp_graphs']} {'graphs' if vulkan else 'compute calls'}" + k]
        if x.get("host"):
            h = x["host"]
            lines.append(f"  host for the prompt: graph build {h[0]:.1f} ms, inputs {h[1]:.1f} ms, submit {h[2]:.1f} ms")
        for label, n, ops in x["tg_groups"]:
            lines.append(f"  {label}: {sum(o[2] for o in ops):.2f} ms of GPU time (mean of {n})")
        run.add(*lines)
    if pp_cols:
        cols = [d for d in depths if d in pp_cols]
        run.add("", "prompt ubatch: GPU ms per kind of op by depth",
                *table(["kind", *heads], [[kind, *[f"{pp_cols[d].get(kind, 0):.1f}" for d in cols]] for kind in kinds] +
                       [["total", *[f"{sum(pp_cols[d].values()):.1f}" for d in cols]]]))
        run.add("", "decode step of the target (its verify with MTP): GPU ms per kind of op by depth",
                *table(["kind", *heads], [[kind, *[f"{tg_cols[d].get(kind, 0):.2f}" for d in cols]] for kind in kinds] +
                       [["total", *[f"{sum(tg_cols[d].values()):.2f}" for d in cols]]]))
        deepest = r.get(str(cols[-1]))
        run.add("", f"the ops of the prompt ubatch at {deepest['depth']} tokens",
                *table(["  ms", "count", "op"], [[f"  {ms:.2f}", f"{c:g}", name] for name, c, ms in deepest["pp"][:25]]))
        main = next((g for g in deepest["tg_groups"] if g[0].startswith(("verify", "decode"))), None)
        if main:
            run.add("", f"the ops of the decode step at {deepest['depth']} tokens ({main[0]})",
                    *table(["  ms", "count", "op"], [[f"  {ms:.3f}", f"{c:g}", name] for name, c, ms in main[2][:25]]))
    def totals(res, d):
        x = res.get(str(d))
        if not x:
            return None
        main = next((g for g in x["tg_groups"] if g[0].startswith(("verify", "decode"))), None)
        tg = by_kind(main[2]) if main else {}
        return sum(o[2] for o in x["pp"]), sum(tg.values()), tg.get("delta net", 0.0), tg.get("rows, copies", 0.0)

    for name, res, kept_arm in results:
        for d in depths:
            tot = totals(res, d)
            if tot:
                point = f"d{d}" if name == "default" else f"{name} d{d}"
                run.record("profile", point, {"gpu_pp": tot[0], "gpu_tg": tot[1]}, kept_arm, exe.parent)
    if len(results) > 1:
        rows = []
        for name, res, kept_arm in results:
            for d in depths:
                tot = totals(res, d)
                if tot:
                    rows.append([name, str(res[str(d)]["depth"]), f"{tot[0]:.1f}", f"{tot[1]:.2f}", f"{tot[2]:.2f}",
                                 f"{tot[3]:.2f}" + (" k" if kept_arm else "")])
        run.add("", "variants: GPU ms of the prompt ubatch and of the decode step (its verify with MTP), with the delta "
                    "net and the row copies of that step",
                *table(["variant", "depth", "prompt", "decode", "delta net", "rows, copies"], rows))
    return run.finish()


# ---------------------------------------------------------------- main

COMMANDS = {"check": cmd_check, "long": cmd_long, "profile": cmd_profile, "build-begin": cmd_build_begin,
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
