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
import bisect
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
# the end of the last build or long model run, for COOL: a build loads every core for minutes and heats the 395 as a
# long run does
LAST_HEAT = CACHE / "last-heat.txt"
# the start of the model process running now, for mark_end()
RUN_STARTED = [None]
DEVICE_LINE = re.compile(r"^\s*(\S+): .*\((\d+) MiB, (\d+) MiB free\)\s*$")
IDLE_USED = CACHE / "idle-used.json"
# the GPU memory counters of Windows (Task Manager shows them): the memory of all processes on all adapters, also the
# memory the driver has not released yet. The Vulkan memory budget the 395 reports does not see it
GPU_COUNTERS = ("Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | "
                "ForEach-Object { [string]$_.DedicatedUsage + ' ' + [string]$_.SharedUsage }")
# the lines of the crash report of a Windows build (src/fork/llama-load-watch-fork.cpp): the event and the thread, the
# last C++ exception of that thread, the step of a model load; the frames of the stack follow as module+offset
CRASH_EVENT = re.compile(r"^(abort\(\)|an invalid parameter of a C runtime function) in thread ")
CRASH_DETAIL = re.compile(r"^(the last C\+\+ exception of this thread|a model load is at)")
CRASH_FRAME = re.compile(r"^  \S+\+0x[0-9a-f]+$")
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

# builds of other refs in their own worktrees (00-build.bat stable | upstream): the git ref (None: UPSTREAMREF, default
# master), the variables of the source and build folders
REF_BUILDS = {"stable": ("stable", "STABLESRC", "STABLEBUILD"), "upstream": (None, "UPSTREAMSRC", "UPSTREAMBUILD")}

def ref_folder(kind):
    return build_folder(Path(env(REF_BUILDS[kind][2], str(ROOT / f"build-{kind}"))))

def stable_folder():
    return ref_folder("stable")

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

# the values of the fork that upstream does not take, as upstream says them: -lzm dio (rows gathered through the row
# cache of the fork) is on there (rows read on demand from a mapping of the file); 12.11 07 failed on both machines
UPSTREAM_VALUES = {("-lzm", "dio"): "on", ("--lazy-mode", "dio"): "on"}

def upstream_args(args):
    """args for a binary of the upstream build"""
    out = [str(a) for a in args]
    for i in range(1, len(out)):
        out[i] = UPSTREAM_VALUES.get((out[i - 1], out[i]), out[i])
    return out

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
    # that set it and the default shared one result. QA_ARM_ARGS: the server arguments of a variant (start_server),
    # without it every such variant shared the result of the default
    return {k: v for k, v in sorted(run_env.items())
            if (k.upper().startswith(ENV_PREFIXES) or k.upper() == "QA_ARM_ARGS") and k.upper() not in ENV_IGNORED
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
    """COOL seconds (0): at least that long after the last build or long model run (COOL_RUN seconds or more, see
    mark_end()), so that the arms of a run start from about the same temperature (the 395 reached 95 C in an hour of
    chats to 213K on 14.10, the ops of the last profile depth ran 2 percent slower than the night before; the first arm
    after the build of 14.10 ran 3 to 5 percent slower in every phase than the second, also in the steps that read
    nothing from the drive). Returns the seconds it waited."""
    cool = env_int("COOL", 0)
    try:
        heat = time.time() - float(LAST_HEAT.read_text())
    except (OSError, ValueError):
        heat = None
    if cool > 0 and heat is not None and heat < cool:
        print(f"cooling {cool - heat:.0f} s after the last build or long model run (COOL)", flush=True)
        time.sleep(cool - heat)
        return cool - heat
    return 0

def settle(timed=True):
    """Before a model run: wait until the GPU memory of the last one is free again. The driver releases the memory of
    a finished process lazily: on the 395 a run right after another failed on memory or hung while loading
    (08.10-11.10). Done at once when the memory in use is within SETTLE_MARGIN MiB (2048) of the least seen on this
    machine (cache/idle-used.json), else the memory is read every SETTLE_STEP seconds (3) until it stops going down,
    SETTLE_MAX seconds (180) at most. Without a reading the old rule: SETTLE seconds (30) after the last model run.
    timed=False: the run measures no speed (the probes of llama-quality-fork, the checkpoint replay), so it does not
    wait for COOL either; a server (start_server: 01 spec probe, 02, 03) always does."""
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
    """The end of a model run: SETTLE counts from every one, COOL only from one of COOL_RUN seconds (300) or more. The
    arms of a profile at depth 0 (2-3 minutes each, mostly the load) matched to 0.23 percent in the ops they do not
    change, twice (07.11, 395, 120 s apart); a profile arm to the end of the context fills 248K tokens of prompt cache,
    about 8 minutes"""
    CACHE.mkdir(parents=True, exist_ok=True)
    now = time.time()
    LAST_RUN.write_text(f"{now}\n")
    started, RUN_STARTED[0] = RUN_STARTED[0], None
    if started is None or now - started >= env_int("COOL_RUN", 300):
        LAST_HEAT.write_text(f"{now}\n")

def exit_text(code):
    return f"exit {code}" if 0 <= code < 256 else f"exit {code & 0xffffffff:#010x}"

def is_crash_line(s):
    return bool(CRASH_EVENT.match(s) or CRASH_DETAIL.match(s))

def last_error(text):
    # the lines of the crash report are not the error: crash_report() adds them
    lines = [s.strip() for s in text.splitlines() if s.strip() and not is_crash_line(s.strip()) and not CRASH_FRAME.match(s)]
    errors = [s for s in lines if re.search(r"error|exception|failed|abort", s, re.IGNORECASE)]
    if errors:
        return errors[-1][:200]
    return "no error message, the log ends with: " + (lines or [""])[-1][:160]

def lifetime(code, seconds):
    """How long a process that ended on an exception code ran: the fail-fast exit 0xc0000409 of a Windows build writes
    nothing into the log, and the log of a model load is silent from the first buffers to the end of the load. 395,
    31.10-01.11: eight loads ended so after 'mapMemory: ErrorMemoryMapFailed' and the 2 s retry of the buffer."""
    return f" after {seconds:.1f} s" if code & 0xffffffff >= 0xC0000000 else ""

def crash_report(text):
    """The crash report of a Windows build in short: the event and the thread, the last C++ exception of that thread,
    the step of a model load."""
    lines = [s.strip() for s in text.splitlines()]
    for i, s in enumerate(lines):
        if CRASH_EVENT.match(s):
            return "; " + ", ".join([s] + [t for t in lines[i + 1:] if CRASH_DETAIL.match(t)])[:400]
    return ""

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
    # the crash report counts as no message, its exception can still name a memory error
    plain = "\n".join(s for s in text.splitlines() if not is_crash_line(s.strip()) and not CRASH_FRAME.match(s))
    return any(s in text for s in MEMORY_ERRORS) or (code & 0xffffffff >= 0xC0000000 and "error" not in plain.lower())

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
            began = time.time()
            if model and RUN_STARTED[0] is None:
                RUN_STARTED[0] = began
            try:
                proc = subprocess.Popen(cmd, stdout=out, stderr=f, env=run_env, cwd=cwd)
                try:
                    code = proc.wait()
                except BaseException:
                    proc.kill()
                    proc.wait()
                    raise
            finally:
                if stdout:
                    out.close()
        lived = time.time() - began
        if model:
            mark_end()
        text = read_log(log)
        if code == 0 or not check:
            return code, text
        why = f"{last_error(text)}{crash_report(text)}"
        if attempt < 3 and model and retryable(code, text):
            wait = retry_wait(attempt)
            print(f"{log.name}: {exit_text(code)}{lifetime(code, lived)}, {why}; one more try" + (f" in {wait:.0f} s" if wait else ""), flush=True)
            time.sleep(wait)
            log = first.with_name(f"{first.stem}-try{attempt + 1}.log")
            continue
        raise Fail(f"{exit_text(code)}{lifetime(code, lived)}: {why} ({log.name}, try {attempt})")

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

# the llama-server processes this run started: one of them still listed after its kill is waited for, any other fails
OWN_SERVER_PIDS = set()


def server_pids():
    """the process ids of the running llama-server.exe (tasklist, CSV without a header)"""
    out = subprocess.run(["tasklist", "/fi", "imagename eq llama-server.exe", "/fo", "csv", "/nh"],
                         capture_output=True).stdout.decode("utf-8", errors="replace")
    pids = set()
    for row in csv.reader(out.splitlines()):
        if len(row) > 1 and row[0].lower() == "llama-server.exe" and row[1].strip().isdigit():
            pids.add(int(row[1]))
    return pids


def no_server_running():
    """no llama-server.exe may hold GPU memory during a measurement. One this run started and killed can stay listed for
    a while: 12.11 on the 3090 a server that aborted stayed about 90 s (Windows Error Reporting writes the dump of a
    crashed process of 70 GB), and 03 and 07 failed on it. Those are waited for up to QA_SERVER_EXIT_WAIT (300) s, any
    other one fails at once"""
    if os.name != "nt":
        return
    wait = env_int("QA_SERVER_EXIT_WAIT", 300)
    start, told = time.time(), False
    while True:
        pids = server_pids()
        if not pids:
            return
        foreign = pids - OWN_SERVER_PIDS
        waited = int(time.time() - start)
        if foreign or waited > wait:
            raise Fail("a llama-server.exe is running: it holds gpu memory and spoils the numbers, stop it first" +
                       ("" if foreign else f" (pid {' '.join(map(str, sorted(pids)))} of this run, still there {waited} s "
                                           f"after its kill, QA_SERVER_EXIT_WAIT)"))
        if not told:
            print(f"waiting for llama-server.exe pid {' '.join(map(str, sorted(pids)))} of this run to exit (a crashed one "
                  f"stays while Windows writes its dump)", flush=True)
            told = True
        time.sleep(5)

def start_server(exe, args, run_env, log):
    """Start llama-server and wait for /health, with the retries of a load that fails on memory or stalls; returns the
    process and the log it writes (a retry writes log-tryN). QA_ARM_ARGS in the environment of a variant: server
    arguments of that variant only, put after the common ones (a later option wins: QA_ARM_ARGS="--moe-cache 0
    --moe-cache-mib 6500 --moe-cache-cpu-share 0.3" runs the expert cache of upstream in place of the fork's)"""
    run_env = dict(run_env)
    arm_extra = split_args(run_env.pop("QA_ARM_ARGS", ""))
    if arm_extra:
        i_host = args.index("--host") if "--host" in args else len(args)
        args = [*args[:i_host], *arm_extra, *args[i_host:]]
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
        OWN_SERVER_PIDS.add(proc.pid)
        start = time.time()
        if RUN_STARTED[0] is None:
            RUN_STARTED[0] = start
        # a load that stops writing its log hangs: 11.10, 395: mtp3 right after auto stopped in the load of the MTP head
        # (as on 08.10) and the arm failed after DECWAIT; a normal load is silent for 20 s at most
        stall = env_int("DECSTALL", 240)
        last_size, last_growth, stalled = -1, time.time(), False
        while proc.poll() is None:
            # wait() rather than sleep(): the time of an exit is exact
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
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
        lived = time.time() - start
        mark_end()
        text = read_log(log)
        if stalled:
            pause = retry_wait(attempt + 1)
            print(f"{log.name}: no log output for {stall} s while loading{load_watch(log)}; one more try" + (f" in {pause:.0f} s" if pause else ""), flush=True)
            time.sleep(pause)
            log = first.with_name(f"{first.stem}-try{attempt + 1}.log")
            continue
        note = lifetime(proc.returncode, lived)
        why = f"{last_error(text)}{crash_report(text)}"
        if attempt < 3 and retryable(proc.returncode, text):
            pause = retry_wait(attempt)
            print(f"{log.name}: {exit_text(proc.returncode)}{note} while loading, {why}; one more try" + (f" in {pause:.0f} s" if pause else ""), flush=True)
            time.sleep(pause)
            log = first.with_name(f"{first.stem}-try{attempt + 1}.log")
            continue
        raise Fail(f"llama-server {exit_text(proc.returncode)}{note} while loading: {why} ({log.name}, try {attempt})")
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

def cuda_reports(text):
    """The GGML_CUDA_OP_TIMING reports of a log part in order: [(compute calls, [[op, count, ms], ...]), ...]. With
    =idle a report comes at a compute call that starts 300 ms or more after the one before and holds the calls since the
    last report, so the first report of a request holds the end of the request before it (when that one computed); on
    the 3090 the CPU experts make such gaps inside a prompt too, and one request leaves many reports. A report lists its
    40 longest labels and, since 31.10, the rest by op ("OP (rest)"); what the lines of an older report leave out of its
    total is a row named (unlisted)"""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"cuda_op_timing \S+: (\d+) compute calls, \d+ ops, ([\d.]+) ms of kernels", line)
        if not m:
            continue
        ops = []
        for x in lines[i + 1:]:
            r = re.match(r"cuda_op_timing \S+:\s+([\d.]+) ms\s+(\d+) x\s+[\d.]+ us\s+(.+)", x)
            if not r:
                break
            ops.append([r[3].strip(), int(r[2]), float(r[1])])
        rest = float(m[2]) - sum(o[2] for o in ops)
        if rest > 0.01:
            ops.append(["(unlisted)", 0, rest])
        out.append((int(m[1]), ops))
    return out


def cuda_work(this_text, next_text):
    """The work of a request: its reports after the first (which holds the request before) and the first report of the
    next request (the end of this one). (compute calls, [[op, count, ms], ...]) or None"""
    reports = cuda_reports(this_text)[1:] + cuda_reports(next_text)[:1]
    if not reports:
        return None
    total = {}
    for _, ops in reports:
        for op, c, ms in ops:
            x = total.setdefault(op, [0, 0.0])
            x[0] += c
            x[1] += ms
    return (sum(n for n, _ in reports),
            sorted(([op, c, ms] for op, (c, ms) in total.items()), key=lambda o: -o[2]))

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

def build_ref(kind):
    """Build a ref (REF_BUILDS) in its own worktree, with the smaller perplexity buffers."""
    ref, src_var, build_var = REF_BUILDS[kind]
    ref = ref or env("UPSTREAMREF", "master")
    run = Run(f"build-{kind}")
    source = Path(env(src_var, str(ROOT / f"build-{kind}-src"))).resolve()
    build = Path(env(build_var, str(ROOT / f"build-{kind}"))).resolve()
    commit = git("rev-parse", f"{ref}^{{commit}}").decode().strip()
    ppl = Path("tools/perplexity/perplexity.cpp")
    marker = source / f".qsa-{kind}-source"
    if source.exists():
        if not marker.is_file():
            raise Fail(f"{source} exists and is not a worktree of this script, set {src_var} to another folder")
        subprocess.run(["git", "-C", str(source), "checkout", "--force", "--detach", commit], check=True)
    else:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(source), commit], check=True)
    marker.write_text(commit + "\n")
    try:
        (source / ppl).write_bytes(baseline_perplexity(commit))
    except Fail as e:
        if kind == "stable":
            raise
        # upstream moves on: its perplexity keeps its buffers, which the context of 07-kld (QA_KLD_CTX) allows
        run.add(f"{e}: the perplexity of {ref} as it is")
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
    sid = f"{kind}-{commit[:12]}-{patch}"
    (folder / BUILD_RECORD).write_text(json.dumps({"id": sid, "head": commit[:12], "time": now()}) + "\n",
                                       encoding="utf-8")
    run.add(f"{kind} {ref} {commit[:12]} built in {folder}, id {sid}" +
            (": 07-kld.bat compares this build with it" if kind == "upstream" else ""))
    return run.finish()


def cmd_build_stable():
    return build_ref("stable")


def cmd_build_upstream():
    return build_ref("upstream")


# ---------------------------------------------------------------- notes from the logs of a run
#
# the lines of the server and tool logs that the summary of a run names (12.11): the device memory limit of the 395
# (ggml_vk_fork_heap_limit: the heap less what other processes hold, in place of the 95% budget of the driver), a device
# allocation refused at that limit, the KV cache spilled to shared memory or stopped at the RAM floor, the QSA block
# lists of the slots, the slots of the expert cache, the prompt cache giving memory back, a fit refused

NOTE_LIMIT = re.compile(r"device memory heap \d+ of ([\d.]+) MiB, other processes hold ([\d.]+) MiB of it, the driver "
                        r"budgets ([\d.]+) MiB: this process takes up to ([\d.]+) MiB")
NOTE_REFUSED = re.compile(r"device memory heap \d+: ([\d.]+) MiB in use of a limit of ([\d.]+) MiB, ([\d.]+) MiB more asked")
# the refusal of the CUDA backend on Windows (ggml-cuda/fork/device-room-fork.inc, 12.11): the driver would back it with
# system memory
NOTE_REFUSED_CUDA = re.compile(r"([\d.]+) MiB on device \d+ do not fit its free memory \(([\d.]+) MiB free of ([\d.]+) MiB")
NOTE_SPILL = re.compile(r"elastic buffer: (?:the device memory is full \((?:([\d.]+) of ([\d.]+) MiB in use|an allocation "
                        r"failed)|granules from the shared memory)")
NOTE_FLOOR = re.compile(r"elastic buffer: the device memory is full and the system has ([\d.]+) MiB free")
NOTE_QSA = re.compile(r"QSA scores the blocks of each slice's own sequences: (\d+) slices, lists of (\d+) of (\d+) blocks")
NOTE_MOE = re.compile(r"moe_cache: (\S+), \d+ of \d+ MoE layers(?: \(\d+ kept their slots\))?, (\d+) slots per layer, "
                      r"[\d.]+ MiB per slot, ([\d.]+) MiB total, (\d+) MiB were free")
NOTE_MOE_NONE = re.compile(r"moe_cache: (\S+) has (\d+) MiB free, one slot on \d+ layers takes [\d.]+ MiB and (\d+) MiB "
                           r"stay free, no slots")
NOTE_RELIEF = re.compile(r"relieving [\d.]+ MiB from the prompt cache")
NOTE_FIT = re.compile(r"would go to host memory, refused")
NOTE_KEYS = (b"device memory heap", b"elastic buffer", b"QSA scores", b"moe_cache: ", b"relieving", b"refused")


def log_first(log, pattern):
    """the groups of the first match of pattern in a log, None without one"""
    try:
        with Path(log).open("rb") as f:
            for raw in f:
                m = pattern.search(raw.decode("utf-8", errors="replace"))
                if m:
                    return list(m.groups())
    except OSError:
        pass
    return None


def log_notes(folder):
    """the notes of the logs in folder (NOTE_*), as lines for the summary"""
    limits, refused, spills, floors, qsa, moe, relief, fits = [], [], [], [], [], [], [], []
    for log in sorted(Path(folder).glob("*.log")):
        first, n_relief = {}, 0
        try:
            with log.open("rb") as f:
                for raw in f:
                    if not any(k in raw for k in NOTE_KEYS):
                        continue
                    line = raw.decode("utf-8", errors="replace")
                    if NOTE_RELIEF.search(line):
                        n_relief += 1
                    for key, pattern in (("limit", NOTE_LIMIT), ("refused", NOTE_REFUSED),
                                         ("refused_cuda", NOTE_REFUSED_CUDA), ("spill", NOTE_SPILL),
                                         ("floor", NOTE_FLOOR), ("qsa", NOTE_QSA), ("moe", NOTE_MOE),
                                         ("moe0", NOTE_MOE_NONE), ("fit", NOTE_FIT)):
                        if key not in first:
                            m = pattern.search(line)
                            if m:
                                first[key] = m.groups()
        except OSError:
            continue
        name = log.name
        if "limit" in first:
            limits.append([float(x) for x in first["limit"]])
        if "refused" in first:
            use, lim, ask = first["refused"]
            refused.append(f"{name} ({use} MiB in use of {lim}, {ask} more asked)")
        if "refused_cuda" in first:
            ask, free, total = first["refused_cuda"]
            refused.append(f"{name} ({ask} MiB asked, {free} MiB free of {total})")
        if "spill" in first:
            use, lim = first["spill"]
            spills.append(name + (f" (at {use} of {lim} MiB)" if use else ""))
        if "floor" in first:
            floors.append(f"{name} ({first['floor'][0]} MiB of RAM free)")
        if "qsa" in first:
            qsa.append(f"{name} {first['qsa'][0]} slices, {first['qsa'][1]} of {first['qsa'][2]} blocks")
        if "moe" in first:
            dev, n, total, free = first["moe"]
            moe.append(f"{name} {n} slots per layer on {dev}, {total} MiB, {free} MiB were free")
        elif "moe0" in first:
            dev, free, margin = first["moe0"]
            moe.append(f"{name} no slots on {dev}: {free} MiB free, {margin} MiB kept free")
        if n_relief:
            relief.append(f"{name} {n_relief}")
        if "fit" in first:
            fits.append(name)
    out = []
    if limits:
        heap = limits[0][0]
        others, budget, limit = ([x[i] for x in limits] for i in (1, 2, 3))
        span = lambda v: f"{min(v):.0f}" if min(v) == max(v) else f"{min(v):.0f}-{max(v):.0f}"
        out.append(f"device memory: a heap of {heap:.0f} MiB, other processes {span(others)} MiB, the driver budget "
                   f"{span(budget)} MiB: this process up to {span(limit)} MiB ({min(limit) - max(budget):+.0f} MiB over the "
                   f"budget at least), {len(limits)} log(s)")
    for title, items in (("device allocations refused at the limit", refused),
                         ("KV cache in shared memory (elastic spill)", spills),
                         ("KV cache stopped at the RAM floor", floors), ("QSA block lists", qsa), ("expert cache", moe),
                         ("the prompt cache gave memory back (times)", relief),
                         ("fit refused, weights would go to host memory", fits)):
        if items:
            out.append(f"{title}: " + "; ".join(items[:8]) + (f"; {len(items) - 8} more" if len(items) > 8 else ""))
    return out


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
        notes = log_notes(self.dir)
        if notes:
            self.add("", "from the logs of this run:", *("  " + n for n in notes))
        if self.new or self.kept:
            self.add("", f"{self.new} new, {self.kept} kept (k): the same build measured them before. FORCE=1 runs "
                         f"them again, FORCE=<regex> the matching parts")
        self.add(f"logs in {self.dir}")
        print("\n" + "\n".join(self.lines), flush=True)
        return 1 if self.failed else 0


def env_pairs(text, where):
    """VAR=value VAR=value ... as a dict"""
    return env_pairs_tokens(split_args(text), where)


def env_pairs_tokens(tokens, where):
    """the same of tokens already split: a quoted "VAR=a b" keeps its spaces (QA_ARM_ARGS)"""
    out = {}
    for t in tokens:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", t, re.S):
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
            out.append((tokens[0], env_pairs_tokens(tokens[1:], name)))
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
    """the probe positions: the last window positions before each depth (the context end is a depth); and the decode
    ranges of the second run of a variant, the whole windows in batches of decode_batch tokens (the decode path)"""
    depths = sorted({min(ctx, d) for d in depths if d >= window} | {ctx})
    positions = sorted({p for d in depths for p in range(d - window, d)})
    ranges = []
    for d in depths if decode_batch > 0 else []:
        if ranges and ranges[-1][1] >= d - window:
            ranges[-1][1] = d
        else:
            ranges.append([d - window, d])
    return depths, positions, ranges


def probe_rows(prompt, decode, depths, window):
    """per depth and path: the KL to the reference (median, 90th percentile), the share of tokens above 0.01 and above
    0.5, the top-1 agreement and the mean NLL difference. prompt: the samples of the prompt path against the reference
    build; decode: the decode path against the prompt path of the same variant, at the same positions"""
    out = []
    for d in depths:
        for path, samples in (("prompt", prompt), ("decode", decode)):
            s = [x for x in samples if d - window <= x["position"] < d]
            if not s:
                continue
            kl = stats([x["kl"] for x in s])
            out.append({"depth": d, "path": path, "n": len(s), "kl_median": kl["median"],
                        "kl_p90": kl["p90"], "kl_max": kl["max"],
                        "over_001": sum(x["kl"] > 0.01 for x in s) / len(s), "over_05": sum(x["kl"] > 0.5 for x in s),
                        "top1": sum(bool(x["top1_equal"]) for x in s) / len(s),
                        "dnll": statistics.mean(x["delta_nll"] for x in s)})
    return out


def probe_flags(rows, ref_p90):
    """per depth: DECODE DIFFERS where the decode path of a variant strays from its prompt path more than the decode
    path of the reference build strays from its own on the same machine (ref_p90, per depth; none without the decode
    run of the reference, QA_REF_DECODE=0): the 90th percentile of the KL above QA_DECODE_RATIO (2) times that of the
    reference. PROMPT DIFFERS (04.11) where the prompt path of a variant strays from the reference build more than
    QA_PROMPT_RATIO (2) times the same noise: until then the prompt path had LOW TOP1 alone, and a shift of the
    distribution that kept the argmax on 85% of the probes passed with code 0. On the 3090 (02.11) the prompt path of the
    default build was 0.75 - 1.26 times the decode noise of the reference at the same depths (0.011 - 0.046 against
    0.014 - 0.052). How far the two paths part by rounding depends on the machine and the model: the test model on the Mac
    Vulkan0 (top-k 16, 31.10) 0.001 - 0.004 with up to 4 tokens above 0.5 of 128, the CPU 0; the real model 0.012 -
    0.074 on the 3090 and the 395 (02.11), where a fixed bar of 0.02 from the Mac fired at every depth. The decode fault
    of 13.10 gave 4-5; a variant run twice, 1.4 times apart at one depth (the 3090, 02.11, before the stable radix-select
    of the block top-k). LOW TOP1 where the prompt path agrees with the reference build on less than 85% of the tokens
    (the noise of the upstream copy on the real model reached 84% with 32 probes, 31.10). Before 31.10 the two paths
    took the two halves of a window, both against the reference: the halves differ in text, and the flag fired on that
    (395, 32768)"""
    flags = {}
    ratio = float(env("QA_DECODE_RATIO", "2"))
    pratio = float(env("QA_PROMPT_RATIO", "2"))
    for r in rows:
        f = []
        if r["path"] == "decode" and r["depth"] in ref_p90 and r["kl_p90"] > ratio * ref_p90[r["depth"]]:
            f.append("DECODE DIFFERS")
        if r["path"] == "prompt" and r["depth"] in ref_p90 and r["kl_p90"] > pratio * ref_p90[r["depth"]]:
            f.append("PROMPT DIFFERS")
        if r["path"] == "prompt" and r["top1"] < 0.85:
            f.append("LOW TOP1")
        if f:
            flags[(r["depth"], r["path"])] = ", ".join(f)
    return flags


def quality(run, label, ctx, depths, qargs=None):
    """The logits at the probe positions, for the default settings and the variants of QA_QARMS: the prompt path
    against the reference (REF_ENV; the same batches, qargs, else QA_QARGS), which also keeps the variant's own logits;
    then a second run of the variant through the decode path (batches of QA_DECODE_BATCH tokens) at the same positions
    against them. The reference takes the decode path as well, against its own prompt path: the rounding between the
    two paths on this machine, the measure of the decode rows (probe_flags). Returns True when nothing was flagged."""
    qargs = quality_args() if qargs is None else qargs
    model = model_path()
    exe = binary("llama-quality-fork")
    native, vocab = native_context(run, exe, model)
    ctx = native if ctx <= 0 else ctx
    if ctx > native:
        raise Fail(f"the context {ctx} is past the native context of the model, {native}")
    window, dbatch = env_int("QA_WINDOW", 128), env_int("QA_DECODE_BATCH", 3)
    depths, positions, ranges = probe_plan(ctx, depths, window, dbatch)
    tokens = corpus_tokens(run, exe, model, ctx + 1)
    plan = {"context": ctx, "positions": positions, "tokens": str(tokens), "max_outputs": 32,
            "decode_batch": dbatch, "decode_ranges": []}
    base = {"v": VERSION, "kind": "quality", "build": build_id(exe.parent), "model": model_identity(model),
            "tokens": digest(tokens), "plan": {k: v for k, v in plan.items() if k != "tokens"}, "decode_ranges": ranges,
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
            f"context of {ctx} tokens: prompt - the prompt batch against the reference ({env('QA_REF_ENV', REF_ENV)}); "
            f"decode - the same positions in batches of {dbatch} (the decode path) against the prompt path of the "
            f"same variant; p90/ref - its 90th percentile over that of the decode path of the reference (the rounding "
            f"between two paths on this machine)")
    clean = True
    header = ["depth", "path", "n", "KL median", "KL p90", "KL max", ">0.01", ">0.5", "top-1", "dNLL"]

    def table_row(r):
        return [r["depth"], r["path"], r["n"], f"{r['kl_median']:.6f}", f"{r['kl_p90']:.5f}", f"{r['kl_max']:.3f}",
                f"{100 * r['over_001']:.1f}%", r["over_05"], f"{100 * r['top1']:.1f}%", f"{r['dnll']:+.4f}"]

    def record_rows(name, rows, kept):
        for r in rows:
            run.record(label, f"{name} d{r['depth']} {r['path']}", {"kl_median": r["kl_median"], "kl_p90": r["kl_p90"],
                       "top1": r["top1"], "dnll": r["dnll"]}, kept, exe.parent)

    # the decode path of the reference against its own prompt path (the reference file)
    ref_p90 = {}
    if ranges and env("QA_REF_DECODE", "1") != "0":
        def go_ref(log):
            rd = tool_run(run, exe, model, f"{label}-reference-decode",
                          dict(plan, action="compare", reference=str(ref), decode_ranges=ranges), ref_env, qargs=qargs)
            if rd["processed"] != ctx or len(rd["samples"]) != len(positions):
                raise Fail(f"the decode run stopped at {rd['processed']} of {ctx} tokens")
            return probe_rows([], [x for x in rd["samples"] if x.get("decode")], depths, window)

        try:
            ref_rows, kept = run.measure(f"{label}-reference-decode",
                                         {**ref_id, "kind": "quality-reference-decode", "reference": digest(ref)},
                                         go_ref)
            ref_p90 = {r["depth"]: r["kl_p90"] for r in ref_rows}
            run.add("", f"{label} reference ({env('QA_REF_ENV', REF_ENV)})" + (" k" if kept else ""),
                    *table(header, [table_row(r) for r in ref_rows]))
            record_rows("reference", ref_rows, kept)
        except Fail as e:
            run.failed = True
            clean = False
            run.add("", f"{label} reference decode: FAIL {e}")

    def vs_ref(r):
        p = ref_p90.get(r["depth"])
        if p is None:
            return ""
        return f"{r['kl_p90'] / p:.2f}" if p > 0 else ("-" if r["kl_p90"] == 0 else "inf")

    same = {}  # per variant: the NLL of every probe on both paths, to compare with default bit for bit
    for name, venv in variants("QA_QARMS", "default"):
        run_env = dict(os.environ, **venv)
        identity = {**base, "env": env_identity(run_env), "reference": digest(ref)}

        def go(log, name=name, run_env=run_env):
            # the variant's own logits of the prompt path, the reference of its decode run; large, so only until then
            own = folder / ("own-" + re.sub(r"[^\w.-]+", "_", name) + ".ref")
            try:
                r = tool_run(run, exe, model, f"{label}-{name}",
                             dict(plan, action="compare", reference=str(ref), capture=str(own)), run_env, qargs=qargs)
                if r["processed"] != ctx or len(r["samples"]) != len(positions):
                    raise Fail(f"the probe run stopped at {r['processed']} of {ctx} tokens")
                dec = []
                if ranges:
                    rd = tool_run(run, exe, model, f"{label}-{name}-decode",
                                  dict(plan, action="compare", reference=str(own), decode_ranges=ranges), run_env,
                                  qargs=qargs)
                    if rd["processed"] != ctx or len(rd["samples"]) != len(positions):
                        raise Fail(f"the decode run stopped at {rd['processed']} of {ctx} tokens")
                    dec = [x for x in rd["samples"] if x.get("decode")]
            finally:
                own.unlink(missing_ok=True)
            return {"rows": probe_rows(r["samples"], dec, depths, window),
                    "positions": [x["position"] for x in r["samples"]],
                    "nll": [x["nll"] for x in r["samples"]], "nll_decode": [x["nll"] for x in dec]}

        try:
            res, kept = run.measure(f"{label}-{name}", identity, go)
        except Fail as e:
            run.failed = True
            clean = False
            run.add("", f"{label} {name}: FAIL {e}")
            continue
        # a result kept before 01.11 holds the rows alone
        rows = res["rows"] if isinstance(res, dict) else res
        if isinstance(res, dict):
            same[name] = res
        flags = probe_flags(rows, ref_p90)
        clean &= not flags
        run.add("", f"{label} {name}" + (" k" if kept else ""),
                *table(header + ["p90/ref", ""],
                       [table_row(r) + [vs_ref(r), flags.get((r["depth"], r["path"]), "")] for r in rows]))
        record_rows(name, rows, kept)
    # which variants give the logits of default: equal logits give an equal NLL. a second default (an unread variable)
    # shows whether a run repeats itself, as the radix-select of the block top-k did not before 01.11
    base_nll = same.get("default")
    others = [n for n in same if n != "default"]
    if base_nll and others:
        cmp_rows = []
        for n in others:
            o = same[n]
            pp = [a == b for a, b in zip(o["nll"], base_nll["nll"])]
            dd = [a == b for a, b in zip(o["nll_decode"], base_nll["nll_decode"])]
            first = next((base_nll["positions"][i] for i, eq in enumerate(pp) if not eq), None)
            cmp_rows.append([n, f"{sum(pp)}/{len(pp)}", f"{sum(dd)}/{len(dd)}", "-" if first is None else first])
        run.add("", f"{label}: probes with the NLL of default bit for bit",
                *table(["variant", "prompt path", "decode path", "first other probe"], cmp_rows))
    if not clean:
        run.failed = True
    return clean


def cmd_check():
    run = Run("check")
    check_ops(run)
    if env("QA_CHECK_GUARDS", "1") != "0":
        try:
            guards_probe(run, model_path())
        except Skip as e:
            run.add("", f"guards: skipped, {e}")
        except Fail as e:
            run.failed = True
            run.add("", f"guards: FAIL {e}")
    if env("QA_CHECK_QUALITY", "1") != "0":
        try:
            quality(run, "quality", env_int("QA_CHECK_CTX", 32768), env_ints("QA_CHECK_DEPTHS", "4096 16384"))
        except (Fail, Skip) as e:
            run.failed = True
            run.add("", f"quality: FAIL {e}")
    if env("QA_CHECK_CKPT", "1") != "0":
        try:
            ckpt_probe(run, model_path())
        except Skip as e:
            run.add("", f"ckpt: skipped, {e}")
        except Fail as e:
            run.failed = True
            run.add("", f"ckpt: FAIL {e}")
    if env("QA_CHECK_SPEC", "1") != "0":
        try:
            spec_probe(run, model_path(), env_int("QA_PORT", 8097))
        except Skip as e:
            run.add("", f"spec: skipped, {e}")
        except Fail as e:
            run.failed = True
            run.add("", f"spec: FAIL {e}")
    # the draft acceptance on fixed text, on request (QA_CHECK_FORCED=1): an A/B of kernels the drafts go through
    if env("QA_CHECK_FORCED", "0") != "0":
        try:
            forced_probe(run, model_path(), env_int("QA_PORT", 8097))
        except Skip as e:
            run.add("", f"forced: skipped, {e}")
        except Fail as e:
            run.failed = True
            run.add("", f"forced: FAIL {e}")
    return run.finish()


# ---------------------------------------------------------------- context checkpoints
#
# between the turns of a conversation the server restores a partial checkpoint (the recurrent state, the attention goes
# back with seq_rm) and reads on from it. test-checkpoint-partial (tests/) reads a prompt in two halves with a checkpoint
# after the first, restores it, reads the second half again and compares the logits of the last token with the
# uninterrupted run; ctest runs it on generated models only, here it runs on the GPU of the machine (04.11, a review)

def ckpt_probe(run, model):
    """test-checkpoint-partial with the model as the server loads it (EXTRA, LOADMODE, DECARGS) on QA_CKPT_TOKENS tokens
    of the corpus: the second half is past the QSA budget (513 blocks of 4 tokens), so the replay takes the sparse path"""
    exe = binary("test-checkpoint-partial")
    corpus = Path(env("QA_CORPUS", env("PPLFILE")))
    if not corpus.is_file():
        raise Fail(f"no corpus {corpus}: set QA_CORPUS to a long text file (or run get-wikitext.bat for PPLFILE)")
    n = env_int("QA_CKPT_TOKENS", 8192)
    # the test parses the common arguments (LLAMA_EXAMPLE_COMMON), which have no --no-warmup: it warms up with one empty run
    args = [str(a) for a in [*model_args(model), *split_args(env("DECARGS")), "-f", corpus, "--tokens", n]
            if a != "--no-warmup"]
    identity = {"v": VERSION, "kind": "ckpt", "build": build_id(exe.parent), "args": args, "model": model_identity(model),
                "corpus": digest(corpus), "env": env_identity(os.environ), "machine": machine()}

    def go(log):
        # a replay that differs exits with 1 and its line, execute() raises it
        _, text = execute([exe, *args], log, os.environ.copy(), timed=False)
        m = re.search(r"partial checkpoint replay passed: checkpoint = (\d+) bytes, max logit diff = (\S+)", text)
        if not m:
            raise Fail(f"no result line ({log.name})")
        return {"bytes": int(m.group(1)), "diff": float(m.group(2))}

    r, kept = run.measure("ckpt", identity, go)
    same = "bit for bit" if r["diff"] == 0 else f"max diff {r['diff']:g} (the test allows 1e-2)"
    run.add("", f"ckpt: {n} tokens of the corpus, a partial checkpoint ({r['bytes'] / 2**20:.1f} MiB) after the first "
                f"half, the second half again after its restore: the logits of the last token {same} with the "
                f"uninterrupted run" + (" k" if kept else ""))
    run.record("check", "ckpt", {"diff": r["diff"], "mib": r["bytes"] / 2**20}, kept, exe.parent)


# ---------------------------------------------------------------- guards of the device memory (12.11)
#
# a model must not land in system memory unasked: on 12.11 a load of the 395 went to the RAM, the system swapped and froze
# until the server fell over. 01 checks the guards on the machine. Vulkan: a load with no device memory allowed
# (GGML_VK_DEVICE_MARGIN_MIB above the heap) fails with the heap numbers, where the UMA fallback took host memory before;
# a fit that would put weights on the host is refused and the model loads on the device; the log has the limit line of
# ggml_vk_fork_heap_limit (Windows). CUDA: a load larger than the card fails, where the sysmem fallback of the NVIDIA
# driver (a setting of the driver) backs the rest with system memory and the runs go 25 times slower (02.11)

GUARD_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
GUARD_FREE = re.compile(r"using device (\S+) \(.*- (\d+) MiB free")
GUARD_BUFFER = re.compile(r"(\S+) model buffer size =\s*([\d.]+) MiB")


def guard_server(exe, args, run_env, log):
    """a server of a guard, once (no retries: a failed load is the answer): (healthy, exit code, log text). A server that
    answers /health is stopped at once"""
    args = [str(a) for a in args]
    url = f"http://127.0.0.1:{args[args.index('--port') + 1]}"
    no_server_running()
    settle(timed=False)
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"command": [str(exe), *args], "env": env_identity(run_env)}) + "\n")
        f.flush()
        proc = subprocess.Popen([str(exe), *args], stdout=f, stderr=f, env=run_env)
    OWN_SERVER_PIDS.add(proc.pid)
    start, healthy = time.time(), False
    if RUN_STARTED[0] is None:
        RUN_STARTED[0] = start
    try:
        while proc.poll() is None and time.time() - start < env_int("DECWAIT", 900):
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            try:
                if http(url + "/health")[0] == 200:
                    healthy = True
                    break
            except (urllib.error.URLError, OSError):
                pass
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        mark_end()
    return healthy, proc.returncode, read_log(log)


def guard_read(healthy, code, text):
    """what a guard needs from the log of its server"""
    off, free = GUARD_OFFLOAD.search(text), GUARD_FREE.search(text)
    refused, limit = NOTE_REFUSED.search(text) or NOTE_REFUSED_CUDA.search(text), NOTE_LIMIT.search(text)
    return {"healthy": healthy, "code": code, "error": "" if healthy else f"{last_error(text)}{crash_report(text)}",
            "memory_error": any(s in text for s in MEMORY_ERRORS),
            "refused": list(refused.groups()) if refused else None,
            "fit_refused": bool(NOTE_FIT.search(text)),
            "offload": [int(off[1]), int(off[2])] if off else None,
            "limit": list(limit.groups()) if limit else None,
            "free": [free[1], int(free[2])] if free else None,
            "buffers": [[b, float(mib)] for b, mib in GUARD_BUFFER.findall(text)]}


def guards_probe(run, model):
    exe = binary("llama-server")
    port = env_int("QA_PORT", 8097)
    backend = env("BACKEND")
    if backend not in ("vulkan", "cuda"):
        raise Skip(f"no guards for the backend {backend}")
    # -lv 4: the INFO lines of the library (the layers on the GPU, the buffers, the free memory) are TRACE in a server
    head = ["-m", model, "-c", 4096, "-np", 1, "-fa", "on", *split_args(env("LOADMODE")), *warmup_args(), "-lv", 4]
    tail = ["--host", "127.0.0.1", "--port", port]
    run.add("", "guards: no weights in the system memory unless asked (12.11)")
    if backend == "cuda":
        # the weights of -ncmoe QA_GUARD_NCMOE layers on the card take more than it has: the backend refuses them
        # (ggml-cuda/fork/device-room-fork.inc) whatever the sysmem fallback policy of the driver, which stays as it is
        n_cpu = env("QA_GUARD_NCMOE", "30")
        args = [str(a) for a in [*head, "-ngl", 99, "-ncmoe", n_cpu, "--no-repack", "-fit", "off", *tail]]
        identity = {"v": VERSION, "kind": "guard", "guard": "sysmem", "build": build_id(exe.parent), "args": args,
                    "model": model_identity(model), "env": env_identity(os.environ), "machine": machine()}
        what = f"no room (-ncmoe {n_cpu}: the expert layers on the card take more than it has, QA_GUARD_NCMOE)"
        try:
            g, kept = run.measure("guard sysmem", identity,
                                  lambda log: guard_read(*guard_server(exe, args, os.environ.copy(), log)))
        except Fail as e:
            run.failed = True
            run.add(f"  {what}: FAIL {e}")
            return
        k = " k" if kept else ""
        if not g["healthy"] and g["refused"]:
            ask, free, total = g["refused"]
            run.add(f"  {what}: the backend refused {ask} MiB with {free} MiB free of {total}, the load failed: ok{k}")
            return
        if not g["healthy"] and g["memory_error"]:
            run.add(f"  {what}: the load failed on the device memory without the refusal of the backend (the driver "
                    f"refused it): ok{k}")
            return
        run.failed = True
        if not g["healthy"]:
            run.add(f"  {what}: FAIL, the load failed otherwise: {g['error']}{k}")
            return
        dev = g["free"][0] if g["free"] else "CUDA0"
        on_card = sum(mib for b, mib in g["buffers"] if b == dev)
        if g["free"] and on_card > g["free"][1]:
            run.add(f"  {what}: FAIL, the weights on {dev} take {on_card:.0f} MiB with {g['free'][1]} MiB free and the "
                    f"server loaded: the driver backs the rest with system memory, the backend did not refuse it "
                    f"(GGML_CUDA_ALLOW_SYSMEM_FALLBACK set, or a build before 12.11){k}")
        elif g["free"] and g["buffers"]:
            run.add(f"  {what}: FAIL, inconclusive: the weights on {dev} take {on_card:.0f} MiB of {g['free'][1]} MiB free, "
                    f"they fit the card; a lower QA_GUARD_NCMOE puts more of them there{k}")
        else:
            run.add(f"  {what}: FAIL, the server loaded (no buffer lines in the log): most likely the driver backs the "
                    f"card with system memory{k}")
        return
    limit = None
    for name, extra_env, extra_args in (("no room", {"GGML_VK_DEVICE_MARGIN_MIB": "1048576"}, ["-ngl", 99, "-fit", "off"]),
                                        ("fit", {}, ["-fit", "on", "-fitt", 1048576])):
        run_env = dict(os.environ, **extra_env)
        args = [str(a) for a in [*head, *extra_args, *tail]]
        identity = {"v": VERSION, "kind": "guard", "guard": name, "build": build_id(exe.parent), "args": args,
                    "model": model_identity(model), "env": env_identity(run_env), "machine": machine()}
        try:
            g, kept = run.measure(f"guard {name}", identity,
                                  lambda log, a=args, e=run_env: guard_read(*guard_server(exe, a, e, log)))
        except Fail as e:
            run.failed = True
            run.add(f"  {name}: FAIL {e}")
            continue
        k = " k" if kept else ""
        limit = limit or g["limit"]
        if name == "no room":
            what = "no room (GGML_VK_DEVICE_MARGIN_MIB=1048576: no device allocation fits)"
            if g["healthy"]:
                run.failed = True
                run.add(f"  {what}: FAIL, the model loaded with no device memory allowed: it went to the system memory{k}")
            elif g["refused"]:
                use, lim, ask = g["refused"]
                run.add(f"  {what}: the load failed at the limit ({use} MiB in use of {lim}, {ask} MiB more asked): ok{k}")
            else:
                run.failed = True
                run.add(f"  {what}: FAIL, the load failed otherwise: {g['error']}{k}")
            continue
        what = "fit (-fit on -fitt 1048576: fit would put every layer on the host)"
        off = g["offload"]
        if not g["healthy"]:
            run.failed = True
            run.add(f"  {what}: FAIL, the load failed: {g['error']}{k}")
        elif not g["fit_refused"]:
            run.failed = True
            run.add(f"  {what}: FAIL, no refusal of the fit in the log{k}")
        elif not off or off[0] < off[1]:
            run.failed = True
            run.add(f"  {what}: FAIL, the fit was refused, but " +
                    (f"{off[0]} of {off[1]} layers are on the GPU" if off else "no line of the layers on the GPU") + k)
        else:
            run.add(f"  {what}: refused, the model loaded as asked, {off[0]}/{off[1]} layers on the GPU: ok{k}")
    if limit:
        heap, others, budget, lim = (float(x) for x in limit)
        run.add(f"  device memory: a heap of {heap:.0f} MiB, other processes {others:.0f} MiB, the driver budget "
                f"{budget:.0f} MiB: this process up to {lim:.0f} MiB ({lim - budget:+.0f} MiB over the budget)")
    elif os.name == "nt":
        run.failed = True
        run.add("  device memory: FAIL, no limit line: the counter of the adapter (\\GPU Adapter Memory(*)\\Dedicated "
                "Usage) was not found, the budget of the driver (95% of the heap) applies")
    else:
        run.add("  device memory: no limit line (not Windows): the budget of the driver applies")


# ---------------------------------------------------------------- speculative decoding
#
# the probes of quality() compare the logits of the target alone (llama-quality-fork drafts nothing): the drafts of the MTP
# head, the rejection sampling of the verify and the rollbacks are not in them (04.11, a review). spec_probe runs the server
# as serve.bat does and asks it with and without drafts ("speculative.n_max": 0, a field of the fork)

def chi2_sf(x, df):
    """P(X > x) of a chi-square of df degrees of freedom: the regularized upper incomplete gamma Q(df/2, x/2)"""
    if x <= 0:
        return 1.0
    a, z = df / 2.0, x / 2.0
    gln = math.lgamma(a)
    if z < a + 1:
        # the series of P
        term = total = 1.0 / a
        n = a
        for _ in range(100000):
            n += 1
            term *= z / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - gln))
    # the continued fraction of Q (Lentz)
    tiny = 1e-300
    b = z + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - gln) * h


def pit_values(tokens, probs, rng):
    """the randomized probability integral transform of each emitted token under the distribution it was drawn from (the
    top n_probs of it in their order, the rest one bucket): uniform on [0, 1) when the tokens follow that distribution.
    Returns the values and the count of tokens outside it (missing from a top that holds all of it)"""
    out, outside = [], 0
    for tok, top in zip(tokens, probs):
        lo, p_tok = 0.0, None
        for e in top:
            p = math.exp(e["logprob"])
            if e["id"] == tok:
                p_tok = p
                break
            lo += p
        if p_tok is None:
            p_tok = max(0.0, 1.0 - lo)
            if p_tok < 1e-6:
                outside += 1
                continue
        out.append(min(1.0 - 1e-12, lo + rng.random() * p_tok))
    return out, outside


def uniform_test(values, bins=20):
    """values in [0, 1) against the uniform distribution: the chi-square over equal bins (its statistic and p-value) for the
    shape, and the p-value of their mean against 1/2 (a z-test, the variance of a uniform 1/12) for a shift, which a
    leaning of the output to the tokens the draft favours makes and the bins see later"""
    n = len(values)
    if n < 5 * bins:
        return None, None, None
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int(v * bins))] += 1
    e = n / bins
    stat = sum((c - e) ** 2 / e for c in counts)
    z = (statistics.mean(values) - 0.5) / math.sqrt(1.0 / (12.0 * n))
    return stat, chi2_sf(stat, bins - 1), math.erfc(abs(z) / math.sqrt(2.0))


def spec_prompts(n, chars):
    """n parts of the corpus of about `chars` characters, spread over it"""
    text = corpus_text()
    step = max(1, (len(text) - chars) // max(1, n))
    return [text[i * step: i * step + chars] for i in range(n)]


def spec_completion(url, body):
    try:
        _, raw = http(url + "/completion", body, env_int("QA_TIMEOUT", 3600))
    except urllib.error.HTTPError as e:
        raise Fail(f"spec: HTTP {e.code} {e.read()[:300]!r}")
    r = json.loads(raw)
    if "tokens" not in r or "completion_probabilities" not in r or "timings" not in r:
        raise Fail(f"spec: no tokens, probabilities or timings in the answer {raw[:300]!r}")
    return r


def logprob_of(top, tok):
    """the logprob of token tok in a list of top_logprobs, None when the list lacks it"""
    return next((e["logprob"] for e in top if e["id"] == tok), None)


def gap_shift(top_a, top_b):
    """the gap of the top two tokens of top_a, and that gap where top_b gives the same two tokens (nats; None when
    top_b lacks one): at a position of the same context, how far the two paths part"""
    if len(top_a) < 2:
        return None, None
    a = top_a[0]["logprob"] - top_a[1]["logprob"]
    b1, b2 = logprob_of(top_b, top_a[0]["id"]), logprob_of(top_b, top_a[1]["id"])
    return a, None if b1 is None or b2 is None else b1 - b2


def spec_greedy(url, prompts, n_predict, n_probs):
    """temperature 0 with and without drafts on one server: the tokens agree up to the first one where the two paths of
    the model round apart, where the margin of the top token over the second (nats, without drafts) is small. A fault of
    the drafts, the verify or a rollback parts them also where the model is sure of the next token. Up to that token the
    context is the same, so the two distributions compare at every position: the gap of the top two tokens of the run
    without drafts against the gap of the same two tokens with drafts (the verify path) there (shift: how far the paths
    part, at the first other token and the largest before it), and every token with drafts must be the top one of its
    own distribution at temperature 0 (not greedy: the acceptance took another, a fault whatever the rounding)"""
    rows = []
    for i, text in enumerate(prompts):
        body = {"prompt": text, "n_predict": n_predict, "temperature": 0, "n_probs": n_probs, "return_tokens": True,
                "cache_prompt": True}
        base = spec_completion(url, dict(body, **{"speculative.n_max": 0}))
        spec = spec_completion(url, body)
        a, b = base["tokens"], spec["tokens"]
        pa = [x["top_logprobs"] for x in base["completion_probabilities"]]
        pb = [x["top_logprobs"] for x in spec["completion_probabilities"]]
        first = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        margin = spec_gap = None
        if first is not None and first < len(pa):
            margin, spec_gap = gap_shift(pa[first], pb[first] if first < len(pb) else [])
            if margin is None:
                margin = float("inf")
        end = min(len(pa), len(pb), len(a), len(b), len(a) if first is None else first)
        shifts = [g[0] - g[1] for g in (gap_shift(pa[k], pb[k]) for k in range(end)) if g[0] is not None and g[1] is not None]
        not_greedy = [k for k in range(min(len(b), len(pb), end + 1)) if pb[k] and pb[k][0]["id"] != b[k]]
        base_not_greedy = [k for k in range(min(len(a), len(pa))) if pa[k] and pa[k][0]["id"] != a[k]]
        t = spec["timings"]
        rows.append({"prompt": i, "n": len(a), "same": len(a) if first is None else first, "margin": margin,
                     "spec_gap": spec_gap, "shift_max": max((abs(x) for x in shifts), default=None),
                     "not_greedy": not_greedy[:1], "base_not_greedy": base_not_greedy[:1],
                     "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0)})
    return rows


def spec_pit(url, prompts, n_predict, n_probs, seed):
    """temperature 1 with the temperature alone in the chain: the probabilities the server gives for every token (before
    sampling, the only ones it gives for a drafted token) are then the distribution it was drawn from, and the transform
    of pit_values is uniform. With drafts every token after the first goes through the verify of the MTP head, its
    rejection sampling (common_sampler_sample_and_accept_n_rejection) on the q of the head; without them, the control"""
    arms = {}
    for name, extra in (("spec", {}), ("nospec", {"speculative.n_max": 0})):
        rng = random.Random(seed)
        values, outside, draft_n, draft_acc, n_tok = [], 0, 0, 0, 0
        for i, text in enumerate(prompts):
            r = spec_completion(url, {"prompt": text, "n_predict": n_predict, "temperature": 1.0, "samplers": ["temperature"],
                                      "n_probs": n_probs, "return_tokens": True, "cache_prompt": True, "seed": seed + i,
                                      **extra})
            v, o = pit_values(r["tokens"], [x["top_logprobs"] for x in r["completion_probabilities"]], rng)
            values += v
            outside += o
            n_tok += len(r["tokens"])
            draft_n += r["timings"].get("draft_n", 0)
            draft_acc += r["timings"].get("draft_n_accepted", 0)
        stat, p, p_mean = uniform_test(values)
        arms[name] = {"tokens": n_tok, "outside": outside, "chi2": stat, "p": p, "p_mean": p_mean,
                      "mean": statistics.mean(values) if values else None, "draft_n": draft_n, "draft_acc": draft_acc}
    return arms


def spec_flags(greedy, pit):
    """SPEC GREEDY DIFFERS: temperature 0 parts with and without drafts where the model without them was sure of the
    token, a margin above QA_SPEC_MARGIN (1.0) nats. SPEC NOT GREEDY: a token with drafts that is not the top one of the
    distribution the server gives for it (the verify path), at temperature 0 a fault of the acceptance. SPEC SAMPLING DIFFERS: the tokens with drafts do not follow the
    distribution they were drawn from (the shape or the mean of uniform_test below QA_SPEC_ALPHA, 1e-4, each) or one came
    out that it does not have. SPEC PROBE DIFFERS: the same without drafts, so the probe itself does not hold. NO DRAFTS: nothing drafted"""
    margin, alpha = float(env("QA_SPEC_MARGIN", "1.0")), float(env("QA_SPEC_ALPHA", "1e-4"))
    flags = []
    for r in greedy:
        if r["margin"] is not None and r["margin"] > margin:
            flags.append(f"SPEC GREEDY DIFFERS (prompt {r['prompt']}, token {r['same']}, margin {r['margin']:.2f})")
        if r.get("not_greedy"):
            flags.append(f"SPEC NOT GREEDY (prompt {r['prompt']}, token {r['not_greedy'][0]}: not the top token of its own "
                         f"distribution at temperature 0)")
        if r.get("base_not_greedy"):
            flags.append(f"SPEC PROBE NOT GREEDY (prompt {r['prompt']}, token {r['base_not_greedy'][0]} without drafts)")
    if not any(r["draft_n"] for r in greedy) or not pit["spec"]["draft_n"]:
        flags.append("NO DRAFTS")
    def off(a):
        return a["outside"] or any(a.get(k) is not None and a[k] < alpha for k in ("p", "p_mean"))

    if off(pit["spec"]):
        flags.append("SPEC SAMPLING DIFFERS")
    if off(pit["nospec"]):
        flags.append("SPEC PROBE DIFFERS")
    return flags


def spec_probe(run, model, port):
    """the speculative decoding of the server as serve.bat runs it (SPECDRAFT, QA_SPEC): greedy tokens with and without
    drafts, and the calibration of sampled tokens with and without them (spec_greedy, spec_pit, spec_flags)"""
    if not env("SPECDRAFT") or env("QA_SPEC", "--spec-auto") == "none":
        raise Skip("no MTP head (SPECDRAFT) or QA_SPEC=none")
    exe = binary("llama-server")
    ctx = env_int("QA_SPEC_CTX", 8192)
    args = server_args(model, ctx, 1, port)
    n_greedy, n_tokens, n_probs = env_int("QA_SPEC_GREEDY", 64), env_int("QA_SPEC_TOKENS", 2048), env_int("QA_SPEC_PROBS", 20)
    prompts = spec_prompts(env_int("QA_SPEC_PROMPTS", 4), env_int("QA_SPEC_CHARS", 4000))
    identity = {"v": VERSION, "kind": "spec", "build": build_id(exe.parent), "args": args, "model": model_identity(model),
                "draft": model_identity(Path(env("SPECDRAFT"))), "env": env_identity(os.environ),
                "prompts": text_digest("\n".join(prompts)), "greedy": n_greedy, "tokens": n_tokens, "probs": n_probs,
                "greedy_probs": n_probs,
                "machine": machine()}

    def go(log):
        proc, log = start_server(exe, args, os.environ.copy(), log)
        try:
            url = f"http://127.0.0.1:{port}"
            return {"greedy": spec_greedy(url, prompts, n_greedy, n_probs), "pit": spec_pit(url, prompts, n_tokens, n_probs, 1000)}
        finally:
            proc.kill()
            proc.wait()
            mark_end()

    r, kept = run.measure("spec", identity, go)
    k = " k" if kept else ""
    flags = spec_flags(r["greedy"], r["pit"])
    def nats(x):
        return "-" if x is None else f"{x:.2f}"

    run.add("", f"spec: temperature 0, {n_greedy} tokens of {len(prompts)} prompts with and without drafts: the same "
                f"tokens up to the first other one; margin - the gap of the top two tokens there without drafts (nats), "
                f"with drafts - the gap of the same two tokens in the verify path (below 0: it takes the second); shift "
                f"before - the largest change of that gap at the tokens before it (the same context, the rounding of the "
                f"two paths)" + k,
            *table(["prompt", "same", "margin", "with drafts", "shift before", "drafted", "accepted"],
                   [[g["prompt"], f"{g['same']}/{g['n']}", nats(g["margin"]), nats(g.get("spec_gap")),
                     nats(g.get("shift_max")), g["draft_n"], g["draft_acc"]] for g in r["greedy"]]))
    run.add("", f"spec: temperature 1 (the temperature alone), {n_tokens} tokens of each prompt: the probability integral "
                f"transform of every token under the distribution it was drawn from, uniform when they follow it" + k,
            *table(["", "tokens", "chi2 (19)", "p", "mean (0.5)", "p", "outside", "drafted", "accepted"],
                   [[name, a["tokens"], "-" if a["chi2"] is None else f"{a['chi2']:.1f}",
                     "-" if a["p"] is None else f"{a['p']:.2g}", "-" if a["mean"] is None else f"{a['mean']:.3f}",
                     "-" if a.get("p_mean") is None else f"{a['p_mean']:.2g}",
                     a["outside"], a["draft_n"], a["draft_acc"]] for name, a in r["pit"].items()]))
    if flags:
        run.failed = True
        run.add("spec: " + "; ".join(flags))
    run.record("check", "spec", {"greedy_same": sum(g["same"] for g in r["greedy"]) / max(1, sum(g["n"] for g in r["greedy"])),
               "pit_p_spec": r["pit"]["spec"]["p"], "pit_p_nospec": r["pit"]["nospec"]["p"],
               "pit_mean_spec": r["pit"]["spec"]["mean"],
               "accept": r["pit"]["spec"]["draft_acc"] / max(1, r["pit"]["spec"]["draft_n"])}, kept, exe.parent)


def gbnf_literal(s):
    """s as one GBNF string literal (the escapes of llama-grammar's parse_char)"""
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7f:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def forced_parts(n, ctx_chars, chars):
    """n pairs (context, continuation) of the corpus spread over it, cut at spaces"""
    text = corpus_text()
    step = max(1, (len(text) - ctx_chars - chars - 200) // max(1, n))
    parts = []
    for i in range(n):
        a = text.find(" ", i * step)
        b = text.find(" ", a + ctx_chars)
        c = text.find(" ", b + chars)
        if a < 0 or b < 0 or c < 0:
            break
        parts.append((text[a + 1:b], text[b:c]))
    return parts


def forced_probe(run, model, port):
    """the draft acceptance on fixed text (09.11): a grammar of one literal makes the target write the continuation of a
    corpus part at temperature 0, the MTP head drafts without the grammar, so every variant drafts over the same tokens
    from the same contexts and a difference of acceptance comes from the numbers, not from the text it wrote (the
    acceptance of free replies differs with the text: 03 and the spec probe gave both signs for one kernel). Drafts of a
    fixed length (QA_FORCED_SPEC in place of QA_SPEC): --spec-auto picks the length from the measured verify cost, so a
    faster verify drafts longer and accepts a smaller share of the drafted tokens (03, 09.11: 81 verifies of 4 tokens
    against 61); at temperature 0 the drafts are the head's argmax. Parts: QA_FORCED_PROMPTS of QA_FORCED_CTX characters
    of context and QA_FORCED_CHARS of forced text; TG t/s includes the grammar on the CPU"""
    if not env("SPECDRAFT") or env("QA_SPEC", "--spec-auto") == "none":
        raise Skip("no MTP head (SPECDRAFT) or QA_SPEC=none")
    exe = binary("llama-server")
    n_ctx = env_int("QA_FORCED_N_CTX", 8192)
    args = server_args(model, n_ctx, 1, port,
                       spec_args=env("QA_FORCED_SPEC", "--spec-draft-n-min 3 --spec-draft-n-max 3 --spec-draft-p-min 0"))
    parts = forced_parts(env_int("QA_FORCED_PROMPTS", 16), env_int("QA_FORCED_CTX", 3000), env_int("QA_FORCED_CHARS", 1500))
    identity = {"v": VERSION, "kind": "forced", "build": build_id(exe.parent), "args": args, "model": model_identity(model),
                "draft": model_identity(Path(env("SPECDRAFT"))), "env": env_identity(os.environ),
                "parts": text_digest("\n".join(a + "\x00" + b for a, b in parts)), "machine": machine()}

    def go(log):
        proc, log = start_server(exe, args, os.environ.copy(), log)
        try:
            url = f"http://127.0.0.1:{port}"
            rows = []
            for i, (prompt, forced) in enumerate(parts):
                body = {"prompt": prompt, "n_predict": len(forced) + 16, "temperature": 0, "cache_prompt": False,
                        "grammar": "root ::= " + gbnf_literal(forced)}
                try:
                    _, raw = http(url + "/completion", body, env_int("QA_TIMEOUT", 3600))
                except urllib.error.HTTPError as e:
                    raise Fail(f"forced part {i}: HTTP {e.code} {e.read()[:300]!r}")
                r = json.loads(raw)
                t = r.get("timings", {})
                rows.append({"part": i, "n": t.get("predicted_n", 0), "same": r.get("content", "") == forced,
                             "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0),
                             "tg": t.get("predicted_per_second", 0.0)})
            return {"rows": rows}
        finally:
            proc.kill()
            proc.wait()
            mark_end()

    r, kept = run.measure("forced", identity, go)
    k = " k" if kept else ""
    rows = r["rows"]
    dn, da, n = sum(x["draft_n"] for x in rows), sum(x["draft_acc"] for x in rows), sum(x["n"] for x in rows)
    run.add("", f"forced: the draft acceptance on {len(rows)} parts of the corpus that a grammar makes the target write at "
                f"temperature 0 (the same tokens for every variant): {da} of {dn} drafted accepted "
                f"({da / max(1, dn):.4f}), {n} tokens" + k,
            *table(["part", "tokens", "as forced", "drafted", "accepted", "rate", "TG t/s"],
                   [[x["part"], x["n"], "yes" if x["same"] else "NO", x["draft_n"], x["draft_acc"],
                     f"{x['draft_acc'] / max(1, x['draft_n']):.3f}", f"{x['tg']:.2f}"] for x in rows]))
    if not all(x["same"] for x in rows):
        run.failed = True
        run.add("forced: FORCED TEXT DIFFERS (the grammar did not hold the target to the text)")
    run.record("check", "forced", {"accept": da / max(1, dn), "drafted": dn, "tokens": n}, kept, exe.parent)


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


def server_args(model, ctx, slots, port, extra=(), spec_args=None):
    """the server as serve.bat runs it: the model, LOADMODE, EXTRA, DECARGS, the MTP head of SPECDRAFT with QA_SPEC (or
    with spec_args in its place)"""
    draft = env("SPECDRAFT") if env("QA_SPEC", "--spec-auto") != "none" else ""
    if draft and not Path(draft).is_file():
        raise Fail(f"SPECDRAFT not found: {draft!r}")
    spec = (split_args(env("QA_SPEC", "--spec-auto")) if spec_args is None else split_args(spec_args)) if draft else []
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
    """several conversations at once in the shared KV of the slots, each with facts, asked at its end. QA_PAR_ARMS:
    variants (name VAR=value ...;...), a server each, side by side (12.11: the QSA block lists of the slots against
    rows0 LLAMA_QSA_ROWS=0)"""
    exe = binary("llama-server")
    par, total = env_int("QA_PAR", 4), env_int("QA_PAR_TOKENS", 49152)
    chunk, reserve = env_int("QA_PAR_CHUNK", 12288), env_int("QA_LONG_RESERVE", 2048)
    if par < 2:
        return
    args = server_args(model, ctx, par, port)
    text = corpus_text()
    done = []    # (name, result, kept)
    for name, arm_env in variants("QA_PAR_ARMS", "default"):
        run_env = dict(os.environ, **arm_env)
        identity = {"v": VERSION, "kind": "long-par", "build": build_id(exe.parent), "args": args,
                    "model": model_identity(model), "env": env_identity(run_env), "corpus": text_digest(text), "par": par,
                    "total": total, "chunk": chunk, "samp": env("QA_SAMP", "1.0 20 0.95"), "machine": machine(),
                    "draft": model_identity(Path(env("SPECDRAFT"))) if "-md" in args else None}

        def go(log, run_env=run_env):
            proc, log = start_server(exe, args, run_env, log)
            try:
                url = f"http://127.0.0.1:{port}"
                ratio = token_ratio(url, text)
                span = max(1, (len(text) - int(total / ratio)) // par)
                results, errors = [None] * par, []

                def one(i):
                    try:
                        chat = Chat(url, log, f"par{i}")
                        facts = make_facts(total // max(chunk // 2, 1) + 8, 100 + i)
                        _, _, depth = converse(chat, text, i * span, ratio, total, chunk, facts, random.Random(i), [],
                                               reserve, ctx)
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
                return {"results": results, "errors": errors, "wall": time.time() - start, "qsa": log_first(log, NOTE_QSA)}
            finally:
                proc.kill()
                proc.wait()
                mark_end()

        label = "long-par" if name == "default" else f"long-par {name}"
        try:
            r, kept = run.measure(label, identity, go)
        except Fail as e:
            run.failed = True
            run.add("", f"{label}: FAIL {e}")
            continue
        rows, all_ok = [], not r["errors"]
        for i, x in enumerate(r["results"]):
            if not x:
                rows.append([f"par{i}", "", "", "FAIL"])
                continue
            ok = sum(y["ok"] for y in x["recall"])
            all_ok &= ok == len(x["recall"])
            n = sum(t["n"] for t in x["turns"])
            rows.append([f"par{i}", x["depth"], n, f"{ok}/{len(x['recall'])}" + ("" if ok == len(x["recall"]) else " WRONG")])
        run.add("", f"{par} conversations at once (-np {par} --kv-unified, -c {ctx}{', ' + name if name != 'default' else ''}), "
                    f"{total} tokens each in parts of {chunk}, {r['wall']:.0f} s" + (" k" if kept else ""),
                *table(["slot", "depth", "generated", "recall"], rows))
        for e in r["errors"]:
            run.add(f"  {e}")
        if not all_ok:
            run.failed = True
        done.append((name, r, kept))
        run.record("long", "par" if name == "default" else f"par {name}", {"wall": r["wall"]}, kept, exe.parent)
    if len(done) > 1:
        rows, ref = [], None
        for name, r, kept in done:
            parts = [t for x in r["results"] if x for t in x["turns"] if t["kind"] == "part"]
            pn, pms = sum(t["prompt_n"] for t in parts), sum(t["prompt_ms"] for t in parts)
            ref = r["wall"] if ref is None else ref
            ok = sum(y["ok"] for x in r["results"] if x for y in x["recall"])
            n_rec = sum(len(x["recall"]) for x in r["results"] if x)
            q = r.get("qsa")
            rows.append([name + (" k" if kept else ""), f"{r['wall']:.0f}",
                         "" if r["wall"] == ref else f"{(r['wall'] / ref - 1) * 100:+.1f}%",
                         f"{rate(pn, pms):.1f}" if pms else "", f"{ok}/{n_rec}",
                         f"{q[0]} slices, {q[1]} of {q[2]} blocks" if q else "all blocks"])
        run.add("", "par: the variants side by side: the wall time of the whole step against the first, the prompt "
                    "tokens of the parts over their prompt time (per request), the facts recalled, the QSA block lists "
                    "(the first line of the server log)",
                *table(["variant", "wall s", "vs first", "PP t/s", "recall", "QSA scores"], rows))


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


# ---------------------------------------------------------------- decode speed by kind of text

def decode_tasks(names=None):
    """the requests of 06-decode (QA_DEC_SET, or names): prose about a corpus part, prose in Russian, new code in two languages, an edit of a source
    file of the repository (QA_DEC_SOURCE; the reply copies most of it, as a coding agent rewriting a file does, and long
    drafts pay there), JSON records; on request a long prompt with a short reply (long)"""
    text = corpus_text()
    src_path = ROOT / env("QA_DEC_SOURCE", "examples/simple/simple.cpp")
    try:
        src = src_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise Fail(f"QA_DEC_SOURCE: {e}")
    a = text.find(" ", len(text) // 3)
    part = text[a + 1:a + 4000]
    # a long prompt as an agent sends (QA_DEC_LONG_CHARS, ~24K tokens by default) with a short reply: what the
    # generating streams of other requests wait for (LLAMA_PREFILL_EXCLUSIVE). Not in the default set
    b = text.find(" ", len(text) // 10)
    long_part = text[b + 1:b + 1 + env_int("QA_DEC_LONG_CHARS", 96000)]
    tasks = {
        "prose": part + "\n\nWrite a long, detailed essay about the subject of this text.",
        "code-py": "Write a complete Python module that implements an LRU cache with a time-to-live per entry, thread "
                   "safety, type hints and docstrings, followed by a full pytest test suite for it. Output only the code.",
        "code-cpp": "Write a C++17 header-only lock-free single-producer single-consumer ring buffer with a templated "
                    "element type, documented with comments, and a small test program that drives it from two threads. "
                    "Output only the code.",
        "edit": "Here is a source file:\n\n```cpp\n" + src + "\n```\n\nRewrite the whole file with two changes: rename the "
                "variable `n_predict` to `n_tokens_to_predict` everywhere and the function `batch_set_tokens` to "
                "`fill_batch`. Keep everything else exactly as it is. Output the complete file only.",
        "json": "Generate a JSON array of 40 objects that describe fictional employees. Every object has the fields id "
                "(integer), name, department, title, salary (integer), start_date (YYYY-MM-DD) and skills (a list of "
                "three strings). Output only the JSON.",
        # prose in Russian (09.11): the drafts of a Cyrillic text, which the vocabulary subset of the MTP head keeps
        # in its base (LLAMA_MTP_VOCAB_SCRIPTS)
        "ru": "\u041e\u0431\u044a\u044f\u0441\u043d\u0438 \u043f\u043e\u0434\u0440\u043e\u0431\u043d\u043e, "
              "\u043a\u0430\u043a \u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 \u0441\u043f\u0435\u043a\u0443\u043b\u044f\u0442\u0438\u0432\u043d\u043e\u0435 "
              "\u0434\u0435\u043a\u043e\u0434\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 \u0432 \u0431\u043e\u043b\u044c\u0448\u0438\u0445 "
              "\u044f\u0437\u044b\u043a\u043e\u0432\u044b\u0445 \u043c\u043e\u0434\u0435\u043b\u044f\u0445: "
              "\u0447\u0435\u0440\u043d\u043e\u0432\u0430\u044f \u043c\u043e\u0434\u0435\u043b\u044c, \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 "
              "\u043e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u044c\u044e, \u043f\u0440\u0438\u043d\u044f\u0442\u0438\u0435 "
              "\u0442\u043e\u043a\u0435\u043d\u043e\u0432, \u043f\u043e\u0447\u0435\u043c\u0443 \u043e\u0442\u0432\u0435\u0442 \u043d\u0435 "
              "\u043c\u0435\u043d\u044f\u0435\u0442\u0441\u044f \u0438 \u043a\u0430\u043a\u0438\u0435 \u0435\u0441\u0442\u044c "
              "\u0441\u043f\u043e\u0441\u043e\u0431\u044b \u0443\u0441\u043a\u043e\u0440\u0438\u0442\u044c \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044e. "
              "\u041f\u0438\u0448\u0438 \u043f\u043e-\u0440\u0443\u0441\u0441\u043a\u0438, \u0440\u0430\u0437\u0432\u0451\u0440\u043d\u0443\u0442\u043e, "
              "\u0441 \u043f\u0440\u0438\u043c\u0435\u0440\u0430\u043c\u0438.",
        "long": long_part + "\n\nSummarize this text in one short paragraph.",
    }
    names = env("QA_DEC_SET", "prose ru code-py code-cpp edit json").split() if names is None else names
    unknown = [n for n in names if n not in tasks]
    if unknown:
        raise Fail(f"QA_DEC_SET: no task {' '.join(unknown)} (tasks: {' '.join(tasks)})")
    return [(n, tasks[n]) for n in names]


def decode_ask(url, prompt, max_tokens, temp, log):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temp, "seed": 1,
            "cache_prompt": False, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    try:
        _, raw = http(url + "/v1/chat/completions", body, env_int("QA_TIMEOUT", 3600))
    except urllib.error.HTTPError as e:
        raise Fail(f"decode: HTTP {e.code} {e.read()[:300]!r}")
    except (urllib.error.URLError, OSError) as e:
        raise Fail(f"decode: {e} ({log.name})")
    answer = json.loads(raw)
    return answer.get("timings", {}), answer["choices"][0]["message"].get("content") or ""


def decode_row(task, t, content, stream=0, t0=0.0, t1=0.0):
    return {"task": task, "stream": stream, "n": t.get("predicted_n", 0), "ms": t.get("predicted_ms", 0.0),
            "prompt_n": t.get("prompt_n", 0), "prompt_ms": t.get("prompt_ms", 0.0), "draft_n": t.get("draft_n", 0),
            "draft_acc": t.get("draft_n_accepted", 0), "text": text_digest(content)[:8], "t0": t0, "t1": t1}


def decode_run(url, tasks, ntok, temp, log, par):
    """the tasks on one server: in order for one stream; for par streams every stream runs all tasks from its own
    offset (stream i starts at task i), all streams start together, so par requests run at once all the time"""
    t0 = time.monotonic()
    if par <= 1:
        rows = []
        for task, prompt in tasks:
            a = time.monotonic() - t0
            t, content = decode_ask(url, prompt, ntok, temp, log)
            rows.append(decode_row(task, t, content, 0, a, time.monotonic() - t0))
        return rows
    rows, errors = [], []
    lock = threading.Lock()
    barrier = threading.Barrier(par)

    def stream(i):
        order = tasks[i % len(tasks):] + tasks[:i % len(tasks)]
        try:
            barrier.wait()
            for task, prompt in order:
                a = time.monotonic() - t0
                t, content = decode_ask(url, prompt, ntok, temp, log)
                with lock:
                    rows.append(decode_row(task, t, content, i, a, time.monotonic() - t0))
        except (Fail, threading.BrokenBarrierError) as e:
            with lock:
                errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(par)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise next((e for e in errors if isinstance(e, Fail)), Fail(f"decode: a stream stopped ({errors[0]!r})"))
    return sorted(rows, key=lambda x: (x["stream"], x["t0"]))


def gguf_token_count(path):
    """the length of tokenizer.ggml.tokens in the header of a GGUF file, None without it"""
    import struct
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        f.read(4)
        _, n_kv = struct.unpack("<QQ", f.read(16))

        def skip(t):
            if t == 8:
                n, = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
            elif t == 9:
                st, n = struct.unpack("<IQ", f.read(12))
                if st in sizes:
                    f.seek(sizes[st] * n, 1)
                else:
                    for _ in range(n):
                        skip(st)
            else:
                f.seek(sizes[t], 1)

        for _ in range(n_kv):
            n, = struct.unpack("<Q", f.read(8))
            key = f.read(n).decode("utf-8", "replace")
            t, = struct.unpack("<I", f.read(4))
            if key == "tokenizer.ggml.tokens" and t == 9:
                _, count = struct.unpack("<IQ", f.read(12))
                return count
            skip(t)
    return None


def vocab_history(model):
    """LLAMA_MTP_VOCAB_STATS=history in a variant: the token counts of texts of the repository that the tasks of 06 do not
    use (QA_VOCAB_HISTORY: globs under the repository root; Russian notes, the fork's code, English docs), as a history
    of conversations would count them. Returns the file in cache/ (by the digest of the texts and the model; it may not
    exist yet, vocab_history_build writes it) and the texts"""
    globs = env("QA_VOCAB_HISTORY", "FORK.md RESEARCH.md scripts/win-qsa/*.md src/fork/*.cpp src/fork/*.h common/fork/*.cpp "
                "common/fork/*.h tools/server/fork/*.cpp ggml/src/ggml-vulkan/fork/*.cpp scripts/win-qsa/_qa.py "
                "src/llama-*.cpp docs/*.md README.md tools/server/README.md").split()
    files = sorted({q for g in globs for q in ROOT.glob(g) if q.is_file()})
    if not files:
        raise Fail(f"QA_VOCAB_HISTORY: no files for {' '.join(globs)}")
    texts = [q.read_text(encoding="utf-8", errors="replace") for q in files]
    digest = text_digest("\x00".join(texts) + "\x00" + json.dumps(model_identity(model), sort_keys=True))[:12]
    return CACHE / f"vocab-stats-{digest}.txt", texts


def vocab_history_build(url, out, texts, model):
    """the counts of the texts of vocab_history, tokenized by /tokenize of the server at url (the vocabulary of the
    model; llama-tokenize makes a context, which a qwen4exp file with an MTP block refuses without its target)"""
    n_vocab = gguf_token_count(model)
    if not n_vocab:
        raise Fail(f"{model.name}: no tokenizer.ggml.tokens for the vocabulary statistics")
    counts, total = {}, 0
    for text in texts:
        # parts of about 64K characters at line ends: a token across a cut is counted as its two pieces
        lines, part, size = text.splitlines(keepends=True), [], 0
        for i, line in enumerate(lines + [None]):
            if line is not None:
                part.append(line)
                size += len(line)
            if part and (line is None or size >= 65536):
                _, raw = http(url + "/tokenize", {"content": "".join(part), "add_special": False}, 600)
                for t in json.loads(raw)["tokens"]:
                    counts[t] = counts.get(t, 0) + 1
                    total += 1
                part, size = [], 0
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(f"llama-mtp-vocab-stats 1 {n_vocab}\n")
        for t in sorted(counts):
            f.write(f"{t} {counts[t]}\n")
    tmp.replace(out)
    print(f"decode: vocabulary statistics of {len(texts)} files, {total} tokens ({len(counts)} distinct): {out.name}",
          flush=True)


def decode_trace(path):
    """the draft steps and rounds of a GGML_TIMELINE trace of one request (QA_DEC_TRACE): a step is a decode of the MTP
    head (track "llama mtp": graph, inputs, submit) followed by another within 20 ms; its period runs to the start of the
    next one and splits into graph (build or reuse), inputs, submit (graph_compute) and after (from the return of the
    decode to the next one: readback, sampling, the draft gate, the next batch); gpu - the time of the gpu spans in the
    period. A round runs from one decode of the target (track "llama") to the next"""
    tracks, ev = {}, []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line.startswith("{"):
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ph") == "M":
                    tracks[e.get("tid")] = e.get("args", {}).get("name", "")
                elif e.get("ph") == "X":
                    ev.append(e)
    except OSError:
        return None
    gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("name") == "gpu")

    def gpu_in(t0, t1):
        busy, last = 0.0, t0
        for a, b in gpu:
            if b <= last:
                continue
            if a >= t1:
                break
            busy += min(b, t1) - max(a, last)
            last = min(b, t1)
        return busy

    def decodes(track):
        out, cur = [], None
        for e in sorted((e for e in ev if tracks.get(e.get("tid")) == track), key=lambda e: e["ts"]):
            kind, _, n = e["name"].partition(" ")
            if kind == "graph":
                cur = {"n": int(n) if n.isdigit() else 0, "t0": e["ts"], "graph": e["dur"]}
            elif cur is not None and kind in ("inputs", "submit"):
                cur[kind] = e["dur"]
                if kind == "submit":
                    cur["t1"] = e["ts"] + e["dur"]
                    out.append(cur)
                    cur = None
        return out

    mtp, tgt = decodes("llama mtp"), decodes("llama")
    steps = []
    for a, b in zip(mtp, mtp[1:]):
        if b["t0"] - a["t1"] > 20000:
            continue
        # a decode of the target in between ends the round: the next mtp decode is a catch-up, not a step
        if any(a["t0"] < t["t0"] < b["t0"] for t in tgt):
            continue
        period = b["t0"] - a["t0"]
        steps.append({"period": period, "gpu": gpu_in(a["t0"], b["t0"]), "graph": a["graph"], "inputs": a.get("inputs", 0),
                      "submit": a.get("submit", 0), "after": b["t0"] - a["t1"]})
    # the server's parts of a round (10.11): "spec process" - the catch-up of the head, "spec accept" - the sampling of
    # the verified positions, "spec tokens" - the work on the accepted tokens (traces before 10.11 have the first only)
    srv = sorted((e for e in ev if tracks.get(e.get("tid")) == "server"), key=lambda e: e["ts"])
    gpu_t0 = [a for a, _ in gpu]
    rounds = []
    for a, b in zip(tgt, tgt[1:]):
        if b["t0"] - a["t0"] > 1e6:
            continue
        inner = [m for m in mtp if a["t0"] < m["t0"] < b["t0"]]
        parts = [e for e in srv if a["t0"] < e["ts"] < b["t0"]]
        proc = next((e for e in parts if e["name"].startswith("spec process")), None)
        first = next((m for m in inner if m["t0"] >= proc["ts"] + proc["dur"]), None) if proc else None
        g = bisect.bisect_left(gpu_t0, a["t0"])
        rounds.append({"period": b["t0"] - a["t0"], "gpu": gpu_in(a["t0"], b["t0"]), "mtp": len(inner),
                       "verify": (inner[0]["t0"] if inner else b["t0"]) - a["t0"],
                       "verify_gpu": gpu_in(a["t0"], inner[0]["t0"] if inner else b["t0"]),
                       "lead": (gpu_t0[g] - a["t0"]) if g < len(gpu_t0) and gpu_t0[g] < b["t0"] else 0,
                       "sample": sum(e["dur"] for e in parts if e["name"].startswith("spec accept")),
                       "tokens": sum(e["dur"] for e in parts if e["name"].startswith("spec tokens")),
                       "gap": (first["t0"] - proc["ts"] - proc["dur"]) if first else 0})
    return {"steps": steps, "rounds": rounds, "n_gpu": len(gpu)}


def cmd_decode():
    """Decode speed on the server as serve.bat runs it (SPECDRAFT, QA_SPEC), by kind of text (09.11): one request per
    task of QA_DEC_SET in a conversation of its own, up to QA_DEC_TOKENS tokens at temperature QA_DEC_TEMP (0), TG t/s
    from the server's timings (no perf logger: the wall clock of the reply), drafted and accepted tokens. QA_DARMS:
    variants (name VAR=value ...;...), each a server of its own, e.g. a draft cap (LLAMA_ARG_SPEC_DRAFT_N_MAX=7). At
    temperature 0 the drafts do not change the text: two variants write the same reply up to a near-tie of the numbers
    (the column "text" names the reply by a digest), so their TG compares on the same tokens.
    QA_DEC_PAR: numbers of streams (1 by default), each a server with -np N --kv-unified where N requests run at once
    (every stream all tasks from its own offset); the total is the tokens of all replies over the wall time from the
    first request to the end of the last, prompts included (agents send prompts while others generate). A warm-up
    server runs first (QA_DEC_WARM)."""
    run = Run("decode")
    model = model_path()
    exe = binary("llama-server")
    ctx, port = env_int("QA_DEC_CTX", 32768), env_int("QA_PORT", 8097)
    ntok, temp = env_int("QA_DEC_TOKENS", 1024), float(env("QA_DEC_TEMP", "0"))
    pars = env_ints("QA_DEC_PAR", "1")
    tasks = decode_tasks()
    arms = variants("QA_DARMS", "default")
    # LLAMA_MTP_VOCAB_STATS=history: the counts of the repository texts (vocab_history), built once by the warm-up server
    hist, hist_texts = (vocab_history(model) if any(e.get("LLAMA_MTP_VOCAB_STATS") == "history" for _, e in arms)
                        else (None, None))
    hist_build = hist is not None and not hist.is_file()
    if hist is not None:
        arms = [(n, {k: (str(hist) if k == "LLAMA_MTP_VOCAB_STATS" and v == "history" else v) for k, v in e.items()})
                for n, e in arms]
    run.add(f"decode: llama-server, tasks {' '.join(n for n, _ in tasks)}, up to {ntok} tokens each at temperature {temp}"
            f"{', streams ' + ' '.join(map(str, pars)) if pars != [1] else ''}, {model.name}, build {current_build()}" +
            (", variants " + "; ".join(f"{name} {' '.join(f'{k}={v}' for k, v in e.items())}".strip() for name, e in arms)
             if len(arms) > 1 else ""))
    # a warm-up server before the measured ones (QA_DEC_WARM=0: none): the first variant after a build compiles its
    # pipelines in its first replies (the driver has no cache of the new shaders yet): prose of the first variant of
    # decode-20261109-203021 34.5 t/s against 42.7 for the same settings later. The first variant with the most
    # streams, every task with 32 tokens, nothing kept
    warm = env("QA_DEC_WARM", "1") != "0"
    if warm or hist_build:
        warm_par = max(1, max(pars)) if warm else 1
        warm_args = server_args(model, ctx, warm_par, port)
        # the first variant without the statistics it may name: they are written here
        warm_env = {k: v for k, v in arms[0][1].items() if not (hist is not None and k == "LLAMA_MTP_VOCAB_STATS")}
        proc, warm_log = start_server(exe, warm_args, dict(os.environ, **warm_env), run.dir / "decode_warmup.log")
        try:
            if hist_build:
                vocab_history_build(f"http://127.0.0.1:{port}", hist, hist_texts, model)
            if warm:
                print(f"decode: warm-up, {warm_par} stream(s), every task with 32 tokens, decode_warmup.log", flush=True)
                decode_run(f"http://127.0.0.1:{port}", tasks, 32, temp, warm_log, warm_par)
        finally:
            proc.kill()
            proc.wait()
            mark_end()
    results = []   # (name, par, rows)
    # QA_DEC_TRACE=<n>: a GGML_TIMELINE trace of the n-th chat request of every one-stream variant (1 is the short reply
    # before the tasks, 2 the first task), trace-<variant>.json in the run folder, and its draft steps in a table.
    # QA_DEC_TRACE=<task>: one more request of that task after the measured ones, traced and left out of the results
    trace_env = env("QA_DEC_TRACE", "")
    trace_extra = None
    if trace_env and not trace_env.isdigit():
        trace_extra = decode_tasks([trace_env])[0]
        trace_req = len(tasks) + 2
    else:
        trace_req = int(trace_env or 0)
    traces = []    # (name, path)
    for name, arm_env in arms:
        for par in pars:
            run_env = dict(os.environ, **arm_env)
            if trace_req > 0 and par == 1:
                trace_path = run.dir / f"trace-{name}.json"
                run_env["GGML_TIMELINE"] = str(trace_path)
                run_env["LLAMA_TRACE_REQUESTS"] = str(trace_req)
                traces.append((name, trace_path))
            args = server_args(model, ctx, max(1, par), port)
            identity = {"v": VERSION, "kind": "decode", "build": build_id(exe.parent), "args": args,
                        "env": env_identity(run_env), "model": model_identity(model),
                        "draft": model_identity(Path(env("SPECDRAFT"))) if "-md" in args else None,
                        "tasks": text_digest("\n".join(n + "\x00" + p for n, p in tasks)), "tokens": ntok, "temp": temp,
                        "machine": machine(), **({"par": par} if par > 1 else {})}

            def go(log, run_env=run_env, args=args, par=par):
                proc, log = start_server(exe, args, run_env, log)
                url = f"http://127.0.0.1:{port}"
                try:
                    # the first use of a pipeline compiles it: a short reply before the measured ones
                    decode_ask(url, "Say hello in one sentence.", 16, temp, log)
                    rows = decode_run(url, tasks, ntok, temp, log, par)
                    if trace_extra and par == 1:
                        decode_ask(url, trace_extra[1], ntok, temp, log)
                    return {"rows": rows}
                finally:
                    proc.kill()
                    proc.wait()
                    mark_end()

            label = ("decode" if name == "default" else f"decode {name}") + (f" x{par}" if par > 1 else "")
            try:
                r, kept = run.measure(label, identity, go)
            except Fail as e:
                run.failed = True
                run.add("", f"{label}: FAIL {e}")
                continue
            rows = r["rows"]
            results.append((name, par, rows))
            head = ["task"] + (["stream"] if par > 1 else []) + ["tokens", "TG t/s", "drafted", "accepted", "rate",
                                                                 "per verify", "prompt", "prompt s", "text"]
            body = [[x["task"]] + ([x["stream"]] if par > 1 else []) +
                    [x["n"], f"{rate(x['n'], x['ms']):.2f}", x["draft_n"], x["draft_acc"],
                     f"{x['draft_acc'] / x['draft_n']:.3f}" if x["draft_n"] else "",
                     # the verify steps: every step but the first token of the reply verifies a draft or decodes one
                     f"{x['n'] / max(1, x['n'] - x['draft_acc']):.2f}", x["prompt_n"],
                     f"{x.get('prompt_ms', 0.0) / 1000:.1f}", x["text"]] for x in rows]
            run.add("", label + (" k" if kept else ""), *table(head, body))
    single = [(name, rows) for name, par, rows in results if par == 1]
    if len(single) > 1:
        base_name, base = single[0]
        groups = [(x["task"], [x["task"]]) for x in base] + [
            ("code", [t for t, _ in tasks if t not in ("prose", "ru", "long")]), ("all", [t for t, _ in tasks])]
        rows = []
        for gname, members in groups:
            row = [gname]
            ref = None
            for name, res in single:
                sel = [x for x in res if x["task"] in members]
                n, ms = sum(x["n"] for x in sel), sum(x["ms"] for x in sel)
                tg = rate(n, ms) if ms else 0.0
                ref = tg if ref is None else ref
                row.append(f"{tg:.2f}" + ("" if name == base_name or not ref else f" ({(tg / ref - 1) * 100:+.1f}%)"))
            same = all(next((y["text"] for y in res if y["task"] == gname), None) ==
                       next((y["text"] for y in base if y["task"] == gname), None) for _, res in single[1:])
            row.append(("same" if same else "differs") if len(members) == 1 else "")
            rows.append(row)
        run.add("", "decode: TG t/s by task and variant (tokens over the time of the replies; the text of every variant "
                    "against the first)",
                *table(["task"] + [name for name, _ in single] + ["text"], rows))
    if any(par > 1 for _, par, _ in results):
        rows = []
        total1 = {}
        for name, par, res in results:
            n = sum(x["n"] for x in res)
            wall = max(x["t1"] for x in res) - min(x["t0"] for x in res)
            total = n / wall if wall > 0 else 0.0
            if par == 1:
                total1[name] = total
            per = statistics.mean(rate(x["n"], x["ms"]) for x in res if x["ms"])
            dn, da = sum(x["draft_n"] for x in res), sum(x["draft_acc"] for x in res)
            ref = total1.get(name)
            rows.append([name, par, len(res), n, f"{wall:.1f}", f"{total:.2f}",
                         "" if par == 1 or not ref else f"{total / ref:.2f}x", f"{per:.2f}",
                         f"{da / dn:.3f}" if dn else "", f"{sum(x['prompt_ms'] for x in res) / 1000 / max(wall, 1e-9) * 100:.0f}%"])
        run.add("", "decode: streams at once: the tokens of all replies over the wall time (prompts included), against "
                    "one stream of the same variant; per stream - the mean TG of a reply; prompts - the share of the "
                    "wall time the prompts took, if one after another",
                *table(["variant", "streams", "replies", "tokens", "wall s", "total t/s", "vs 1", "per stream t/s",
                        "accepted", "prompts"], rows))
    if traces:
        srows, rrows = [], []
        for name, path in traces:
            tr = decode_trace(path)
            if not tr or not tr["steps"]:
                srows.append([name, "no steps" if tr else "no trace"])
                continue
            st, rd = tr["steps"], tr["rounds"]

            def ms(rows, key):
                return f"{statistics.mean(x[key] for x in rows) / 1000:.3f}"

            srows.append([name, len(st), ms(st, "period"), ms(st, "gpu"),
                          f"{statistics.mean(x['period'] - x['gpu'] for x in st) / 1000:.3f}",
                          ms(st, "graph"), ms(st, "inputs"), ms(st, "submit"), ms(st, "after"), tr["n_gpu"]])
            if rd:
                rrows.append([name, len(rd), ms(rd, "period"), ms(rd, "gpu"), f"{statistics.mean(x['mtp'] for x in rd):.2f}",
                              ms(rd, "verify"), ms(rd, "verify_gpu"), ms(rd, "lead"),
                              ms(rd, "sample") if any(x["sample"] for x in rd) else "",
                              ms(rd, "tokens") if any(x["tokens"] for x in rd) else "", ms(rd, "gap")])
        run.add("", f"decode: draft steps of the traced request "
                    f"{trace_req if not trace_extra else trace_extra[0] + ' after the measured ones'} "
                    "(GGML_TIMELINE, ms per step): period - to the "
                    "start of the next step, gpu - gpu spans in it, host - the rest; graph, inputs, submit - the parts "
                    "of the decode; after - from its return to the next step (readback, sampling, gate)",
                *table(["variant", "steps", "period", "gpu", "host", "graph", "inputs", "submit", "after", "gpu spans"],
                       srows))
        if rrows:
            run.add("", "decode: rounds of the traced request (ms): from one decode of the target to the next; mtp - "
                        "the number of decodes of the head in it (catch-up and steps), not ms; verify - from the target "
                        "decode to the first decode of the head; lead - from the target decode to the first gpu span; "
                        "sample - the sampling of the verified positions, tokens - the work on the accepted tokens "
                        "(the server's \"spec accept\" and \"spec tokens\", builds since 10.11); gap - from the end of "
                        "the catch-up to the first draft step, the gpu idle but for the tail of the catch-up",
                    *table(["variant", "rounds", "period", "gpu", "mtp", "verify", "verify gpu", "lead", "sample",
                            "tokens", "gap"], rrows))
    return run.finish()


# ---------------------------------------------------------------- profile

# the kinds of op of the profile tables, by the start of the op name (the first match counts); op_kind sorts the top-k
# ops and the expert router of CUDA first
OP_KINDS = [("attention", ("FLASH_ATTN", "SOFT_MAX")),
            ("qsa select", ("LIGHTNING_INDEXER", "MUL_MAT_HEADSUM", "MUL_MAT_RELU")),
            ("moe", ("MUL_MAT_ID", "ADD_ID", "MULTI_ADD", "TOPK_MOE")),
            ("matmul", ("MUL_MAT",)),
            ("delta net", ("GATED_DELTA_NET", "SSM_CONV", "CONCAT_SSM_CONV")),
            ("hyper conn", ("DSV4_HC", "HC_POST")),
            ("norm", ("RMS_NORM", "NORM", "L2_NORM")),
            ("rows, copies", ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT", "REPEAT", "DUP")),
            ("elementwise", ("ADD", "SUB", "MUL", "DIV", "SCALE", "SILU", "SIGMOID", "GELU", "RELU", "GLU", "SWIGLU",
                             "EXP", "LOG", "SOFTPLUS", "STEP", "ABS", "SQR", "SUM", "ROPE", "CLAMP", "FILL"))]

# the form of a kept profile result: the log parts of the requests since 31.10, read into tables when reported, so that
# a fix of the reading applies to the kept results too. 3 (08.11): a warm-up request before the parts
PROFILE_FORM = 3


def op_kind(name):
    op = op_group(name)
    if op.startswith("FUSED "):
        parts = op[6:].split("+")
        # CUDA fuses the expert router: the softmax of the gate, argsort, gather, norm
        if "ARGSORT" in parts:
            return "moe"
        op = parts[0]
    if op == "TOP_K":
        # the block selection of QSA keeps 513 blocks (Vulkan names K=513, CUDA the dst [513,...]); a small K is the top-k
        # of the MTP draft over the vocabulary
        m = re.search(r"K=(\d+)|\[(\d+),", name)
        return "qsa select" if m and int(m[1] or m[2]) > 64 else "other"
    for kind, prefixes in OP_KINDS:
        if op.startswith(prefixes):
            return kind
    return "other"


def by_kind(ops):
    out = {}
    for name, _, ms in ops:
        out[op_kind(name)] = out.get(op_kind(name), 0.0) + ms
    return out


def graph_tokens(block):
    """the tokens of a GGML_VK_PERF_LOGGER block: the dst of its attention, else the n of the first plain matmul (the
    attention can run elsewhere), None without either"""
    n = block_kind(block)[1]
    if n is None:
        m = next((re.search(r"\bn=(\d+)", op) for op, _, _ in block if op.startswith(("MUL_MAT ", "MUL_MAT_VEC "))), None)
        n = int(m[1]) if m else None
    return n


def profile_read(raw, depths, vulkan, mtp):
    """The tables of one profile run out of its log parts, per depth: the target's graphs of the prompt (more than 8
    tokens: the checkpoint splits the last tokens off into a graph of their own), the decode groups (the verify of each
    size and the head's steps with MTP), the GPU time of the reply per generated token and the draft acceptance"""
    s = raw["slices"]
    out = {}
    for d in depths:
        if f"pp{d}" not in s:
            continue
        pp_text, tg_text, end_text = s[f"pp{d}"], s[f"tg{d}"], s[f"end{d}"]
        host = [[float(m[2]), float(m[3]), float(m[4])] for m in
                (re.match(r".*ubatch timing: (\d+) tokens, graph ([\d.]+) ms, inputs ([\d.]+) ms, submit ([\d.]+) ms", x)
                 for x in pp_text.splitlines()) if m and int(m[1]) > 64]
        n_tg = max(1, int(s.get(f"tg{d}:n", 1)))
        acc = re.search(r"draft acceptance = ([\d.]+)", tg_text)
        prompt_n = re.search(r"prompt eval time = +[\d.]+ ms / +(\d+) tokens", pp_text)
        entry = {"depth": s[f"depth{d}"], "host": [sum(h[i] for h in host) for i in range(3)] if host else None,
                 "acceptance": float(acc[1]) if acc else None}
        if vulkan:
            # the target has an expert matmul per layer (a fused gate and up count once), the MTP head one layer
            is_target = lambda b: not mtp or block_kind(b)[0] == "target"
            pp_blocks = [b for b in vulkan_blocks(pp_text) if is_target(b) and (graph_tokens(b) or 0) > 8]
            groups, step = {}, []
            for b in vulkan_blocks(tg_text):
                n = graph_tokens(b)
                if n is not None and n > 8 or is_target(b) and n is None:
                    continue  # the prompt of the reply, the target's and the head's
                label = (("verify of %d tokens" % n if mtp else "decode step") if is_target(b)
                         else "head draft step" if n else "head catch-up")
                groups.setdefault(label, []).append(b)
                step.append(b)
            entry["pp"] = sum_ops(pp_blocks)
            entry["pp_graphs"] = len(pp_blocks)
            entry["pp_tokens"] = sum(graph_tokens(b) for b in pp_blocks)
            entry["tg_groups"] = [[label, len(bl), [[o, c / len(bl), t / len(bl)] for o, c, t in sum_ops(bl)]]
                                  for label, bl in sorted(groups.items(), key=lambda g: -len(g[1]))]
            entry["per_token"] = sum(t for b in step for _, _, t in b) / n_tg
        else:
            # the target and the MTP head together: the reports do not tell the graphs apart
            pp_r = cuda_work(pp_text, tg_text)
            tg_r = cuda_work(tg_text, end_text)
            entry["pp"] = pp_r[1] if pp_r else []
            entry["pp_graphs"] = pp_r[0] if pp_r else 0
            entry["pp_tokens"] = int(prompt_n[1]) if prompt_n else None
            entry["tg_groups"] = [["decode, all graphs per token", n_tg, [[o, c / n_tg, t / n_tg] for o, c, t in tg_r[1]]]] if tg_r else []
            entry["per_token"] = sum(o[2] for o in tg_r[1]) / n_tg if tg_r else None
        out[str(d)] = entry
    return out


def verify_size(reads, depths):
    """the largest verify that every depth of every run has, so that a column compares the same step; None without MTP"""
    common = None
    for r in reads:
        for d in depths:
            x = r.get(str(d))
            if x:
                have = {int(m[1]) for m in (re.match(r"verify of (\d+) tokens", g[0]) for g in x["tg_groups"]) if m}
                common = have if common is None else common & have
    return max(common) if common else None


def step_group(x, nv):
    """the decode group of a depth that the tables compare: the verify of nv tokens, or the one decode group"""
    want = f"verify of {nv} tokens" if nv else None
    return next((g for g in x["tg_groups"] if (g[0] == want if want else g[0].startswith("decode"))), None)


def cmd_profile():
    """Where the time goes on the server at each depth of QA_PROF_DEPTHS (tokens): a conversation reaches the depth
    with the prompt cache, then a part of QA_PROF_PART tokens (pp: the prompt ubatches at that depth) and a reply of
    QA_PROF_TG tokens (tg: the decode steps). Vulkan: GGML_VK_PERF_LOGGER, a block per graph compute; CUDA:
    GGML_CUDA_OP_TIMING=idle, a report when the GPU has been idle. LLAMA_INPUT_TIMING gives the host steps.
    QA_PARMS: variants to compare (name VAR=value ...;...), each a run of its own; the tables are those of the first,
    then the totals of every variant side by side. With MTP the tables compare the verify of one size, the largest that
    every depth of every variant has: a reply mixes verifies of 1 to 4 tokens, and their mix differs per text."""
    run = Run("profile")
    model = model_path()
    exe = binary("llama-server")
    vulkan = env("BACKEND") == "vulkan"
    ctx, port = env_int("QA_PROF_CTX", 262144), env_int("QA_PORT", 8097)
    # a part of 4000 tokens plus the chat template stays within one prompt ubatch of 4096
    part, ntg = env_int("QA_PROF_PART", 4000), env_int("QA_PROF_TG", 32)
    depths = [d for d in env_ints("QA_PROF_DEPTHS", f"0 65536 131072 {ctx - 3 * part - 2048}") if d + 2 * part + 1024 < ctx]
    args = server_args(model, ctx, 1, port, ["-lv", "4"])
    mtp = "-md" in args
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
                slices[name + ":n"] = answer["timings"].get("predicted_n", n)
                return answer["timings"]

            # a request that computes before the first part: CUDA's first report of a request holds the request
            # before it, and without one the first report of the part of depth 0 was its own prompt
            ask("start", 1, [{"role": "user", "content": "ok"}])
            # a part and a short reply of their own before the measured ones: the first use of a pipeline compiles it
            # (without the driver's cache: the shaders of a new build) and the GPU waits meanwhile; the perf logger
            # takes the time between two timestamps, so that wait went to the next op. 08.11, 395, depth 0 right after
            # a build: q5_K gate/up 150 ms in the first variant against 21.7 in the next one with the same pipeline.
            # the end of the corpus, a conversation of its own: the parts share no prefix with it
            n = int(part / ratio)
            ask("warm", 8, [{"role": "user", "content": text[len(text) - n:] + "\n\nWhat is this part about?"}])
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
        # the compute buffers the server reserved (the largest size of each buffer type over the run), MiB
        buffers = {}
        for m in re.finditer(r"(\S+) compute buffer size =\s*([\d.]+) MiB", log.read_text(encoding="utf-8", errors="replace")):
            buffers[m[1]] = max(buffers.get(m[1], 0.0), float(m[2]))
        # the parts the tables need: the fills are left out
        return {"buffers": buffers, "slices": {k: v for k, v in slices.items() if not k.startswith(("fill", "start", "warm"))}}

    results = []
    for name, arm_env in arms:
        run_env = dict(os.environ, LLAMA_INPUT_TIMING="1", **({"GGML_VK_PERF_LOGGER": "1"} if vulkan else {"GGML_CUDA_OP_TIMING": "idle"}),
                       **arm_env)
        identity = {"v": VERSION, "kind": "profile", "form": PROFILE_FORM, "build": build_id(exe.parent), "args": args,
                    "env": env_identity(run_env), "model": model_identity(model), "depths": depths, "part": part,
                    "tg": ntg, "corpus": text_digest(text), "machine": machine(),
                    "draft": model_identity(Path(env("SPECDRAFT"))) if mtp else None}
        label = "profile" if name == "default" else f"profile {name}"
        try:
            r_arm, kept_arm = run.measure(label, identity, lambda log, e=run_env: go(log, e))
        except Fail as e:
            run.failed = True
            run.add("", f"{label}: FAIL {e}")
            continue
        results.append((name, r_arm, profile_read(r_arm, depths, vulkan, mtp), kept_arm))
    if not results:
        return run.finish()
    nv = verify_size([x[2] for x in results], depths) if mtp else None
    step_name = f"verify of {nv} tokens" if nv else "decode step"
    name0, _, r, kept = results[0]
    k = " k" if kept else ""
    kinds = [kind for kind, _ in OP_KINDS] + ["other"]
    pp_cols, tg_cols, heads = {}, {}, []
    for d in depths:
        x = r.get(str(d))
        if not x:
            continue
        heads.append(str(x["depth"]))
        pp_cols[d] = by_kind(x["pp"])
        main = step_group(x, nv)
        tg_cols[d] = by_kind(main[2]) if main else {}
        pp_total = sum(o[2] for o in x["pp"])
        tps = f" ({rate(x['pp_tokens'], pp_total):.0f} t/s)" if x.get("pp_tokens") and pp_total else ""
        lines = [f"", f"at {x['depth']} tokens: prompt of {x.get('pp_tokens') or part} tokens {pp_total:.1f} ms of GPU time"
                      f"{tps} over {x['pp_graphs']} {'graphs' if vulkan else 'compute calls'}" + k]
        if x.get("host"):
            h = x["host"]
            lines.append(f"  host for the prompt: graph build {h[0]:.1f} ms, inputs {h[1]:.1f} ms, submit {h[2]:.1f} ms")
        for label, n, ops in sorted(x["tg_groups"], key=lambda g: g[0]):
            lines.append(f"  {label}: {sum(o[2] for o in ops):.2f} ms of GPU time (mean of {n})")
        if x.get("per_token") is not None:
            lines.append(f"  the reply: {x['per_token']:.2f} ms of GPU time per generated token" +
                         (f", draft acceptance {x['acceptance']:.2f}" if x.get("acceptance") is not None else ""))
        run.add(*lines)
    if pp_cols:
        cols = [d for d in depths if d in pp_cols]
        run.add("", "prompt: GPU ms per kind of op by depth",
                *table(["kind", *heads], [[kind, *[f"{pp_cols[d].get(kind, 0):.1f}" for d in cols]] for kind in kinds] +
                       [["total", *[f"{sum(pp_cols[d].values()):.1f}" for d in cols]]]))
        run.add("", f"{step_name} of the target: GPU ms per kind of op by depth",
                *table(["kind", *heads], [[kind, *[f"{tg_cols[d].get(kind, 0):.2f}" for d in cols]] for kind in kinds] +
                       [["total", *[f"{sum(tg_cols[d].values()):.2f}" for d in cols]]]))
        deepest = r.get(str(cols[-1]))
        run.add("", f"the ops of the prompt at {deepest['depth']} tokens",
                *table(["  ms", "count", "op"], [[f"  {ms:.2f}", f"{c:g}", name] for name, c, ms in deepest["pp"][:25]]))
        main = step_group(deepest, nv)
        if main:
            run.add("", f"the ops of the {main[0]} at {deepest['depth']} tokens",
                    *table(["  ms", "count", "op"], [[f"  {ms:.3f}", f"{c:g}", name] for name, c, ms in main[2][:25]]))

    def totals(x):
        main = step_group(x, nv)
        tg = by_kind(main[2]) if main else {}
        pp = by_kind(x["pp"])
        pp_ms = sum(o[2] for o in x["pp"])
        return {"pp": pp_ms, "pp_tps": x["pp_tokens"] / pp_ms * 1000 if x.get("pp_tokens") and pp_ms else None,
                "pp_qsa": pp.get("qsa select", 0.0), "tg": sum(tg.values()) if main else None,
                "tg_qsa": tg.get("qsa select", 0.0), "delta": tg.get("delta net", 0.0), "rows": tg.get("rows, copies", 0.0),
                "tok": x.get("per_token"), "acc": x.get("acceptance")}

    def device_mib(res):
        # the compute buffers on the devices, without the host ones
        return sum(v for k, v in res.get("buffers", {}).items() if not re.match(r"(?i)^(cpu|.*host|.*_Host)", k))

    tg_metric = f"gpu_verify{nv}" if nv else "gpu_tg"
    for name, _, rd, kept_arm in results:
        for d in depths:
            if str(d) in rd:
                t = totals(rd[str(d)])
                point = f"d{d}" if name == "default" else f"{name} d{d}"
                run.record("profile", point, {"gpu_pp": t["pp"], "gpu_pp_tps": t["pp_tps"], tg_metric: t["tg"],
                                              "gpu_tok": t["tok"]}, kept_arm, exe.parent)
    if len(results) > 1:
        rows = []
        f2 = lambda v, fmt: "-" if v is None else fmt % v
        for name, raw, rd, kept_arm in results:
            for d in depths:
                if str(d) in rd:
                    t = totals(rd[str(d)])
                    rows.append([name, str(rd[str(d)]["depth"]), f2(t["pp"], "%.1f"), f2(t["pp_tps"], "%.0f"),
                                 f2(t["pp_qsa"], "%.1f"), f2(t["tg"], "%.2f"), f2(t["tg_qsa"], "%.2f"), f2(t["delta"], "%.2f"),
                                 f2(t["rows"], "%.2f"), f2(t["tok"], "%.2f"), f2(t["acc"], "%.2f"),
                                 f"{device_mib(raw):.0f}" + (" k" if kept_arm else "")])
        run.add("", f"variants: GPU ms of the prompt (t/s) and of the {step_name} of the target, the block selection of "
                    "QSA in both, the delta net and the row copies of that step, GPU ms of the reply per generated token "
                    "(with MTP it depends on the acceptance, which differs with the text), the compute buffers on the "
                    "device (MiB)",
                *table(["variant", "depth", "prompt", "t/s", "qsa", nv and f"verify {nv}" or "decode", "qsa", "delta net",
                        "rows, copies", "per token", "accept", "compute"], rows))
    return run.finish()


# ---------------------------------------------------------------- micro

def cmd_micro():
    """The QSA indexer alone on the device of the machine (test-micro-fork): the lightning indexer, the head matmul of the
    Vulkan graph and of the other backends at the token counts of decode, MTP checks and prompts and the depths to the
    end of the context; microseconds per graph, TFLOP/s, GB/s and the largest difference to the first case.
    QA_MICRO_ARGS: arguments of the tool (-c, -t, -b, --ms); QA_MARMS: variants (name VAR=value ...;...), a run each."""
    run = Run("micro")
    exe = binary("test-micro-fork")
    device = env("CHECKDEVICE", DEVICES.get(env("BACKEND"), "CPU"))
    args = ["-d", device, *split_args(env("QA_MICRO_ARGS", ""))]
    arms = variants("QA_MARMS", "default")
    run.add(f"micro: the QSA indexer on {device}, build {current_build()}, {' '.join(args)}")
    for name, arm_env in arms:
        run_env = dict(os.environ, **arm_env)
        identity = {"v": VERSION, "kind": "micro", "build": build_id(exe.parent), "args": args,
                    "env": env_identity(run_env), "machine": machine()}
        label = "micro" if name == "default" else f"micro {name}"

        def go(log, run_env=run_env):
            execute([exe, *args], log, run_env)
            return [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if re.match(r"^(device |\s*blocks|\s*\d+\s+\d+\s)", l)]

        try:
            lines, kept = run.measure(label, identity, go)
        except Fail as e:
            run.failed = True
            run.add("", f"{label}: FAIL {e}")
            continue
        run.add("", label + (" k" if kept else ""), *lines)
        for l in lines:
            m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+([\d.]+)\s", l)
            if m:
                point = f"b{m[1]} t{m[2]} {m[3]}" if name == "default" else f"{name} b{m[1]} t{m[2]} {m[3]}"
                run.record("micro", point, {"us": float(m[4])}, kept, exe.parent)
    return run.finish()


# ---------------------------------------------------------------- mm

# the tiles of mul_mm_f16reg.comp under test: <r|l><rows>x<tokens> of a subgroup / subgroups along M x N; r: the codes
# decoded in registers, l: decoded to f16 in shared memory; a last n: the B tile of the next stage not held in registers,
# h: held as f16; at most 64 x 64 a subgroup. 07.11 run 1 (logs/395/mm-20261107-201543): best r16x64/8x1 (1324 ms of
# dense q8_0 per ubatch against 1470 of int8), the 64x64 tiles of a subgroup used scratch (544-640 bytes); run 2
# (mm-20261107-220919): the same without the B tile in f32 registers, 544 bytes of scratch at the least, r16x64/8x1
# 1305 ms against 1464. run 3: the sums named by constants in the shader instead of an array indexed in loops - do the
# 64x64 and 48x64 tiles fit now, and do they go past ~24 TFLOP/s. run 3 (mm-20261107-225950): they fit (no scratch
# but the 128-invocation 64x64/2x2), none passes r16x64/8x1: 1303 ms, r48x64/4x2 1361, l64x64/4x2n 1374, r64x64/4x2 1385
# run 4 (08.11): up to 8 B fragments a subgroup (128 tokens), so that one decoded A fragment serves 8 multiplies
# instead of 4 and a stage (two barriers) carries twice the work: does that pass the ~24-25 TFLOP/s of every 64-token
# tile. r16x64/8x1 is the default for comparison
MM_VARIANTS = ["r16x64/8x1", "r16x128/8x1", "r16x96/8x1", "r16x128/4x2", "r32x128/4x2", "l16x128/8x1"]


def cmd_mm():
    """The dense q8_0 matmuls of a prompt on the device of the machine (test-backend-ops): the default path and the
    variants of mul_mm_f16reg.comp (GGML_VK_MM_F16REG). For each variant first the MUL_MAT tests of q8_0 against the CPU,
    all of them through the variant (GGML_VK_MM_F16REG_MIN_N=1, _MIN_M=1, _MIN_K=1), then perf of the q8_0 shapes of qwen4exp at QA_MM_N
    tokens (4096): TFLOP/s per shape, and the registers, the scratch (registers spilled to memory) and the shared memory
    the driver gave the shader. QA_MMARMS: variants (name VAR=value ...;...), a run each, the int8 path first."""
    run = Run("mm")
    exe = binary("test-backend-ops")
    device = env("CHECKDEVICE", DEVICES.get(env("BACKEND"), "CPU"))
    n_tok = env("QA_MM_N", "4096")
    # the int8 path by name: since 08.11 r16x64/8x1 is the default on RDNA3
    arms = variants("QA_MMARMS", "int8 GGML_VK_MM_F16REG=0;" + ";".join(f"{v} GGML_VK_MM_F16REG={v}" for v in MM_VARIANTS))
    bid = build_id(exe.parent)
    run.add(f"mm: dense q8_0 matmuls on {device}, build {current_build()}, perf at {n_tok} tokens")
    test_rows, stat_rows, fails = [], [], []
    perf = {}      # (type_b, m, k) -> {arm: TFLOP/s}
    perf_us = {}   # (type_b, m, k) -> {arm: us per matmul}
    names = []
    for name, arm_env in arms:
        names.append(name)
        run_env = dict(os.environ, **arm_env)

        # the tests: every q8_0 case through the variant, its log line counts the matmuls it took
        test_env = dict(run_env, GGML_VK_MM_F16REG_MIN_N="1", GGML_VK_MM_F16REG_MIN_M="1", GGML_VK_MM_F16REG_MIN_K="1",
                        GGML_VK_MM_F16REG_LOG="1")
        cmd = [exe, "test", "-b", device, "-o", "MUL_MAT", "-p", "type_a=q8_0"]
        identity = {"v": VERSION, "kind": "mm-test", "build": bid, "args": [str(c) for c in cmd[1:]],
                    "env": env_identity(test_env), "machine": machine()}

        def go_test(log, cmd=cmd, test_env=test_env):
            code, text = execute(cmd, log, test_env, model=False, check=False)
            counts = re.findall(r"(\d+)/(\d+) tests passed", text)
            if not counts:
                raise Fail(f"{exit_text(code)}, no test result: {last_error(text)} ({log.name})")
            return {"passed": sum(int(a) for a, _ in counts), "total": sum(int(b) for _, b in counts),
                    "via": len(re.findall(r"MUL_MAT q8_0 \d+ x \d+ x \d+ on", text)),
                    "fails": [s.strip() for s in text.splitlines() if "FAIL" in s][:8],
                    # the tile and shared memory of the variant, or why it is off
                    "notes": [s.strip()[len("ggml_vulkan: "):] for s in text.splitlines()
                              if s.startswith("ggml_vulkan: ") and ("GGML_VK_MM_F16REG" in s or "tiles of" in s)][:3]}

        try:
            r, kept = run.measure(f"mm test {name}", identity, go_test)
        except Fail as e:
            run.failed = True
            test_rows.append([name, f"FAIL {e}"])
            continue
        ok = r["passed"] == r["total"]
        run.failed |= not ok
        run.record("mm", f"test {name}", {"passed": r["passed"], "total": r["total"], "via": r["via"]}, kept, exe.parent)
        test_rows.append([name, f"{r['passed']}/{r['total']}" + ("" if ok else " FAIL") + (" k" if kept else ""),
                          str(r["via"]), "; ".join(r["notes"])])
        fails.extend(f"  {name}: {s}" for s in r["fails"])
        if not ok:
            continue
        # a variant the device refused (shared memory, coopmat) would measure the default path under its name
        if arm_env.get("GGML_VK_MM_F16REG", "0") not in ("", "0") and r["via"] == 0:
            test_rows[-1][1] += " (off: no matmul took it, no perf)"
            continue

        # perf of the q8_0 shapes and the statistics of the pipelines the variant compiled
        perf_env = dict(run_env, GGML_VK_PIPELINE_STATS="mul_mm_f16reg")
        cmd = [exe, "perf", "-b", device, "-o", "MUL_MAT", "-p", f"type_a=q8_0,type_b=f(32|16),m=\\d+,n={n_tok},k="]
        identity = {"v": VERSION, "kind": "mm-perf", "build": bid, "args": [str(c) for c in cmd[1:]],
                    "env": env_identity(perf_env), "machine": machine()}

        def go_perf(log, cmd=cmd, perf_env=perf_env):
            execute(cmd, log, perf_env, model=False)
            text = read_log(log)
            shapes, stats, current = [], {}, None
            for line in text.splitlines():
                m = re.search(r"MUL_MAT\(type_a=q8_0,type_b=(\w+),m=(\d+),n=(\d+),k=(\d+).*?([\d.]+) us/run.*?([\d.]+) ([GT])FLOPS", line)
                if m:
                    tf = float(m[6]) / (1000.0 if m[7] == "G" else 1.0)
                    shapes.append({"b": m[1], "m": int(m[2]), "n": int(m[3]), "k": int(m[4]), "us": float(m[5]), "tflops": tf})
                    continue
                m = re.match(r"ggml_vulkan: pipeline stats for (\S+):", line)
                if m:
                    current = m[1]
                    stats[current] = {}
                    continue
                m = re.match(r"ggml_vulkan:\s{2,}(.+?): (.+)$", line)
                if m and current:
                    stats[current][m[1].strip()] = m[2].strip()
            if not shapes:
                raise Fail(f"no perf line ({log.name})")
            return {"shapes": shapes, "stats": stats}

        try:
            r, kept = run.measure(f"mm perf {name}", identity, go_perf)
        except Fail as e:
            run.failed = True
            test_rows.append([name, f"perf FAIL {e}"])
            continue
        for s in r["shapes"]:
            perf.setdefault((s["b"], s["m"], s["k"]), {})[name] = s["tflops"]
            perf_us.setdefault((s["b"], s["m"], s["k"]), {})[name] = s["us"]
            run.record("mm", f"{name} {s['b']} {s['m']}x{s['k']} n{s['n']}", {"tflops": s["tflops"], "us": s["us"]}, kept, exe.parent)
        for pipe, st in r["stats"].items():
            keep = {k: v for k, v in st.items() if re.search(r"vgpr|sgpr|scratch|spill|lds|shared|wave|subgroup", k, re.I)}
            stat_rows.append([name, pipe, ", ".join(f"{k} {v}" for k, v in keep.items()) or ", ".join(f"{k} {v}" for k, v in st.items())])

    run.add("", "tests: MUL_MAT q8_0 against the CPU, passed / cases, matmuls through the variant", *table(["variant", "tests", "via", "note"], test_rows))
    if fails:
        run.add("", "failed:", *fails)
    if perf:
        rows = []
        for (b, mm_, k), by in sorted(perf.items(), key=lambda kv: (kv[0][0], -kv[0][1] * kv[0][2])):
            best = max(by.values())
            rows.append([f"q8_0 x {b} {mm_}x{k}", *[(f"{by[a]:.1f}" + ("*" if by[a] == best else "")) if a in by else "" for a in names]])
        # the dense q8_0 matmuls of one 4096-token ubatch of qwen4exp as the 31.10 profile of the 395 counts them (the hc
        # down projection reads f16 with LLAMA_HC_F16): their time on each variant
        counts = {(10240, 2560): 37, (2560, 6144): 48, (6144, 2560): 36, (12288, 2560): 12, (320, 10240): 96,
                  (10240, 320): 96, (640, 2560): 96, (2560, 640): 48, (512, 2560): 24, (2560, 2560): 1}
        if n_tok == "4096":
            total = []
            for a in names:
                ms = 0.0
                for (mm_, k), c in counts.items():
                    us = perf_us.get(("f16", mm_, k), {}).get(a) if (mm_, k) == (320, 10240) else None
                    us = us if us is not None else perf_us.get(("f32", mm_, k), {}).get(a)
                    if us is None:
                        ms = None
                        break
                    ms += c * us / 1000.0
                total.append(f"{ms:.0f}" if ms is not None else "")
            rows.append(["ubatch ms (31.10 counts)", *total])
        run.add("", f"TFLOP/s at {n_tok} tokens (* best of the shape); the last row: ms of the dense q8_0 matmuls of a "
                    f"4096-token ubatch of qwen4exp", *table(["shape", *names], rows))
    if stat_rows:
        run.add("", "pipeline statistics of the driver", *table(["variant", "pipeline", "statistics"], stat_rows))
    return run.finish()


# ---------------------------------------------------------------- kld: this build against upstream

# the lines of llama-perplexity --kl-divergence (tools/perplexity/perplexity.cpp)
# (a mean KLD of equal logits rounds to -0.000001). PPL(base) there comes from the log-probs of the base stored in 16
# bits with a floor, and a token below it counts as more likely than it was: the ratio takes the PPL that upstream
# itself printed when it wrote the base (Final estimate)
KLD_STATS = {"ppl": r"Mean PPL\(Q\)\s*:\s*(-?[0-9.]+)", "kld_mean": r"Mean\s+KLD:\s*(-?[0-9.]+)",
             "kld_median": r"Median\s+KLD:\s*(-?[0-9.]+)", "kld_p99": r"99\.0%\s+KLD:\s*(-?[0-9.]+)",
             "kld_max": r"Maximum KLD:\s*(-?[0-9.]+)", "top1": r"Same top p:\s*(-?[0-9.]+)"}
KLD_BASE_PPL = r"Final estimate: PPL = ([0-9.]+)"


def kld_read(log):
    text = read_log(log)
    out = {}
    for key, pattern in KLD_STATS.items():
        m = re.search(pattern, text)
        if m is None:
            raise Fail(f"no {key} in {log}: {last_error(text)}")
        out[key] = float(m[1])
    return out


# QA_KLD_BASE=exact: the base is this build with every matmul in f32 (Vulkan: no int8, no f16 shaders, no coopmat, the
# f16 paths of the fork off), the exact math of the same weights, against which the 8-bit and 16-bit activations of an
# arm show their own cost (12.11; the base of upstream carries the q8_1 activations of upstream itself)
KLD_EXACT_ENV = {
    "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT": "1", "GGML_VK_DISABLE_F16": "1", "GGML_VK_DISABLE_COOPMAT": "1",
    "GGML_VK_DISABLE_COOPMAT2": "1", "GGML_VK_MM_F16REG": "0", "GGML_VK_MMID_F16REG": "0", "GGML_VK_MMV_ID_FORK": "0",
    "LLAMA_HC_F16": "0", "LLAMA_MOE_DOWN_F16": "0",
}


def kld_arms():
    """QA_KARMS: "name [@upstream] [VAR=value ...];...", @upstream runs the upstream build. QA_ARM_UB=<n> in an arm sets
    its ubatch: 1 to 8 tokens take the decode kernels (mat-vec, the expert mat-vec of the fork), which a prompt-sized
    ubatch never runs. "QA_ARM_ARGS=<args>" (quoted whole): arguments of the arm after the model arguments"""
    out = []
    for part in env("QA_KARMS", "fork;upstream @upstream").split(";"):
        tokens = split_args(part)
        if not tokens:
            continue
        up = "@upstream" in tokens[1:]
        out.append((tokens[0], up, env_pairs_tokens([t for t in tokens[1:] if t != "@upstream"], "QA_KARMS")))
    return out


def kld_before(run, arm, bid):
    """the last result of this arm in history-qa.csv from another build: the state before a merge"""
    path = run.dir.parent / "history-qa.csv"
    last = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["test"] == "kld" and row["point"] == arm and row["build"] != bid and row["host"] == platform.node():
                    if last.get("build") != row["build"]:
                        last = {"build": row["build"], "head": row["head"]}
                    last[row["metric"]] = float(row["value"])
    except (OSError, ValueError, KeyError):
        return {}
    return last


def cmd_kld():
    """The logits of this build against those of upstream (00-build.bat upstream, UPSTREAMREF, default master) on the
    same model, text and arguments: llama-perplexity of upstream writes the base, each arm of QA_KARMS compares with it.
    The default arms: this build as it runs, and upstream again (the noise of the base: 0 when upstream repeats itself)"""
    run = Run("kld")
    model = model_path()
    text = Path(env("PPLFILE"))
    if not text.is_file():
        raise Fail(f"no text {text}: run get-wikitext.bat or set PPLFILE")
    up_folder = ref_folder("upstream")
    up_exe = up_folder / ("llama-perplexity" + EXE)
    base_kind = env("QA_KLD_BASE", "upstream")
    if base_kind not in ("upstream", "exact"):
        raise Fail(f"QA_KLD_BASE={base_kind}: upstream or exact")
    if base_kind == "exact" and env("BACKEND") == "cuda":
        raise Skip("QA_KLD_BASE=exact: no f32 mode of the CUDA matmuls yet")
    uses_up = base_kind == "upstream" or any(up for _, up, _ in kld_arms())
    if uses_up and not up_exe.is_file():
        raise Skip(f"no {up_exe}: build upstream with 00-build.bat upstream")
    # the upstream of the comparison is UPSTREAMREF: a build of another commit (an older merge, a failed build of the
    # new one that left the old binaries) would measure the distance to something else
    if uses_up:
        want = env("UPSTREAMREF", "master")
        try:
            commit = git("rev-parse", f"{want}^{{commit}}").decode().strip()
        except (OSError, subprocess.CalledProcessError):
            commit = ""
        up_head, _ = build_head(up_folder)
        if commit and up_head and not commit.startswith(up_head):
            raise Fail(f"the upstream build is {up_head}, UPSTREAMREF {want} is {commit[:12]}: run 00-build.bat upstream")
    base_folder = up_folder if base_kind == "upstream" else Path(env("BIN"))
    base_exe = up_exe if base_kind == "upstream" else binary("llama-perplexity")
    ctx = env_int("QA_KLD_CTX", 8192)
    chunks = env_int("QA_KLD_CHUNKS", 2)
    ub = env("QA_KLD_UB", "8192" if env("BACKEND") == "cuda" else "4096")
    args = ["-f", str(text), "-c", str(ctx), "-b", str(ctx), "-ub", ub, "--chunks", str(chunks),
            *split_args(env("QA_KLD_ARGS"))]
    margs = model_args(model)
    base_margs = upstream_args(margs) if base_kind == "upstream" else margs
    base_env = os.environ.copy()
    if base_kind == "exact":
        base_env.update(KLD_EXACT_ENV)
    base_env.update(env_pairs(env("QA_KLD_BASE_ENV"), "QA_KLD_BASE_ENV"))
    ident_base = {"v": VERSION, "kind": "kld-base", "base": base_kind, "build": build_id(base_folder),
                  "model": model_identity(model), "text": digest(text), "args": [str(a) for a in base_margs + args],
                  "env": env_identity(base_env)}
    folder = CACHE / "qa"
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / (identity_key(ident_base)[:24] + ".kld")
    head_base, _ = build_head(base_folder)
    base_desc = "upstream" if base_kind == "upstream" else "this build, every matmul in f32 (QA_KLD_BASE=exact)"
    run.add(f"base: {base_desc} {head_base or build_id(base_folder)}, {chunks} chunks of {ctx} tokens of {text.name} "
            f"(the second half of each is compared), ubatch {ub}, {' '.join(map(str, base_margs[2:]))}")
    base_ppl = base.with_suffix(".ppl")
    if forced("kld base") or not base.is_file() or not base_ppl.is_file():
        part = base.with_suffix(".part")
        print(f"kld base: running {base_kind}, {base.name}", flush=True)
        execute([base_exe, *base_margs, *args, "--kl-divergence-base", part], run.dir / "base.log", base_env)
        m = re.search(KLD_BASE_PPL, read_log(run.dir / "base.log"))
        if m is None:
            raise Fail(f"no PPL in {run.dir / 'base.log'}")
        part.replace(base)
        base_ppl.write_text(m[1] + "\n", encoding="utf-8")
    ppl_up = float(base_ppl.read_text(encoding="utf-8"))
    run.add(f"base PPL {ppl_up:.4f}")
    rows = []
    for name, up, arm_env in kld_arms():
        exe = up_exe if up else binary("llama-perplexity")
        bfolder = up_folder if up else Path(env("BIN"))
        arm_margs = upstream_args(margs) if up else margs
        arm_env = dict(arm_env)
        arm_ub = arm_env.pop("QA_ARM_UB", None)
        # QA_ARM_ARGS: arguments of this arm after the model arguments, a later value of an option wins (12.11: -lzm on
        # for this build, as upstream reads the rows of the per-layer embeddings)
        arm_extra = split_args(arm_env.pop("QA_ARM_ARGS", ""))
        if up:
            arm_extra = upstream_args(arm_extra)
        arm_args = args if arm_ub is None else [a if i == 0 or args[i - 1] != "-ub" else arm_ub for i, a in enumerate(args)]
        run_env = os.environ.copy()
        run_env.update(arm_env)
        identity = {"v": VERSION, "kind": "kld", "build": build_id(bfolder), "base": ident_base,
                    "env": env_identity(run_env), "args": [str(a) for a in arm_args]}
        if arm_extra:
            identity["arm_args"] = arm_extra

        def measure(log, exe=exe, run_env=run_env, arm_args=arm_args, arm_margs=arm_margs, arm_extra=arm_extra):
            execute([exe, *arm_margs, *arm_extra, *arm_args, "--kl-divergence-base", base, "--kl-divergence"], log,
                    run_env)
            return dict(kld_read(log), ppl_up=ppl_up)

        result, kept = run.measure(f"kld {name}", identity, measure)
        result["ppl_ratio"] = result["ppl"] / result["ppl_up"]
        run.record("kld", name, result, kept, bfolder)
        before = kld_before(run, name, build_id(bfolder))
        prev = (f"{before['kld_mean']:.6f} {before['ppl_ratio']:.4f} ({before['head'] or before['build']})"
                if "kld_mean" in before and "ppl_ratio" in before else "-")
        rows.append([name + (" (k)" if kept else ""), f"{result['ppl']:.4f}", f"{result['ppl_ratio']:.4f}",
                     f"{result['kld_mean']:.6f}", f"{result['kld_median']:.6f}", f"{result['kld_p99']:.5f}",
                     f"{result['kld_max']:.3f}", f"{result['top1']:.2f}", prev])
    base_name = "upstream" if base_kind == "upstream" else "exact"
    run.add("", *table(["arm", "PPL", f"PPL/{base_name}", "KLD mean", "KLD median", "KLD p99", "KLD max", "top-1 %",
                        f"before: KLD mean, PPL/{base_name} (build)"], rows),
            "", f"KLD to the logits of the base ({base_desc}) on the same model and text; PPL/{base_name} above 1 is "
                "worse than the base. before: the last result of the arm from another build on this machine "
                "(history-qa.csv)")
    return run.finish()


# ---------------------------------------------------------------- main

COMMANDS = {"check": cmd_check, "long": cmd_long, "profile": cmd_profile, "micro": cmd_micro, "mm": cmd_mm, "decode": cmd_decode,
            "kld": cmd_kld, "build-begin": cmd_build_begin, "build-end": cmd_build_end, "build-stable": cmd_build_stable,
            "build-upstream": cmd_build_upstream}


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
