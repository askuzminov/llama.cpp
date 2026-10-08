#!/usr/bin/env python3
"""Checks and measurements of this fork on the 395 (Vulkan) and the 3090 (CUDA), see README.md.

Each result is kept in cache/ with the identity of what made it: the source of the build (00-build.bat records it),
the arguments, the GGML_/LLAMA_/CUDA_ environment, the model files, the inputs and the machine. When that identity
did not change, the script prints the kept result and does not run the measurement again. FORCE=1 runs it again."""

import argparse
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
# the speed results of 02 are kept under their own version: 2 since 07.10, when 02 got its warmup process (the
# numbers before it had a cold first process in every step), 3 since 07.10 evening, when the warmup started to read
# the PLE rows of every token llama-bench draws (warm_page_cache)
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
    (DECPREFIX), @c= and @ub= set the -c and the -ub of 02 and 05 for this arm, @o=OP,... limits it to these ops of
    01 and 06, the rest goes to the tool."""

    def __init__(self, text):
        tokens = split_args(text)
        if not tokens:
            raise Fail("empty arm")
        self.name = tokens[0]
        self.env = {}
        self.args = []
        self.build = None
        self.stable = False
        self.draft = False
        self.depths = None
        self.prefixes = None
        self.ctx = None
        self.ub = None
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
    directories of the other backends are left out, so a change there keeps the results of this machine."""
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
    h.update(git("diff", "HEAD", "--binary", "--", *entries))
    for path in git("ls-files", "--others", "--exclude-standard", "--", *entries).decode().splitlines():
        h.update(f"{path} {digest(ROOT / path)}\n".encode())
    return "src-" + h.hexdigest()[:16]


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


def settle():
    try:
        wait = float(env("SETTLE", "30")) - (time.time() - float(LAST_RUN.read_text()))
    except (OSError, ValueError):
        return
    if wait > 0:
        print(f"waiting {wait:.0f} s for the driver to release the memory of the last model run", flush=True)
        time.sleep(wait)


def mark_end():
    CACHE.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(f"{time.time()}\n")


def exit_text(code):
    return f"exit {code}" if 0 <= code < 256 else f"exit {code & 0xffffffff:#010x}"


def last_error(text):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    errors = [s for s in lines if re.search(r"error|exception|failed|abort", s, re.IGNORECASE)]
    if errors:
        return errors[-1][:200]
    return "no error message, the log ends with: " + (lines or [""])[-1][:160]


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
    # an exception exit code without a message: the process died before it could print one (a load or an abort)
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
            wait = float(env("SETTLE", "30")) * attempt
            print(f"{log.name}: {exit_text(code)}, {last_error(text)}; one more try in {wait:.0f} s", flush=True)
            time.sleep(wait)
            log = first.with_name(f"{first.stem}-try{attempt + 1}.log")
            continue
        raise Fail(f"{exit_text(code)}: {last_error(text)} ({log.name}, try {attempt})")


class Run:
    """One call of a script: its log folder, its summary and the counts of new and kept results."""

    def __init__(self, name):
        self.dir = Path(env("LOGS", str(HERE / "logs"))) / f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
        self.dir.mkdir(parents=True, exist_ok=True)
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

def warm_page_cache(run, model, ub, done):
    """One llama-bench process that is not measured, before the first measured one of a step. llama-bench draws its
    tokens with std::rand(): the same tokens in every process, below 32768 with msvc. The PLE rows of those tokens
    (llama_row_cache, a 26.8 GiB table on disk) come from the page cache once read, from the drive before. 07.10: the
    arm that ran first read them cold, 530 against 170 ms of inputs per 4096-token ubatch, PP 786 against 840 t/s at
    depth 0 on the 395, TG -6 percent at 32768 on the 3090. BENCHWARMTOKENS draws (98304: 24 prompts of 4096) read
    95 percent of those rows, and the gpu clocks are up after it."""
    if done or env("BENCHWARMUP", "1") != "1":
        return
    done.append(True)
    exe = Arm("warmup").binary("llama-bench")
    reps = max(1, -(-env_int("BENCHWARMTOKENS", 98304) // ub))
    print(f"warmup: one llama-bench process of {reps} prompts of {ub} tokens, not measured", flush=True)
    try:
        execute([exe, *model_args(model), "-c", 16384, "-b", ub, "-ub", ub, "-p", ub, "-n", 0, "-r", reps, "-o", "json"],
                run.dir / "warmup.log", os.environ.copy(), stdout=run.dir / "warmup.json")
    except Fail as e:
        print(f"warmup: {e}", flush=True)


def cmd_bench():
    run = Run("bench")
    model = model_path()
    depths = env_ints("BENCHDEPTHS", "0,32768,65536,122880")
    pp, tg = env_int("BENCHPP", 4096), env_int("BENCHTG", 64)
    ub, reps = env_int("BENCHUB", 4096), env_int("BENCHREPS", 3)
    # auto: the context of each depth is just large enough, as --phase-mem sizes the buffers in the server. a fixed
    # -c reserves the compute buffer for the full context at every depth
    auto = env("BENCHCTX", "262144") == "auto"
    ctx = 0 if auto else env_int("BENCHCTX", 262144)
    arms = parse_arms("BENCHARMS", "default")
    if not auto and max(depths) + max(pp, tg) > ctx:
        raise Fail("the deepest BENCHDEPTHS plus the prompt does not fit BENCHCTX")

    warmed = []

    def warm_up():
        warm_page_cache(run, model, ub, warmed)
    results = {}
    for i, depth in enumerate(depths):
        # the order turns at each depth, so that no arm always runs right after the same one
        for arm in (arms if i % 2 == 0 else arms[::-1]):
            label = f"{arm.name}-d{depth}"
            if arm.depths is not None and depth not in arm.depths:
                continue
            try:
                exe = arm.binary("llama-bench")
            except Skip as e:
                results[arm.name, depth] = f"skip: {e}"
                continue
            c = arm.ctx or (-(-(depth + max(pp, tg)) // 256) * 256 if auto else ctx)
            u = arm.ub or ub
            cmd = [exe, *model_args(model), "-c", c, "-b", max(pp, u), "-ub", u, "-p", pp, "-n", tg, "-d", depth,
                   "-r", reps, "-o", "json", *arm.args]
            run_env = arm_env(arm, {"GGML_SCHED_LOG_REALLOC": "1"})
            identity = {"v": BENCH_VERSION, "kind": "bench", "build": build_id(exe.parent), "args": [str(c) for c in cmd[1:]],
                        "env": env_identity(run_env), "model": model_identity(model), "machine": machine()}

            def go(log):
                warm_up()
                out = log.with_suffix(".json")
                _, text = execute(cmd, log, run_env, stdout=out)
                lines = text.splitlines()
                r = {"replans": text.count("sched re-reserve:"), "graph_timing": graph_timing(lines),
                     "host_tg": host_timing(lines, 1)}
                for row in json.loads(out.read_text(encoding="utf-8")):
                    samples = sorted(row.get("samples_ts") or [row["avg_ts"]])
                    r["pp" if row["n_prompt"] else "tg"] = [row["avg_ts"], row["stddev_ts"], statistics.median(samples),
                                                            samples[0], samples[-1]]
                if (pp and "pp" not in r) or (tg and "tg" not in r):
                    raise Fail(f"llama-bench gave no PP or TG row ({log.name})")
                return r

            try:
                results[arm.name, depth] = run.measure(label, identity, go)
            except Fail as e:
                run.failed = True
                results[arm.name, depth] = f"FAIL {e}"
    run.add(f"bench: llama-bench pp{pp} tg{tg} -ub {ub} -c {'depth + pp' if auto else ctx}, median of {reps} reps "
            f"(min-max), {model.name}, build {current_build()}")
    for kind in [k for k, n in (("pp", pp), ("tg", tg)) if n]:
        rows = []
        for depth in depths:
            row = [depth]
            for arm in arms:
                r = results.get((arm.name, depth))
                if isinstance(r, tuple):
                    _, _, med, lo, hi = r[0][kind]
                    row.append(f"{med:.2f} ({lo:.1f}-{hi:.1f})" + (" k" if r[1] else ""))
                else:
                    row.append("" if r is None else "FAIL" if r.startswith("FAIL") else "skip")
            rows.append(row)
        run.add("", f"{kind.upper()} t/s", *table(["depth", *[a.name for a in arms]], rows))
    # prompts of an agent turn at depth 0: one llama-bench run per arm over the sizes of BENCHSMALL. a small -c is
    # enough here, the graph of a short prompt at depth 0 does not depend on it
    small = env_ints("BENCHSMALL", "") if env("BENCHSMALL") else []
    if small:
        small_arms = parse_arms("BENCHSMALLARMS", "default")
        small_ctx = env_int("BENCHSMALLCTX", 16384)
        small_res = {}
        for arm in small_arms:
            try:
                exe = arm.binary("llama-bench")
            except Skip as e:
                small_res[arm.name] = f"skip: {e}"
                continue
            cmd = [exe, *model_args(model), "-c", arm.ctx or small_ctx, "-b", max(arm.ub or ub, max(small)), "-ub", arm.ub or ub,
                   "-p", ",".join(map(str, small)), "-n", 0, "-r", reps, "-o", "json", *arm.args]
            run_env = arm_env(arm)
            identity = {"v": BENCH_VERSION, "kind": "bench-small", "build": build_id(exe.parent),
                        "args": [str(c) for c in cmd[1:]], "env": env_identity(run_env), "model": model_identity(model),
                        "machine": machine()}

            def go(log):
                warm_up()
                out = log.with_suffix(".json")
                execute(cmd, log, run_env, stdout=out)
                r = {}
                for row in json.loads(out.read_text(encoding="utf-8")):
                    samples = sorted(row.get("samples_ts") or [row["avg_ts"]])
                    r[str(row["n_prompt"])] = [statistics.median(samples), samples[0], samples[-1]]
                if any(str(n) not in r for n in small):
                    raise Fail(f"llama-bench gave no row for every size of BENCHSMALL ({log.name})")
                return r

            try:
                small_res[arm.name] = run.measure(f"{arm.name}-small", identity, go)
            except Fail as e:
                run.failed = True
                small_res[arm.name] = f"FAIL {e}"
        rows = []
        for n in small:
            row = [n]
            for arm in small_arms:
                r = small_res.get(arm.name)
                if isinstance(r, tuple):
                    med, lo, hi = r[0][str(n)]
                    row.append(f"{med:.1f} ({lo:.1f}-{hi:.1f})" + (" k" if r[1] else ""))
                else:
                    row.append("skip" if r is None or r.startswith("skip") else "FAIL")
            rows.append(row)
        run.add("", f"PP t/s of short prompts at depth 0, -c {small_ctx}", *table(["prompt", *[a.name for a in small_arms]], rows))
        results.update({(a, "small"): r for a, r in small_res.items() if isinstance(r, str)})
    notes = sorted({f"  {a}: {r}" for (a, d), r in results.items() if isinstance(r, str) and r.startswith("skip")})
    notes += [f"  {a}-d{d}: {r}" for (a, d), r in results.items() if isinstance(r, str) and not r.startswith("skip")]
    notes += [f"  {a}-d{d}: {r[0]['replans']} re-plans of the buffers while computing" for (a, d), r in results.items()
              if isinstance(r, tuple) and r[0]["replans"]]
    notes += [f"  {a}-d{d}: {graph_timing_text(r[0]['graph_timing'])}" for (a, d), r in results.items()
              if isinstance(r, tuple) and r[0].get("graph_timing")]
    notes += [f"  {a}-d{d}: host per decode step (LLAMA_INPUT_TIMING), median of {h['n']}: graph build {h['graph']:.2f} ms, "
              f"inputs {h['inputs']:.2f} ms, submit {h['submit']:.2f} ms" for (a, d), r in results.items()
              if isinstance(r, tuple) and (h := r[0].get("host_tg"))]
    if notes:
        run.add("", "notes:", *notes)
    return run.finish()


# ---------------------------------------------------------------- quality

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


def serve(exe, args, run_env, log, requests, sampling, port, rows_text=""):
    """Start llama-server, send the warmup and the requests one after another, stop it. The answers stay next to
    the log. rows_text, one token long and not measured, reads the PLE rows of the prompts into the page cache first:
    otherwise the arm that runs first reads them from the drive (see warm_page_cache)."""
    wait, timeout = env_int("DECWAIT", 900), env_int("DECTIMEOUT", 1800)
    url = f"http://127.0.0.1:{port}"
    run_env = dict(run_env, LLAMA_SPEC_TRACE=str(log.with_suffix(".trace.csv")))
    no_server_running()
    settle()
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"command": [str(exe), *args], "env": env_identity(run_env)}) + "\n")
        f.flush()
        proc = subprocess.Popen([str(exe), *args], stdout=f, stderr=f, env=run_env)
    rows = []
    try:
        start = time.time()
        while True:
            time.sleep(3)
            if proc.poll() is not None:
                raise Fail(f"llama-server {exit_text(proc.returncode)} while loading: "
                           f"{last_error(read_log(log))} ({log.name})")
            try:
                if http(url + "/health")[0] == 200:
                    break
            except (urllib.error.URLError, OSError):
                pass
            if time.time() - start > wait:
                raise Fail(f"no answer from /health in {wait} s ({log.name})")
        temp, top_k, top_p = sampling
        warm = [("warmup", WARMUP, 0, 256)] + ([("rows", rows_text, 0, 1)] if rows_text else [])
        for name, text, seed, n in [*warm, *requests]:
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": temp,
                    "top_k": top_k, "top_p": top_p, "seed": seed, "cache_prompt": False, "stream": False}
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
            rows.append({"name": f"{name}-s{seed}", "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"],
                         "n": t["predicted_n"], "ms": t["predicted_ms"], "draft_n": t.get("draft_n", 0),
                         "draft_acc": t.get("draft_n_accepted", 0),
                         "answer": text_digest(f"{message.get('content')}\0{message.get('reasoning_content')}")[:16]})
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
            "graph_timing": graph_timing(lines)}


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
    prefixes = env_ints("DECPREFIX", "0")
    corpus = Path(env("PPLFILE")).read_text(encoding="utf-8", errors="replace") if max(prefixes) > 0 else ""
    results = {}
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
            args = [str(a) for a in ["-m", model, *(["-md", draft] if arm.draft else []), "-c", ctx, "-np", 1,
                    "-fa", "on", "--repeat-penalty", "1.0", *split_args(env("LOADMODE")), *split_args(env("EXTRA")),
                    *warmup_args(), *split_args(env("DECARGS")), *arm.args, "--host", "127.0.0.1", "--port", port, "-lv", 4]]
            run_env = arm_env(arm)
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
        for pair in split_args(env("DECSAME")):
            a, _, b = pair.partition(":")
            ra, rb = results.get((a, prefix)), results.get((b, prefix))
            if not (isinstance(ra, tuple) and isinstance(rb, tuple)):
                run.add(f"same answers {pair}: no result for both arms")
                continue
            other = [x["name"] for x, y in zip(ra[0]["rows"], rb[0]["rows"]) if x["answer"] != y["answer"]]
            run.failed |= bool(other)
            run.add(f"same answers {pair}: " + ("all" if not other else "differ in " + " ".join(other)))
    run.add(*details)
    return run.finish()


# ---------------------------------------------------------------- profile

def op_group(name):
    return re.split(r"[ (]", name, 1)[0]


def cmd_profile():
    run = Run("profile")
    model = model_path()
    vulkan = env("BACKEND") == "vulkan"
    depths, pp, tg = env_ints("PROFDEPTH", "65536"), env_int("PROFPP", 4096), env_int("PROFTG", 4)
    ub = env_int("PROFUB", 4096)
    # auto: -c of each depth is depth + prompt; a profile asks where the time of a graph goes, and the buffers of a
    # full context put the 395 at the edge of its memory (05.10: ErrorOutOfDeviceMemory at 122880)
    prof_auto = env("PROFCTX", "auto") == "auto"
    ctx = 0 if prof_auto else env_int("PROFCTX", 262144)
    arms = parse_arms("PROFARMS", "default")
    run.add(f"profile: {'GGML_VK_PERF_LOGGER per op' if vulkan else 'GGML_SCHED_PROF per split'}, host time per ubatch "
            f"from LLAMA_INPUT_TIMING, llama-bench -ub {ub} -c {'depth + prompt' if prof_auto else ctx}, {model.name}, "
            f"build {current_build()}")
    # a kind is pp (PROFPP tokens), ppN (N tokens) or tg, with @d1,d2 for its own depths instead of PROFDEPTH
    plan = []
    for k in split_args(env("PROFKINDS", "pp")):
        name, _, at = k.partition("@")
        if not re.fullmatch(r"pp\d*|tg", name):
            raise Fail(f"PROFKINDS: {k} is not pp, ppN or tg")
        plan += [(name, d) for d in ([int(x) for x in at.split(",") if x] if at else depths)]
    warmed = []
    for kind, depth in plan:
        n_pp = int(kind[2:]) if kind.startswith("pp") and len(kind) > 2 else pp
        done = {}
        for arm in arms:
            label = f"{kind}-d{depth}-{arm.name}"
            if arm.depths is not None and depth not in arm.depths:
                continue
            try:
                exe = arm.binary("llama-bench")
            except Skip as e:
                run.add("", f"{label}: skip: {e}")
                continue
            sizes = ["-p", n_pp, "-n", 0] if kind != "tg" else ["-p", 0, "-n", tg]
            # two reps: the last graph then comes from the second, warm, without the warmup run (WARMUP); the depth
            # state of the first rep is restored, not computed again
            c = arm.ctx or (-(-(depth + max(n_pp, tg)) // 256) * 256 if prof_auto else ctx)
            cmd = [exe, *model_args(model), "-c", c, "-b", max(n_pp, arm.ub or ub), "-ub", arm.ub or ub, "-d", depth, *sizes, "-r", 2,
                   *arm.args]
            run_env = arm_env(arm, {"LLAMA_INPUT_TIMING": "1",
                                    **({"GGML_VK_PERF_LOGGER": "1"} if vulkan else {"GGML_SCHED_PROF": "1"})})
            identity = {"v": VERSION, "kind": "profile", "build": build_id(exe.parent), "args": [str(c) for c in cmd[1:]],
                        "env": env_identity(run_env), "model": model_identity(model), "machine": machine()}
            n_tokens = min(n_pp, arm.ub or ub) if kind != "tg" else 1

            def go(log):
                # the host time of the inputs depends on the PLE rows in the page cache (warm_page_cache)
                warm_page_cache(run, model, ub, warmed)
                _, text = execute(cmd, log, run_env)
                lines = text.splitlines()
                # host time of the last ubatches of the measured size; a build without LLAMA_INPUT_TIMING has none
                host = [[float(m[2]), float(m[3]), float(m[4])] for m in
                        (re.match(r"ubatch timing: (\d+) tokens, graph ([\d.]+) ms, inputs ([\d.]+) ms, submit ([\d.]+) ms", x)
                         for x in lines) if m and int(m[1]) == n_tokens][-3:]
                host = [statistics.median(h[i] for h in host) for i in range(3)] if host else None
                if not vulkan:
                    return {"lines": [x.strip() for x in lines if "sched_prof" in x][-4:], "host": host}
                # the last graph is the measured one: the ubatch of the prompt or the last decode step at the depth
                starts = [i for i, x in enumerate(lines) if x.startswith("Vulkan Timings:")]
                if not starts:
                    raise Fail(f"no Vulkan Timings in the log ({log.name})")
                ops = []
                for x in lines[starts[-1] + 1:]:
                    m = re.match(r"(.+): (\d+) x [\d.e+-]+ us = ([\d.e+-]+) us", x)
                    if m:
                        ops.append([m[1], int(m[2]), float(m[3]) / 1000])
                return {"ops": sorted(ops, key=lambda o: -o[2]), "host": host}

            try:
                r, kept = run.measure(label, identity, go)
            except Fail as e:
                run.failed = True
                run.add("", f"{label}: FAIL {e}")
                continue
            done[arm.name] = r
            title = f"{label}: {'prompt ' + str(n_pp) if kind != 'tg' else 'decode step'} at depth {depth}" + (" k" if kept else "")
            host = r.get("host")
            host_line = (f"  host: graph build {host[0]:.2f} ms, inputs {host[1]:.2f} ms, compute call {host[2]:.2f} ms"
                         if host else "  host: no LLAMA_INPUT_TIMING in this build")
            if not vulkan:
                run.add("", title, host_line, *[f"  {x}" for x in r["lines"]])
                continue
            total = sum(o[2] for o in r["ops"]) or 1
            run.add("", f"{title}, {total:.2f} ms of gpu time", host_line,
                    *table(["  ms", "count", "op"], [[f"  {ms:.2f}", count, name] for name, count, ms in r["ops"][:25]]))
        if vulkan and len(done) > 1:
            # gpu time per kind of op, side by side: where one arm loses against the other
            groups = {}
            for name, r in done.items():
                for op, _, ms in r["ops"]:
                    groups.setdefault(op_group(op), {}).setdefault(name, 0.0)
                    groups[op_group(op)][name] += ms
            top = sorted(groups, key=lambda g: -max(groups[g].values()))[:20]
            rows = [[g, *[f"{groups[g].get(n, 0):.2f}" for n in done]] for g in top]
            rows.append(["total", *[f"{sum(o[2] for o in r['ops']):.2f}" for r in done.values()]])
            run.add("", f"{kind} at depth {depth}: gpu ms per kind of op", *table(["op", *done], rows))
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
        sid = source_id(env("BACKEND", "cpu"))
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"no source id ({e}): the results of this build are kept under the hash of its binaries")
        return 0
    head = git("rev-parse", "--short", "HEAD").decode().strip()
    (build / "qsa-build.pending.json").write_text(json.dumps({"id": sid, "head": head}) + "\n", encoding="utf-8")
    print(f"source id {sid}")
    return 0


def cmd_build_end():
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
    """Build the tag stable in its own worktree, with the llama-bench of this checkout (it reserves one output, as
    the server does) and the smaller perplexity buffers."""
    run = Run("build-stable")
    source = Path(env("STABLESRC", str(ROOT / "build-stable-src"))).resolve()
    build = Path(env("STABLEBUILD", str(ROOT / "build-stable"))).resolve()
    commit = git("rev-parse", "stable^{commit}").decode().strip()
    bench, ppl = Path("tools/llama-bench/llama-bench.cpp"), Path("tools/perplexity/perplexity.cpp")
    marker = source / ".qsa-stable-source"
    if source.exists():
        if not marker.is_file():
            raise Fail(f"{source} exists and is not a worktree of this script, set STABLESRC to another folder")
        subprocess.run(["git", "-C", str(source), "checkout", "--force", "--detach", commit], check=True)
    else:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(source), commit], check=True)
    marker.write_text(commit + "\n")
    shutil.copy2(ROOT / bench, source / bench)
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
    patch = hashlib.sha256((ROOT / bench).read_bytes() + (source / ppl).read_bytes()).hexdigest()[:8]
    sid = f"stable-{commit[:12]}-{patch}"
    (folder / BUILD_RECORD).write_text(json.dumps({"id": sid, "head": commit[:12], "time": now()}) + "\n",
                                       encoding="utf-8")
    run.add(f"stable {commit[:12]} built in {folder}, id {sid}. arms with @stable run it")
    return run.finish()


COMMANDS = {"check": cmd_check, "bench": cmd_bench, "quality": cmd_quality, "decode": cmd_decode,
            "profile": cmd_profile, "kernels": cmd_kernels, "build-begin": cmd_build_begin, "build-end": cmd_build_end,
            "build-stable": cmd_build_stable}


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
