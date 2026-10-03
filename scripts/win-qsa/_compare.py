#!/usr/bin/env python3
"""Compare stable and the current build without changing model or KV precision."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]
KNOBS = (
    "LLAMA_QSA_SPLIT_HEADS", "LLAMA_REPLAN_DISABLE", "LLAMA_INPUT_THREADS",
    "GGML_VK_INT_COOPMAT", "GGML_VK_F16ACC", "GGML_VK_DISABLE_FUSION",
    "GGML_VK_DISABLE_GRAPH_OPTIMIZE", "GGML_VK_DISABLE_MM_HEADSUM",
    "GGML_VK_DISABLE_MM_RELU", "GGML_VK_TOPK_RADIX", "GGML_VK_FA_SPARSE_DISABLE",
    "GGML_VK_FA_SPARSE_GROUP", "GGML_VK_FA_SPARSE_GQA", "GGML_VK_PERF_LOGGER",
    "LLAMA_INPUT_TIMING", "GGML_SCHED_LOG_REALLOC",
)


def digest(path):
    with Path(path).open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def executable(folder, name):
    path = Path(folder) / (name + (".exe" if os.name == "nt" else ""))
    if not path.is_file():
        raise RuntimeError(f"Missing {path}; build it first")
    return path.resolve()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def arguments(value):
    # Keep Windows paths intact. Quote the whole argument if it contains spaces.
    return [s[1:-1] if s.startswith('"') and s.endswith('"') else s
            for s in shlex.split(value, posix=False)]


def run(command, log, env=None, result=None):
    print(f"{log.name}: running {Path(command[0]).name}", flush=True)
    with log.open("w", encoding="utf-8") as err:
        err.write(json.dumps({"command": [str(a) for a in command],
                              "knobs": {k: (env or os.environ).get(k) for k in KNOBS}}) + "\n")
        err.flush()
        if result:
            with result.open("w", encoding="utf-8") as out:
                p = subprocess.run(command, stdout=out, stderr=err, env=env)
        else:
            p = subprocess.run(command, stdout=err, stderr=err, env=env)
    if p.returncode:
        raise RuntimeError(f"Exit {p.returncode}: {log}")
    return log.read_text(encoding="utf-8", errors="replace")


def prepare(logs):
    source = Path(os.environ.get("ABSOURCE", ROOT / "build-stable-src")).resolve()
    build = Path(os.environ.get("ABBUILD", ROOT / "build-stable")).resolve()
    commit = git("rev-parse", "stable^{commit}")
    bench = Path("tools/llama-bench/llama-bench.cpp")
    marker = source / ".qsa-stable-source"
    if source.exists():
        if not marker.is_file() or marker.read_text().strip() != commit or git("rev-parse", "HEAD", cwd=source) != commit:
            raise RuntimeError(f"Refusing to overwrite {source}; choose a new ABSOURCE")
        allowed = {bench.as_posix(), "scripts/win-qsa/00-build.bat", "scripts/win-qsa/_config.bat"}
        if set(git("diff", "HEAD", "--name-only", cwd=source).splitlines()) - allowed:
            raise RuntimeError(f"Baseline core was modified in {source}; choose a new ABSOURCE")
    else:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(source), commit], check=True)
        marker.write_text(commit + "\n")
    shutil.copy2(ROOT / bench, source / bench)
    if os.name == "nt":
        scripts = source / "scripts/win-qsa"
        for name in ("00-build.bat", "_config.bat"):
            shutil.copy2(ROOT / "scripts/win-qsa" / name, scripts / name)
        settings = {"BUILD": build, "CMAKE_BACKEND": os.environ["CMAKE_BACKEND"],
                    "GENERATOR": os.environ.get("GENERATOR", "auto")}
        (scripts / "_local.bat").write_text("@echo off\n" + "".join(f'set "{k}={v}"\n' for k, v in settings.items()))
        run(["cmd.exe", "/d", "/c", "call", str(scripts / "00-build.bat")], logs / "build.log")
    else:
        flags = arguments(os.environ.get("CMAKE_BACKEND", "-DGGML_METAL=OFF"))
        run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
             "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_CURL=OFF", *flags], logs / "configure.log")
        run(["cmake", "--build", str(build), "--config", "Release", "--target",
             "llama-bench", "llama-perplexity", "-j", str(os.cpu_count() or 4)], logs / "build.log")
    folder = build / "bin/Release" if (build / "bin/Release").is_dir() else build / "bin"
    data = {"commit": commit, "bench_source_sha256": digest(ROOT / bench),
            "binaries": {n: digest(executable(folder, n)) for n in ("llama-bench", "llama-perplexity")}}
    (folder / "qsa-baseline.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Baseline ready. Set ABBIN={folder} in _local.bat", flush=True)


def base_env(accurate=False, heads=None, timing=False):
    env = os.environ.copy()
    for key in KNOBS:
        env.pop(key, None)
    if accurate:
        env.update(GGML_VK_INT_COOPMAT="0", GGML_VK_F16ACC="0")
    if heads is not None:
        env["LLAMA_QSA_SPLIT_HEADS"] = str(heads)
    if timing:
        env.update(LLAMA_INPUT_TIMING="1", GGML_SCHED_LOG_REALLOC="1")
    return env


def checks(folder, logs, device, vulkan):
    tool = executable(folder, "test-backend-ops")
    tests = [
        ("gdn", "GATED_DELTA_NET_CACHE_FUSION", ""),
        ("headsum", "MUL_MAT_HEADSUM,MUL_MAT_RELU", ""),
        ("moe-deps", "MUL_MAT_VEC_FUSION", "gate_cont=1"),
        ("moe-glu", "MUL_MAT_VEC_FUSION", "glu_op=2,m=(16|64|200),.*use_id=1,.*with_bias=0,with_gate=1,with_lane_scale=0,.*gate_cont=0"),
    ]
    for accurate in ([False, True] if vulkan else [False]):
        for name, op, pattern in tests:
            cmd = [str(tool), "test", "-b", device, "-o", op]
            if pattern:
                cmd += ["-p", pattern]
            log = logs / f"check-{name}-{'accurate' if accurate else 'default'}.log"
            output = run(cmd, log, base_env(accurate))
            counts = re.findall(r"(\d+)/(\d+) tests passed", output)
            if not counts or not any(int(n) > 0 for _, n in counts) or any(a != b for a, b in counts):
                raise RuntimeError(f"Tests failed or no cases ran: {log}")


def compare(mode, logs):
    folder = Path(os.environ["BIN"]).resolve()
    default_baseline = ROOT / "build-stable/bin/Release"
    if not default_baseline.is_dir():
        default_baseline = ROOT / "build-stable/bin"
    baseline = Path(os.environ.get("ABBIN") or default_baseline).resolve()
    info = json.loads((baseline / "qsa-baseline.json").read_text())
    if info["commit"] != git("rev-parse", "stable^{commit}") or info["bench_source_sha256"] != digest(ROOT / "tools/llama-bench/llama-bench.cpp"):
        raise RuntimeError("Baseline tag or benchmark source differs; run 35-build-stable.bat again")
    for name, sha in info["binaries"].items():
        if digest(executable(baseline, name)) != sha:
            raise RuntimeError(f"Baseline binary changed: {name}")
    model = Path(os.environ["MODEL"]).resolve()
    if not model.is_file():
        raise RuntimeError(f"Missing model: {model}")
    vulkan = os.environ.get("BACKEND") == "vulkan"
    default_device = {"vulkan": "Vulkan0", "cuda": "CUDA0"}.get(os.environ.get("BACKEND"), "CPU")
    device = os.environ.get("ABDEVICE", default_device)
    metadata = {"baseline": info, "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
                "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD"])).hexdigest(),
                "model": {"path": str(model), "size": model.stat().st_size, "mtime_ns": model.stat().st_mtime_ns},
                "device": device,
                "environment": {k: v for k, v in os.environ.items() if (k.startswith(("AB", "GGML_", "LLAMA_")) or k in ("EXTRA", "LOADMODE", "BACKEND", "CMAKE_BACKEND", "GENERATOR", "SETTLE")) and not re.search("TOKEN|SECRET|PASS|API_KEY", k)},
                "binaries": {n: digest(executable(folder, n)) for n in ("llama-bench", "llama-perplexity", "test-backend-ops")}}
    (logs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checks(folder, logs, device, vulkan)
    common = ["-m", str(model), "-fa", "on", *arguments(os.environ.get("LOADMODE", "")), *arguments(os.environ.get("EXTRA", ""))]
    arms = [("stable-default", baseline, False, None), ("new-default", folder, False, None),
            ("new-split", folder, False, 1), ("new-joint", folder, False, 0)]
    if vulkan:
        arms.append(("new-accurate", folder, True, None))
    summary = []
    failed = False
    settle = float(os.environ.get("SETTLE", "30"))
    if mode == "quality":
        ctx = int(os.environ.get("ABQCTX", "32768"))
        chunks = int(os.environ.get("ABQCHUNKS", "1"))
        ub = int(os.environ.get("ABQUB", "2048"))
        if ctx <= 4096 or chunks < 1 or ub < 1:
            raise RuntimeError("Quality requires ABQCTX > 4096, ABQCHUNKS >= 1 and ABQUB >= 1")
        text = Path(os.environ["PPLFILE"]).resolve()
        if not text.is_file():
            raise RuntimeError(f"Missing corpus: {text}; run get-wikitext.bat")
        metadata["corpus"] = {"path": str(text), "sha256": digest(text)}
        metadata["parameters"] = {"ctx": ctx, "chunks": chunks, "ubatch": ub}
        (logs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        params = [*common, "-f", str(text), "-c", str(ctx), "--chunks", str(chunks), "-b", str(ub), "-ub", str(ub)]
        print(f"Free disk: {shutil.disk_usage(logs).free / 2**30:.1f} GiB. At 248320 vocab, base logits need about {ctx * chunks * 248320 / 2**30:.1f} GiB.", flush=True)
        stable_arms = [("stable-repeat", baseline, False, None), *arms[1:]]
        references = [("stable", baseline, False, stable_arms)]
        if vulkan:
            references.append(("accurate", folder, True, [("accurate-repeat", folder, True, None), arms[1]]))
        for ref_name, ref_bin, ref_accurate, candidates in references:
            data = logs / f"{ref_name}-base.dat"
            ref_params = [*params, "--kl-divergence-base", str(data)]
            time.sleep(settle)
            output = run([str(executable(ref_bin, "llama-perplexity")), *ref_params], logs / f"{ref_name}-base.log", base_env(ref_accurate))
            if "Final estimate" not in output:
                raise RuntimeError(f"Baseline did not finish; see {ref_name}-base.log")
            with data.open("rb") as f:
                header = f.read(20)
            if len(header) != 20 or header[:8] != b"_logits_":
                raise RuntimeError("Invalid baseline logits header")
            nctx, vocab, nchunk = struct.unpack("<Iii", header[8:])
            expected = 20 + nctx * nchunk * 4 + nchunk * (nctx - nctx // 2 - 1) * (2 * ((vocab + 1) // 2) + 4) * 2
            if (nctx, nchunk) != (ctx, chunks) or vocab < 1 or data.stat().st_size != expected:
                raise RuntimeError("Baseline logits incomplete or corpus too short for requested chunks")
            for name, arm_bin, accurate, heads in candidates:
                time.sleep(settle)
                label = f"{ref_name}-{name}"
                try:
                    output = run([str(executable(arm_bin, "llama-perplexity")), *ref_params, "--kl-divergence"], logs / f"{label}.log", base_env(accurate, heads))
                    kld = re.search(r"Mean\s+KLD:\s*(\S+)", output)
                    ratio = re.search(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*(\S+)", output)
                    if not kld or not ratio or not math.isfinite(float(kld[1])) or not math.isfinite(float(ratio[1])) or float(ratio[1]) <= 0:
                        raise RuntimeError(f"Missing KLD: {label}.log")
                    summary.append(label + "\n" + "\n".join(line for line in output.splitlines() if any(s in line for s in ("Mean PPL", "Mean    KLD", "Maximum KLD", "99.9%", "99.0%", "Same top p"))))
                except (RuntimeError, ValueError) as e:
                    failed = True
                    summary.append(f"{label}: FAIL: {e}")
            if not failed and os.environ.get("ABKEEPLOGITS", "0") != "1":
                data.unlink()
    else:
        ctx = int(os.environ.get("ABCTX", "262144"))
        pp = int(os.environ.get("ABPP", "4096"))
        tg = int(os.environ.get("ABTG", "64"))
        ub = int(os.environ.get("ABUB", "4096"))
        reps = int(os.environ.get("ABREPS", "3"))
        depths = [int(d) for d in os.environ.get("ABDEPTHS", "0,32768,65536,122880,139264").split(",")]
        if min(ctx, pp, tg, ub, reps) < 1 or min(depths) < 0 or max(depths) + max(pp, tg) > ctx:
            raise RuntimeError("Invalid benchmark sizes or depth exceeds ABCTX")
        metadata["parameters"] = {"ctx": ctx, "prompt": pp, "generation": tg, "ubatch": ub, "reps": reps, "depths": depths}
        (logs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for i, depth in enumerate(depths):
            for name, arm_bin, accurate, heads in (arms if i % 2 == 0 else list(reversed(arms))):
                time.sleep(settle)
                label = f"{name}-d{depth}"
                result = logs / f"{label}.json"
                cmd = [str(executable(arm_bin, "llama-bench")), *common, "-c", str(ctx), "-b", str(pp), "-ub", str(ub),
                       "-p", str(pp), "-n", str(tg), "-d", str(depth), "-r", str(reps), "--progress", "-o", "json"]
                try:
                    run(cmd, logs / f"{label}.log", base_env(accurate, heads, True), result)
                    rows = json.loads(result.read_text())
                    if len(rows) != 2 or any(not math.isfinite(r["avg_ts"]) or r["avg_ts"] <= 0 or not math.isfinite(r["stddev_ts"]) for r in rows):
                        raise RuntimeError("Missing or invalid PP/TG results")
                    summary.append(label + " " + " ".join(f"{'PP' if r['n_prompt'] else 'TG'}={r['avg_ts']:.2f} +/- {r['stddev_ts']:.2f}" for r in rows))
                    output = (logs / f"{label}.log").read_text(errors="replace")
                    summary.append(f"  reserve={output.count('sched reserve:')} re-reserve={output.count('sched re-reserve:')}")
                except (RuntimeError, ValueError, KeyError) as e:
                    failed = True
                    summary.append(f"{label}: FAIL: {e}")
    (logs / "summary.txt").write_text("\n\n".join(summary) + "\n")
    print("\n\n".join(summary), flush=True)
    if failed:
        raise RuntimeError("One or more arms failed; results retained in logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "quality", "speed"))
    mode = parser.parse_args().mode
    logs = Path(os.environ.get("LOGS", ROOT / "scripts/win-qsa/logs")) / (f"{mode}-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    logs.mkdir(parents=True)
    print(f"Logs: {logs}", flush=True)
    try:
        if mode == "prepare":
            prepare(logs)
        else:
            compare(mode, logs)
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as e:
        (logs / "failure.txt").write_text(str(e) + "\n")
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
