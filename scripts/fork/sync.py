#!/usr/bin/env python3
"""Upstream changes to the originals of the fork's copies (copies.txt).

    sync.py                         commits upstream made to each original since the copy's commit
    sync.py --diff [copy ...]       the same as one diff per original
    sync.py --mark <commit> copy..  record that the copies have every upstream change up to <commit>
    sync.py --ref <ref>             the upstream ref to compare with (default: github.com/master, else origin/master)

copies.txt has one line per copy: copy, original, commit, component and, optionally, a line range of the original
in the form git log -L takes (/start-regex/,/end-regex/) when the copy holds only a part of it.

Run `git fetch <remote> master` first. Nothing here changes a source file: porting a change into a copy is done by
hand, then --mark records it.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIST = HERE / "copies.txt"


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def read_list():
    rows = []
    for line in LIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            parts = line.split(None, 4)
            # an optional fifth column limits the original to a line range, as git log -L takes it
            rows.append(parts + [""] * (5 - len(parts)))
    return rows


def default_ref():
    for ref in ("github.com/master", "upstream/master", "origin/master"):
        if subprocess.run(["git", "rev-parse", "--verify", "-q", ref], cwd=ROOT, capture_output=True).returncode == 0:
            return ref
    sys.exit("no upstream ref found, pass --ref")


def mark(rows, commit, copies):
    full = git("rev-parse", "--short=9", commit).strip()
    lines = LIST.read_text(encoding="utf-8").splitlines(keepends=True)
    hit = set()
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 4 and not line.lstrip().startswith("#") and parts[0] in copies:
            lines[i] = line.replace(parts[2], full.ljust(len(parts[2])), 1)
            hit.add(parts[0])
    missing = set(copies) - hit
    if missing:
        sys.exit(f"not in copies.txt: {' '.join(sorted(missing))}")
    LIST.write_text("".join(lines), encoding="utf-8")
    print(f"marked {len(hit)} copies at {full}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--mark", metavar="COMMIT")
    ap.add_argument("copies", nargs="*")
    a = ap.parse_args()
    rows = read_list()
    if a.mark:
        mark(rows, a.mark, a.copies)
        return
    ref = a.ref or default_ref()
    for copy, original, commit, component, lines in rows:
        if a.copies and copy not in a.copies:
            continue
        if lines:
            out = git("log", "-L", f"{lines}:{original}", "--oneline", "-s", "--no-merges", f"{commit}..{ref}")
            log = "\n".join(x for x in out.splitlines() if x[:1].isalnum() and " " in x and len(x.split()[0]) >= 7).strip()
        else:
            log = git("log", "--oneline", "--no-merges", f"{commit}..{ref}", "--", original).strip()
        print(f"{copy} <- {original} ({component}, at {commit}): " + (f"{len(log.splitlines())} upstream commits" if log else "up to date"))
        if log and not a.diff:
            print("    " + log.replace("\n", "\n    "))
        if log and a.diff:
            print(git("log", "-L", f"{lines}:{original}", "--no-merges", f"{commit}..{ref}") if lines
                  else git("diff", commit, ref, "--", original))


if __name__ == "__main__":
    main()
