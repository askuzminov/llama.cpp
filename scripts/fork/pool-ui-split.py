#!/usr/bin/env python3
"""The parts of the pool dashboard (tools/server/fork/server-pool-ui-fork.inc) split again.

MSVC takes a string literal of up to 16 KB, so the page is an array of raw strings that the server joins. Edit the page
in the .inc as one text, then run this: it joins the parts, checks that the page is ASCII, and splits it again at line
boundaries into parts of at most 15000 bytes. The header comment of the file stays.

    pool-ui-split.py                 split again in place
    pool-ui-split.py --page out.html also write the joined page, to open it in a browser (it reads /pool/status)
"""

import argparse
import re
import sys
from pathlib import Path

INC = Path(__file__).resolve().parents[2] / "tools" / "server" / "fork" / "server-pool-ui-fork.inc"
LIMIT = 15000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", help="write the joined page to this file")
    args = ap.parse_args()

    text = INC.read_text()
    head, sep, _ = text.partition("static const char * const POOL_UI_PARTS[] = {\n")
    if not sep:
        print(f"{INC}: no POOL_UI_PARTS array", file=sys.stderr)
        return 1
    parts = re.findall(r'R"POOLUI\((.*?)\)POOLUI"', text, flags=re.S)
    page = "".join(parts)

    bad = [(i + 1, ch) for i, line in enumerate(page.splitlines()) for ch in line if ord(ch) > 126]
    if bad:
        print(f"non-ASCII characters (line, char): {bad[:5]}; write them as \\u escapes or HTML entities", file=sys.stderr)
        return 1

    out, cur = [], ""
    for line in page.splitlines(keepends=True):
        if len(line.encode()) > LIMIT:
            print(f"a line of {len(line.encode())} bytes is longer than a part", file=sys.stderr)
            return 1
        if cur and len((cur + line).encode()) > LIMIT:
            out.append(cur)
            cur = ""
        cur += line
    out.append(cur)

    INC.write_text(head + sep + "".join(f'R"POOLUI({p})POOLUI",\n' for p in out) + "};\n")
    print(f"{INC.name}: {len(page.encode())} bytes in {len(out)} parts ({', '.join(str(len(p.encode())) for p in out)})")
    if args.page:
        Path(args.page).write_text(page)
        print(f"wrote {args.page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
