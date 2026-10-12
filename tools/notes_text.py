#!/usr/bin/env python3
"""Extract one of the official CS229 class notes to page-addressable text, plus a heading map.

The 2018 lectures are taught at a blackboard, so there is no slide deck to map; the
official notes (cs229-notes1.pdf, cs229-notes-dt.pdf, ...) are the secondary source,
and they are organised by numbered sections rather than by slides. This produces two
files beside the PDF:

    cs229-notes/text/cs229-notes1.txt    every page, wrapped in page markers
    cs229-notes/text/cs229-notes1.map    one line per section heading, with its page

Read the .map to find the section you need, then pull only those pages with
`tools/notes_pages.py`. Never read the .txt end to end.

    python3 tools/notes_text.py cs229-notes/cs229-notes1.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

PAGE_BREAK = "\f"
# The notes set headings flush left: "Part III", or a section number, a run of
# spaces and a capitalised title ("2.1    Matrix derivatives"). Displayed maths is
# indented and body text has single spaces, so neither matches.
HEADING_RE = re.compile(r"^(Part [IVX]+\b.*|\d+(\.\d+)*\s{2,}[A-Z].*)$")


def headings(page: str) -> list[str]:
    """Every section heading on a page, whitespace collapsed."""
    out = []
    for line in page.splitlines():
        line = line.rstrip()
        if len(line) <= 90 and HEADING_RE.match(line):
            out.append(" ".join(line.split()))
    return out


def first_line(page: str) -> str:
    """Fallback label for a page with no heading: its first substantial line."""
    for line in page.splitlines():
        line = " ".join(line.split())
        if len(line) > 3 and not line.isdigit():
            return line[:90]
    return "(no text — a full-page figure)"


def main() -> int:
    """Run pdftotext, split on form feeds, and write the page text and heading map."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=pathlib.Path)
    ap.add_argument("--outdir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if not args.pdf.is_file():
        print(f"no such file: {args.pdf}", file=sys.stderr)
        return 1

    outdir = args.outdir or args.pdf.parent / "text"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        raw = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(args.pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except FileNotFoundError:
        print("pdftotext not found — install poppler (brew install poppler)", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"pdftotext failed: {exc.stderr[:300]}", file=sys.stderr)
        return 1

    pages = raw.split(PAGE_BREAK)
    if pages and not pages[-1].strip():
        pages.pop()

    body, contents = [], []
    for n, page in enumerate(pages, start=1):
        text = "\n".join(line.rstrip() for line in page.splitlines()).strip()
        body.append(f"=== PAGE {n} ===\n{text}\n")
        found = headings(page)
        if found:
            contents.extend(f"{n:>4}  {h}" for h in found)
        elif n == 1:
            contents.append(f"{n:>4}  {first_line(page)}")

    stem = args.pdf.stem
    txt, mp = outdir / f"{stem}.txt", outdir / f"{stem}.map"
    banner = "# Stanford course material — local working copy, never commit or redistribute.\n"
    txt.write_text(banner + "\n".join(body), encoding="utf-8")
    mp.write_text(banner + f"# {len(pages)} pages\n" + "\n".join(contents) + "\n", encoding="utf-8")

    print(f"{txt}: {len(pages)} pages")
    print(f"{mp}: {len(contents)} headings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
