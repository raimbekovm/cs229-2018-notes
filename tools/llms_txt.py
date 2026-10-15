#!/usr/bin/env python3
"""Write llms.txt, llms-full.txt and a Markdown twin of every chapter into _site/.

Answer engines and coding assistants read a site far more reliably from plain
Markdown than from rendered HTML: the maths on these pages is typeset by KaTeX
into hundreds of nested spans, which a crawler flattens into noise. So beside
every chapter's HTML this writes the chapter as Markdown, with the maths left
as the TeX it was written in, and two index files in the llmstxt.org format:

    _site/llms.txt                         the book's map: one line per chapter
    _site/llms-full.txt                    every chapter, in order, in one file
    _site/lectures/02-linear-regression.md one per chapter

The Markdown is built from the .qmd sources rather than from the HTML, because
the source is the cleanest form of the text there is. The Quarto-only syntax is
resolved on the way out, the same way the site resolves it: `[12:34]{.t}`
becomes a link into the lecture video at that second, `@eq-…`, `@fig-…` and
`@sec-…` become "Equation 2.3", "Figure 2.1" and "Section 2.4" with the book's
own numbering, figure paths become absolute URLs, and links between chapters
point at the published pages.

Quarto 1.10 has a native llms.txt generator, but it is gated on a `website:`
key and silently inert in a book project; see the lecture-notes skill.

    python3 tools/llms_txt.py    # post-render; does nothing without _site/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site"
BASE = "https://raimbekovm.github.io/cs229-2018-notes/"
REPO = "https://github.com/raimbekovm/cs229-2018-notes"


def config() -> tuple[dict[str, str], list[tuple[str | None, str]]]:
    """Book metadata and the chapter list, in order, with the part each sits in.

    `_quarto.yml` is read with patterns rather than a YAML parser so that the build needs nothing beyond the standard
    library; the shape of the chapter list is fixed by Quarto and the patterns are anchored to it.
    """
    text = (ROOT / "_quarto.yml").read_text()
    book = text.split("\nbook:\n", 1)[1].split("\nformat:\n", 1)[0]
    meta = {
        key: re.search(rf'^  {key}: "(.*)"$', book, re.M).group(1)
        for key in ("title", "subtitle", "author")
    }
    chapters: list[tuple[str | None, str]] = []
    part = None
    for line in book.splitlines():
        if m := re.match(r'\s*- part: "(.*)"$', line):
            part = m.group(1)
        elif m := re.match(r"\s*- (\S+\.qmd)$", line):
            chapters.append((part, m.group(1)))
    return meta, chapters


def front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a .qmd into its scalar front-matter keys and its body."""
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    keys = {}
    for line in m.group(1).splitlines():
        if km := re.match(r'(\w[\w-]*): "(.*)"$', line):
            keys[km.group(1)] = km.group(2).replace('\\"', '"')
    return keys, text[m.end() :]


def playlist() -> dict[str, tuple[str, str]]:
    """Lecture key -> (video id, video title), from tools/playlist.tsv."""
    videos = {}
    for line in (ROOT / "tools" / "playlist.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            videos[parts[0]] = (parts[1], parts[3])
    return videos


def video_key(raw: str) -> str:
    """'2' -> '02' and '8b' -> '08b', as filters/timecodes.lua normalises it."""
    m = re.match(r"(\d+)(.*)", raw)
    return f"{int(m.group(1)):02d}{m.group(2)}" if m else raw


def seconds(stamp: str) -> int:
    """'1:02:38' -> 3758."""
    total = 0
    for part in stamp.split(":"):
        total = total * 60 + int(part)
    return total


class Chapter:
    """One chapter of the book: its metadata, its body, and where it is published."""

    def __init__(self, part: str | None, source: str, number: int | None):
        """Read the chapter's source; `number` is None for an unnumbered chapter."""
        self.part = part
        self.source = source
        self.number = number
        keys, self.body = front_matter((ROOT / source).read_text())
        self.description = keys.get("description", "")
        self.subtitle = keys.get("subtitle")
        title = keys.get("title")
        if not title:
            # An unnumbered chapter carries its title as the first heading.
            m = re.search(r"^# (.+?)(?: \{[^}]*\})?$", self.body, re.M)
            title = m.group(1) if m else Path(source).stem
            self.body = self.body.replace(m.group(0), "", 1) if m else self.body
        self.title = title
        self.html = BASE + source.removesuffix(".qmd") + ".html"
        self.md_path = source.removesuffix(".qmd") + ".md"
        self.md = BASE + self.md_path

    def heading(self) -> str:
        """The title as the book shows it, with its chapter number."""
        return f"{self.number}. {self.title}" if self.number else self.title

    def labels(self) -> dict[str, str]:
        """Every cross-reference label in the chapter, mapped to its reader-facing name."""
        names = {}
        prefix = f"{self.number}." if self.number else ""
        counters = {"eq": 0, "fig": 0, "tbl": 0}
        section = [0, 0, 0]
        for line in self.body.splitlines():
            if hm := re.match(r"(#{2,4}) (.+?)\s*\{#(sec-[\w-]+)[^}]*\}\s*$", line):
                depth = len(hm.group(1)) - 2
                section[depth] += 1
                section[depth + 1 :] = [0] * (len(section) - depth - 1)
                number = ".".join(str(n) for n in section[: depth + 1])
                names[hm.group(3)] = (
                    f"Section {prefix}{number}"
                    if self.number
                    else f"the section “{hm.group(2)}” of {self.title}"
                )
            for kind in ("eq", "fig", "tbl"):
                for label in re.findall(rf"\{{#({kind}-[\w-]+)", line):
                    counters[kind] += 1
                    word = {"eq": "Equation", "fig": "Figure", "tbl": "Table"}[kind]
                    names[label] = f"{word} {prefix}{counters[kind]}"
        return names


def to_markdown(
    chapter: Chapter,
    labels: dict[str, str],
    by_source: dict[str, Chapter],
    videos: dict[str, tuple[str, str]],
) -> str:
    """The chapter as standalone Markdown, with Quarto\'s own syntax resolved."""
    body = chapter.body
    video_id = None

    def marker(m: re.Match) -> str:
        nonlocal video_id
        video_id, title = videos[video_key(m.group(1))]
        return f"Lecture video: [{title}](https://www.youtube.com/watch?v={video_id})\n"

    body = re.sub(
        r'^::: \{\.lecture-video lecture="([^"]+)"\}\n:::\n', marker, body, flags=re.M
    )

    def stamp(m: re.Match) -> str:
        if not video_id:
            return f"[{m.group(1)}]"
        return f"[{m.group(1)}](https://www.youtube.com/watch?v={video_id}&t={seconds(m.group(1))}s)"

    body = re.sub(r"\[(\d+(?::\d\d){1,2})\]\{\.t\}", stamp, body)

    # Callouts become a bold lead-in; the closing fence goes with them.
    body = re.sub(
        r'^::: \{\.callout-\w+ title="([^"]+)"\}\n', r"**\1.** ", body, flags=re.M
    )
    body = re.sub(r"^:::\s*$\n?", "", body, flags=re.M)

    chapter_dir = (ROOT / chapter.source).parent

    def figure(m: re.Match) -> str:
        url = BASE + (chapter_dir / m.group(2)).resolve().relative_to(ROOT).as_posix()
        label = re.search(r"#(fig-[\w-]+)", m.group(3) or "")
        caption = " ".join(m.group(1).split())
        name = labels.get(label.group(1)) if label else None
        lead = f"{name}. " if name else ""
        return f"![{lead}{caption}]({url})"

    body = re.sub(
        r"!\[(.*?)\]\(([^)\s]+\.(?:svg|png|jpe?g|avif|webp))\)(\{[^}]*\})?",
        figure,
        body,
        flags=re.S,
    )

    # Equation labels become a visible tag, so "Equation 2.3" in the prose can be found.
    def equation(m: re.Match) -> str:
        name = labels.get(m.group(1), m.group(1))
        return f"$$ ({name.split(' ', 1)[1]})"

    body = re.sub(r"\$\$\s*\{#(eq-[\w-]+)\}", equation, body)

    def table(m: re.Match) -> str:
        return f": {labels.get(m.group(2), '')}. {m.group(1)}"

    body = re.sub(r"^: (.+?)\s*\{#(tbl-[\w-]+)\}[ \t]*$", table, body, flags=re.M)

    # Heading attributes carry nothing a reader of the Markdown needs.
    body = re.sub(r"^(#{1,6} .+?)[ \t]*\{[^}]*\}[ \t]*$", r"\1", body, flags=re.M)
    # One level down, so the chapter title can be the only H1.
    body = re.sub(r"^(#{1,5}) ", r"#\1 ", body, flags=re.M)

    def link(m: re.Match) -> str:
        target, anchor = m.group(2), m.group(3) or ""
        resolved = (chapter_dir / target).resolve().relative_to(ROOT).as_posix()
        other = by_source.get(resolved)
        url = (
            other.html if other else BASE + resolved.removesuffix(".qmd") + ".html"
        ) + anchor
        return f"[{m.group(1)}]({url})"

    body = re.sub(r"\[([^\]]+)\]\(([\w./-]+\.qmd)(#[\w-]+)?\)", link, body)

    def ref(m: re.Match) -> str:
        return labels.get(m.group(1), m.group(0))

    body = re.sub(r"@((?:sec|eq|fig|tbl)-[\w-]+)", ref, body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    lines = [f"# {chapter.heading()}", ""]
    if chapter.description:
        lines += [f"> {chapter.description}", ""]
    lines += [f"Web version: {chapter.html}", "", body, ""]
    return "\n".join(lines)


def main() -> int:
    """Write the three outputs; a no-op when nothing has been rendered."""
    if not SITE.is_dir():
        return 0
    meta, entries = config()
    videos = playlist()

    chapters: list[Chapter] = []
    number = 0
    for part, source in entries:
        if source == "index.qmd":
            continue
        keys, _ = front_matter((ROOT / source).read_text())
        numbered = "title" in keys
        number += numbered
        chapters.append(Chapter(part, source, number if numbered else None))

    labels: dict[str, str] = {}
    for chapter in chapters:
        labels.update(chapter.labels())
    by_source = {c.source: c for c in chapters}

    full = [
        f"# {meta['title']}",
        "",
        f"> {meta['subtitle']}. Lecture notes by {meta['author']}, written from the "
        "recorded lectures of Stanford CS229 Machine Learning, Autumn 2018. "
        "Unofficial; not affiliated with Stanford University. MIT licence.",
        "",
        f"Web version: {BASE}",
        "",
    ]
    for chapter in chapters:
        text = to_markdown(chapter, labels, by_source, videos)
        out = SITE / chapter.md_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        full += [re.sub(r"^#", "##", text, flags=re.M), ""]
    (SITE / "llms-full.txt").write_text("\n".join(full))

    index = [
        f"# {meta['title']}",
        "",
        "> Free, unofficial lecture notes for Stanford CS229: Machine Learning, "
        "written from Andrew Ng's Autumn 2018 lectures. Every lecture of the course "
        "has a chapter of continuous prose with the board derivations written out "
        "in full, links to the moment in the YouTube recording where each section "
        "is taught, and pointers to the official class notes (cs229-notes1 to "
        "cs229-notes13) where they go further.",
        "",
        "The notes cover supervised learning (linear and logistic regression, "
        "generalized linear models, Gaussian discriminant analysis, naive Bayes, "
        "support vector machines and kernels), generalization (bias and variance, "
        "regularization, cross-validation, learning theory and VC dimension), "
        "decision trees, ensembles and neural networks, practical advice for "
        "debugging learning algorithms, unsupervised learning (k-means, the EM "
        "algorithm, factor analysis, PCA, ICA) and reinforcement learning (MDPs, "
        "value and policy iteration, fitted value iteration, LQR, policy search). "
        "Each link below is the chapter as Markdown, with the maths in TeX; "
        "the same chapter as a web page is at the same path ending in .html.",
        "",
    ]
    part = object()
    for chapter in chapters:
        if chapter.part != part:
            part = chapter.part
            index += ["", f"## {part}", ""]
        label = (
            chapter.heading()
            if chapter.number
            else f"{chapter.title} (discussion section)"
        )
        index.append(f"- [{label}]({chapter.md}): {chapter.description}")
    index += [
        "",
        "## Optional",
        "",
        f"- [Every chapter in one file]({BASE}llms-full.txt): the whole book as Markdown",
        f"- [The book as a PDF]({BASE}CS229-2018-Lecture-Notes.pdf)",
        "- [The 2018 lecture videos](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU): "
        "Stanford Online's YouTube playlist",
        "- [Official course site](https://cs229.stanford.edu/): Stanford CS229",
        f"- [Source repository]({REPO}): corrections are welcome as issues or pull requests",
        "",
    ]
    (SITE / "llms.txt").write_text("\n".join(index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
