# Contributing

Thank you for your interest in improving these lecture notes. Corrections of any
size are welcome, from a typo to a wrong sign in a derivation.

## The quickest way

Every page on the [site](https://raimbekovm.github.io/cs229-2018-notes/) has an
**Edit this page** link at the bottom. It opens the source file in GitHub's
editor and turns your change into a pull request; no local setup is needed.

For anything larger, or if you are not sure something is wrong, open an
[issue](https://github.com/raimbekovm/cs229-2018-notes/issues/new/choose). The
templates ask for the lecture, the section and, for a correction, a source.

## What is in scope

| Type          | Example                                                        | Label         |
| ------------- | -------------------------------------------------------------- | ------------- |
| Typo          | "recieve" in section 2.1                                       | `typo`        |
| Correction    | A missing transpose in an equation, a wrong claim              | `correction`  |
| Timestamp     | A `[12:34]` link that opens the video at the wrong moment      | `bug`         |
| Clarification | A step in a derivation that is skipped without saying so       | `enhancement` |
| Broken link   | A paper link that returns 404                                  | `bug`         |

The notes follow what was taught in the Autumn 2018 lectures. A correction is
in scope when the notes misstate the lecture or the mathematics; a request to
cover material the lecture did not is better raised as a suggestion first.

## Working locally

The notes are a [Quarto](https://quarto.org/) book: one `.qmd` per lecture in
`lectures/`, rendered to the site and to a single PDF by CI on every push to
`main`.

```bash
git clone https://github.com/<you>/cs229-2018-notes.git
cd cs229-2018-notes
quarto preview           # live-reloading site at localhost:4321
quarto render --to html  # the site only; the PDF additionally needs LaTeX
```

Before opening a pull request, check that the page renders without warnings in
the terminal, that equations and cross-references resolve, and that any links
you added work. Three scripts catch most of what Quarto only warns about:

```bash
python3 tools/check_anchors.py lectures/02-linear-regression.qmd
python3 tools/check_links.py   lectures/02-linear-regression.qmd
python3 tools/check_figure_labels.py figures/02-linear-regression/
```

## Style

### Front matter

```yaml
---
title: "Linear Regression and Gradient Descent"
description: "One sentence, used for search results and link previews."
lecture: 2
---
```

`lecture:` ties the chapter to its recording. It is looked up in
`tools/playlist.tsv` (`2`, or `08b` for the learning-theory section), which is
the only place a video id is written down. Section numbering is automatic —
do not number headings by hand.

### Timestamps

Write a moment in the lecture as `[12:34]{.t}` or `[1:02:03]{.t}`. It renders
as a link that opens the video at that point, and the build fails if the time
is past the end of the recording, so a typo cannot ship. Every section opens
with one.

### Mathematics

Inline maths uses single dollars, display maths double, with the label on the
closing line:

```markdown
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$ {#eq-gd-update}
```

Reference it as `@eq-gd-update`; Quarto inserts the word "Equation" itself.
Keep a display block contiguous — a blank line inside it detaches the label.

### Figures

Diagrams are drawn by a script in `figures/<lecture-slug>/make_diagrams.py`
and committed as SVG beside it, so a figure can be corrected by editing code.
Every figure has a row in [`figures/SOURCES.md`](figures/SOURCES.md).

```markdown
![What the reader should see in it.](../figures/02-linear-regression/diagram_contours.svg){#fig-contours}
```

Leave a blank line above and below every figure: text on the same line turns it
into an inline image and its `@fig-` references stop resolving.

### Callouts

```markdown
::: {.callout-note title="In the official notes"}
Where the formal treatment continues: cs229-notes1, section 2.
:::

::: {.callout-warning title="A common slip"}
A caveat, or a place where the lecture and the notes differ.
:::
```

### Links

Link the original paper or official documentation the first time a result is
mentioned — an arXiv abstract page, the proceedings entry, the journal DOI.
Never a blog summary or a PDF mirror.

## Commit messages

Imperative mood, under 72 characters, saying what changed and why:

```
Fix the sign of the gradient in the LMS update
Clarify why the normal equations need XᵀX to be invertible
```

## Project structure

```
cs229-2018-notes/
├── index.qmd           # landing page
├── lectures/           # one .qmd per lecture — the source of truth
├── figures/            # one directory per lecture, plus the shared _svg.py
├── filters/            # Lua filters: timestamps, SEO metadata, images
├── tools/              # ingest and checking scripts, playlist.tsv
├── theme-common.scss   # type, geometry and structure — no colours
├── theme.scss          # light colour tokens
├── theme-dark.scss     # dark colour tokens
└── _quarto.yml         # book, site and PDF configuration
```

`_site/` is build output: generated, gitignored, rebuilt by CI. Never edit or
commit it.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
