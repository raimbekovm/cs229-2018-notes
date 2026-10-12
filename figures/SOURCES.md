# Figure provenance

One row per figure used in the notes. This file is the authoritative record; the
short credit in each figure caption is a summary of the row here.

**Status column:**

- **own work** — drawn for this project by the `make_diagrams.py` beside it. If
  it redraws a figure the lecture drew at the board or the official notes print,
  the row says which, so the debt is recorded even though no pixels were copied.
- **verified** — a third-party figure compared directly against its primary
  source (the paper PDF) and matching. The comparison is reproducible: fetch the
  source listed and look at the page given.
- **attributed** — a recognisable, canonical illustration of the cited work
  whose specific bitmap has not been matched against the publication.

A figure whose origin cannot be established does not go into the notes. A
missing credit would read as a claim of authorship, and a plausible-looking
invented one as a verified credit.

## Site images

| File | Status | Source |
| :--- | :----- | :----- |
| `readme-hero.svg` | own work | Drawn by `make_readme_hero.py`: batch gradient descent on a quadratic bowl, computed from the update rule it illustrates. |
| `social-card.svg`, `social-card.png` | own work | Drawn by `make_readme_hero.py`, same geometry as the banner; the PNG is rasterised from the SVG. |

## Lecture 1 — Introduction

| File | Status | Source |
| :--- | :----- | :----- |
| `01-introduction/diagram_regression.svg` | own work | Drawn by `make_diagrams.py`: synthetic house sizes and prices, with a line and a quadratic fitted by least squares. Redraws the housing-price sketch Ng drew at the board. |
| `01-introduction/diagram_tumour_1d.svg` | own work | Drawn by `make_diagrams.py`: synthetic tumour sizes, shown as label-against-size and projected onto one axis. Redraws the board sketch of the same two views. |
| `01-introduction/diagram_tumour_2d.svg` | own work | Drawn by `make_diagrams.py`: synthetic size-and-age data with the boundary fitted by logistic regression (Newton's method). Redraws the two-feature board sketch. |
| `01-introduction/diagram_clusters.svg` | own work | Drawn by `make_diagrams.py`: synthetic unlabelled points clustered by k-means (Lloyd's algorithm). Redraws the unlabelled-data board sketch. |

## Lecture 2 — Linear Regression and Gradient Descent

| File | Status | Source |
| :--- | :----- | :----- |
| `02-linear-regression/diagram_local_minima.svg` | own work | Drawn by `make_diagrams.py`: gradient descent run from two starts on a made-up one-parameter cost with two minima. Stands in for the hilly 3-D surface Ng showed. |
| `02-linear-regression/diagram_contours.svg` | own work | Drawn by `make_diagrams.py`: contours of the least-squares cost on 47 synthetic houses, with the batch and stochastic gradient descent paths actually run on them. Redraws the contour sketches Ng drew at the board. |
| `02-linear-regression/diagram_learning_rate.svg` | own work | Drawn by `make_diagrams.py`: gradient descent on J(θ) = θ² at three learning rates. |
| `02-linear-regression/diagram_gd_lines.svg` | own work | Drawn by `make_diagrams.py`: the hypothesis after successive gradient descent steps on the synthetic housing data. Redraws the idea of the slide sequence Ng showed, from computed values. |
