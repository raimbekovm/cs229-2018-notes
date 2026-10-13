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

## Lecture 3 — Locally Weighted and Logistic Regression

| File | Status | Source |
| :--- | :----- | :----- |
| `03-locally-weighted-logistic/diagram_lwr.svg` | own work | Drawn by `make_diagrams.py`: locally weighted regression on synthetic data, at one query point and for three bandwidths. Redraws the board sketches of the local fit and the weight bump. |
| `03-locally-weighted-logistic/diagram_linear_vs_logistic.svg` | own work | Drawn by `make_diagrams.py`: least-squares and logistic fits to synthetic tumour labels, with and without one far example. Redraws Ng's board argument against linear regression for classification. |
| `03-locally-weighted-logistic/diagram_links.svg` | own work | Drawn by `make_diagrams.py`: the logistic, normal and Cauchy CDFs and their logarithms. |
| `03-locally-weighted-logistic/diagram_newton.svg` | own work | Drawn by `make_diagrams.py`: Newton's method run on the derivative of a one-parameter logistic log-likelihood. Redraws the tangent-line sketch from the board. |

## Lecture 4 — Perceptron and Generalized Linear Models

| File | Status | Source |
| :--- | :----- | :----- |
| `04-perceptron-glm/diagram_perceptron_update.svg` | own work | Drawn by `make_diagrams.py`: one perceptron update on hand-placed points, with the old and new boundaries. Redraws the board sketch of θ rotating toward a misclassified example. |
| `04-perceptron-glm/diagram_soft_hard.svg` | own work | Drawn by `make_diagrams.py`: the logistic function at three slopes against the step function. |
| `04-perceptron-glm/diagram_glm_generative.svg` | own work | Drawn by `make_diagrams.py`: samples from a Gaussian and a Bernoulli GLM. Redraws the board sketches of the data-generating process. |
| `04-perceptron-glm/diagram_parametrisations.svg` | own work | Drawn by `make_diagrams.py`: the model, natural and canonical parameters and the maps between them. |
| `04-perceptron-glm/diagram_softmax_pipeline.svg` | own work | Drawn by `make_diagrams.py`: logits, exponentials, softmax probabilities and a one-hot label for one example. |
| `04-perceptron-glm/diagram_softmax_shift.svg` | own work | Drawn by `make_diagrams.py`: a softmax regression fitted to synthetic clusters, before and after a common shift of every parameter vector. |

## Lecture 5 — GDA and Naive Bayes

| File | Status | Source |
| :--- | :----- | :----- |
| `05-gda-naive-bayes/diagram_gaussians.svg` | own work | Drawn by `make_diagrams.py`: contours of six bivariate Gaussians at fixed density heights. Recomputes the surface plots shown in lecture. |
| `05-gda-naive-bayes/diagram_gda_fit.svg` | own work | Drawn by `make_diagrams.py`: GDA with shared and separate covariances fitted to synthetic classes, with logistic regression's boundary on the same data. |
| `05-gda-naive-bayes/diagram_posterior_1d.svg` | own work | Drawn by `make_diagrams.py`: the GDA posterior along one feature at two priors. Redraws the board sketch of the sigmoid built point by point. |
| `05-gda-naive-bayes/diagram_learning_curves.svg` | own work | Drawn by `make_diagrams.py`: simulated test error of GDA and logistic regression against training-set size, on Gaussian and Poisson data. |
| `05-gda-naive-bayes/diagram_nb_graph.svg` | own work | Drawn by `make_diagrams.py`: the naive Bayes graphical model. Redraws the board sketch. |

## Lecture 6 — Support Vector Machines

| File | Status | Source |
| :--- | :----- | :----- |
| `06-support-vector-machines/diagram_event_models.svg` | own work | Drawn by `make_diagrams.py`: the lecture's example e-mail under the multivariate Bernoulli and multinomial event models. |
| `06-support-vector-machines/diagram_quad_boundary.svg` | own work | Drawn by `make_diagrams.py`: logistic regression fitted to synthetic data with raw and with quadratic features. Redraws the board sketch of a curved boundary. |
| `06-support-vector-machines/diagram_margins.svg` | own work | Drawn by `make_diagrams.py`: the geometric margin of one example for the hyperplane w = (2, 1), b = −2. Redraws the board sketch. |
| `06-support-vector-machines/diagram_optimal_margin.svg` | own work | Drawn by `make_diagrams.py`: the maximum-margin separator of a synthetic separable set beside a line that barely separates it. Redraws the board sketch of the green and red lines. |

## Lecture 7 — Kernels

| File | Status | Source |
| :--- | :----- | :----- |
| `07-kernels/diagram_lift.svg` | own work | Drawn by `make_diagrams.py`: a one-dimensional set lifted by φ(x) = (x, x²) and separated by an SVM. Recomputes, one dimension lower, the idea of the third-party video shown in lecture; nothing is traced from it. |
| `07-kernels/diagram_rbf_boundary.svg` | own work | Drawn by `make_diagrams.py`: a Gaussian-kernel SVM, solved by the script's own SMO, on synthetic rings. |
| `07-kernels/diagram_soft_margin.svg` | own work | Drawn by `make_diagrams.py`: hard- and soft-margin SVMs on lecture 6's synthetic data with one outlier added. Redraws the board sketch of the swinging boundary. |
