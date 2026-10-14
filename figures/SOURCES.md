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

## Lecture 8 — Data Splits, Models and Cross-Validation

| File | Status | Source |
| :--- | :----- | :----- |
| `08-cross-validation/diagram_three_fits.svg` | own work | Drawn by `make_diagrams.py`: polynomials of degree 1, 2 and 5 fitted to six synthetic houses. Redraws the board sketch of underfitting and overfitting. |
| `08-cross-validation/diagram_variance.svg` | own work | Drawn by `make_diagrams.py`: degree-1 and degree-5 fits to twelve independent synthetic datasets, computing the lecture's "a friend collects different data" thought experiment. |
| `08-cross-validation/diagram_lambda.svg` | own work | Drawn by `make_diagrams.py`: ridge-regularised degree-5 fits with λ = 0, a leave-one-out choice of λ, and λ = 10¹⁰⁰. |
| `08-cross-validation/diagram_error_curves.svg` | own work | Drawn by `make_diagrams.py`: training and expected test error against degree and against λ, averaged over synthetic datasets. Redraws the board sketch of error against model complexity. |
| `08-cross-validation/diagram_kfold.svg` | own work | Drawn by `make_diagrams.py`: schematic of five-fold cross-validation on 100 examples, as described in lecture. |

## Discussion section — Learning Theory

| File | Status | Source |
| :--- | :----- | :----- |
| `08b-learning-theory/diagram_sampling.svg` | own work | Drawn by `make_diagrams.py`: sampling distributions of least squares and ridge estimates on a synthetic two-feature regression. Recomputes the board's four-target sketch of bias and variance with real estimators. |
| `08b-learning-theory/diagram_decomposition.svg` | own work | Drawn by `make_diagrams.py`: schematic of g, h* and ĥ in hypothesis space beside both error decompositions computed for straight lines fitted to a parabola. |
| `08b-learning-theory/diagram_hoeffding.svg` | own work | Drawn by `make_diagrams.py`: exact binomial tail probabilities against Hoeffding's bound. |
| `08b-learning-theory/diagram_uniform.svg` | own work | Drawn by `make_diagrams.py`: true and empirical error of ten threshold classifiers on synthetic noisy data. Recomputes the board sketch of the error curves. |
| `08b-learning-theory/diagram_shatter.svg` | own work | Drawn by `make_diagrams.py`: labellings of three and four points with separating lines found by exhaustive search. |

## Lecture 9 — Decision Trees and Ensemble Methods

| File | Status | Source |
| :--- | :----- | :----- |
| `09-decision-trees-ensembles/diagram_ski.svg` | own work | Drawn by `make_diagrams.py`: a decision tree grown by the script's own CART implementation on synthetic month/latitude data. Recomputes the lecture's ski example. |
| `09-decision-trees-ensembles/diagram_impurity.svg` | own work | Drawn by `make_diagrams.py`: misclassification, cross-entropy and Gini impurities with the lecture's 900/100 split. Redraws the board sketch of the concave loss and its chord. |
| `09-decision-trees-ensembles/diagram_staircase.svg` | own work | Drawn by `make_diagrams.py`: trees of several depths fitted to synthetic data with a diagonal boundary. Redraws the board sketch of a staircase approximation. |
| `09-decision-trees-ensembles/diagram_correlation.svg` | own work | Drawn by `make_diagrams.py`: the variance of an average of correlated variables, in closed form and by simulation. |
| `09-decision-trees-ensembles/diagram_forest.svg` | own work | Drawn by `make_diagrams.py`: test error of bagged regression trees and a random forest on synthetic data as trees are added. |
| `09-decision-trees-ensembles/diagram_adaboost.svg` | own work | Drawn by `make_diagrams.py`: AdaBoost with decision stumps on synthetic data with a diagonal boundary. Redraws the board sketch of reweighted mistakes. |

## Lecture 10 — Introduction to Neural Networks

| File | Status | Source |
| :--- | :----- | :----- |
| `10-neural-networks/diagram_neuron.svg` | own work | Drawn by `make_diagrams.py`: a flattened image feeding one sigmoid neuron. Redraws the board sketch of logistic regression as a neuron. |
| `10-neural-networks/diagram_softmax.svg` | own work | Drawn by `make_diagrams.py`: one set of logits through three sigmoids and through a softmax, computed. |
| `10-neural-networks/diagram_house.svg` | own work | Drawn by `make_diagrams.py`: the house-price network wired by hand and fully connected. Redraws the board sketch. |
| `10-neural-networks/diagram_shapes.svg` | own work | Drawn by `make_diagrams.py`: the 3-2-1 network with the shape of every parameter. Redraws the board sketch. |

## Lecture 11 — Backprop and Improving Neural Networks

| File | Status | Source |
| :--- | :----- | :----- |
| `11-backprop/diagram_backprop.svg` | own work | Drawn by `make_diagrams.py`: the forward and backward passes of the 3-2-1 network, with the backpropagation recurrence the script checks against finite differences. |
| `11-backprop/diagram_activations.svg` | own work | Drawn by `make_diagrams.py`: sigmoid, tanh and ReLU and their derivatives, computed. |
| `11-backprop/diagram_normalise.svg` | own work | Drawn by `make_diagrams.py`: gradient descent on a synthetic least-squares problem with raw and standardised inputs. Recomputes the board sketch of elongated and round contours. |
| `11-backprop/diagram_init.svg` | own work | Drawn by `make_diagrams.py`: activation sizes through a randomly initialised 30-layer ReLU network under three weight variances. |
| `11-backprop/diagram_momentum.svg` | own work | Drawn by `make_diagrams.py`: gradient descent and momentum on a 100:1 quadratic. Recomputes the board sketch of momentum damping a zig-zag. |

## Lecture 12 — Debugging ML Models and Error Analysis

| File | Status | Source |
| :--- | :----- | :----- |
| `12-debugging-ml/diagram_learning_curves.svg` | own work | Drawn by `make_diagrams.py`: closed-form expected training and test error of least squares, checked against simulation. Recomputes the board sketch of high-variance and high-bias learning curves. |
| `12-debugging-ml/diagram_diagnostic.svg` | own work | Drawn by `make_diagrams.py`: logistic regression's objective along gradient ascent on a synthetic spam problem, against a class-weighted linear SVM. |
| `12-debugging-ml/diagram_helicopter.svg` | own work | Drawn by `make_diagrams.py`: the lecture's helicopter diagnostics as a decision flow. |
| `12-debugging-ml/diagram_pipeline.svg` | own work | Drawn by `make_diagrams.py`: the lecture's face-recognition pipeline and the illustrative error-analysis accuracies it quotes. |

## Lecture 13 — Expectation-Maximization Algorithms

| File | Status | Source |
| :--- | :----- | :----- |
| `13-em-algorithms/diagram_kmeans.svg` | own work | Drawn by `make_diagrams.py`: k-means run on synthetic data. Recomputes the lecture's animation. |
| `13-em-algorithms/diagram_mixture.svg` | own work | Drawn by `make_diagrams.py`: a two-component Gaussian mixture and its E-step posterior, computed. |
| `13-em-algorithms/diagram_jensen.svg` | own work | Drawn by `make_diagrams.py`: a convex function and a chord. Redraws the board sketch of Jensen's inequality. |
| `13-em-algorithms/diagram_em_bound.svg` | own work | Drawn by `make_diagrams.py`: the log-likelihood of a one-parameter mixture and EM's lower bounds, computed. Recomputes the board sketch. |

## Lecture 14 — EM Algorithm and Factor Analysis

| File | Status | Source |
| :--- | :----- | :----- |
| `14-factor-analysis/diagram_covariance.svg` | own work | Drawn by `make_diagrams.py`: maximum-likelihood Gaussians under full, diagonal and spherical covariances on synthetic data. Recomputes the board sketches. |
| `14-factor-analysis/diagram_fa_generative.svg` | own work | Drawn by `make_diagrams.py`: samples from the factor-analysis model with the lecture's parameters. Recomputes the board example. |
| `14-factor-analysis/diagram_fa_em.svg` | own work | Drawn by `make_diagrams.py`: EM for factor analysis on synthetic 100-dimensional data, and held-out likelihoods, computed. |

## Lecture 15 — PCA and ICA

| File | Status | Source |
| :--- | :----- | :----- |
| `15-pca-ica/diagram_pca_projection.svg` | own work | Drawn by `make_diagrams.py`: projections of synthetic height data onto two directions. Recomputes the board sketch. |
| `15-pca-ica/diagram_pca_preprocessing.svg` | own work | Drawn by `make_diagrams.py`: PCA on uncentred and unscaled synthetic data, computed. |
| `15-pca-ica/diagram_pca_stability.svg` | own work | Drawn by `make_diagrams.py`: angles between principal directions and subspaces of resampled synthetic data, computed. |
| `15-pca-ica/diagram_ica_mixing.svg` | own work | Drawn by `make_diagrams.py`: uniform sources, their linear mixture and an ambiguous unmixing, computed. Recomputes the slide shown in the lecture. |

## Lecture 16 — Independent Component Analysis and RL

| File | Status | Source |
| :--- | :----- | :----- |
| `16-ica-rl/diagram_source_densities.svg` | own work | Drawn by `make_diagrams.py`: contours of products of Gaussian, logistic and Laplace densities, computed. Answers the student question in the lecture. |
| `16-ica-rl/diagram_change_of_variables.svg` | own work | Drawn by `make_diagrams.py`: the lecture's one-dimensional uniform example and the three log densities, computed. Recomputes the board sketch. |
| `16-ica-rl/diagram_ica_ml.svg` | own work | Drawn by `make_diagrams.py`: ICA by batch gradient ascent on synthetic mixed Laplace sources. |
| `16-ica-rl/diagram_gridworld.svg` | own work | Drawn by `make_diagrams.py` and `figures/_gridworld.py`: the lecture's gridworld (after Russell and Norvig) and its optimal policy, computed by value iteration. Recomputes the board drawing. |

## Lecture 17 — MDPs and Value/Policy Iteration

| File | Status | Source |
| :--- | :----- | :----- |
| `17-value-policy-iteration/diagram_values.svg` | own work | Drawn by `make_diagrams.py`: the value of an always-East policy and the optimal value function of the gridworld, computed. Recomputes the board values. |
| `17-value-policy-iteration/diagram_vi_convergence.svg` | own work | Drawn by `make_diagrams.py`: value iteration's error per sweep on the gridworld, computed. |
| `17-value-policy-iteration/diagram_exploration.svg` | own work | Drawn by `make_diagrams.py`: simulated model-based learners on a corridor MDP of our own, a one-dimensional version of the lecture's +1/+10 example. |

## Lecture 18 — Continuous State MDPs and Model Simulation

| File | Status | Source |
| :--- | :----- | :----- |
| `18-continuous-mdps/diagram_staircase.svg` | own work | Drawn by `make_diagrams.py`: a one-dimensional MDP of our own solved on a fine grid, by a six-cell discretization and by fitted value iteration, computed; the k^n curve is arithmetic. Recomputes the lecture's staircase sketch. |
| `18-continuous-mdps/diagram_divergence.svg` | own work | Drawn by `make_diagrams.py`: least-squares fitted value iteration on the two-state example of Tsitsiklis and Van Roy (1996), computed from the example's definition. |

## Lecture 19 — Finite-Horizon MDPs and the Linear Quadratic Regulator

| File | Status | Source |
| :--- | :----- | :----- |
| `19-lqr/diagram_horizon.svg` | own work | Drawn by `make_diagrams.py`: finite-horizon dynamic programming on a corridor of our own, a version of the lecture's +1/+10 example. |
| `19-lqr/diagram_pendulum.svg` | own work | Drawn by `make_diagrams.py`: a simple inverted pendulum, its linearization and an LQR controller, simulated. Recomputes the lecture's tangent-line sketch. |
| `19-lqr/diagram_gains.svg` | own work | Drawn by `make_diagrams.py`: the Riccati recursion for a cart model of our own, computed. |
