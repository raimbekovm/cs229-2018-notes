"""Figures for lecture 8: bias, variance, regularisation and cross-validation.

Every figure is computed from one synthetic housing problem: six (or ten)
houses whose price rises with size and then levels off, the shape Ng draws at
the board, generated as a concave quadratic plus Gaussian noise. Polynomials of
each degree are fitted by solving the (regularised) normal equations exactly.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: the ridge solution in closed form
against gradient descent; the MAP estimate under a Gaussian prior, found by
gradient ascent on the log posterior, against ridge with λ = σ²/τ², for linear
and for logistic regression; the leave-one-out error computed by refitting m
times against the one-fit shortcut through the hat matrix; the expected
minimum of six noisy dev-set scores by simulation against numerical
integration; the variance of a 10-fold estimate across datasets against the
naive estimate from its own ten fold scores; and the number of fits forward
search makes against n(n+1)/2 + 1.

Plain Python on purpose, as in lectures 1–7.

Run from the repository root.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import (
    INDIGO,
    PURPLE,
    TEAL,
    axes,
    circle,
    curve,
    line,
    mapper,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent
SIGMA = 0.06  # noise standard deviation of the housing prices


# --- the problem -------------------------------------------------------------


def truth(x):
    """Price as a function of size: rises, then levels off (Ng's board sketch)."""
    return 0.2 + 1.6 * x - 1.0 * x * x


def design(m):
    """Evenly spaced house sizes in (0, 1)."""
    return [(i + 0.5) / m for i in range(m)]


def sample(xs, rng):
    """One noisy draw of prices at the given sizes — 'a friend's dataset'."""
    return [truth(x) + rng.gauss(0, SIGMA) for x in xs]


def features(x, d):
    """Polynomial features of degree d in t = 2x − 1, which keeps them well scaled."""
    t = 2 * x - 1
    return [t**j for j in range(d + 1)]


# --- linear algebra ----------------------------------------------------------


def solve(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-14:
            raise ValueError("singular")
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def min_pivot(a):
    """Smallest pivot met while eliminating a — zero means a is singular."""
    n = len(a)
    m = [row[:] for row in a]
    smallest = math.inf
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        smallest = min(smallest, abs(m[c][c]))
        if abs(m[c][c]) < 1e-300:
            return 0.0
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n):
                m[r][k] -= f * m[c][k]
    return smallest


def gram(rows):
    """XᵀX for a list of feature rows."""
    n = len(rows[0])
    return [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]


def ridge(rows, ys, lam, penalise_intercept=False):
    """Argmin ½Σ(θᵀx − y)² + (λ/2)‖θ‖², the intercept optionally left unpenalised."""
    a = gram(rows)
    for j in range(len(a)):
        if j > 0 or penalise_intercept:
            a[j][j] += lam
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(len(rows[0]))]
    return solve(a, b)


def predict(theta, row):
    """Θᵀx."""
    return sum(t * v for t, v in zip(theta, row))


def fit_poly(xs, ys, d, lam=0.0, penalise_intercept=False):
    """Fit a degree-d polynomial; return it as a function of x."""
    theta = ridge([features(x, d) for x in xs], ys, lam, penalise_intercept)
    return lambda x: predict(theta, features(x, d))


def mse(h, xs, ys):
    """Mean squared error of h on (xs, ys)."""
    return sum((h(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def test_error(h):
    """Expected squared error of h on a fresh house drawn uniformly in [0, 1]."""
    g = [(i + 0.5) / 400 for i in range(400)]
    return sum((h(x) - truth(x)) ** 2 for x in g) / len(g) + SIGMA**2


# --- the checks the prose rests on -------------------------------------------


def gradient_descent_ridge(rows, ys, lam, steps=20000):
    """Ridge by plain gradient descent, as a second route to the closed form."""
    n = len(rows[0])
    a = gram(rows)
    lmax = max(sum(abs(v) for v in row) for row in a) + lam  # Gershgorin bound
    alpha = 1.0 / lmax
    theta = [0.0] * n
    for _ in range(steps):
        r = [predict(theta, x) - y for x, y in zip(rows, ys)]
        g = [sum(ri * x[j] for ri, x in zip(r, rows)) + (lam * theta[j] if j > 0 else 0.0) for j in range(n)]
        theta = [t - alpha * gj for t, gj in zip(theta, g)]
    return theta


def map_linear(rows, ys, sigma, tau, steps=20000):
    """MAP for y = θᵀx + N(0, σ²) under θ ~ N(0, τ²I), by ascent on the log posterior."""
    n = len(rows[0])
    a = gram(rows)
    lmax = (max(sum(abs(v) for v in row) for row in a)) / sigma**2 + 1 / tau**2
    alpha = 1.0 / lmax
    theta = [0.0] * n
    for _ in range(steps):
        r = [y - predict(theta, x) for x, y in zip(rows, ys)]
        g = [sum(ri * x[j] for ri, x in zip(r, rows)) / sigma**2 - theta[j] / tau**2 for j in range(n)]
        theta = [t + alpha * gj for t, gj in zip(theta, g)]
    return theta


def sigmoid(z):
    """Logistic function."""
    return 1 / (1 + math.exp(-z))


def logistic_toy():
    """A small two-feature classification set with overlap, so the MLE is finite."""
    rng = random.Random(8)
    rows, ys = [], []
    for _ in range(40):
        y = rng.random() < 0.5
        c = 0.8 if y else -0.8
        rows.append([1.0, c + rng.gauss(0, 1), 0.5 * c + rng.gauss(0, 1)])
        ys.append(1 if y else 0)
    return rows, ys


def newton_penalised_logistic(rows, ys, lam):
    """Argmax ℓ(θ) − λ‖θ‖² by Newton's method (Ng's form of regularised logistic regression)."""
    n = len(rows[0])
    theta = [0.0] * n
    for _ in range(50):
        p = [sigmoid(predict(theta, x)) for x in rows]
        g = [sum((y - pi) * x[j] for x, y, pi in zip(rows, ys, p)) - 2 * lam * theta[j] for j in range(n)]
        h = [
            [
                -sum(pi * (1 - pi) * x[i] * x[j] for x, pi in zip(rows, p)) - (2 * lam if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]
        step = solve(h, g)
        theta = [t - s for t, s in zip(theta, step)]
    return theta


def map_logistic(rows, ys, tau, steps=40000):
    """MAP logistic regression under θ ~ N(0, τ²I), by gradient ascent on the log posterior."""
    n = len(rows[0])
    theta = [0.0] * n
    alpha = 0.02
    for _ in range(steps):
        p = [sigmoid(predict(theta, x)) for x in rows]
        g = [sum((y - pi) * x[j] for x, y, pi in zip(rows, ys, p)) - theta[j] / tau**2 for j in range(n)]
        theta = [t + alpha * gj for t, gj in zip(theta, g)]
    return theta


def loocv_brute(xs, ys, d, lam):
    """Leave-one-out error by refitting m times."""
    total = 0.0
    for i in range(len(xs)):
        h = fit_poly(xs[:i] + xs[i + 1 :], ys[:i] + ys[i + 1 :], d, lam)
        total += (h(xs[i]) - ys[i]) ** 2
    return total / len(xs)


def loocv_shortcut(xs, ys, d, lam):
    """Leave-one-out error from one fit: residual_i / (1 − H_ii), H the ridge hat matrix."""
    rows = [features(x, d) for x in xs]
    a = gram(rows)
    for j in range(1, len(a)):
        a[j][j] += lam
    theta = ridge(rows, ys, lam)
    total = 0.0
    for r, y in zip(rows, ys):
        hii = predict(r, solve(a, r))
        total += ((predict(theta, r) - y) / (1 - hii)) ** 2
    return total / len(xs)


def kfold(xs, ys, d, k, lam=0.0):
    """K-fold cross-validation: the k fold errors (folds are contiguous blocks of a shuffled set)."""
    m = len(xs)
    errs = []
    for f in range(k):
        lo, hi = f * m // k, (f + 1) * m // k
        h = fit_poly(xs[:lo] + xs[hi:], ys[:lo] + ys[hi:], d, lam)
        errs.append(mse(h, xs[lo:hi], ys[lo:hi]))
    return errs


def expected_min_normal(k):
    """E[min of k iid N(0,1)] by numerical integration of k·z·φ(z)·(1−Φ(z))^(k−1)."""
    total, dz = 0.0, 0.001
    z = -8.0
    while z < 8.0:
        phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        big_phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        total += k * z * phi * (1 - big_phi) ** (k - 1) * dz
        z += dz
    return total


def forward_search(rows, ys, dev_rows, dev_ys):
    """Greedy forward feature selection; return the order features were added and the fit count."""
    n = len(rows[0])
    chosen, order, fits = [], [], 0

    def dev_err(cols):
        nonlocal fits
        fits += 1
        r = [[1.0] + [x[c] for c in cols] for x in rows]
        th = ridge(r, ys, 1e-9)
        return sum((predict(th, [1.0] + [x[c] for c in cols]) - y) ** 2 for x, y in zip(dev_rows, dev_ys)) / len(dev_ys)

    history = [dev_err([])]
    while len(chosen) < n:
        best = min((c for c in range(n) if c not in chosen), key=lambda c: dev_err([*chosen, c]))
        chosen.append(best)
        order.append(best)
        history.append(dev_err(chosen))
        fits -= 1  # the re-evaluation of the winner is bookkeeping, not a new fit
    return order, fits, history


# --- figures -----------------------------------------------------------------


def housing():
    """The one six-house dataset every single-fit figure uses."""
    xs = design(6)
    return xs, sample(xs, random.Random(3))


GRID = [i / 400 for i in range(401)]


def dots(xs, ys, to_px):
    """The training points."""
    return "".join(circle(*to_px(x, y), 4.2, fill=PURPLE, stroke=PURPLE) for x, y in zip(xs, ys))


def panel(x0, xs, ys, h, colour, title, ydom, xlabel="size", ylabel="price"):
    """One small plot of a fit over the data."""
    xdom = (0.0, 1.0)
    to_px = mapper(xdom, ydom, (x0, x0 + 180), (46, 216))
    s = axes(to_px, xdom, ydom, [0, 0.5, 1], [0, 0.5, 1], xlabel, ylabel)
    s += curve([(x, h(x)) for x in GRID], to_px, xdom, ydom, stroke=colour, sw=2.2)
    s += dots(xs, ys, to_px)
    s += text(x0 + 90, 26, title, size=13, weight="600")
    return s


def fig_three_fits():
    """Degree 1, 2 and 5 fits to six houses: underfit, just right, overfit."""
    xs, ys = housing()
    ydom = (-0.1, 1.1)
    s = ""
    for i, (d, colour, title) in enumerate(
        ((1, INDIGO, "degree 1: underfits"), (2, TEAL, "degree 2: just right"), (5, PURPLE, "degree 5: overfits"))
    ):
        s += panel(60 + 240 * i, xs, ys, fit_poly(xs, ys, d), colour, title, ydom)
    return wrap(
        720,
        270,
        "figThreeFits",
        "Three polynomial fits to six houses",
        "Six points rising then levelling off. A straight line misses the curvature, a quadratic follows it, and a fifth-degree polynomial passes through every point while bending between them; beyond the outermost houses it leaves the plotted range.",
        s,
    )


def friends(d, n=12, seed=11):
    """Degree-d fits to n independent draws of the same six houses."""
    xs = design(6)
    rng = random.Random(seed)
    return [fit_poly(xs, sample(xs, rng), d) for _ in range(n)]


def fig_variance():
    """Twelve friends' datasets: degree-1 fits agree and are wrong, degree-5 fits disagree."""
    xdom, ydom = (0.0, 1.0), (-0.1, 1.1)
    s = ""
    for i, (d, colour, title) in enumerate(
        ((1, INDIGO, "degree 1 on 12 datasets"), (5, PURPLE, "degree 5 on 12 datasets"))
    ):
        x0 = 70 + 340 * i
        to_px = mapper(xdom, ydom, (x0, x0 + 260), (46, 226))
        s += axes(to_px, xdom, ydom, [0, 0.5, 1], [0, 0.5, 1], "size", "price")
        for h in friends(d):
            s += curve([(x, h(x)) for x in GRID], to_px, xdom, ydom, stroke=colour, sw=1.2, op=0.55)
        s += curve([(x, truth(x)) for x in GRID], to_px, xdom, ydom, stroke="currentColor", sw=1.6, dash="5 4")
        s += text(x0 + 130, 26, title, size=13, weight="600")
    s += line(250, 278, 280, 278, sw=1.6, dash="5 4") + text(286, 282, "the true price curve", size=11, anchor="start")
    return wrap(
        720,
        296,
        "figVariance",
        "Fits to twelve independent datasets",
        "Left: twelve straight-line fits, close to one another and all missing the curvature of the dashed true curve. Right: twelve fifth-degree fits that swing far apart, especially beyond the outermost houses, where several leave the plotted range.",
        s,
    )


def fig_lambda():
    """Degree-5 fits with λ = 0, a cross-validated λ, and an enormous λ."""
    xs, ys = housing()
    lam_cv = best_lambda(xs, ys)
    ydom = (-0.1, 1.1)
    xdom = (0.0, 1.0)
    s = panel(60, xs, ys, fit_poly(xs, ys, 5, 0.0), PURPLE, "λ = 0", ydom)
    s += panel(300, xs, ys, fit_poly(xs, ys, 5, lam_cv), TEAL, f"λ = {lam_cv:g} (by LOOCV)", ydom)
    to_px = mapper(xdom, ydom, (540, 720), (46, 216))
    s += axes(to_px, xdom, ydom, [0, 0.5, 1], [0, 0.5, 1], "size", "price")
    big = 1e100
    h_free = fit_poly(xs, ys, 5, big)
    h_all = fit_poly(xs, ys, 5, big, penalise_intercept=True)
    s += curve([(x, h_free(x)) for x in GRID], to_px, xdom, ydom, stroke=INDIGO, sw=2.2)
    s += curve([(x, h_all(x)) for x in GRID], to_px, xdom, ydom, stroke=INDIGO, sw=2.2, dash="6 4")
    s += dots(xs, ys, to_px)
    s += text(630, 26, "λ = 10¹⁰⁰", size=13, weight="600")
    s += text(716, to_px(0, h_free(0.5))[1] - 8, "θ₀ free: h = ȳ", size=11, anchor="end", fill=INDIGO)
    s += text(716, to_px(0, 0)[1] - 8, "θ₀ penalised: h = 0", size=11, anchor="end", fill=INDIGO)
    return wrap(
        760,
        270,
        "figLambda",
        "Regularised fifth-degree fits",
        "Left: with no penalty the fifth-degree polynomial interpolates the six points. Middle: a moderate penalty chosen by leave-one-out cross-validation gives a smooth rising-then-levelling curve. Right: an enormous penalty flattens the fit to the mean price, or to zero if the intercept is penalised too.",
        s,
    )


LAMBDAS = [10 ** (e / 4) for e in range(-24, 9)]  # 10⁻⁶ … 10²


def best_lambda(xs, ys):
    """Λ (on a coarse grid) that minimises leave-one-out error of a degree-5 fit."""
    grid = [10.0**e for e in range(-6, 2)]
    return min(grid, key=lambda lam: loocv_shortcut(xs, ys, 5, lam))


def error_curves(reps=400):
    """Average training and test error against degree, and against λ for degree 9, on m = 10."""
    xs = design(10)
    rng = random.Random(5)
    datasets = [sample(xs, rng) for _ in range(reps)]
    by_deg = []
    for d in range(0, 9):
        tr = te = 0.0
        for ys in datasets:
            h = fit_poly(xs, ys, d)
            tr += mse(h, xs, ys)
            te += test_error(h)
        by_deg.append((d, tr / reps, te / reps))
    by_lam = []
    for lam in LAMBDAS:
        tr = te = 0.0
        for ys in datasets[:150]:
            h = fit_poly(xs, ys, 9, lam)
            tr += mse(h, xs, ys)
            te += test_error(h)
        by_lam.append((lam, tr / 150, te / 150))
    return by_deg, by_lam


CURVES = None


def curves():
    """Cached error curves (they take a few seconds)."""
    global CURVES
    if CURVES is None:
        CURVES = error_curves()
    return CURVES


def fig_error_curves():
    """Training error falls with complexity; test error falls, then rises."""
    by_deg, by_lam = curves()
    ydom = (-4.5, -0.5)  # log10 of squared error
    s = ""
    xdom = (0, 8)
    to_px = mapper(xdom, ydom, (70, 330), (46, 226))
    s += axes(to_px, xdom, ydom, list(range(0, 9, 2)), [], "polynomial degree", "squared error (log scale)")
    s += log_ticks(to_px, xdom, ydom)
    s += curve([(d, math.log10(tr)) for d, tr, _ in by_deg], to_px, xdom, ydom, stroke=INDIGO, sw=2.0)
    s += curve([(d, math.log10(te)) for d, _, te in by_deg], to_px, xdom, ydom, stroke=TEAL, sw=2.4)
    for d, tr, te in by_deg:
        s += circle(*to_px(d, math.log10(te)), 3, fill=TEAL, stroke=TEAL)
        if math.log10(tr) >= ydom[0]:
            s += circle(*to_px(d, math.log10(tr)), 3, fill=INDIGO, stroke=INDIGO)
    s += text(200, 26, "degree 0 … 8, no regularisation", size=13, weight="600")

    xdom2 = (2.0, -6.0)  # λ from large (simple) to small (complex), left to right
    to_px = mapper(xdom2, ydom, (430, 690), (46, 226))
    s += axes(to_px, (2.0, -6.0), ydom, [2, 0, -2, -4, -6], [], "log₁₀ λ (complexity →)", "")
    s += log_ticks(to_px, xdom2, ydom)
    lo = (min(xdom2), max(xdom2))
    s += curve(
        [(math.log10(lam), math.log10(tr)) for lam, tr, _ in by_lam],
        lambda u, v: to_px(u, v),
        lo,
        ydom,
        stroke=INDIGO,
        sw=2.0,
    )
    s += curve(
        [(math.log10(lam), math.log10(te)) for lam, _, te in by_lam],
        lambda u, v: to_px(u, v),
        lo,
        ydom,
        stroke=TEAL,
        sw=2.4,
    )
    s += text(560, 26, "degree 9, penalty λ", size=13, weight="600")
    s += line(250, 284, 280, 284, stroke=TEAL, sw=2.4) + text(286, 288, "test error", size=11, anchor="start")
    s += line(390, 284, 420, 284, stroke=INDIGO, sw=2.0) + text(426, 288, "training error", size=11, anchor="start")
    return wrap(
        720,
        300,
        "figErrorCurves",
        "Training and test error against model complexity",
        "Averaged over 400 datasets of ten houses. Left: as the degree rises, training error falls steadily while test error falls to a minimum at degree 2 and then climbs. Right: the same shape for a ninth-degree polynomial as the penalty λ is relaxed from large to tiny.",
        s,
    )


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_ticks(to_px, xdom, ydom):
    """Y ticks at powers of ten for an axis whose data are log10 values."""
    s = ""
    x0 = to_px(xdom[0], 0)[0]
    for e in range(math.ceil(ydom[0]), math.floor(ydom[1]) + 1):
        py = to_px(xdom[0], e)[1]
        s += line(x0 - 4, py, x0, py, op=0.7) + text(
            x0 - 8,
            py + 4,
            "10" + str(e).translate(SUPERSCRIPT),
            size=11,
            anchor="end",
            op=0.8,
        )
    return s


def fig_kfold():
    """Five-fold cross-validation: each fold is held out once."""
    s = ""
    w, h = 84, 26
    x0, y0 = 150, 50
    for r in range(5):
        y = y0 + r * 38
        s += text(x0 - 14, y + 18, f"round {r + 1}", size=12, anchor="end")
        for c in range(5):
            held = c == 4 - r
            s += rect(
                x0 + c * (w + 6), y, w, h, fill=TEAL if held else INDIGO, stroke="none", op=0.85 if held else 0.22
            )
            s += text(
                x0 + c * (w + 6) + w / 2,
                y + 18,
                "test" if held else "train",
                size=12,
                fill="#ffffff" if held else "currentColor",
            )
        s += text(x0 + 5 * (w + 6) + 14, y + 18, f"error ε̂{chr(0x2081 + r)}", size=12, anchor="start")
    for c in range(5):
        s += text(x0 + c * (w + 6) + w / 2, y0 - 12, f"S{chr(0x2081 + c)} (20 examples)", size=11, op=0.8)
    s += text(360, y0 + 5 * 38 + 22, "estimate for this model = (ε̂₁ + ε̂₂ + ε̂₃ + ε̂₄ + ε̂₅) / 5", size=13, weight="600")
    return wrap(
        720,
        290,
        "figKfold",
        "Five-fold cross-validation",
        "A hundred examples split into five folds of twenty. In each of five rounds one fold is held out for testing and the model is trained on the other four; the five test errors are averaged.",
        s,
    )


FIGURES = {
    "diagram_three_fits.svg": fig_three_fits,
    "diagram_variance.svg": fig_variance,
    "diagram_lambda.svg": fig_lambda,
    "diagram_error_curves.svg": fig_error_curves,
    "diagram_kfold.svg": fig_kfold,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    xs, ys = housing()

    # 0. Extents: the degree-5 interpolant and friends' fits against the plotted range.
    h5 = fit_poly(xs, ys, 5)
    print(f"degree-5 fit on [0,1]: min {min(h5(x) for x in GRID):.3f}, max {max(h5(x) for x in GRID):.3f}")
    ext = [h(x) for h in friends(5) for x in GRID]
    print(f"friends' degree-5 fits: min {min(ext):.3f}, max {max(ext):.3f} (plot range -0.1..1.1)")

    # 1. Bias and variance at x = 0.5 and at the edge x = 0.02, over 4000 friends.
    rng = random.Random(1)
    for d in (1, 2, 5):
        for x0 in (0.5, 0.02):
            preds = []
            for _ in range(4000):
                preds.append(fit_poly(design(6), sample(design(6), rng), d)(x0))
            mean = sum(preds) / len(preds)
            var = sum((p - mean) ** 2 for p in preds) / len(preds)
            print(f"degree {d} at x={x0}: bias {mean - truth(x0):+.4f}, sd {math.sqrt(var):.4f}")

    # 2. Ridge closed form against gradient descent.
    rows = [features(x, 5) for x in xs]
    closed = ridge(rows, ys, 1e-2)
    gd = gradient_descent_ridge(rows, ys, 1e-2, steps=200000)
    gap = max(abs(a - b) for a, b in zip(closed, gd))
    print(f"ridge closed form vs gradient descent: max |Δθ| = {gap:.2e}")
    assert gap < 1e-6

    # 3. Huge λ: penalised intercept gives h ≈ 0, free intercept gives h = mean(y).
    big = 1e100
    h_all, h_free = fit_poly(xs, ys, 5, big, True), fit_poly(xs, ys, 5, big)
    ybar = sum(ys) / len(ys)
    print(
        f"λ=1e100: max|h| with θ₀ penalised = {max(abs(h_all(x)) for x in GRID):.1e}; with θ₀ free h(0.5) = {h_free(0.5):.4f}, ȳ = {ybar:.4f}"
    )
    assert max(abs(h_all(x)) for x in GRID) < 1e-90 and abs(h_free(0.5) - ybar) < 1e-12
    print(f"LOOCV-chosen λ for the degree-5 housing fit: {best_lambda(xs, ys):g}")

    # 4. MAP with a Gaussian prior is ridge with λ = σ²/τ² (linear) and λ = 1/(2τ²) (Ng's logistic form).
    tau = 0.5
    rows2 = [features(x, 3) for x in xs]
    mp = map_linear(rows2, ys, SIGMA, tau, steps=400000)
    lam = SIGMA**2 / tau**2
    a = gram(rows2)
    for j in range(len(a)):
        a[j][j] += lam
    rd = solve(a, [sum(r[i] * y for r, y in zip(rows2, ys)) for i in range(len(rows2[0]))])
    gap = max(abs(u - v) for u, v in zip(mp, rd))
    print(f"MAP (linear, σ={SIGMA}, τ={tau}) vs ridge λ=σ²/τ²={lam:.4f}: max |Δθ| = {gap:.2e}")
    assert gap < 1e-6
    lr_rows, lr_ys = logistic_toy()
    tau = 0.7
    nt = newton_penalised_logistic(lr_rows, lr_ys, 1 / (2 * tau**2))
    ga = map_logistic(lr_rows, lr_ys, tau)
    gap = max(abs(u - v) for u, v in zip(nt, ga))
    print(
        f"MAP logistic (τ={tau}) vs Newton on ℓ − λ‖θ‖², λ=1/(2τ²)={1 / (2 * tau**2):.4f}: max |Δθ| = {gap:.2e}; θ = {[round(t, 3) for t in nt]}"
    )
    assert gap < 1e-6

    # 5. More features than examples: XᵀX singular, XᵀX + λI not.
    rng = random.Random(2)
    wide = [[rng.gauss(0, 1) for _ in range(10)] for _ in range(4)]
    g = gram(wide)
    print(
        f"n=10 features, m=4 examples: smallest pivot of XᵀX = {min_pivot(g):.1e}; of XᵀX + 0.1I = {min_pivot([[v + (0.1 if i == j else 0) for j, v in enumerate(r)] for i, r in enumerate(g)]):.3f}"
    )
    # Unregularised logistic regression on separable n > m data: ‖θ‖ keeps growing.
    lab = [1, 0, 1, 0]
    theta = [0.0] * 10
    norms = {}
    for step in range(1, 10001):
        p = [sigmoid(predict(theta, x)) for x in wide]
        theta = [t + 0.5 * sum((y - pi) * x[j] for x, y, pi in zip(wide, lab, p)) for j, t in enumerate(theta)]
        if step in (100, 1000, 10000):
            norms[step] = math.sqrt(sum(t * t for t in theta))
    print("unregularised logistic, ‖θ‖ after 100/1000/10000 steps:", {k: round(v, 2) for k, v in norms.items()})
    assert norms[100] < norms[1000] < norms[10000]

    # 6. Training error falls monotonically with degree; test error is U-shaped.
    by_deg, by_lam = curves()
    for d, tr, te in by_deg:
        print(f"  degree {d}: train {tr:.5f}  test {te:.5f}")
    assert all(by_deg[i][1] >= by_deg[i + 1][1] - 1e-12 for i in range(len(by_deg) - 1))
    best_d = min(by_deg, key=lambda r: r[2])[0]
    best_l = min(by_lam, key=lambda r: r[2])
    print(f"test error minimised at degree {best_d}; for degree 9 at λ = {best_l[0]:.2g}")
    print("degree-9 curve extremes (log10):", round(math.log10(by_lam[0][2]), 2), round(math.log10(by_lam[0][1]), 2), "at λ=", by_lam[0][0])
    print(f"noise floor σ² = {SIGMA**2:.5f}")
    tr_l = [r[1] for r in by_lam]
    assert all(tr_l[i] <= tr_l[i + 1] + 1e-12 for i in range(len(tr_l) - 1))

    # 7. Ng's dev-set example: six models with true error 5.0, dev noise sd s. The minimum is optimistic.
    k = 6
    em = expected_min_normal(k)
    rng = random.Random(4)
    sim = sum(min(rng.gauss(0, 1) for _ in range(k)) for _ in range(200000)) / 200000
    print(f"E[min of {k} N(0,1)]: integration {em:.4f}, simulation {sim:.4f}")
    assert abs(em - sim) < 0.01
    print(
        f"  so with dev noise sd 0.1 the winner's reported error is {5.0 + 0.1 * em:.3f} on average, though all are 5.0"
    )

    # 8. How big a dev set resolves which accuracy difference (standard error of an accuracy at p=0.9).
    for m in (100, 1000, 10000, 1000000):
        print(f"  m={m:>8}: one example = {100 / m:.4f}%, SE of a 90% accuracy = {100 * math.sqrt(0.09 / m):.3f}%")
    m_needed = 4 * 0.09 / (1e-4) ** 2
    print(f"  unpaired, two SEs below 0.01%: m ≈ {m_needed:.2e}")

    # 9. Leave-one-out: m refits against the hat-matrix shortcut.
    xs10 = design(10)
    ys10 = sample(xs10, random.Random(6))
    for d, lam in ((2, 0.0), (5, 1e-3), (9, 1e-4)):
        bf, sc = loocv_brute(xs10, ys10, d, lam), loocv_shortcut(xs10, ys10, d, lam)
        print(f"LOOCV degree {d}, λ={lam:g}: refit {bf:.6f}, shortcut {sc:.6f}")
        assert abs(bf - sc) < 1e-9 * max(1, bf)

    # 10. The variance of a 10-fold estimate across datasets versus the naive one from its own folds.
    rng = random.Random(9)
    m, kf = 40, 10
    ests, naive = [], []
    for _ in range(1500):
        xs40 = [rng.random() for _ in range(m)]
        ys40 = [truth(x) + rng.gauss(0, SIGMA) for x in xs40]
        errs = kfold(xs40, ys40, 2, kf)
        mean = sum(errs) / kf
        ests.append(mean)
        naive.append(sum((e - mean) ** 2 for e in errs) / (kf - 1) / kf)
    mu = sum(ests) / len(ests)
    actual = sum((e - mu) ** 2 for e in ests) / (len(ests) - 1)
    print(
        f"10-fold CV, m=40: actual variance of the estimate {actual:.3e}, mean naive variance {sum(naive) / len(naive):.3e}, ratio {actual / (sum(naive) / len(naive)):.2f}"
    )

    # 11. Forward search on 5 features, two of them relevant; count the fits.
    rng = random.Random(12)

    def draw(mm):
        """Rows of five features where only x₂ and x₄ (indices 1 and 3) matter."""
        r = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(mm)]
        return r, [1.5 * x[1] - 1.0 * x[3] + rng.gauss(0, 0.5) for x in r]

    tr_r, tr_y = draw(60)
    dv_r, dv_y = draw(60)
    order, fits, hist = forward_search(tr_r, tr_y, dv_r, dv_y)
    n = 5
    print(
        f"forward search order (1-indexed): {[c + 1 for c in order]}; fits {fits}, n(n+1)/2 + 1 = {n * (n + 1) // 2 + 1}; dev errors {[round(h, 3) for h in hist]}"
    )
    assert fits == n * (n + 1) // 2 + 1 and order[:2] == [1, 3]
    print(f"n = 10,000: forward search to the end = {10000 * 10001 // 2 + 1:,} fits, against 2^10000 subsets")
