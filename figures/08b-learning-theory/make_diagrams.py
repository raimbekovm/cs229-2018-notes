"""Figures for the learning-theory discussion section.

Three small worlds carry every figure and every number. A two-parameter linear
regression with correlated features, fitted by least squares and by ridge,
gives the sampling distributions of the parameter view of bias and variance.
A one-dimensional regression in which the truth is a parabola and the class
is straight lines gives both decompositions of the error. And a threshold
classifier on [0, 1] with 15 % label noise, restricted to ten thresholds,
gives a finite hypothesis class small enough that the true error of every
hypothesis is a closed form, so uniform convergence can be watched rather
than assumed.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: Hoeffding's bound against the exact
binomial tail; the union bound against the simulated probability that some
hypothesis in the class deviates by more than γ; the guarantee
ε(ĥ) ≤ ε(h*) + 2γ against simulated runs of empirical risk minimisation;
the approximation-plus-estimation split against the bias-plus-variance split
of the same expected error, including the Pythagorean identity that links
them; the variance of the mean and the median against 1/m and π/(2m); the
bias of ridge at growing m (it is consistent); and every labelling of three
points, and of the two kinds of four-point set, against an exhaustive
search for a separating line.

Plain Python on purpose, as in lectures 1–8.

Run from the repository root.
"""

import itertools
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
    polyline,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# --- world 1: two correlated features, least squares and ridge ---------------

THETA_STAR = (1.0, 0.5)
RHO = 0.9
NOISE = 1.0


def draw_regression(m, rng):
    """M examples with features of correlation RHO and y = θ*ᵀx + N(0, 1)."""
    rows, ys = [], []
    for _ in range(m):
        z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
        x = (z1, RHO * z1 + math.sqrt(1 - RHO**2) * z2)
        rows.append(x)
        ys.append(THETA_STAR[0] * x[0] + THETA_STAR[1] * x[1] + rng.gauss(0, NOISE))
    return rows, ys


def fit2(rows, ys, lam):
    """Two-parameter ridge (λ = 0 is least squares), no intercept."""
    a = sum(x[0] * x[0] for x in rows) + lam
    b = sum(x[0] * x[1] for x in rows)
    d = sum(x[1] * x[1] for x in rows) + lam
    u = sum(x[0] * y for x, y in zip(rows, ys))
    v = sum(x[1] * y for x, y in zip(rows, ys))
    det = a * d - b * b
    return ((d * u - b * v) / det, (a * v - b * u) / det)


def sampling(m, lam, n=300, seed=0):
    """N draws of θ̂ from the sampling distribution at training-set size m."""
    rng = random.Random(seed)
    return [fit2(*draw_regression(m, rng), lam) for _ in range(n)]


def moments(points):
    """Bias vector length and total variance (trace of the covariance)."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    bias = math.hypot(mx - THETA_STAR[0], my - THETA_STAR[1])
    var = sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in points) / n
    return bias, var


# --- world 2: a parabola fitted with straight lines ---------------------------

SIGMA = 0.3


def g(x):
    """The best possible predictor E[y | x]: a parabola on [0, 1]."""
    return 4 * (x - 0.5) ** 2


def best_line():
    """H*: the line minimising E[(g(x) − a − bx)²] for x uniform on [0, 1] (closed form)."""
    # Moments of x ~ U(0,1): E x = 1/2, E x² = 1/3; E g = 1/3, E[x g] = 1/6.
    eg, exg = 1 / 3, 1 / 6
    b = (exg - 0.5 * eg) / (1 / 3 - 0.25)
    return eg - b * 0.5, b


def l2(f1, f2, n=2000):
    """∫₀¹ (f1 − f2)² dx by the midpoint rule."""
    return sum((f1((i + 0.5) / n) - f2((i + 0.5) / n)) ** 2 for i in range(n)) / n


def fit_line(xs, ys):
    """Least-squares line through (xs, ys)."""
    m = len(xs)
    mx, my = sum(xs) / m, sum(ys) / m
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def decompositions(m=8, reps=20000, seed=1):
    """Both decompositions of E ε(ĥ) for lines fitted to m points from the parabola."""
    rng = random.Random(seed)
    a_star, b_star = best_line()
    fits = []
    for _ in range(reps):
        xs = [rng.random() for _ in range(m)]
        ys = [g(x) + rng.gauss(0, SIGMA) for x in xs]
        fits.append(fit_line(xs, ys))
    ma = sum(f[0] for f in fits) / reps
    mb = sum(f[1] for f in fits) / reps

    def h_star(x):
        """Best line in the class."""
        return a_star + b_star * x

    def h_bar(x):
        """The average fitted line, E[ĥ]."""
        return ma + mb * x

    # E ε(ĥ) - σ² = E ∫(ĥ - g)², and ∫(ĥ - h̄)² averaged is the variance. Both in closed form for lines:
    # ∫₀¹ (c + d x)² dx = c² + c d + d²/3.
    def sq(c, d):
        """∫₀¹ (c + d x)² dx."""
        return c * c + c * d + d * d / 3

    var = sum(sq(f[0] - ma, f[1] - mb) for f in fits) / reps
    excess = sum(l2(lambda x, f=f: f[0] + f[1] * x, g, 200) for f in fits[:2000]) / 2000
    bias2 = l2(h_bar, g)
    approx = l2(h_star, g)
    est_bias2 = sq(ma - a_star, mb - b_star)
    est = var + est_bias2  # E ∫(ĥ - h*)², since h̄ - h* is fixed
    return {
        "bayes": SIGMA**2,
        "approx": approx,
        "estimation": est,
        "bias2": bias2,
        "variance": var,
        "est_bias2": est_bias2,
        "excess_mc": excess,
        "h_star": (a_star, b_star),
        "h_bar": (ma, mb),
        "fits": fits,
    }


# --- world 3: thresholds on [0, 1] with label noise ---------------------------

T_TRUE = 0.4
FLIP = 0.15
THRESHOLDS = [(j + 0.5) / 10 for j in range(10)]  # the class H: k = 10 hypotheses


def true_error(t):
    """ε(h_t) for h_t(x) = 1{x ≥ t}: the flip rate, plus 0.7 on the strip between t and 0.4."""
    return FLIP + (1 - 2 * FLIP) * abs(t - T_TRUE)


def draw_labels(m, rng):
    """M examples (x, y) with y = 1{x ≥ 0.4}, flipped with probability 0.15."""
    out = []
    for _ in range(m):
        x = rng.random()
        y = int(x >= T_TRUE)
        if rng.random() < FLIP:
            y = 1 - y
        out.append((x, y))
    return out


def emp_error(t, data):
    """ε̂_S(h_t)."""
    return sum(int(x >= t) != y for x, y in data) / len(data)


def erm(data):
    """The empirical risk minimiser over the ten thresholds (ties to the smaller t)."""
    return min(THRESHOLDS, key=lambda t: (emp_error(t, data), t))


def binom_tail(m, phi, gamma):
    """Exact P(|φ̂ − φ| > γ) for φ̂ the mean of m Bernoulli(φ) draws."""
    total = 0.0
    for j in range(m + 1):
        if abs(j / m - phi) > gamma + 1e-12:
            total += math.exp(
                math.lgamma(m + 1)
                - math.lgamma(j + 1)
                - math.lgamma(m - j + 1)
                + j * math.log(phi)
                + (m - j) * math.log(1 - phi)
            )
    return total


def hoeffding(m, gamma):
    """Hoeffding's bound 2 exp(−2γ²m)."""
    return 2 * math.exp(-2 * gamma * gamma * m)


def sample_complexity(k, gamma, delta):
    """Smallest m with 2k exp(−2γ²m) ≤ δ."""
    return math.ceil(math.log(2 * k / delta) / (2 * gamma * gamma))


# --- shattering -------------------------------------------------------------


def separable(points, labels, n_dir=7200):
    """Whether some line puts every 1 strictly on one side and every 0 on the other."""
    if len(set(labels)) < 2:
        return True, None
    for i in range(n_dir):
        a = math.pi * i / n_dir
        for w in ((math.cos(a), math.sin(a)), (-math.cos(a), -math.sin(a))):
            pos = [w[0] * p[0] + w[1] * p[1] for p, y in zip(points, labels) if y]
            neg = [w[0] * p[0] + w[1] * p[1] for p, y in zip(points, labels) if not y]
            if max(neg) < min(pos) - 1e-9:
                return True, (w, (max(neg) + min(pos)) / 2)
    return False, None


TRIANGLE = [(0.2, 0.25), (0.8, 0.3), (0.45, 0.8)]
COLLINEAR = [(0.15, 0.2), (0.5, 0.5), (0.85, 0.8)]
SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
INNER = [(0.15, 0.2), (0.85, 0.2), (0.5, 0.85), (0.5, 0.42)]


def shatters(points):
    """Whether lines realise all 2^n labellings of the points; also the first failure."""
    for labels in itertools.product((0, 1), repeat=len(points)):
        ok, _ = separable(points, labels)
        if not ok:
            return False, labels
    return True, None


# --- figures -----------------------------------------------------------------


def fig_sampling():
    """Sampling distributions of θ̂ for least squares and ridge at m = 10 and m = 100."""
    xdom, ydom = (-1.0, 3.0), (-1.5, 2.5)
    s = ""
    cells = (
        (0, 0, 10, 0.0, "least squares, m = 10", "low bias, high variance"),
        (1, 0, 10, 10.0, "ridge λ = 10, m = 10", "higher bias, low variance"),
        (0, 1, 100, 0.0, "least squares, m = 100", "low bias, low variance"),
        (1, 1, 100, 10.0, "ridge λ = 10, m = 100", "bias shrinking with m"),
    )
    for col, row, m, lam, title, sub in cells:
        x0, y0 = 70 + 330 * col, 50 + 300 * row
        to_px = mapper(xdom, ydom, (x0, x0 + 220), (y0, y0 + 220))
        s += axes(to_px, xdom, ydom, [-1, 0, 1, 2, 3], [-1, 0, 1, 2], "θ₁", "θ₂")
        pts = sampling(m, lam, seed=10 * m + int(lam))
        for p in pts:
            if xdom[0] <= p[0] <= xdom[1] and ydom[0] <= p[1] <= ydom[1]:
                s += circle(*to_px(*p), 1.8, fill=TEAL if lam == 0 else INDIGO, stroke="none", op=0.55)
        cx, cy = to_px(*THETA_STAR)
        s += line(cx - 7, cy - 7, cx + 7, cy + 7, stroke=PURPLE, sw=2.4) + line(
            cx - 7, cy + 7, cx + 7, cy - 7, stroke=PURPLE, sw=2.4
        )
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        s += circle(*to_px(mx, my), 4.5, fill="#ffffff", stroke=PURPLE, sw=1.6)
        s += text(x0 + 110, y0 - 22, title, size=13, weight="600")
        s += text(x0 + 110, y0 - 6, sub, size=11, op=0.85)
    s += line(250, 612, 264, 626, stroke=PURPLE, sw=2.4) + line(250, 626, 264, 612, stroke=PURPLE, sw=2.4)
    s += text(272, 623, "true θ*", size=11, anchor="start")
    s += circle(380, 619, 4.5, fill="#ffffff", stroke=PURPLE, sw=1.6) + text(
        392, 623, "mean of θ̂", size=11, anchor="start"
    )
    return wrap(
        690,
        640,
        "figSampling",
        "Sampling distributions of two estimators",
        "Each dot is the estimate from one training set; 300 training sets per panel. Least squares is centred on the true parameter and spreads along the direction the correlated features cannot distinguish; ridge is pulled towards the origin and spreads far less. With ten times the data both clouds tighten and ridge's offset shrinks.",
        s,
    )


def fig_decomposition():
    """The class H, the best predictor g, the best-in-class h*, and the two error splits."""
    d = decompositions()
    s = ""
    # Left: the hypothesis-space picture, drawn from the regression world's geometry.
    s += f'  <ellipse cx="170" cy="190" rx="130" ry="95" fill="{TEAL}" fill-opacity="0.12" stroke="{TEAL}" stroke-width="1.4"/>\n'
    s += text(88, 132, "H: all straight lines", size=12, anchor="start", fill=TEAL)
    gx, gy = 340, 110
    hx, hy = 270, 129
    s += circle(gx, gy, 5, fill=PURPLE, stroke=PURPLE) + text(gx + 10, gy + 4, "g = E[y | x]", size=12, anchor="start")
    s += circle(hx, hy, 5, fill=INDIGO, stroke=INDIGO) + text(hx - 6, hy - 10, "h*", size=12, anchor="end")
    s += line(hx, hy, gx, gy, stroke=PURPLE, sw=1.2, dash="4 3")
    s += text(318, 150, "approximation", size=11, anchor="start", fill=PURPLE)
    rng = random.Random(4)
    for _ in range(40):
        r, a = 38 * math.sqrt(rng.random()), 2 * math.pi * rng.random()
        s += circle(hx - 40 + r * math.cos(a) * 1.3, hy + 55 + r * math.sin(a), 2.0, fill=INDIGO, stroke="none", op=0.5)
    s += text(170, 258, "ĥ from 40 training sets", size=11, fill=INDIGO)
    s += line(hx, hy, hx - 40, hy + 55, stroke=INDIGO, sw=1.2, dash="4 3")
    s += text(150, 175, "estimation", size=11, fill=INDIGO)

    # Right: stacked bars of the two decompositions, to scale.
    total = d["bayes"] + d["approx"] + d["estimation"]
    scale = 220 / total
    base = 290
    bars = (
        (
            440,
            (
                ("Bayes error σ²", d["bayes"], "#c9cdd6"),
                ("approximation", d["approx"], PURPLE),
                ("estimation", d["estimation"], INDIGO),
            ),
        ),
        (
            650,
            (("noise σ²", d["bayes"], "#c9cdd6"), ("bias²", d["bias2"], PURPLE), ("variance", d["variance"], INDIGO)),
        ),
    )
    for x0, parts in bars:
        y = base
        for label, val, colour in parts:
            hgt = val * scale
            s += rect(x0, y - hgt, 50, hgt, fill=colour, stroke="none", op=0.85 if colour != "#c9cdd6" else 1.0)
            s += text(x0 + 58, y - hgt / 2 + 4, f"{label} {val:.4f}", size=11, anchor="start")
            y -= hgt
    s += text(465, base + 20, "by class", size=12, op=0.85)
    s += text(675, base + 20, "by estimator", size=12, op=0.85)
    s += text(600, 40, f"expected test error {total:.4f}, split two ways", size=13, weight="600")
    return wrap(
        820,
        320,
        "figDecomposition",
        "Two decompositions of the expected error",
        "Left: the hypothesis class drawn as a region; the best predictor g lies outside it, the best line h* is the closest point inside, and lines fitted to finite training sets scatter around a point near h*. Right: for lines fitted to eight points from a parabola, the expected test error split into Bayes error, approximation and estimation error, and into noise, squared bias and variance; the totals agree.",
        s,
    )


def fig_hoeffding():
    """Exact deviation probabilities of a Bernoulli mean against Hoeffding's bound."""
    xdom, ydom = (0, 600), (-6.0, 0.5)
    to_px = mapper(xdom, ydom, (80, 640), (40, 250))
    s = axes(
        to_px, xdom, ydom, [0, 100, 200, 300, 400, 500, 600], [], "number of examples m", "probability (log scale)"
    )
    for e in range(-6, 1):
        py = to_px(0, e)[1]
        s += line(76, py, 80, py, op=0.7) + text(
            72, py + 4, "10" + str(e).translate(SUPERSCRIPT), size=11, anchor="end", op=0.8
        )
    gamma = 0.1
    ms = list(range(10, 601, 10))
    s += curve([(m, math.log10(min(1.0, hoeffding(m, gamma)))) for m in ms], to_px, xdom, ydom, stroke=PURPLE, sw=2.2)
    for phi, colour, dash in ((0.5, TEAL, None), (0.15, INDIGO, "6 4")):
        pts = [(m, math.log10(binom_tail(m, phi, gamma))) for m in ms if binom_tail(m, phi, gamma) > 0]
        s += curve(pts, to_px, xdom, ydom, stroke=colour, sw=2.0, dash=dash)
    s += text(
        to_px(250, 0)[0],
        to_px(250, math.log10(hoeffding(250, gamma)))[1] - 12,
        "Hoeffding: 2 exp(−2γ²m)",
        size=12,
        anchor="start",
        fill=PURPLE,
    )
    s += text(
        250,
        to_px(250, math.log10(binom_tail(250, 0.5, gamma)))[1] + 22,
        "exact, φ = 0.5",
        size=12,
        anchor="start",
        fill=TEAL,
    )
    s += text(
        150,
        to_px(150, math.log10(binom_tail(150, 0.15, gamma)))[1] + 20,
        "exact, φ = 0.15",
        size=12,
        anchor="start",
        fill=INDIGO,
    )
    s += text(360, 22, "P(|φ̂ − φ| > 0.1) for the mean of m coin flips", size=13, weight="600")
    return wrap(
        680,
        300,
        "figHoeffding",
        "Hoeffding's inequality against the exact tail",
        "The probability that the mean of m Bernoulli draws misses its parameter by more than 0.1, computed exactly from the binomial distribution for φ = 0.5 and φ = 0.15, lies below Hoeffding's bound at every m. The bound has the right exponential rate and is loosest when φ is far from one half.",
        s,
    )


def fig_uniform():
    """True and empirical error across the ten threshold hypotheses, for three training sets."""
    xdom, ydom = (0.0, 1.0), (0.0, 0.6)
    to_px = mapper(xdom, ydom, (80, 640), (50, 270))
    s = axes(
        to_px,
        xdom,
        ydom,
        [0.05, 0.25, 0.45, 0.65, 0.85],
        [0, 0.2, 0.4, 0.6],
        "threshold t of the hypothesis h_t(x) = 1{x ≥ t}",
        "error",
    )
    true_pts = [(t, true_error(t)) for t in THRESHOLDS]
    rng = random.Random(21)
    m = 60
    for i in range(3):
        data = draw_labels(m, rng)
        emp = [(t, emp_error(t, data)) for t in THRESHOLDS]
        s += polyline([to_px(*p) for p in emp], stroke=INDIGO, sw=1.3, op=0.7, dash="5 4")
        for p in emp:
            s += circle(*to_px(*p), 2.6, fill=INDIGO, stroke="none", op=0.7)
        if i == 0:
            th = erm(data)
            px, py = to_px(th, emp_error(th, data))
            s += circle(px, py, 7, stroke=PURPLE, sw=1.6)
            s += line(px, py, px, to_px(th, true_error(th))[1], stroke=PURPLE, sw=1.2)
            s += text(px - 10, py + 22, "ĥ: lowest training error", size=11, anchor="end", fill=PURPLE)
    s += polyline([to_px(*p) for p in true_pts], stroke=TEAL, sw=2.6)
    for p in true_pts:
        s += circle(*to_px(*p), 3.4, fill=TEAL, stroke=TEAL)
    hs = min(THRESHOLDS, key=true_error)
    px, py = to_px(hs, true_error(hs))
    s += text(px + 6, py + 28, "h*: lowest true error", size=11, anchor="start", fill=TEAL)
    s += line(px, py + 4, px + 4, py + 16, stroke=TEAL, sw=1.0)
    s += line(110, 330, 140, 330, stroke=TEAL, sw=2.6) + text(146, 334, "ε(h), true error", size=11, anchor="start")
    s += line(300, 330, 330, 330, stroke=INDIGO, sw=1.3, dash="5 4") + text(
        336, 334, f"ε̂_S(h) on three training sets of m = {m}", size=11, anchor="start"
    )
    return wrap(
        680,
        346,
        "figUniform",
        "Empirical error curves around the true error curve",
        "A finite class of ten threshold classifiers. The solid curve is each hypothesis's true error, lowest at t = 0.35 and 0.45 where the class comes closest to the real threshold 0.4. The dashed curves are training errors on three samples of sixty examples; they scatter around the true curve, and the hypothesis with the lowest training error on the first sample is circled, with its true error marked above it.",
        s,
    )


def fig_shatter():
    """Lines shatter three points; no line separates the XOR labelling of four."""
    s = ""
    w = 120
    for i, labels in enumerate(itertools.product((0, 1), repeat=3)):
        x0 = 20 + (i % 4) * (w + 20)
        y0 = 40 + (i // 4) * (w + 30)
        s += mini(TRIANGLE, labels, x0, y0, w)
    x0 = 610
    s += text(x0 + 70, 22, "four points: XOR", size=13, weight="600")
    s += mini(SQUARE, (1, 0, 1, 0), x0 + 10, 40, w)
    s += text(x0 + 70, 40 + w + 18, "no line separates", size=11, op=0.85)
    s += mini(INNER, (1, 1, 1, 0), x0 + 10, 40 + w + 30, w)
    s += text(x0 + 70, 40 + 2 * w + 48, "nor here", size=11, op=0.85)
    s += text(290, 22, "three points: all eight labellings separable", size=13, weight="600")
    return wrap(
        760,
        360,
        "figShatter",
        "Shattering with lines in the plane",
        "Left: three points in general position under each of their eight labellings, each with a line that separates the filled points from the open ones, so lines shatter this set. Right: the two possible shapes of four points in general position, a convex quadrilateral and a triangle with a point inside, each with a labelling no line can separate.",
        s,
    )


def mini(points, labels, x0, y0, w):
    """One small panel: points, labels, and a separating line when one exists."""
    s = rect(x0, y0, w, w, stroke="currentColor", sw=0.8, so=0.35)
    ok, sep = separable(points, labels)
    if ok and sep is not None:
        (a, b), c = sep
        # The line a·x + b·y = c, clipped to the unit square, drawn in panel coordinates.
        ends = []
        for t in (0.0, 1.0):
            if abs(b) > 1e-9:
                yv = (c - a * t) / b
                if 0 <= yv <= 1:
                    ends.append((t, yv))
            if abs(a) > 1e-9:
                xv = (c - b * t) / a
                if 0 <= xv <= 1:
                    ends.append((xv, t))
        if len(ends) >= 2:
            (u0, v0), (u1, v1) = ends[0], ends[-1]
            s += line(x0 + u0 * w, y0 + (1 - v0) * w, x0 + u1 * w, y0 + (1 - v1) * w, stroke=INDIGO, sw=1.6)
    elif not ok:
        s += rect(x0, y0, w, w, stroke=PURPLE, sw=1.6, so=0.8)
    for (u, v), y in zip(points, labels):
        px, py = x0 + u * w, y0 + (1 - v) * w
        s += circle(px, py, 6, fill=TEAL if y else "none", stroke=TEAL, sw=1.8)
    return s


FIGURES = {
    "diagram_sampling.svg": fig_sampling,
    "diagram_decomposition.svg": fig_decomposition,
    "diagram_hoeffding.svg": fig_hoeffding,
    "diagram_uniform.svg": fig_uniform,
    "diagram_shatter.svg": fig_shatter,
}

if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    # 1. Parameter view: bias and variance of least squares and ridge; ridge is consistent.
    for m in (10, 100):
        for lam in (0.0, 10.0):
            b, v = moments(sampling(m, lam, n=4000, seed=7 * m + int(lam)))
            print(f"m={m:>4} λ={lam:>4}: |bias| {b:.3f}, total variance {v:.4f}")
    for m in (10, 100, 1000, 10000):
        b, _ = moments(sampling(m, 10.0, n=1000 if m < 10000 else 200, seed=m))
        print(f"  ridge λ=10 bias at m={m}: {b:.4f}")
    b10 = moments(sampling(10, 10.0, n=2000, seed=1))[0]
    b1000 = moments(sampling(1000, 10.0, n=400, seed=2))[0]
    assert b1000 < b10 / 10

    # 2. Efficiency: the mean and the median of a Gaussian sample both shrink like 1/m, at different rates.
    rng = random.Random(3)
    for m in (11, 101):
        means, medians = [], []
        for _ in range(20000):
            xs = sorted(rng.gauss(0, 1) for _ in range(m))
            means.append(sum(xs) / m)
            medians.append(xs[m // 2])
        vm = sum(v * v for v in means) / len(means)
        vd = sum(v * v for v in medians) / len(medians)
        print(
            f"m={m}: m·Var(mean) {m * vm:.3f}, m·Var(median) {m * vd:.3f} (π/2 = {math.pi / 2:.3f}), ratio {vd / vm:.3f}"
        )

    # 3. Two decompositions of the same expected error, and the Pythagorean link between them.
    d = decompositions()
    print("decomposition (lines on a parabola, m = 8):")
    for key in ("bayes", "approx", "estimation", "bias2", "variance", "est_bias2"):
        print(f"  {key:>10}: {d[key]:.5f}")
    lhs = d["bayes"] + d["approx"] + d["estimation"]
    rhs = d["bayes"] + d["bias2"] + d["variance"]
    print(
        f"  total via approximation+estimation {lhs:.5f}; via bias²+variance {rhs:.5f}; Monte Carlo E∫(ĥ−g)² + σ² = {d['excess_mc'] + d['bayes']:.5f}"
    )
    assert abs(lhs - rhs) < 1e-6
    assert abs(d["bias2"] - (d["approx"] + d["est_bias2"])) < 1e-6  # h* - g ⟂ the class of lines
    assert abs((d["excess_mc"] + d["bayes"]) - lhs) / lhs < 0.03
    print(f"  h* = {d['h_star'][0]:.4f} + {d['h_star'][1]:.4f}x; E[ĥ] = {d['h_bar'][0]:.4f} + {d['h_bar'][1]:.4f}x")

    # 4. Hoeffding against the exact binomial tail.
    worst = 0.0
    for m in (10, 50, 100, 200, 400, 600):
        for phi in (0.5, 0.15, 0.02):
            ex, hb = binom_tail(m, phi, 0.1), hoeffding(m, 0.1)
            assert ex <= hb
            worst = max(worst, ex / hb)
    print(f"Hoeffding holds at every (m, φ) tried; largest exact/bound ratio {worst:.3f}")
    for m in (100, 300):
        print(
            f"  m={m}, γ=0.1: exact φ=0.5 {binom_tail(m, 0.5, 0.1):.2e}, φ=0.15 {binom_tail(m, 0.15, 0.1):.2e}, bound {hoeffding(m, 0.1):.2e}"
        )

    # 5. Uniform convergence over the ten thresholds: union bound against simulation.
    rng = random.Random(5)
    k = len(THRESHOLDS)
    for m, gamma in ((100, 0.15), (200, 0.1), (500, 0.07)):
        hits = 0
        reps = 4000
        for _ in range(reps):
            data = draw_labels(m, rng)
            if max(abs(emp_error(t, data) - true_error(t)) for t in THRESHOLDS) > gamma:
                hits += 1
        bound = 2 * k * math.exp(-2 * gamma * gamma * m)
        print(
            f"m={m}, γ={gamma}: P(some h off by > γ) simulated {hits / reps:.4f}; one-h bound {hoeffding(m, gamma):.4f}; union bound {bound:.4f}"
        )
        assert hits / reps <= bound

    # 6. Sample complexity m ≥ log(2k/δ) / (2gamma²).
    for kk in (10, 1000, 10**6):
        print(f"  k={kk:>8}, γ=0.05, δ=0.05: m ≥ {sample_complexity(kk, 0.05, 0.05):,}")
    for dd in (1, 10, 100):
        print(
            f"  64-bit argument, d={dd}: k = 2^(64d), m ≥ {math.ceil((64 * dd * math.log(2) + math.log(2 / 0.05)) / (2 * 0.05**2)):,}"
        )

    # 7. ε(ĥ) ≤ ε(h*) + 2gamma with probability ≥ 1 - δ, for ERM on the threshold class.
    rng = random.Random(6)
    delta = 0.05
    e_star = min(true_error(t) for t in THRESHOLDS)
    print(f"Bayes error {FLIP}; ε(h*) = {e_star:.3f}; approximation error {e_star - FLIP:.3f}")
    for m in (50, 200, 1000):
        gamma = math.sqrt(math.log(2 * k / delta) / (2 * m))
        excess, ok = [], 0
        for _ in range(3000):
            th = erm(draw_labels(m, rng))
            ex = true_error(th) - e_star
            excess.append(ex)
            ok += ex <= 2 * gamma + 1e-12
        print(
            f"  m={m:>5}: 2γ = {2 * gamma:.3f}; guarantee held in {ok / 3000:.4f} of runs; mean excess {sum(excess) / len(excess):.4f}, worst {max(excess):.3f}"
        )
        assert ok / 3000 >= 1 - delta

    # 8. Shattering: three points yes, three collinear no, both kinds of four no.
    ok3, _ = shatters(TRIANGLE)
    okc, failc = shatters(COLLINEAR)
    oks, fails = shatters(SQUARE)
    oki, faili = shatters(INNER)
    print(
        f"lines shatter the triangle: {ok3}; collinear: {okc} (fails {failc}); square: {oks} (fails {fails}); triangle+inner: {oki} (fails {faili})"
    )
    assert ok3 and not okc and not oks and not oki
    assert not separable(SQUARE, (1, 0, 1, 0))[0] and not separable(INNER, (1, 1, 1, 0))[0]
