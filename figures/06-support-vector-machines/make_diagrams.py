"""Figures for lecture 6: Laplace smoothing, event models, and margins.

Every figure is computed or drawn from the lecture's own example: the two
event models applied to the e-mail "drugs buy drugs now", logistic regression
given hand-picked quadratic features, the geometric margin of one example,
and the maximum-margin separator of a small separable set beside a separator
that only just separates it.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: Laplace's estimate against the
posterior mean under a uniform prior, integrated numerically; the smoothed
multinomial estimates summing to one; the multinomial model's log-odds
against a linear function of word counts; the functional margin's scaling
against the unchanged predictions; the geometric-margin formula against an
explicit projection onto the hyperplane; and the maximum-margin separator
found by a search over directions against the optimality (KKT) conditions of
min ½‖w‖², solved independently for the support vectors' multipliers.

Plain Python on purpose, as in lectures 1–5.

Run from the repository root.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import (
    FONT,
    INDIGO,
    MONO,
    PURPLE,
    TEAL,
    arrow,
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


# ---------------------------------------------------------------- algebra


def dot(a, b):
    """Inner product of two equal-length sequences."""
    return sum(x * y for x, y in zip(a, b))


def solve(a, b):
    """Solve a small dense linear system by Gaussian elimination with pivoting."""
    n = len(b)
    m = [[*row, bi] for row, bi in zip(a, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def sigmoid(z):
    """The logistic function, written to avoid overflow."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def logistic_fit(feats, ys, ridge=1e-3, iters=60):
    """Newton's method for L2-regularised logistic regression; feats include the intercept."""
    d = len(feats[0])
    th = [0.0] * d
    for _ in range(iters):
        g = [ridge * t for t in th]
        h = [[ridge * (i == j) for j in range(d)] for i in range(d)]
        for f, y in zip(feats, ys):
            p = sigmoid(dot(th, f))
            for i in range(d):
                g[i] += (p - y) * f[i]
                for j in range(d):
                    h[i][j] += p * (1 - p) * f[i] * f[j]
        step = solve(h, g)
        th = [t - s for t, s in zip(th, step)]
    return th


def contour(f, xdom, ydom, n=140):
    """Zero level set of f on a grid, by marching squares, chained into polylines."""
    xs = [xdom[0] + (xdom[1] - xdom[0]) * i / n for i in range(n + 1)]
    ys = [ydom[0] + (ydom[1] - ydom[0]) * j / n for j in range(n + 1)]
    v = [[f(x, y) for y in ys] for x in xs]

    def cut(i0, j0, i1, j1):
        a, b = v[i0][j0], v[i1][j1]
        t = a / (a - b)
        return (xs[i0] + t * (xs[i1] - xs[i0]), ys[j0] + t * (ys[j1] - ys[j0]))

    segs = []
    for i in range(n):
        for j in range(n):
            edges = (
                ((i, j), (i + 1, j)),
                ((i + 1, j), (i + 1, j + 1)),
                ((i, j + 1), (i + 1, j + 1)),
                ((i, j), (i, j + 1)),
            )
            pts = [cut(*p, *q) for p, q in edges if (v[p[0]][p[1]] >= 0) != (v[q[0]][q[1]] >= 0)]
            if len(pts) == 2:
                segs.append(tuple(pts))
            elif len(pts) == 4:
                segs += [(pts[0], pts[1]), (pts[2], pts[3])]
    key = lambda p: (round(p[0], 9), round(p[1], 9))  # noqa: E731
    ends = {}
    for k, (a, b) in enumerate(segs):
        ends.setdefault(key(a), []).append(k)
        ends.setdefault(key(b), []).append(k)
    used, runs = set(), []
    for k in range(len(segs)):
        if k in used:
            continue
        used.add(k)
        run = list(segs[k])
        for forward in (True, False):
            while True:
                tip = run[-1] if forward else run[0]
                nxt = [s for s in ends[key(tip)] if s not in used]
                if not nxt:
                    break
                used.add(nxt[0])
                a, b = segs[nxt[0]]
                new = b if key(a) == key(tip) else a
                if forward:
                    run.append(new)
                else:
                    run.insert(0, new)
        runs.append(run)
    return runs


# ---------------------------------------------------------------- Laplace smoothing


def laplace(counts):
    """Laplace-smoothed estimates of a k-way distribution from outcome counts."""
    n, k = sum(counts), len(counts)
    return [(c + 1) / (n + k) for c in counts]


def posterior_mean_uniform(wins, losses, steps=20000):
    """Posterior mean of a Bernoulli parameter under a uniform prior, by Simpson's rule."""
    h = 1.0 / steps

    def lik(p):
        return p**wins * (1 - p) ** losses

    def simpson(g):
        s = g(0) + g(1)
        s += sum((4 if i % 2 else 2) * g(i * h) for i in range(1, steps))
        return s * h / 3

    return simpson(lambda p: p * lik(p)) / simpson(lik)


# ---------------------------------------------------------------- event models

VOCAB = {"a": 1, "aardvark": 2, "buy": 800, "drugs": 1600, "now": 6200, "zymurgy": 10000}
EMAIL = ["drugs", "buy", "drugs", "now"]


def bernoulli_features(words):
    """Multivariate Bernoulli representation: the set of vocabulary indices present."""
    return sorted({VOCAB[w] for w in words})


def multinomial_features(words):
    """Multinomial representation: the sequence of vocabulary indices, one per word."""
    return [VOCAB[w] for w in words]


def fig_event_models():
    """The e-mail "drugs buy drugs now" under the two event models."""
    s = ""
    s += text(360, 26, "e-mail:  “drugs buy drugs now”", size=14, weight="600")
    s += text(
        24, 66, "multivariate Bernoulli: x ∈ {0, 1}¹⁰ ⁰⁰⁰, one entry per dictionary word", size=12, anchor="start"
    )
    cells = [
        ("a", 1),
        ("aardvark", 2),
        ("…", None),
        ("buy", 800),
        ("…", None),
        ("drugs", 1600),
        ("…", None),
        ("now", 6200),
        ("…", None),
        ("zymurgy", 10000),
    ]
    present = set(bernoulli_features(EMAIL))
    x0, w = 30, 66
    for k, (word, idx) in enumerate(cells):
        x = x0 + k * w
        if idx is None:
            s += text(x + w / 2, 104, "⋯", size=14, op=0.7)
            continue
        on = idx in present
        s += rect(
            x + 4,
            84,
            w - 8,
            30,
            fill=TEAL if on else "none",
            stroke=TEAL if on else "currentColor",
            op=0.25 if on else 1.0,
            so=0.8,
        )
        s += text(x + w / 2, 104, "1" if on else "0", size=14, font=MONO, weight="600" if on else None)
        s += text(x + w / 2, 130, word, size=11)
        s += text(x + w / 2, 144, str(idx), size=10, op=0.7, font=MONO)
    s += text(
        24,
        186,
        "multinomial event model: x ∈ {1, …, 10 000}ⁿ, one entry per word of the e-mail (here n = 4)",
        size=12,
        anchor="start",
    )
    seq = multinomial_features(EMAIL)
    for k, (word, idx) in enumerate(zip(EMAIL, seq)):
        x = 30 + k * 96
        s += rect(x + 4, 202, 88, 30, fill=INDIGO, stroke=INDIGO, op=0.18, so=0.8)
        s += text(x + 48, 222, str(idx), size=14, font=MONO, weight="600")
        s += text(x + 48, 248, f"x{'₁₂₃₄'[k]} = {word}", size=11)
    s += text(430, 216, "“drugs” counted twice;", size=12, anchor="start", fill=PURPLE)
    s += text(430, 234, "the Bernoulli vector records it once", size=12, anchor="start", fill=PURPLE)
    return wrap(
        720,
        268,
        "figEventModels",
        "The same e-mail under the multivariate Bernoulli and the multinomial event models",
        "Top: a row of cells indexed by dictionary word, with a 1 under buy, drugs and now and a 0 under a, aardvark and zymurgy. Bottom: four cells holding the dictionary indices 1600, 800, 1600, 6200, one per word of the e-mail, so the repeated word drugs appears twice.",
        s,
    )


def nb_multinomial_fit(docs, ys, v):
    """Multinomial naive Bayes with Laplace smoothing; docs are lists of word ids in range(v)."""
    phi_y = sum(ys) / len(ys)
    phi = {}
    for c in (0, 1):
        counts = [0] * v
        total = 0
        for d, y in zip(docs, ys):
            if y == c:
                for w in d:
                    counts[w] += 1
                total += len(d)
        phi[c] = [(counts[k] + 1) / (total + v) for k in range(v)]
    return phi_y, phi


def nb_multinomial_log_odds(doc, phi_y, phi):
    """Log-odds of spam for one document, by Bayes' rule on the generative model."""
    l1 = math.log(phi_y) + sum(math.log(phi[1][w]) for w in doc)
    l0 = math.log(1 - phi_y) + sum(math.log(phi[0][w]) for w in doc)
    return l1 - l0


# ---------------------------------------------------------------- non-linear boundary


def ring_data(seed, m=90):
    """Points labelled by whether they fall inside a tilted ellipse, with a few labels flipped."""
    rng = random.Random(seed)
    xs, ys = [], []
    while len(xs) < m:
        x = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        q = (x[0] - 0.3) ** 2 / 4.0 + (x[1] + 0.2) ** 2 / 1.6 + 0.35 * x[0] * x[1] / 2
        y = 1 if q < 1 else 0
        if rng.random() < 0.06:
            y = 1 - y
        xs.append(x)
        ys.append(y)
    return xs, ys


def quad_features(x):
    """Intercept, raw features and all degree-two monomials of a two-dimensional input."""
    return [1.0, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1]]


def fig_quad_boundary():
    """Logistic regression on raw features against the same model given quadratic features."""
    xs, ys = ring_data(3)
    lin = logistic_fit([[1.0, *x] for x in xs], ys)
    quad = logistic_fit([quad_features(x) for x in xs], ys)
    dom = (-3.0, 3.0)
    s = ""
    for k, (title, th, feats) in enumerate(
        (
            ("features x₁, x₂: a straight line", lin, lambda x: [1.0, *x]),
            ("adding x₁², x₂², x₁x₂: an ellipse", quad, quad_features),
        )
    ):
        ox = 60 + k * 360
        to_px = mapper(dom, dom, (ox, ox + 260), (40, 300))
        s += axes(to_px, dom, dom, [-2, 0, 2], [-2, 0, 2], "x₁", "x₂")
        for x, y in zip(xs, ys):
            px, py = to_px(*x)
            s += circle(px, py, 3.4, fill=TEAL, stroke=TEAL) if y else rect(px - 3, py - 3, 6, 6, stroke=PURPLE, sw=1.3)
        for run in contour(lambda a, b, th=th, feats=feats: dot(th, feats((a, b))), dom, dom):
            s += curve(run, to_px, dom, dom, stroke=INDIGO, sw=2.2)
        s += text(ox + 130, 22, title, size=13, weight="600")
    return wrap(
        720,
        350,
        "figQuadBoundary",
        "Logistic regression with raw and with quadratic features",
        "Two scatter plots of the same points, one class clustered inside a tilted ellipse and the other around it. Left: logistic regression on the two raw features draws a straight line that cannot separate them. Right: the same algorithm given the squares and the product of the features draws an ellipse around the inner class.",
        s,
    )


# ---------------------------------------------------------------- margins


def functional_margin(w, b, x, y):
    """Y (wᵀx + b): positive when the example is classified correctly."""
    return y * (dot(w, x) + b)


def geometric_margin(w, b, x, y):
    """The functional margin divided by ‖w‖: the signed distance to the hyperplane."""
    return functional_margin(w, b, x, y) / math.sqrt(dot(w, w))


def project(w, b, x):
    """The closest point to x on the hyperplane wᵀx + b = 0."""
    t = (dot(w, x) + b) / dot(w, w)
    return [xi - t * wi for xi, wi in zip(x, w)]


W_EX, B_EX, X_EX = (2.0, 1.0), -2.0, (2.0, 1.6)


def fig_margins():
    """The geometric margin of one positive example and the normal vector w."""
    dom = (-0.5, 3.0)
    to_px = mapper(dom, dom, (70, 380), (30, 340))
    s = axes(to_px, dom, dom, [0, 1, 2, 3], [0, 1, 2, 3], "x₁", "x₂")
    w, b = W_EX, B_EX
    grid = [dom[0] + (dom[1] - dom[0]) * i / 200 for i in range(201)]
    s += curve([(t, (-b - w[0] * t) / w[1]) for t in grid], to_px, dom, dom, stroke=INDIGO, sw=2.2)
    foot = project(w, b, X_EX)
    ax, ay = to_px(*X_EX)
    fx, fy = to_px(*foot)
    s += line(ax, ay, fx, fy, stroke=PURPLE, sw=1.6, dash="5 3")
    s += circle(ax, ay, 5, fill=TEAL, stroke=TEAL)
    s += text(ax + 10, ay - 8, "A = x⁽ⁱ⁾, y⁽ⁱ⁾ = +1", size=12, anchor="start")
    s += circle(fx, fy, 3.5, fill="currentColor", stroke="currentColor")
    s += text(fx - 10, fy + 16, "B", size=12, anchor="end")
    mx, my = (ax + fx) / 2, (ay + fy) / 2
    s += text(mx + 8, my + 18, "γ⁽ⁱ⁾", size=13, anchor="start", fill=PURPLE, style="italic")
    base = (0.15, 1.7)
    bx, by = to_px(*base)
    tip = (base[0] + 0.45 * w[0], base[1] + 0.45 * w[1])
    tx, ty = to_px(*tip)
    s += arrow(bx, by, tx, ty, sw=1.8, op=0.9, stroke=TEAL)
    s += text(tx + 6, ty - 4, "w", size=13, anchor="start", fill=TEAL, style="italic", weight="600")
    s += text(300, 70, "wᵀx + b > 0: predict +1", size=12, anchor="middle")
    s += text(150, 300, "wᵀx + b &lt; 0: predict −1", size=12, anchor="middle")
    s += text(130, 200, "wᵀx + b = 0", size=12, anchor="end", fill=INDIGO)
    s += text(560, 120, "γ⁽ⁱ⁾ = y⁽ⁱ⁾ (wᵀx⁽ⁱ⁾ + b) / ‖w‖", size=13, font=FONT)
    s += text(560, 146, "= γ̂⁽ⁱ⁾ / ‖w‖", size=13, font=FONT)
    s += text(560, 190, "w = (2, 1), b = −2", size=12, op=0.85)
    s += text(560, 210, "A = (2, 1.6):  γ̂ = 3.6,  γ ≈ 1.61", size=12, op=0.85)
    s += text(560, 230, "scale w, b by 10:  γ̂ = 36,  γ unchanged", size=12, op=0.85)
    return wrap(
        720,
        380,
        "figMargins",
        "The geometric margin of one training example",
        "A decision boundary w transpose x plus b equals zero, with w equal to (2, 1) and b equal to minus 2, drawn as a line. The vector w points away from the line at a right angle, into the half-plane where the classifier predicts plus one. A positive example A sits in that half-plane, and a dashed segment drops from it at a right angle to the point B on the line; the length of that segment is the geometric margin gamma i.",
        s,
    )


def separable_data(seed=11, m=24):
    """Two linearly separable clusters with a guaranteed gap between them."""
    rng = random.Random(seed)
    xs, ys = [], []
    while len(xs) < m:
        y = 1 if len(xs) % 2 else -1
        c = (2.4, 2.2) if y == 1 else (0.6, 0.5)
        x = (c[0] + rng.gauss(0, 0.6), c[1] + rng.gauss(0, 0.6))
        if y * (x[0] + 0.8 * x[1] - 2.9) < 0.55:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def margin_of_direction(u, xs, ys):
    """Best geometric margin and offset for unit direction u: half the gap between the classes."""
    lo = min(dot(u, x) for x, y in zip(xs, ys) if y == 1)
    hi = max(dot(u, x) for x, y in zip(xs, ys) if y == -1)
    return (lo - hi) / 2, -(lo + hi) / 2


def max_margin(xs, ys):
    """Maximum-margin separator in two dimensions, by a grid over directions then golden-section refinement."""

    def score(a):
        return margin_of_direction((math.cos(a), math.sin(a)), xs, ys)[0]

    a = max((2 * math.pi * k / 3600 for k in range(3600)), key=score)
    lo, hi = a - 2 * math.pi / 3600, a + 2 * math.pi / 3600
    g = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
        if score(c) > score(d):
            hi = d
        else:
            lo = c
    a = (lo + hi) / 2
    u = (math.cos(a), math.sin(a))
    gam, b = margin_of_direction(u, xs, ys)
    return [ui / gam for ui in u], b / gam, gam


def kkt_multipliers(w, b, xs, ys, tol=1e-6):
    """Solve w = Σ αᵢ yᵢ xᵢ, Σ αᵢ yᵢ = 0 over the support vectors, in least squares."""
    sv = [i for i, (x, y) in enumerate(zip(xs, ys)) if abs(functional_margin(w, b, x, y) - 1) < tol]
    rows = [[ys[i] * xs[i][0] for i in sv], [ys[i] * xs[i][1] for i in sv], [ys[i] for i in sv]]
    rhs = [w[0], w[1], 0.0]
    ata = [[dot([r[i] for r in rows], [r[j] for r in rows]) for j in range(len(sv))] for i in range(len(sv))]
    atb = [dot([r[i] for r in rows], rhs) for i in range(len(sv))]
    alpha = solve(ata, atb)
    resid = max(abs(dot(r, alpha) - t) for r, t in zip(rows, rhs))
    return sv, alpha, resid


def close_separator(xs, ys, turn=0.45, gap=0.1):
    """A separator turned away from the maximum-margin direction and pushed to within gap of the positive class."""
    w, _, _ = max_margin(xs, ys)
    a = math.atan2(w[1], w[0]) + turn
    u = (math.cos(a), math.sin(a))
    lo = min(dot(u, x) for x, y in zip(xs, ys) if y == 1)
    hi = max(dot(u, x) for x, y in zip(xs, ys) if y == -1)
    assert hi < lo - gap, "the turned direction no longer separates the data"
    return u, -(lo - gap)


def fig_optimal_margin():
    """A separator that only just separates the data, beside the maximum-margin separator."""
    xs, ys = separable_data()
    w, b, gam = max_margin(xs, ys)
    red_w, red_b = close_separator(xs, ys)
    dom = (-1.0, 4.0)
    to_px = mapper(dom, dom, (70, 400), (30, 360))
    s = axes(to_px, dom, dom, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], "x₁", "x₂")
    grid = [dom[0] + (dom[1] - dom[0]) * i / 300 for i in range(301)]

    def hyper(ww, bb, off):
        return [(t, (off - bb - ww[0] * t) / ww[1]) for t in grid]

    for off in (1, -1):
        s += curve(hyper(w, b, off), to_px, dom, dom, stroke=INDIGO, sw=1.2, dash="5 4", op=0.8)
    s += curve(hyper(w, b, 0), to_px, dom, dom, stroke=INDIGO, sw=2.4)
    s += curve(hyper(red_w, red_b, 0), to_px, dom, dom, stroke=PURPLE, sw=2.0, dash="2 3")
    for x, y in zip(xs, ys):
        px, py = to_px(*x)
        s += (
            circle(px, py, 4, fill=TEAL, stroke=TEAL)
            if y == 1
            else rect(px - 3.6, py - 3.6, 7.2, 7.2, stroke=PURPLE, sw=1.4)
        )
        if abs(functional_margin(w, b, x, y) - 1) < 1e-6:
            s += circle(px, py, 9, stroke="currentColor", sw=1.2, op=0.9)
    lx = 450
    s += line(lx, 100, lx + 28, 100, stroke=INDIGO, sw=2.4) + text(
        lx + 36, 104, "maximum-margin separator", size=12, anchor="start"
    )
    s += line(lx, 124, lx + 28, 124, stroke=INDIGO, sw=1.2, dash="5 4") + text(
        lx + 36, 128, "wᵀx + b = ±1", size=12, anchor="start"
    )
    s += line(lx, 148, lx + 28, 148, stroke=PURPLE, sw=2.0, dash="2 3") + text(
        lx + 36, 152, "also separates, barely", size=12, anchor="start"
    )
    s += circle(lx + 14, 172, 9, sw=1.2) + text(lx + 36, 176, "support vectors", size=12, anchor="start")
    red_gam = min(geometric_margin(red_w, red_b, x, y) for x, y in zip(xs, ys))
    s += text(lx, 220, f"geometric margin γ = {gam:.2f}", size=12, anchor="start", fill=INDIGO)
    s += text(lx, 240, f"versus γ = {red_gam:.2f} for the other line", size=12, anchor="start", fill=PURPLE)
    return wrap(
        720,
        400,
        "figOptimalMargin",
        "Two separating lines and the maximum-margin one",
        "A small two-class data set, separable by a straight line. A dotted line separates it but passes very close to a few points. A solid line, the maximum-margin separator, sits midway between the classes with two dashed lines parallel to it touching the closest points of each class; those closest points, the support vectors, are circled.",
        s,
    )


FIGURES = {
    "diagram_event_models.svg": fig_event_models,
    "diagram_quad_boundary.svg": fig_quad_boundary,
    "diagram_margins.svg": fig_margins,
    "diagram_optimal_margin.svg": fig_optimal_margin,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    # 1. Laplace smoothing: the football example and the rule of succession.
    mle = 0 / 4
    lap = laplace([0, 4])[0]
    pm = posterior_mean_uniform(0, 4)
    assert abs(lap - 1 / 6) < 1e-12 and abs(pm - lap) < 1e-9
    for wins, losses in ((3, 7), (10, 0), (0, 0)):
        assert abs(posterior_mean_uniform(wins, losses) - laplace([wins, losses])[0]) < 1e-9
    print(f"laplace: 0 wins, 4 losses -> MLE {mle:.3f}, Laplace {lap:.4f}; posterior mean under uniform prior {pm:.6f}")
    for counts in ([0, 0, 5, 2], [3] * 10, [0] * 9 + [1]):
        assert abs(sum(laplace(counts)) - 1) < 1e-12 and min(laplace(counts)) > 0
    print("laplace: k-way smoothed estimates are positive and sum to 1 (checked k = 4, 10)")

    # 2. Event models: the Bernoulli vector forgets the repeat; multinomial log-odds is linear in counts.
    assert bernoulli_features(EMAIL) == bernoulli_features(["drugs", "buy", "now"])
    assert multinomial_features(EMAIL) != multinomial_features(["drugs", "buy", "now"])
    print(
        f"event models: Bernoulli {bernoulli_features(EMAIL)} for both e-mails; multinomial {multinomial_features(EMAIL)}"
    )
    rng = random.Random(7)
    v = 12
    docs = [[rng.randrange(v) for _ in range(rng.randint(1, 9))] for _ in range(40)]
    ys = [rng.random() < 0.4 for _ in docs]
    phi_y, phi = nb_multinomial_fit(docs, ys, v)
    assert abs(sum(phi[0]) - 1) < 1e-12 and abs(sum(phi[1]) - 1) < 1e-12
    wts = [math.log(phi[1][k] / phi[0][k]) for k in range(v)]
    b0 = math.log(phi_y / (1 - phi_y))
    for _ in range(200):
        d = [rng.randrange(v) for _ in range(rng.randint(0, 15))]
        counts = [d.count(k) for k in range(v)]
        assert abs(nb_multinomial_log_odds(d, phi_y, phi) - (b0 + dot(wts, counts))) < 1e-9
    print(
        "event models: multinomial log-odds = b + Σ_k count_k · log(φ_k|1 / φ_k|0), on 200 random e-mails; smoothed φ sum to 1"
    )

    # 3. Logistic regression with and without quadratic features.
    xs, ys = ring_data(3)
    lin = logistic_fit([[1.0, *x] for x in xs], ys)
    quad = logistic_fit([quad_features(x) for x in xs], ys)
    e_lin = sum((dot(lin, [1.0, *x]) >= 0) != y for x, y in zip(xs, ys)) / len(xs)
    e_quad = sum((dot(quad, quad_features(x)) >= 0) != y for x, y in zip(xs, ys)) / len(xs)
    disc = quad[5] ** 2 - 4 * quad[3] * quad[4]
    assert disc < 0 and e_quad < e_lin
    print(
        f"quad: training error raw {e_lin:.3f}, quadratic {e_quad:.3f}; boundary discriminant {disc:.3f} < 0 (an ellipse)"
    )

    # 4. Functional margin scales with (w, b); predictions and geometric margin do not.
    for c in (2.0, 10.0, 1 / 17):
        wc, bc = [c * t for t in W_EX], c * B_EX
        assert abs(functional_margin(wc, bc, X_EX, 1) - c * functional_margin(W_EX, B_EX, X_EX, 1)) < 1e-12
        assert abs(geometric_margin(wc, bc, X_EX, 1) - geometric_margin(W_EX, B_EX, X_EX, 1)) < 1e-12
        for x in xs:
            assert (dot(wc, x) + bc >= 0) == (dot(W_EX, x) + B_EX >= 0)
    print(
        f"margins: A = {X_EX}: functional {functional_margin(W_EX, B_EX, X_EX, 1):.2f}, geometric {geometric_margin(W_EX, B_EX, X_EX, 1):.4f}; ×10 gives functional {functional_margin([20, 10], -20, X_EX, 1):.1f}"
    )

    # 5. The geometric-margin formula is the distance to the hyperplane.
    prng = random.Random(2)
    for _ in range(500):
        w = [prng.gauss(0, 2) for _ in range(3)]
        b = prng.gauss(0, 2)
        x = [prng.gauss(0, 3) for _ in range(3)]
        p = project(w, b, x)
        assert abs(dot(w, p) + b) < 1e-9
        dist = math.dist(x, p)
        assert abs(abs(geometric_margin(w, b, x, 1)) - dist) < 1e-9
    print("margins: |γ| = distance to the projection onto the hyperplane, 500 random cases in R³")

    # 6. Maximum margin: direction search against the KKT conditions of min ½‖w‖².
    xs, ys = separable_data()
    w, b, gam = max_margin(xs, ys)
    assert all(functional_margin(w, b, x, y) >= 1 - 1e-9 for x, y in zip(xs, ys))
    sv, alpha, resid = kkt_multipliers(w, b, xs, ys)
    assert len(sv) >= 2 and min(alpha) > 0 and resid < 1e-6
    assert abs(gam - 1 / math.hypot(*w)) < 1e-9
    red_w, red_b = close_separator(xs, ys)
    red = min(geometric_margin(red_w, red_b, x, y) for x, y in zip(xs, ys))
    assert 0 < red < gam
    print(
        f"optimal margin: γ = {gam:.4f} = 1/‖w‖; {len(sv)} support vectors with α = {[round(a, 4) for a in alpha]} > 0 "
        f"(KKT residual {resid:.1e}); barely-separating line γ = {red:.4f}"
    )
