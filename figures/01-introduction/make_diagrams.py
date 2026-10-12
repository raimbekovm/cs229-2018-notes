"""Figures for lecture 1: the board pictures of supervised and unsupervised learning.

Every figure is computed. The data are synthetic, drawn from a seeded generator
so the files are reproducible, and shaped after what Ng drew at the board: a
housing-price scatter with a line and a quadratic through it, the tumour-size
classification problem in its two views, the same problem with a second
feature, and a cloud of unlabelled points that k-means splits into two groups.

Each model in a figure is actually fitted — least squares by the normal
equations, logistic regression by Newton's method, k-means by Lloyd's
algorithm — so a curve or a boundary on the page is the output of the algorithm
the chapter names, not a line placed by eye. `__main__` prints every number the
prose quotes and asserts the identities the chapter states.

Plain Python on purpose: nothing here is large enough to need numpy, and the
three fitting routines are short enough to read.

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
    circle,
    line,
    mapper,
    polyline,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------- linear algebra


def solve(a, b):
    """Solve a small dense system a x = b by Gaussian elimination with pivoting."""
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in reversed(range(n)):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def least_squares(xs, ys, degree):
    """Fit a polynomial of the given degree by the normal equations (XᵀX)θ = Xᵀy."""
    rows = [[x**d for d in range(degree + 1)] for x in xs]
    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(degree + 1)] for i in range(degree + 1)]
    xty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(degree + 1)]
    return solve(xtx, xty)


def poly_eval(theta, x):
    """Evaluate θ₀ + θ₁x + θ₂x² + … at x."""
    return sum(t * x**d for d, t in enumerate(theta))


def rmse(theta, xs, ys):
    """Root-mean-square training error of a polynomial fit."""
    return math.sqrt(sum((poly_eval(theta, x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def sigmoid(z):
    """The logistic function."""
    return 1.0 / (1.0 + math.exp(-z))


def logistic_newton(feats, labels, iters=25):
    """Fit logistic regression h(x) = g(θᵀx) by Newton's method; feats include the 1."""
    n = len(feats[0])
    theta = [0.0] * n
    for _ in range(iters):
        grad = [0.0] * n
        hess = [[0.0] * n for _ in range(n)]
        for f, y in zip(feats, labels):
            h = sigmoid(sum(t * v for t, v in zip(theta, f)))
            for i in range(n):
                grad[i] += (y - h) * f[i]
                for j in range(n):
                    hess[i][j] += h * (1 - h) * f[i] * f[j]
        step = solve(hess, grad)
        theta = [t + s for t, s in zip(theta, step)]
    return theta


def kmeans(points, centres, iters=50):
    """Lloyd's algorithm: assign each point to its nearest centre, move centres to means."""
    assign = []
    for it in range(iters):
        new = [
            min(range(len(centres)), key=lambda k: (p[0] - centres[k][0]) ** 2 + (p[1] - centres[k][1]) ** 2)
            for p in points
        ]
        if new == assign:
            return centres, assign, it
        assign = new
        centres = [
            (
                sum(p[0] for p, a in zip(points, assign) if a == k) / assign.count(k),
                sum(p[1] for p, a in zip(points, assign) if a == k) / assign.count(k),
            )
            for k in range(len(centres))
        ]
    return centres, assign, iters


def gaussian_feature_inner(x, z, terms):
    """φ(x)·φ(z) with φ_k(x) = e^{-x²/2} x^k / √k!, truncated after `terms` coordinates."""
    return sum(
        math.exp(-(x * x) / 2)
        * x**k
        / math.sqrt(math.factorial(k))
        * math.exp(-(z * z) / 2)
        * z**k
        / math.sqrt(math.factorial(k))
        for k in range(terms)
    )


# ---------------------------------------------------------------- data


def housing_data():
    """Sizes in thousands of square feet and prices in thousands of dollars."""
    rng = random.Random(229)
    xs = sorted(rng.uniform(0.5, 2.5) for _ in range(26))
    ys = [40 + 310 * x - 72 * x * x + rng.gauss(0, 22) for x in xs]
    return xs, ys


def tumour_1d():
    """Tumour sizes (cm) and labels: 0 benign, 1 malignant, overlapping a little."""
    rng = random.Random(1)
    neg = [rng.uniform(0.4, 2.3) for _ in range(9)]
    pos = [rng.uniform(1.9, 4.2) for _ in range(9)]
    return [(x, 0) for x in neg] + [(x, 1) for x in pos]


def tumour_2d():
    """(size cm, age years, label) drawn from two overlapping Gaussian classes."""
    rng = random.Random(7)
    pts = [(rng.gauss(1.7, 0.55), rng.gauss(42, 8), 0) for _ in range(22)]
    pts += [(rng.gauss(3.0, 0.6), rng.gauss(60, 8), 1) for _ in range(22)]
    return pts


def blobs():
    """Unlabelled points in two groups, the board's 'two clusters'."""
    rng = random.Random(11)
    pts = [(rng.gauss(1.3, 0.38), rng.gauss(1.4, 0.38)) for _ in range(20)]
    pts += [(rng.gauss(3.1, 0.38), rng.gauss(3.0, 0.38)) for _ in range(20)]
    return pts


# ---------------------------------------------------------------- drawing helpers


def axes(to_px, xdom, ydom, xticks, yticks, xlabel, ylabel, fmt="{:g}"):
    """Axis lines, ticks and labels for a mapped box."""
    x0, y0 = to_px(xdom[0], ydom[0])
    x1, _ = to_px(xdom[1], ydom[0])
    _, y1 = to_px(xdom[0], ydom[1])
    s = line(x0, y0, x1, y0, op=0.7) + line(x0, y0, x0, y1, op=0.7)
    for t in xticks:
        px, _ = to_px(t, ydom[0])
        s += line(px, y0, px, y0 + 4, op=0.7) + text(px, y0 + 17, fmt.format(t), size=11, op=0.8)
    for t in yticks:
        _, py = to_px(xdom[0], t)
        s += line(x0 - 4, py, x0, py, op=0.7) + text(x0 - 8, py + 4, fmt.format(t), size=11, anchor="end", op=0.8)
    s += text((x0 + x1) / 2, y0 + 36, xlabel, size=12)
    s += f'  <text x="{x0 - 40:.1f}" y="{(y0 + y1) / 2:.1f}" font-size="12" fill="currentColor" text-anchor="middle" transform="rotate(-90 {x0 - 40:.1f} {(y0 + y1) / 2:.1f})">{ylabel}</text>\n'
    return s


def cross(cx, cy, r=4.5, stroke=PURPLE, sw=1.8):
    """A positive example, drawn as an ×."""
    return line(cx - r, cy - r, cx + r, cy + r, stroke=stroke, sw=sw) + line(
        cx - r, cy + r, cx + r, cy - r, stroke=stroke, sw=sw
    )


def nought(cx, cy, r=4.5, stroke=TEAL):
    """A negative example, drawn as an O."""
    return circle(cx, cy, r, stroke=stroke, sw=1.6)


def legend_row(x, y, items):
    """A horizontal legend of (glyph-drawer, label) pairs."""
    s = ""
    for draw, label in items:
        s += draw(x + 6, y - 4) + text(x + 18, y, label, size=11, anchor="start")
        x += 26 + 6.2 * len(label)
    return s


# ---------------------------------------------------------------- figures


def fig_regression():
    """Housing prices with a straight line and a quadratic fitted by least squares."""
    xs, ys = housing_data()
    lin = least_squares(xs, ys, 1)
    quad = least_squares(xs, ys, 2)
    xdom, ydom = (0.0, 3.0), (0.0, 500.0)
    to_px = mapper(xdom, ydom, (80, 610), (30, 300))
    s = axes(
        to_px,
        xdom,
        ydom,
        [0, 0.5, 1, 1.5, 2, 2.5, 3],
        [0, 100, 200, 300, 400, 500],
        "size of house (1000 ft²)",
        "price ($1000s)",
    )
    for x, y in zip(xs, ys):
        s += circle(*to_px(x, y), 3.4, fill="currentColor", stroke="none", op=0.55)
    grid = [0.3 + i * 0.025 for i in range(97)]
    s += polyline([to_px(x, poly_eval(lin, x)) for x in grid], stroke=INDIGO, sw=2)
    s += polyline([to_px(x, poly_eval(quad, x)) for x in grid], stroke=PURPLE, sw=2, dash="7 4")
    q = 1.25
    qx, qy0 = to_px(q, 0)
    for theta, col in ((lin, INDIGO), (quad, PURPLE)):
        _, py = to_px(q, poly_eval(theta, q))
        s += line(qx, qy0, qx, py, stroke=col, sw=1, op=0.6, dash="2 3")
        s += line(to_px(0, 0)[0], py, qx, py, stroke=col, sw=1, op=0.6, dash="2 3")
    s += text(qx + 4, qy0 - 8, "your house", size=11, anchor="start", op=0.85)
    s += legend_row(
        100,
        46,
        [
            (lambda x, y: line(x - 8, y, x + 8, y, stroke=INDIGO, sw=2), "straight line"),
            (lambda x, y: line(x - 8, y, x + 8, y, stroke=PURPLE, sw=2, dash="7 4"), "quadratic"),
        ],
    )
    return wrap(
        660,
        350,
        "figRegression",
        "Housing prices: regression",
        "Scatter of house size against price with a fitted straight line and a fitted quadratic, and the prices both read off at 1250 square feet.",
        s,
    )


def fig_tumour_1d():
    """The tumour problem as a plot of y against size, then pushed onto one line."""
    data = tumour_1d()
    xdom = (0.0, 4.5)
    to_px = mapper(xdom, (0.0, 1.0), (80, 610), (60, 170))
    s = axes(to_px, xdom, (0.0, 1.0), [0, 1, 2, 3, 4], [0, 1], "tumour size (cm)", "malignant?")
    for x, y in data:
        px, py = to_px(x, y)
        s += cross(px, py) if y else nought(px, py)
    # The second view: every example dropped onto a single axis, label kept as the glyph.
    ly = 268
    lx0, _ = to_px(0, 0)
    lx1, _ = to_px(4.5, 0)
    s += line(lx0, ly, lx1, ly, op=0.7)
    for t in [0, 1, 2, 3, 4]:
        px, _ = to_px(t, 0)
        s += line(px, ly, px, ly + 4, op=0.7) + text(px, ly + 17, f"{t}", size=11, op=0.8)
    for x, y in data:
        px, _ = to_px(x, 0)
        s += cross(px, ly - 10) if y else nought(px, ly - 10)
    s += text(lx0, ly - 30, "the same examples on one axis", size=12, anchor="start", op=0.85)
    s += legend_row(
        400, 30, [(lambda x, y: nought(x, y), "benign (y = 0)"), (lambda x, y: cross(x, y), "malignant (y = 1)")]
    )
    return wrap(
        660,
        300,
        "figTumour1d",
        "Tumour classification with one feature",
        "Tumour size on the horizontal axis with labels zero or one plotted as height, and below it the same examples placed on a single line, noughts for benign and crosses for malignant.",
        s,
    )


def fig_tumour_2d():
    """Two features, and the straight boundary logistic regression fits between the classes."""
    pts = tumour_2d()
    # Standardise before Newton so the Hessian is well conditioned, then map the
    # boundary back to raw units for drawing.
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pts) / len(pts))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pts) / len(pts))
    feats = [[1.0, (p[0] - mx) / sx, (p[1] - my) / sy] for p in pts]
    theta = logistic_newton(feats, [p[2] for p in pts])
    xdom, ydom = (0.0, 5.0), (20.0, 80.0)
    to_px = mapper(xdom, ydom, (80, 610), (30, 300))
    s = axes(to_px, xdom, ydom, [0, 1, 2, 3, 4, 5], [20, 40, 60, 80], "tumour size (cm)", "age (years)")

    def age_on_boundary(size):
        """Solve θ₀ + θ₁ẑ₁ + θ₂ẑ₂ = 0 for age at a given size."""
        return my - sy * (theta[0] + theta[1] * (size - mx) / sx) / theta[2]

    seg = [(x, age_on_boundary(x)) for x in [xdom[0] + i * 0.05 for i in range(101)]]
    seg = [to_px(x, a) for x, a in seg if ydom[0] <= a <= ydom[1]]
    s += polyline(seg, stroke=INDIGO, sw=2)
    for x, a, y in pts:
        px, py = to_px(x, a)
        s += cross(px, py) if y else nought(px, py)
    s += legend_row(400, 52, [(lambda x, y: nought(x, y), "benign"), (lambda x, y: cross(x, y), "malignant")])
    acc = sum((sigmoid(sum(t * v for t, v in zip(theta, f))) > 0.5) == bool(p[2]) for f, p in zip(feats, pts)) / len(
        pts
    )
    return wrap(
        660,
        350,
        "figTumour2d",
        "Tumour classification with two features",
        "Tumour size against patient age, benign and malignant examples, and the straight decision boundary logistic regression fits between them.",
        s,
    ), acc


def fig_clusters():
    """Unlabelled points, then the two groups k-means finds in them."""
    pts = blobs()
    centres, assign, iters = kmeans(pts, [pts[0], pts[1]])
    s = ""
    for panel, (left, title) in enumerate(((60, "the input: x only, no y"), (370, "after k-means, k = 2"))):
        to_px = mapper((0.0, 4.5), (0.0, 4.5), (left, left + 240), (40, 280))
        s += axes(to_px, (0.0, 4.5), (0.0, 4.5), [0, 2, 4], [0, 2, 4], "x₁", "x₂")
        s += text(left + 120, 24, title, size=12)
        for p, a in zip(pts, assign):
            if panel == 0:
                s += circle(*to_px(*p), 3.6, fill="currentColor", stroke="none", op=0.55)
            else:
                s += circle(*to_px(*p), 3.6, fill=(INDIGO, TEAL)[a], stroke="none", op=0.85)
        if panel == 1:
            for c in centres:
                cx, cy = to_px(*c)
                s += cross(cx, cy, r=6, stroke=PURPLE, sw=2.4)
    return wrap(
        660,
        330,
        "figClusters",
        "Unsupervised learning: clustering",
        "On the left a cloud of unlabelled points; on the right the same points coloured by the cluster k-means assigns them, with each cluster centre marked.",
        s,
    ), iters


def misclassified_2d():
    """The training points the fitted logistic boundary puts on the wrong side."""
    pts = tumour_2d()
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pts) / len(pts))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pts) / len(pts))
    feats = [[1.0, (p[0] - mx) / sx, (p[1] - my) / sy] for p in pts]
    theta = logistic_newton(feats, [p[2] for p in pts])
    return [p for f, p in zip(feats, pts) if (sigmoid(sum(t * v for t, v in zip(theta, f))) > 0.5) != bool(p[2])]


FIGURES = {
    "diagram_regression.svg": fig_regression,
    "diagram_tumour_1d.svg": fig_tumour_1d,
    "diagram_tumour_2d.svg": fig_tumour_2d,
    "diagram_clusters.svg": fig_clusters,
}


if __name__ == "__main__":
    extras = {}
    for name, fn in FIGURES.items():
        out = fn()
        if isinstance(out, tuple):
            out, extras[name] = out
        (OUT / name).write_text(out)
        print("wrote", name)

    xs, ys = housing_data()
    lin, quad = least_squares(xs, ys, 1), least_squares(xs, ys, 2)
    # The quadratic family contains every line, so its least-squares fit can never
    # have the larger training error — the point the prose makes about "fits better".
    assert rmse(quad, xs, ys) <= rmse(lin, xs, ys) + 1e-9
    print(f"housing: line predicts ${poly_eval(lin, 1.25):.0f}k, quadratic ${poly_eval(quad, 1.25):.0f}k at 1250 ft²")
    print(f"housing: training RMSE line {rmse(lin, xs, ys):.1f}, quadratic {rmse(quad, xs, ys):.1f}")
    # The quadratic is concave, so it has a peak; past it the model prices a bigger
    # house lower. The prose uses this as the reason training error alone cannot
    # choose between the two models.
    peak = -quad[1] / (2 * quad[2])
    assert quad[2] < 0 and 2.0 < peak < 3.0
    print(f"housing: quadratic peaks at {peak * 1000:.0f} ft2 (${poly_eval(quad, peak):.0f}k) and falls after it")
    print(f"tumour 2-D: logistic regression training accuracy {extras['diagram_tumour_2d.svg']:.2f}")
    # The caption names the class of the one example on the wrong side of the line.
    wrong = misclassified_2d()
    assert [p[2] for p in wrong] == [1]
    print(f"tumour 2-D: misclassified {wrong} (label 1 = malignant)")
    print(f"k-means: converged after {extras['diagram_clusters.svg']} reassignment rounds")

    # The kernel identity behind "an infinite list of features": with
    # φ_k(x) = e^{-x²/2} x^k / √k!, the infinite inner product is exp(-(x-z)²/2).
    x, z = 0.7, -0.4
    exact = math.exp(-((x - z) ** 2) / 2)
    for terms in (1, 2, 5, 10, 20):
        approx = gaussian_feature_inner(x, z, terms)
        print(f"kernel: {terms:2d} features -> {approx:.12f} (exact {exact:.12f}, gap {abs(approx - exact):.1e})")
    assert abs(gaussian_feature_inner(x, z, 30) - exact) < 1e-12
