"""Figures for lecture 5: Gaussian discriminant analysis and naive Bayes.

Every figure is computed: contours of two-dimensional Gaussian densities at
fixed heights, GDA and its separate-covariance variant fitted to synthetic
classes, the GDA posterior along one feature, learning curves for GDA against
logistic regression on data that do and do not satisfy GDA's assumptions, and
the naive Bayes graphical model.

`__main__` prints every number the prose quotes and asserts the claims the
chapter makes by two routes that must agree: GDA's closed-form estimates
against perturbations of the joint likelihood, its posterior computed by
Bayes' rule against the logistic function of the implied θ, the same
logistic form for two Poisson classes, the posterior's crossing point against
the prior, parameter counts, and naive Bayes' log-odds against a linear
function — together with the two places where naive Bayes and GDA part from
what they look like: GDA's θ is not the maximiser of the conditional
likelihood, and naive Bayes counts a duplicated word twice.

Plain Python on purpose, as in lectures 1–4.

Run from the repository root.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import (
    GREEN,
    INDIGO,
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


# ---------------------------------------------------------------- 2x2 algebra


def det2(s):
    """Determinant of a 2x2 matrix."""
    return s[0][0] * s[1][1] - s[0][1] * s[1][0]


def inv2(s):
    """Inverse of a 2x2 matrix."""
    d = det2(s)
    return [[s[1][1] / d, -s[0][1] / d], [-s[1][0] / d, s[0][0] / d]]


def mv(a, v):
    """Matrix-vector product for 2x2."""
    return [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]]


def dot(a, b):
    """Inner product."""
    return sum(x * y for x, y in zip(a, b))


def chol2(s):
    """Lower Cholesky factor of a 2x2 positive-definite matrix."""
    l11 = math.sqrt(s[0][0])
    l21 = s[1][0] / l11
    return [[l11, 0.0], [l21, math.sqrt(s[1][1] - l21 * l21)]]


def eig2(s):
    """Eigenvalues (descending) and the unit eigenvector of the larger, for a symmetric 2x2."""
    a, b, d = s[0][0], s[0][1], s[1][1]
    tr, disc = a + d, math.sqrt((a - d) ** 2 + 4 * b * b)
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    v = [b, l1 - a] if abs(b) > 1e-12 else ([1.0, 0.0] if a >= d else [0.0, 1.0])
    n = math.hypot(*v)
    return l1, l2, [v[0] / n, v[1] / n]


def sigmoid(z):
    """The logistic function, overflow-safe."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def gauss2(x, mu, s):
    """Bivariate normal density."""
    d = [x[0] - mu[0], x[1] - mu[1]]
    return math.exp(-0.5 * dot(d, mv(inv2(s), d))) / (2 * math.pi * math.sqrt(det2(s)))


def ellipse(mu, s, r, n=160):
    """Points on the Mahalanobis-radius-r contour of N(mu, s)."""
    lo = chol2(s)
    return [
        (
            mu[0] + r * (lo[0][0] * math.cos(t)),
            mu[1] + r * (lo[1][0] * math.cos(t) + lo[1][1] * math.sin(t)),
        )
        for t in [2 * math.pi * i / n for i in range(n + 1)]
    ]


def level_radius(p, s):
    """Mahalanobis radius at which N(0, s) has density p, or None if p exceeds the peak."""
    r2 = -2 * math.log(p * 2 * math.pi * math.sqrt(det2(s)))
    return math.sqrt(r2) if r2 > 0 else None


# ---------------------------------------------------------------- the Gaussian gallery

I2 = [[1.0, 0.0], [0.0, 1.0]]
GALLERY = [
    ("Σ = I", I2),
    ("Σ = 0.6 I", [[0.6, 0.0], [0.0, 0.6]]),
    ("Σ = 2 I", [[2.0, 0.0], [0.0, 2.0]]),
    ("off-diagonal 0.5", [[1.0, 0.5], [0.5, 1.0]]),
    ("off-diagonal 0.8", [[1.0, 0.8], [0.8, 1.0]]),
    ("off-diagonal −0.8", [[1.0, -0.8], [-0.8, 1.0]]),
]
LEVELS = [0.02, 0.06, 0.12, 0.2]
LEVEL_COLS = [GREEN, TEAL, INDIGO, PURPLE]


def fig_gaussians():
    """Contours of six bivariate Gaussians at the same four density heights."""
    s = ""
    dom = (-3.0, 3.0)
    for k, (title, sig) in enumerate(GALLERY):
        col, row = k % 3, k // 3
        x0, y0 = 50 + col * 230, 40 + row * 235
        to_px = mapper(dom, dom, (x0, x0 + 170), (y0, y0 + 170))
        s += axes(to_px, dom, dom, [-3, 0, 3], [-3, 0, 3], "z₁", "z₂")
        for p, c in zip(LEVELS, LEVEL_COLS):
            r = level_radius(p, sig)
            if r:
                s += curve(ellipse((0, 0), sig, r), to_px, dom, dom, stroke=c, sw=1.6)
        _, _, v = eig2(sig)
        if abs(sig[0][1]) > 0:
            a = 2.6
            s += line(*to_px(-a * v[0], -a * v[1]), *to_px(a * v[0], a * v[1]), sw=1, op=0.5, dash="3 3")
        peak = 1 / (2 * math.pi * math.sqrt(det2(sig)))
        s += text(x0 + 85, y0 - 14, f"{title}  (peak {peak:.3f})", size=12, weight="600")
    ly = 498
    s += text(70, ly + 4, "density levels:", size=11, anchor="start")
    for i, (p, c) in enumerate(zip(LEVELS, LEVEL_COLS)):
        x = 175 + i * 80
        s += line(x, ly, x + 22, ly, stroke=c, sw=2) + text(x + 27, ly + 4, f"{p:g}", size=11, anchor="start")
    s += line(500, ly, 522, ly, sw=1, op=0.5, dash="3 3") + text(
        527, ly + 4, "leading eigenvector", size=11, anchor="start"
    )
    return wrap(
        720,
        515,
        "figGaussians",
        "Contours of two-dimensional Gaussian densities",
        "Six panels of contour lines, all drawn at the same four density heights. With covariance the identity the contours are circles; at 0.6 times the identity they are smaller and the highest level appears, because the peak is taller; at twice the identity they are larger and only the lowest level exists. With positive off-diagonal entries the circles become ellipses tilted along the line z2 equals z1, narrower as the entry grows to 0.8; with a negative entry they tilt the other way. A dashed line marks the leading eigenvector, which is the long axis of each ellipse.",
        s,
    )


# ---------------------------------------------------------------- GDA


def gda_fit(xs, ys):
    """Maximum-likelihood GDA with a shared covariance: φ, μ0, μ1, Σ (divided by m)."""
    m = len(xs)
    phi = sum(ys) / m
    mus = []
    for c in (0, 1):
        pts = [x for x, y in zip(xs, ys) if y == c]
        mus.append([sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)])
    s = [[0.0, 0.0], [0.0, 0.0]]
    for x, y in zip(xs, ys):
        d = [x[0] - mus[y][0], x[1] - mus[y][1]]
        for i in range(2):
            for j in range(2):
                s[i][j] += d[i] * d[j] / m
    return phi, mus[0], mus[1], s


def gda_theta(phi, mu0, mu1, s):
    """The logistic parameters GDA implies: P(y=1|x) = g(θᵀx + θ0)."""
    si = inv2(s)
    th = mv(si, [mu1[0] - mu0[0], mu1[1] - mu0[1]])
    th0 = -0.5 * (dot(mu1, mv(si, mu1)) - dot(mu0, mv(si, mu0))) + math.log(phi / (1 - phi))
    return th0, th


def gda_posterior(x, phi, mu0, mu1, s0, s1=None):
    """P(y = 1 | x) by Bayes' rule, with separate covariances if s1 is given."""
    s1 = s1 or s0
    a = gauss2(x, mu1, s1) * phi
    b = gauss2(x, mu0, s0) * (1 - phi)
    return a / (a + b)


def gda_joint_ll(params, xs, ys):
    """Joint log-likelihood Σ log p(x, y) under GDA."""
    phi, mu0, mu1, s = params
    return sum(math.log(gauss2(x, mu1 if y else mu0, s)) + math.log(phi if y else 1 - phi) for x, y in zip(xs, ys))


def gda_params(n, shared):
    """Parameter count of GDA with n features: φ, two means, one or two covariances."""
    cov = n * (n + 1) // 2
    return 1 + 2 * n + (cov if shared else 2 * cov)


# ---------------------------------------------------------------- logistic regression


def lr_fit(xs, ys, ridge=0.0, iters=50):
    """Logistic regression with features (1, x1, x2) by Newton's method; ridge on the weights only."""
    th = [0.0, 0.0, 0.0]
    for _ in range(iters):
        g = [0.0] * 3
        h = [[0.0] * 3 for _ in range(3)]
        for x, y in zip(xs, ys):
            f = [1.0, x[0], x[1]]
            p = sigmoid(dot(th, f))
            for i in range(3):
                g[i] += (y - p) * f[i]
                for j in range(3):
                    h[i][j] -= p * (1 - p) * f[i] * f[j]
        for i in (1, 2):
            g[i] -= ridge * th[i]
            h[i][i] -= ridge
        step = solve3(h, g)
        th = [t - d for t, d in zip(th, step)]
        if max(abs(d) for d in step) < 1e-10:
            break
    return th


def solve3(a, b):
    """Solve a 3x3 system by Gaussian elimination with partial pivoting."""
    m = [[*row, v] for row, v in zip(a, b)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, 3):
            f = m[r][c] / m[c][c]
            m[r] = [u - f * w for u, w in zip(m[r], m[c])]
    x = [0.0] * 3
    for r in (2, 1, 0):
        x[r] = (m[r][3] - sum(m[r][k] * x[k] for k in range(r + 1, 3))) / m[r][r]
    return x


def cond_ll(th, xs, ys):
    """Conditional log-likelihood Σ log p(y | x) for a logistic model with features (1, x1, x2)."""
    out = 0.0
    for x, y in zip(xs, ys):
        p = sigmoid(th[0] + th[1] * x[0] + th[2] * x[1])
        out += math.log(p if y else 1 - p)
    return out


# ---------------------------------------------------------------- data

G_MU0, G_MU1 = (0.0, 0.0), (1.6, 1.2)
G_SIG = [[1.0, 0.45], [0.45, 0.8]]
P_LAM0, P_LAM1 = (1.0, 4.0), (3.5, 6.0)
Q_SIG0 = [[1.3, 0.0], [0.0, 0.35]]
Q_SIG1 = [[0.35, 0.0], [0.0, 1.4]]
Q_MU0, Q_MU1 = (-0.3, -0.6), (1.3, 1.0)


def mvn_sample(rng, mu, s):
    """One draw from N(mu, s)."""
    lo = chol2(s)
    z = [rng.gauss(0, 1), rng.gauss(0, 1)]
    return (mu[0] + lo[0][0] * z[0], mu[1] + lo[1][0] * z[0] + lo[1][1] * z[1])


def poisson(rng, lam):
    """One Poisson draw by inversion."""
    k, p, c = 0, rng.random(), math.exp(-lam)
    acc = c
    while p > acc:
        k += 1
        c *= lam / k
        acc += c
    return k


def sample(kind, m, rng, phi=0.5):
    """M labelled points from the Gaussian, Poisson or separate-covariance world, with ≥ 2 of each class."""
    while True:
        ys = [1 if rng.random() < phi else 0 for _ in range(m)]
        if min(sum(ys), m - sum(ys)) >= 2:
            break
    xs = []
    for y in ys:
        if kind == "gauss":
            xs.append(mvn_sample(rng, G_MU1 if y else G_MU0, G_SIG))
        elif kind == "quad":
            xs.append(mvn_sample(rng, Q_MU1 if y else Q_MU0, Q_SIG1 if y else Q_SIG0))
        else:
            lam = P_LAM1 if y else P_LAM0
            xs.append((float(poisson(rng, lam[0])), float(poisson(rng, lam[1]))))
    return xs, ys


def bayes_rule(kind, x):
    """The true posterior P(y = 1 | x) in each world (prior 0.5)."""
    if kind == "gauss":
        return gda_posterior(x, 0.5, G_MU0, G_MU1, G_SIG)
    lo = sum(xi * math.log(l1 / l0) - (l1 - l0) for xi, l0, l1 in zip(x, P_LAM0, P_LAM1))
    return sigmoid(lo)


def boundary_line(th0, th, dom):
    """Dense points on th0 + th1 x1 + th2 x2 = 0 across the x1 domain."""
    return [(u, -(th0 + th[0] * u) / th[1]) for u in [dom[0] + (dom[1] - dom[0]) * i / 300 for i in range(301)]]


def fig_gda_fit():
    """GDA with a shared covariance (linear boundary, with logistic regression's) and with separate ones (quadratic)."""
    s = ""
    rng = random.Random(21)
    xs, ys = sample("gauss", 80, rng)
    params = gda_fit(xs, ys)
    _, mu0, mu1, sig = params
    th0, th = gda_theta(*params)
    lr = lr_fit(xs, ys)
    dom = (-3.0, 4.5)
    to_px = mapper(dom, dom, (60, 320), (40, 300))
    s += axes(to_px, dom, dom, [-2, 0, 2, 4], [-2, 0, 2, 4], "x₁", "x₂")
    for mu, c in ((mu0, PURPLE), (mu1, TEAL)):
        for r in (1, 2):
            s += curve(ellipse(mu, sig, r), to_px, dom, dom, stroke=c, sw=1.2)
    for x, y in zip(xs, ys):
        px, py = to_px(*x)
        s += (
            circle(px, py, 3.4, fill=TEAL, stroke=TEAL)
            if y
            else rect(px - 3, py - 3, 6, 6, fill="none", stroke=PURPLE, sw=1.3)
        )
    s += curve(boundary_line(th0, th, dom), to_px, dom, dom, stroke=INDIGO, sw=2.2)
    s += curve(boundary_line(lr[0], lr[1:], dom), to_px, dom, dom, stroke=GREEN, sw=2.2, dash="6 4")
    s += text(190, 22, "shared Σ: a straight boundary", size=13, weight="600")

    qx, qy = sample("quad", 120, random.Random(5))
    q0 = [x for x, y in zip(qx, qy) if y == 0]
    q1 = [x for x, y in zip(qx, qy) if y == 1]
    qphi = len(q1) / len(qx)
    m0, s0 = mean_cov(q0)
    m1, s1 = mean_cov(q1)
    to_px = mapper(dom, dom, (420, 680), (40, 300))
    s += axes(to_px, dom, dom, [-2, 0, 2, 4], [-2, 0, 2, 4], "x₁", "x₂")
    for mu, sg, c in ((m0, s0, PURPLE), (m1, s1, TEAL)):
        for r in (1, 2):
            s += curve(ellipse(mu, sg, r), to_px, dom, dom, stroke=c, sw=1.2)
    for x, y in zip(qx, qy):
        px, py = to_px(*x)
        s += (
            circle(px, py, 3.4, fill=TEAL, stroke=TEAL)
            if y
            else rect(px - 3, py - 3, 6, 6, fill="none", stroke=PURPLE, sw=1.3)
        )
    for branch in quad_boundary(qphi, m0, m1, s0, s1, dom):
        s += curve(branch, to_px, dom, dom, stroke=INDIGO, sw=2.2)
    s += text(550, 22, "separate Σ₀, Σ₁: a conic", size=13, weight="600")
    s += line(80, 362, 104, 362, stroke=INDIGO, sw=2.2) + text(
        110, 366, "GDA boundary, P(y = 1 | x) = 0.5", size=11, anchor="start"
    )
    s += line(330, 362, 354, 362, stroke=GREEN, sw=2.2, dash="6 4") + text(
        360, 366, "logistic regression on the same data", size=11, anchor="start"
    )
    return wrap(
        720,
        380,
        "figGdaFit",
        "Gaussian discriminant analysis with a shared and with separate covariances",
        "Left: two classes of synthetic points with the contours of two Gaussians of the same shape and orientation fitted to them, the straight GDA decision boundary, and the slightly different straight boundary logistic regression finds on the same data. Right: two classes with very different spreads, one wide horizontally and one tall vertically, each fitted with its own covariance; the decision boundary is now curved.",
        s,
    )


def mean_cov(pts):
    """Sample mean and maximum-likelihood covariance of 2-D points."""
    n = len(pts)
    mu = [sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n]
    s = [[0.0, 0.0], [0.0, 0.0]]
    for p in pts:
        d = [p[0] - mu[0], p[1] - mu[1]]
        for i in range(2):
            for j in range(2):
                s[i][j] += d[i] * d[j] / n
    return mu, s


def log_odds(x, phi, mu0, mu1, s0, s1):
    """Log P(y=1|x)/P(y=0|x) with separate covariances."""
    return math.log(gauss2(x, mu1, s1) * phi) - math.log(gauss2(x, mu0, s0) * (1 - phi))


def quad_boundary(phi, mu0, mu1, s0, s1, dom):
    """The zero set of the quadratic log-odds, as up to two branches of (x1, x2) points."""
    lo, hi = [], []
    for i in range(601):
        u = dom[0] + (dom[1] - dom[0]) * i / 600
        f = [log_odds((u, v), phi, mu0, mu1, s0, s1) for v in (-1.0, 0.0, 1.0)]
        a, b, c = (f[2] + f[0]) / 2 - f[1], (f[2] - f[0]) / 2, f[1]
        disc = b * b - 4 * a * c
        if disc < 0 or abs(a) < 1e-12:
            continue
        r = sorted([(-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)])
        lo.append((u, r[0]))
        hi.append((u, r[1]))
    return [lo, hi]


# ---------------------------------------------------------------- the 1-D posterior

D_MU0, D_MU1, D_SD = 0.0, 2.0, 1.0
D_X0 = [-1.2, -0.6, -0.1, 0.4, 0.9]
D_X1 = [1.1, 1.7, 2.2, 2.6, 3.3]


def post1d(x, phi):
    """P(y = 1 | x) for one feature, shared variance."""
    a = math.exp(-((x - D_MU1) ** 2) / (2 * D_SD**2)) * phi
    b = math.exp(-((x - D_MU0) ** 2) / (2 * D_SD**2)) * (1 - phi)
    return a / (a + b)


def crossing(phi):
    """Where post1d = 0.5, by bisection."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if post1d(mid, phi) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fig_posterior_1d():
    """Two Gaussian class densities on one feature and the posterior they imply, at two priors."""
    s = ""
    xdom, ydom = (-3.0, 5.0), (0.0, 1.0)
    to_px = mapper(xdom, ydom, (70, 560), (30, 250))
    s += axes(to_px, xdom, ydom, [-2, 0, 2, 4], [0, 0.5, 1], "x", "")
    grid = [xdom[0] + i * 0.02 for i in range(401)]
    for mu, c in ((D_MU0, PURPLE), (D_MU1, TEAL)):
        s += curve(
            [(x, math.exp(-((x - mu) ** 2) / 2) / math.sqrt(2 * math.pi)) for x in grid],
            to_px,
            xdom,
            ydom,
            stroke=c,
            sw=1.4,
            op=0.8,
        )
    s += curve([(x, post1d(x, 0.5)) for x in grid], to_px, xdom, ydom, stroke=INDIGO, sw=2.4)
    s += curve([(x, post1d(x, 0.2)) for x in grid], to_px, xdom, ydom, stroke=INDIGO, sw=1.8, dash="6 4")
    for phi in (0.5, 0.2):
        c = crossing(phi)
        s += line(*to_px(c, 0), *to_px(c, 0.5), sw=1, op=0.6, dash="2 3")
    s += line(*to_px(xdom[0], 0.5), *to_px(xdom[1], 0.5), sw=0.8, op=0.3)
    for x in D_X0:
        px, py = to_px(x, 0)
        s += rect(px - 3.5, py - 12, 7, 7, fill="none", stroke=PURPLE, sw=1.4)
    for x in D_X1:
        px, py = to_px(x, 0)
        s += circle(px, py - 8.5, 3.8, fill=TEAL, stroke=TEAL)
    lx = 585
    s += line(lx, 60, lx + 22, 60, stroke=PURPLE, sw=1.4) + text(lx + 28, 64, "p(x | y = 0)", size=11, anchor="start")
    s += line(lx, 80, lx + 22, 80, stroke=TEAL, sw=1.4) + text(lx + 28, 84, "p(x | y = 1)", size=11, anchor="start")
    s += line(lx, 100, lx + 22, 100, stroke=INDIGO, sw=2.4) + text(
        lx + 28, 104, "P(y=1|x), φ = 0.5", size=11, anchor="start"
    )
    s += line(lx, 120, lx + 22, 120, stroke=INDIGO, sw=1.8, dash="6 4") + text(
        lx + 28, 124, "P(y=1|x), φ = 0.2", size=11, anchor="start"
    )
    return wrap(
        720,
        290,
        "figPost1d",
        "The posterior GDA implies along one feature",
        "Two unit-variance Gaussian densities centred at 0 and 2, with five training points from each class along the axis. The posterior probability of class 1, computed point by point by Bayes' rule, is an S-shaped curve crossing one half at x equals 1, midway between the means, when the prior is one half. With a prior of 0.2 the same curve shifts right and crosses one half at about 1.69.",
        s,
    )


# ---------------------------------------------------------------- learning curves

MS = [6, 10, 16, 25, 40, 64, 100, 160, 250, 400]
TRIALS = 400


def test_set(kind, n, seed):
    """A large labelled test set from one world."""
    return sample(kind, n, random.Random(seed))


def error(pred_fn, xs, ys):
    """Misclassification rate of a decision function."""
    return sum((pred_fn(x) >= 0) != (y == 1) for x, y in zip(xs, ys)) / len(xs)


def learning_curves(kind):
    """Mean test error of GDA and logistic regression against training-set size."""
    txs, tys = test_set(kind, 4000, 99)
    bayes = error(lambda x: bayes_rule(kind, x) - 0.5, txs, tys)
    rng = random.Random(7)
    out = []
    for m in MS:
        eg = el = 0.0
        for _ in range(TRIALS):
            xs, ys = sample(kind, m, rng)
            th0, th = gda_theta(*gda_fit(xs, ys))
            lr = lr_fit(xs, ys, ridge=0.001)
            eg += error(lambda x: th0 + dot(th, x), txs, tys)
            el += error(lambda x: lr[0] + lr[1] * x[0] + lr[2] * x[1], txs, tys)
        out.append((m, eg / TRIALS, el / TRIALS))
    big = sample(kind, 20000, random.Random(3))
    th0, th = gda_theta(*gda_fit(*big))
    lr = lr_fit(*big)
    asym = (error(lambda x: th0 + dot(th, x), txs, tys), error(lambda x: lr[0] + lr[1] * x[0] + lr[2] * x[1], txs, tys))
    return out, bayes, asym


def fig_learning_curves(curves):
    """Test error against m for GDA and logistic regression in the two worlds."""
    s = ""
    for k, (kind, title) in enumerate([("gauss", "x | y Gaussian, shared Σ"), ("pois", "x | y Poisson")]):
        out, bayes, _ = curves[kind]
        x0 = 70 + k * 340
        hi = max(max(e for _, e, _ in out), max(e for _, _, e in out))
        ydom = (math.floor((bayes - 0.02) * 50) / 50, math.ceil(hi * 50) / 50)
        xdom = (math.log10(MS[0]), math.log10(MS[-1]))
        to_px = mapper(xdom, ydom, (x0, x0 + 260), (40, 250))
        s += axes(
            to_px,
            xdom,
            ydom,
            [],
            [ydom[0], round((ydom[0] + ydom[1]) / 2, 2), ydom[1]],
            "training examples m",
            "test error" if k == 0 else "",
            fmt="{:.2f}",
        )
        for m in (10, 100):
            px, py = to_px(math.log10(m), ydom[0])
            s += line(px, py, px, py + 4, op=0.7) + text(px, py + 17, str(m), size=11, op=0.8)
        s += curve(
            [(math.log10(m), bayes) for m in [MS[0] * 1.01**i for i in range(400)] if m <= MS[-1]],
            to_px,
            xdom,
            ydom,
            sw=1,
            op=0.6,
            dash="2 3",
        )
        s += curve([(math.log10(m), e) for m, e, _ in out], to_px, xdom, ydom, stroke=INDIGO, sw=2.2)
        s += curve([(math.log10(m), e) for m, _, e in out], to_px, xdom, ydom, stroke=GREEN, sw=2.2, dash="6 4")
        for m, eg, el in out:
            s += circle(*to_px(math.log10(m), eg), 2.6, fill=INDIGO, stroke=INDIGO)
            s += circle(*to_px(math.log10(m), el), 2.6, fill=GREEN, stroke=GREEN)
        s += text(x0 + 130, 22, title, size=13, weight="600")
    s += line(90, 302, 114, 302, stroke=INDIGO, sw=2.2) + text(120, 306, "GDA", size=11, anchor="start")
    s += line(180, 302, 204, 302, stroke=GREEN, sw=2.2, dash="6 4") + text(
        210, 306, "logistic regression", size=11, anchor="start"
    )
    s += line(360, 302, 384, 302, sw=1, op=0.6, dash="2 3") + text(390, 306, "Bayes error", size=11, anchor="start")
    return wrap(
        720,
        320,
        "figLearning",
        "Learning curves for GDA and logistic regression",
        f"Mean test error over {TRIALS} training sets of each size, on a logarithmic axis from 6 to 400 examples. Left, data that satisfy GDA's assumptions: GDA's error is slightly lower than logistic regression's from 10 to about 100 examples, though not at 6, and the two converge on the Bayes error. Right, data whose classes are Poisson: logistic regression is correctly specified and GDA is not; GDA is still competitive at the smallest sizes, but logistic regression ends lower.",
        s,
    )


def fig_nb_graph():
    """The naive Bayes assumption as a graphical model: y points to every word indicator."""
    s = ""
    yx, yy = 360, 46
    s += circle(yx, yy, 22, fill=TEAL, stroke=INDIGO, sw=1.4, op=0.15)
    s += text(yx, yy + 5, "y", size=16, style="italic")
    s += text(yx + 34, yy + 5, "spam or not", size=11, anchor="start", op=0.8)
    words = [("x₁", "a"), ("x₂", "aardvark"), ("x₃", "aardwolf"), (None, None), ("x₁₀₀₀₀", "zymurgy")]
    for i, (sym, w) in enumerate(words):
        cx = 120 + i * 120
        if sym is None:
            s += text(cx, 158, "⋯", size=20)
            continue
        s += arrow(yx + (cx - yx) * 0.18, yy + 22, cx - (cx - yx) * 0.12, 136, sw=1.4, op=0.7)
        s += circle(cx, 158, 22, stroke=INDIGO, sw=1.4)
        s += text(cx, 163, sym, size=13 if len(sym) < 4 else 10, style="italic")
        s += text(cx, 198, f"“{w}”", size=11, op=0.8)
    return wrap(
        720,
        215,
        "figNbGraph",
        "The naive Bayes assumption as a graphical model",
        "A node for the label y with arrows to one node per word indicator, x1 for the word a, x2 for aardvark, x3 for aardwolf, through x10000 for zymurgy. There are no edges between the word nodes: given the label, each word's presence is independent of every other's.",
        s,
    )


FIGURES = {
    "diagram_gaussians.svg": fig_gaussians,
    "diagram_gda_fit.svg": fig_gda_fit,
    "diagram_posterior_1d.svg": fig_posterior_1d,
    "diagram_nb_graph.svg": fig_nb_graph,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)
    curves = {k: learning_curves(k) for k in ("gauss", "pois")}
    (OUT / "diagram_learning_curves.svg").write_text(fig_learning_curves(curves))
    print("wrote diagram_learning_curves.svg")

    # 1. Shrinking the covariance raises the peak: in 2-D the peak scales as 1/det^(1/2).
    for title, sig in GALLERY:
        l1, l2, v = eig2(sig)
        assert all(abs(a - b * l1) < 1e-12 for a, b in zip(mv(sig, v), v))
        print(
            f"gaussian: {title}: peak {1 / (2 * math.pi * math.sqrt(det2(sig))):.4f}, eigenvalues {l1:.2f}, {l2:.2f}, axis {v[0]:.3f}, {v[1]:.3f}"
        )

    # 2. GDA closed forms maximise the joint likelihood: every small perturbation lowers it.
    rng = random.Random(21)
    xs, ys = sample("gauss", 80, rng)
    params = gda_fit(xs, ys)
    base = gda_joint_ll(params, xs, ys)
    prng = random.Random(0)
    for _ in range(500):
        phi, mu0, mu1, s = params
        e = 1e-3
        pert = (
            phi + prng.uniform(-e, e),
            [v + prng.uniform(-e, e) for v in mu0],
            [v + prng.uniform(-e, e) for v in mu1],
            [
                [s[0][0] + prng.uniform(-e, e), s[0][1] + (d := prng.uniform(-e, e))],
                [s[1][0] + d, s[1][1] + prng.uniform(-e, e)],
            ],
        )
        assert gda_joint_ll(pert, xs, ys) < base
    print(f"gda: closed forms beat 500 random perturbations of the joint log-likelihood ({base:.3f})")

    # 3. Bayes' rule with GDA's parameters is exactly the logistic function of the implied θ.
    th0, th = gda_theta(*params)
    for _ in range(200):
        x = (prng.uniform(-4, 5), prng.uniform(-4, 5))
        assert abs(gda_posterior(x, params[0], params[1], params[2], params[3]) - sigmoid(th0 + dot(th, x))) < 1e-12
    print(f"gda: posterior = g(θᵀx + θ0) with θ0 = {th0:.3f}, θ = ({th[0]:.3f}, {th[1]:.3f})")

    # 4. ... but that θ is not logistic regression's: it does not maximise the conditional likelihood.
    lr = lr_fit(xs, ys)
    gl, ll = cond_ll([th0, *th], xs, ys), cond_ll(lr, xs, ys)
    assert ll > gl
    print(
        f"gda vs lr: logistic regression θ = ({lr[0]:.3f}, {lr[1]:.3f}, {lr[2]:.3f}); conditional log-lik {ll:.3f} vs GDA's {gl:.3f}"
    )

    # 5. Separate covariances: the log-odds is quadratic (constant, non-zero second difference).
    qphi, qm0, qm1 = 0.5, list(Q_MU0), list(Q_MU1)
    sec = []
    for u in (-1.0, 0.0, 1.5):
        f = [log_odds((u + t, 0.3), qphi, qm0, qm1, Q_SIG0, Q_SIG1) for t in (-0.1, 0.0, 0.1)]
        sec.append((f[0] - 2 * f[1] + f[2]) / 0.01)
    assert max(sec) - min(sec) < 1e-6 and abs(sec[0]) > 0.1
    print(f"qda: second difference of the log-odds along x1 is constant at {sec[0]:.4f}")
    for n in (2, 10, 100):
        a, b = gda_params(n, True), gda_params(n, False)
        print(f"params: n = {n}: shared Σ {a}, separate Σ {b}, ratio {b / a:.2f}")

    # 6. The posterior crosses 1/2 midway between the means only when φ = 1/2.
    for phi in (0.5, 0.2):
        c = crossing(phi)
        closed = (D_MU0 + D_MU1) / 2 + D_SD**2 * math.log((1 - phi) / phi) / (D_MU1 - D_MU0)
        assert abs(c - closed) < 1e-9
        print(f"posterior 1-d: φ = {phi}: P(y=1|x) = 0.5 at x = {c:.3f}")

    # 7. Two Poisson classes also give a logistic posterior: log-odds is linear in x.
    lo = [sum(xi * math.log(l1 / l0) - (l1 - l0) for xi, l0, l1 in zip((k, 2), P_LAM0, P_LAM1)) for k in range(6)]
    direct = [
        math.log(math.exp(-P_LAM1[0]) * P_LAM1[0] ** k / math.factorial(k))
        - math.log(math.exp(-P_LAM0[0]) * P_LAM0[0] ** k / math.factorial(k))
        for k in range(6)
    ]
    assert all(abs((lo[k + 1] - lo[k]) - (direct[k + 1] - direct[k])) < 1e-12 for k in range(5))
    assert max(abs(lo[k + 2] - 2 * lo[k + 1] + lo[k]) for k in range(4)) < 1e-12
    print(
        f"poisson: log-odds rises by log(λ1/λ0) = {math.log(P_LAM1[0] / P_LAM0[0]):.4f} per count; second difference 0"
    )

    # 8. Learning curves.
    for kind, (out, bayes, asym) in curves.items():
        cells = ", ".join(f"m={m}: {eg:.3f}/{el:.3f}" for m, eg, el in out)
        print(f"curves {kind}: Bayes {bayes:.3f}; GDA/LR {cells}; at m = 20000: GDA {asym[0]:.3f}, LR {asym[1]:.3f}")

    # 9. Naive Bayes: parameter counts, linear log-odds, duplicated features, zeros.
    n = 10000
    print(
        f"nb: full joint per class needs 2^{n} - 1 parameters (a {len(str(2**n))}-digit number); naive Bayes needs {2 * n + 1} in all"
    )
    wrng = random.Random(4)
    p1 = [wrng.uniform(0.05, 0.9) for _ in range(6)]
    p0 = [wrng.uniform(0.05, 0.9) for _ in range(6)]
    pri = 0.3
    w = [math.log(a * (1 - b) / (b * (1 - a))) for a, b in zip(p1, p0)]
    b0 = math.log(pri / (1 - pri)) + sum(math.log((1 - a) / (1 - b)) for a, b in zip(p1, p0))
    for code in range(64):
        x = [(code >> j) & 1 for j in range(6)]
        num = pri * math.prod(a if xi else 1 - a for a, xi in zip(p1, x))
        den = num + (1 - pri) * math.prod(b if xi else 1 - b for b, xi in zip(p0, x))
        assert abs(math.log(num / (den - num)) - (b0 + dot(w, x))) < 1e-9
    print("nb: log-odds = b + wᵀx exactly, over all 64 binary inputs of a six-word vocabulary")

    a1, a0 = 0.6, 0.1
    true = a1 / (a1 + a0)
    nb_dup = a1**2 / (a1**2 + a0**2)
    print(
        f"nb: word seen in 60% of spam, 10% of ham, prior 0.5: P(spam | word) = {true:.3f}; with an exact copy of the word as a second feature, naive Bayes says {nb_dup:.3f}"
    )

    spam = [[1, 0], [1, 1], [0, 1]]
    ham = [[0, 0], [0, 1], [0, 0]]
    ph1 = [sum(e[j] for e in spam) / len(spam) for j in range(2)] + [0.0]
    ph0 = [sum(e[j] for e in ham) / len(ham) for j in range(2)] + [0.0]
    x = [1, 1, 1]
    l1 = 0.5 * math.prod(p if xi else 1 - p for p, xi in zip(ph1, x))
    l0 = 0.5 * math.prod(p if xi else 1 - p for p, xi in zip(ph0, x))
    assert l1 == 0.0 and l0 == 0.0
    print(
        "nb: a word absent from every training e-mail gets φ = 0 in both classes, so P(x | y) = 0 for both and the posterior is 0/0"
    )
