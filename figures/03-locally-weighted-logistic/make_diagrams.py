"""Figures for lecture 3: locally weighted regression, logistic regression, Newton's method.

Every figure is computed from synthetic data drawn by a seeded generator, and
every curve on the page is the output of the algorithm the chapter names:
locally weighted regression solved by its weighted normal equations, linear
and logistic regression fitted to the same labels, and Newton's method run on
the derivative of a logistic log-likelihood.

`__main__` prints every number the prose quotes and asserts the claims the
chapter makes by two routes that must agree: the logistic gradient against
finite differences, the Hessian's sign, locally weighted regression against
ordinary least squares in the wide-bandwidth limit, maximum likelihood under
Gaussian and Laplace noise against the mean and the median, and the
log-concavity that makes the logistic likelihood free of local maxima —
checked for two other S-shaped links as well, one of which lacks it.

Plain Python on purpose, as in lectures 1 and 2.

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


def weighted_ls(xs, ys, ws):
    """Θ minimising Σ wᵢ (yᵢ − θ₀ − θ₁xᵢ)², from the 2×2 weighted normal equations XᵀWXθ = XᵀWy."""
    s0 = sum(ws)
    s1 = sum(w * x for w, x in zip(ws, xs))
    s2 = sum(w * x * x for w, x in zip(ws, xs))
    t0 = sum(w * y for w, y in zip(ws, ys))
    t1 = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    return solve([[s0, s1], [s1, s2]], [t0, t1])


# ---------------------------------------------------------------- locally weighted regression


def wavy_data(n=60, seed=3):
    """A curve no single low-order feature fits: a rise, a dip and a plateau."""
    rng = random.Random(seed)
    xs = sorted(rng.uniform(0.0, 10.0) for _ in range(n))
    ys = [2.0 + 1.6 * math.sin(0.9 * x) + 0.25 * x + rng.gauss(0, 0.28) for x in xs]
    return xs, ys


def lwr_weights(xs, x, tau):
    """Wᵢ = exp(−(xᵢ − x)² / 2τ²): a bump of height one, not a probability density."""
    return [math.exp(-((xi - x) ** 2) / (2 * tau * tau)) for xi in xs]


def lwr_predict(xs, ys, x, tau):
    """Locally weighted linear regression at one query point: a fresh weighted fit, then θᵀx."""
    t = weighted_ls(xs, ys, lwr_weights(xs, x, tau))
    return t[0] + t[1] * x, t


TAUS = (0.12, 0.6, 4.0)
QUERY, QUERY_TAU = 6.0, 0.6


def fig_lwr():
    """LWR at one query point, and the whole LWR curve for three bandwidths."""
    xs, ys = wavy_data()
    xdom, ydom = (0.0, 10.0), (0.0, 7.0)
    s = ""
    # Left: the mechanism at one query point.
    to_px = mapper(xdom, ydom, (60, 330), (40, 270))
    s += axes(to_px, xdom, ydom, [0, 2, 4, 6, 8, 10], [0, 2, 4, 6], "x", "y")
    ws = lwr_weights(xs, QUERY, QUERY_TAU)
    for x, y, w in zip(xs, ys, ws):
        s += circle(*to_px(x, y), 3.0, fill=PURPLE, stroke="none", op=0.15 + 0.8 * w)
    grid = [i / 20 for i in range(201)]
    s += curve(
        [(x, 2.2 * math.exp(-((x - QUERY) ** 2) / (2 * QUERY_TAU**2))) for x in grid],
        to_px,
        xdom,
        ydom,
        stroke=TEAL,
        sw=1.4,
        op=0.8,
    )
    yq, t = lwr_predict(xs, ys, QUERY, QUERY_TAU)
    s += curve([(QUERY - 2.2 + i * 0.022, t[0] + t[1] * (QUERY - 2.2 + i * 0.022)) for i in range(201)], to_px, xdom, ydom, stroke=INDIGO, sw=2)
    qx, qy = to_px(QUERY, yq)
    s += line(qx, to_px(0, 0)[1], qx, qy, stroke=INDIGO, sw=1, op=0.6, dash="2 3")
    s += circle(qx, qy, 4.2, fill=INDIGO, stroke="none")
    s += text(to_px(QUERY + 1.1, 0)[0], to_px(0, 2.35)[1], "weights wᵢ", size=11, anchor="start", fill=TEAL)
    s += text(qx - 10, qy - 14, "prediction at x", size=11, anchor="end", fill=INDIGO)
    s += text(195, 22, "one query, τ = 0.6", size=13, weight="600")
    # Right: the curve traced by repeating that at every x.
    to_px = mapper(xdom, ydom, (420, 690), (40, 270))
    s += axes(to_px, xdom, ydom, [0, 2, 4, 6, 8, 10], [0, 2, 4, 6], "x", "y")
    for x, y in zip(xs, ys):
        s += circle(*to_px(x, y), 2.6, fill="currentColor", stroke="none", op=0.45)
    styles = [(PURPLE, "3 3", 1.3), (INDIGO, None, 2.0), (TEAL, "7 4", 1.8)]
    for tau, (col, dash, sw) in zip(TAUS, styles):
        pts = [(x, lwr_predict(xs, ys, x, tau)[0]) for x in grid]
        s += curve(pts, to_px, xdom, ydom, stroke=col, sw=sw, dash=dash)
    lx = 440
    for tau, (col, dash, sw) in zip(TAUS, styles):
        s += line(lx, 58, lx + 18, 58, stroke=col, sw=sw, dash=dash) + text(
            lx + 22, 62, f"τ = {tau:g}", size=11, anchor="start"
        )
        lx += 76
    s += text(555, 22, "the curve, three bandwidths", size=13, weight="600")
    return wrap(
        720,
        320,
        "figLwr",
        "Locally weighted regression",
        "Left: a wavy scatter, with the points near a query at x = 6 drawn darker; a bell-shaped weight curve centred on the query; and the straight line fitted to the weighted points, whose value at x = 6 is the prediction. Right: repeating this at every x traces a curve; a very narrow bandwidth gives a jagged curve, a moderate one follows the data, and a wide one is nearly a straight line.",
        s,
    )


# ---------------------------------------------------------------- classification


def sigmoid(z):
    """g(z) = 1 / (1 + e^{−z}), written to avoid overflow for large |z|."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def tumour_data():
    """Tumour sizes, benign (0) below about 3.5 and malignant (1) above, with one overlap so the MLE is finite."""
    xs0 = [0.6, 1.0, 1.4, 1.7, 2.0, 2.3, 2.6, 3.6]
    xs1 = [3.4, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6]
    return xs0 + xs1, [0] * len(xs0) + [1] * len(xs1)


FAR_POINT = 20.0


def loglik(theta, xs, ys):
    """ℓ(θ) = Σ y log h + (1 − y) log(1 − h) for one feature plus intercept."""
    out = 0.0
    for x, y in zip(xs, ys):
        h = sigmoid(theta[0] + theta[1] * x)
        out += math.log(h) if y else math.log(1 - h)
    return out


def loglik_grad(theta, xs, ys):
    """∂ℓ/∂θⱼ = Σ (yᵢ − h(xᵢ)) xⱼ⁽ⁱ⁾, the same surface form as the LMS rule."""
    g = [0.0, 0.0]
    for x, y in zip(xs, ys):
        e = y - sigmoid(theta[0] + theta[1] * x)
        g[0] += e
        g[1] += e * x
    return g


def loglik_hessian(theta, xs):
    """H = −Σ h(1 − h) x xᵀ, negative semidefinite for every θ."""
    h = [[0.0, 0.0], [0.0, 0.0]]
    for x in xs:
        p = sigmoid(theta[0] + theta[1] * x)
        w = p * (1 - p)
        v = (1.0, x)
        for i in range(2):
            for j in range(2):
                h[i][j] -= w * v[i] * v[j]
    return h


def newton(xs, ys, iters=8, theta=(0.0, 0.0)):
    """Newton's method θ := θ − H⁻¹∇ℓ on the logistic log-likelihood; returns every iterate."""
    t = list(theta)
    path = [t[:]]
    for _ in range(iters):
        g, h = loglik_grad(t, xs, ys), loglik_hessian(t, xs)
        d = solve(h, g)
        t = [t[0] - d[0], t[1] - d[1]]
        path.append(t[:])
    return path


def gradient_ascent(xs, ys, alpha, iters, theta=(0.0, 0.0)):
    """Batch gradient ascent θ := θ + α∇ℓ; returns every iterate."""
    t = list(theta)
    path = [t[:]]
    for _ in range(iters):
        g = loglik_grad(t, xs, ys)
        t = [t[0] + alpha * g[0], t[1] + alpha * g[1]]
        path.append(t[:])
    return path


def threshold_linear(xs, ys):
    """Where the least-squares line crosses 0.5, the boundary linear regression would use."""
    t = weighted_ls(xs, ys, [1.0] * len(xs))
    return (0.5 - t[0]) / t[1], t


def threshold_logistic(xs, ys):
    """Where the maximum-likelihood logistic model crosses probability 0.5."""
    t = newton(xs, ys, iters=30)[-1]
    return -t[0] / t[1], t


def fig_linear_vs_logistic():
    """One far malignant tumour drags linear regression's threshold; logistic regression's barely moves."""
    xs, ys = tumour_data()
    xs2, ys2 = [*xs, FAR_POINT], [*ys, 1]
    xdom, ydom = (0.0, 21.0), (-0.3, 1.4)
    to_px = mapper(xdom, ydom, (70, 680), (40, 250))
    s = axes(to_px, xdom, ydom, [0, 3, 6, 9, 12, 15, 18, 21], [0, 0.5, 1], "tumour size", "y")
    s += line(*to_px(0, 0.5), *to_px(21, 0.5), op=0.3, dash="2 4")
    for x, y in zip(xs2, ys2):
        px, py = to_px(x, y)
        if y:
            s += line(px - 4, py - 4, px + 4, py + 4, stroke=PURPLE, sw=1.8) + line(
                px - 4, py + 4, px + 4, py - 4, stroke=PURPLE, sw=1.8
            )
        else:
            s += circle(px, py, 4.2, stroke=TEAL, sw=1.6)
    s += text(to_px(FAR_POINT, 1)[0], to_px(0, 1)[1] - 12, "one more example", size=11)
    b1, t1 = threshold_linear(xs, ys)
    b2, t2 = threshold_linear(xs2, ys2)
    s += curve([(i * 0.05, t1[0] + t1[1] * i * 0.05) for i in range(421)], to_px, xdom, ydom, stroke=INDIGO, sw=1.6, dash="6 4")
    s += curve([(i * 0.05, t2[0] + t2[1] * i * 0.05) for i in range(421)], to_px, xdom, ydom, stroke=INDIGO, sw=2)
    c1, _ = threshold_logistic(xs, ys)
    c2, t4 = threshold_logistic(xs2, ys2)
    grid = [i * 0.05 for i in range(421)]
    s += polyline([to_px(x, sigmoid(t4[0] + t4[1] * x)) for x in grid], stroke=TEAL, sw=2)
    base = to_px(0, ydom[0])[1]
    for b, col, lab, dy in (
        (b1, INDIGO, "line, before", 0),
        (b2, INDIGO, "line, after", 0),
        (c2, TEAL, "logistic", 14),
    ):
        bx = to_px(b, 0)[0]
        s += line(bx, base, bx, to_px(0, 1.25)[1], stroke=col, sw=1, op=0.7, dash="2 3")
    s += text(to_px(b1, 0)[0] - 4, to_px(0, 1.3)[1], "line before", size=11, anchor="end", fill=INDIGO)
    s += text(to_px(b2, 0)[0] + 4, to_px(0, 1.3)[1], "line after", size=11, anchor="start", fill=INDIGO)
    s += text(to_px(c2, 0)[0] + 4, to_px(0, -0.15)[1], "logistic", size=11, anchor="start", fill=TEAL)
    return wrap(
        720,
        295,
        "figLinearVsLogistic",
        "Linear regression is a poor classifier",
        "Benign tumours at y = 0 and malignant ones at y = 1, with one extra malignant tumour far to the right. The least-squares line fitted with that example is flatter than the one fitted without it, and the size where it crosses 0.5 moves right past several malignant tumours. The logistic regression curve fitted with the extra example keeps its boundary between the classes.",
        s,
    ), (b1, b2, c1, c2)


# ---------------------------------------------------------------- links and log-concavity


def probit(z):
    """Φ(z), the standard normal CDF."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def cauchit(z):
    """The Cauchy CDF, ½ + arctan(z)/π: S-shaped and bounded, with heavy tails."""
    return 0.5 + math.atan(z) / math.pi


LINKS = (("logistic", sigmoid, PURPLE, None), ("probit", probit, TEAL, "7 4"), ("Cauchy", cauchit, INDIGO, "3 3"))


def second_diff(f, z, h=1e-3):
    """Central second difference."""
    return (f(z + h) - 2 * f(z) + f(z - h)) / (h * h)


def fig_links():
    """Three S-shaped functions, and the logs of them that decide whether ℓ is concave."""
    s = ""
    zdom = (-6.0, 6.0)
    grid = [zdom[0] + i * 0.05 for i in range(241)]
    to_px = mapper(zdom, (0.0, 1.0), (60, 320), (40, 230))
    s += axes(to_px, zdom, (0.0, 1.0), [-6, -3, 0, 3, 6], [0, 0.5, 1], "z", "g(z)")
    for name, f, col, dash in LINKS:
        s += polyline([to_px(z, f(z)) for z in grid], stroke=col, sw=2, dash=dash)
    s += text(190, 22, "three choices of g", size=13, weight="600")
    ldom = (-6.0, 0.0)
    to_px = mapper(zdom, ldom, (420, 690), (40, 230))
    s += axes(to_px, zdom, ldom, [-6, -3, 0, 3, 6], [-6, -4, -2, 0], "z", "log g(z)")
    for name, f, col, dash in LINKS:
        s += curve([(z, math.log(f(z))) for z in grid], to_px, zdom, ldom, stroke=col, sw=2, dash=dash)
    s += text(555, 22, "their logs", size=13, weight="600")
    ly = 60
    for name, f, col, dash in LINKS:
        s += line(76, ly, 94, ly, stroke=col, sw=2, dash=dash) + text(98, ly + 4, name, size=11, anchor="start")
        ly += 16
    return wrap(
        720,
        280,
        "figLinks",
        "Three link functions and their logs",
        "Left: the logistic function, the normal CDF and the Cauchy CDF, all rising from 0 to 1. Right: their logarithms. The logistic and normal curves bend downward everywhere; the Cauchy curve flattens into a straight-ish tail on the left and bends upward there, so it is not concave.",
        s,
    )


# ---------------------------------------------------------------- Newton's method


NEWTON_START = 0.0


def newton_1d_data():
    """Sizes centred near the boundary so a model with no intercept is sensible: y = 1 when x > 0, with noise."""
    xs = [-2.4, -1.9, -1.5, -1.1, -0.7, -0.4, 0.3, -0.1, 0.5, 0.9, 1.2, 1.6, 2.1, 2.6]
    ys = [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    return xs, ys


def dl(theta, xs, ys):
    """ℓ′(θ) for the one-parameter model P(y = 1) = g(θx)."""
    return sum((y - sigmoid(theta * x)) * x for x, y in zip(xs, ys))


def d2l(theta, xs):
    """ℓ″(θ) = −Σ g(1 − g) x², always negative."""
    return -sum(sigmoid(theta * x) * (1 - sigmoid(theta * x)) * x * x for x in xs)


def newton_1d(xs, ys, theta, iters):
    """Newton's method for ℓ′(θ) = 0; returns every iterate."""
    path = [theta]
    for _ in range(iters):
        theta -= dl(theta, xs, ys) / d2l(theta, xs)
        path.append(theta)
    return path


def fig_newton():
    """Newton's method on f(θ) = ℓ′(θ): tangent, step to its root, repeat."""
    xs, ys = newton_1d_data()
    path = newton_1d(xs, ys, NEWTON_START, 6)
    tdom, fdom = (-0.5, 4.0), (-4.0, 8.0)
    to_px = mapper(tdom, fdom, (70, 650), (30, 250))
    s = axes(to_px, tdom, fdom, [0, 1, 2, 3, 4], [-4, 0, 4, 8], "θ", "f(θ) = ℓ′(θ)")
    s += line(*to_px(tdom[0], 0), *to_px(tdom[1], 0), op=0.5)
    grid = [tdom[0] + i * 0.02 for i in range(226)]
    s += curve([(t, dl(t, xs, ys)) for t in grid], to_px, tdom, fdom, stroke="currentColor", sw=1.8, op=0.75)
    cols = (PURPLE, INDIGO, TEAL)
    for k in range(3):
        t0, t1 = path[k], path[k + 1]
        f0 = dl(t0, xs, ys)
        p0 = to_px(t0, f0)
        s += line(*to_px(t0, 0), *p0, stroke=cols[k], sw=1, op=0.6, dash="2 3")
        s += line(*p0, *to_px(t1, 0), stroke=cols[k], sw=1.6)
        s += circle(*p0, 3.4, fill=cols[k], stroke="none")
        s += text(to_px(t0, 0)[0], to_px(0, 0)[1] + (16 if f0 > 0 else -8), f"θ{'₀₁₂'[k]}", size=11, fill=cols[k])
    s += circle(*to_px(path[-1], 0), 3.4, fill=PURPLE, stroke="none")
    return wrap(
        700,
        290,
        "figNewton",
        "Newton's method",
        "The derivative of a logistic log-likelihood, falling through zero. From the starting point a tangent line is drawn down to the horizontal axis, giving the next guess; two more tangents land almost on the root.",
        s,
    ), path


FIGURES = {
    "diagram_lwr.svg": fig_lwr,
    "diagram_linear_vs_logistic.svg": fig_linear_vs_logistic,
    "diagram_links.svg": fig_links,
    "diagram_newton.svg": fig_newton,
}


if __name__ == "__main__":
    extras = {}
    for name, fn in FIGURES.items():
        out = fn()
        if isinstance(out, tuple):
            out, extras[name] = out
        (OUT / name).write_text(out)
        print("wrote", name)

    # 1. LWR: wide bandwidth recovers ordinary least squares; the system is 2x2, not m x m.
    xs, ys = wavy_data()
    ols = weighted_ls(xs, ys, [1.0] * len(xs))
    _, wide = lwr_predict(xs, ys, 5.0, 1e4)
    assert max(abs(a - b) for a, b in zip(ols, wide)) < 1e-4
    print(f"lwr: m = {len(xs)}, tau = 1e4 reproduces the OLS line {ols[0]:.3f} + {ols[1]:.3f}x")
    yq, t = lwr_predict(xs, ys, QUERY, QUERY_TAU)
    w = lwr_weights(xs, QUERY, QUERY_TAU)
    print(
        f"lwr: at x = {QUERY}, tau = {QUERY_TAU}: prediction {yq:.3f}; {sum(v > 0.1 for v in w)} of {len(xs)} points have weight > 0.1"
    )
    for tau in TAUS:
        rss = sum((lwr_predict(xs, ys, x, tau)[0] - y) ** 2 for x, y in zip(xs, ys)) / len(xs)
        print(f"lwr: tau = {tau:g}: training MSE {rss:.4f}")

    # 2. Maximum likelihood with an intercept only: Gaussian noise gives the mean,
    # Laplace noise the median. Grid search on each log-likelihood, against the
    # closed forms.
    rng = random.Random(11)
    sample = [rng.gauss(3.0, 1.0) for _ in range(20)] + [25.0]
    grid = [i / 1000 for i in range(0, 10001)]
    g_ll = max(grid, key=lambda mu: -sum((v - mu) ** 2 for v in sample))
    l_ll = max(grid, key=lambda mu: -sum(abs(v - mu) for v in sample))
    mean = sum(sample) / len(sample)
    med = sorted(sample)[len(sample) // 2]
    assert abs(g_ll - mean) < 1e-3 and abs(l_ll - med) < 1e-3
    print(
        f"mle: Gaussian noise -> mean {mean:.2f}; Laplace noise -> median {med:.2f} (one outlier at 25 among 20 draws around 3)"
    )

    # 3. Linear regression as a classifier.
    b1, b2, c1, c2 = extras["diagram_linear_vs_logistic.svg"]
    xt, yt = tumour_data()
    xt2 = [*xt, FAR_POINT]
    yt2 = [*yt, 1]
    wrong0 = sum((x > b1) != bool(y) for x, y in zip(xt, yt))
    wrong = sum((x > b2) != bool(y) for x, y in zip(xt2, yt2))
    print(f"classify: misclassified by the line before {wrong0}, after {wrong}")
    print(
        f"classify: linear threshold {b1:.2f} -> {b2:.2f} with one far example ({wrong} misclassified after); logistic {c1:.2f} -> {c2:.2f}"
    )
    assert b2 - b1 > 0.5 and abs(c2 - c1) < 0.2

    # 4. The logistic gradient and Hessian.
    th = [-2.0, 0.7]
    g = loglik_grad(th, xt, yt)
    for j in range(2):
        e = [0.0, 0.0]
        e[j] = 1e-5
        fd = (loglik([a + b for a, b in zip(th, e)], xt, yt) - loglik([a - b for a, b in zip(th, e)], xt, yt)) / 2e-5
        assert abs(fd - g[j]) < 1e-5
    rng = random.Random(5)
    for _ in range(200):
        tt = [rng.uniform(-10, 10), rng.uniform(-5, 5)]
        h = loglik_hessian(tt, xt)
        assert h[0][0] <= 0 and h[0][0] * h[1][1] - h[0][1] ** 2 >= -1e-12
    print("logistic: gradient Σ(y−h)x matches finite differences; Hessian negative semidefinite at 200 random θ")

    # 5. Log-concavity of g and 1 - g, which is what makes the log-likelihood concave. Logistic and
    # probit have it; the Cauchy CDF does not, so its likelihood can have more
    # than one local maximum.
    zs = [-8 + i * 0.1 for i in range(161)]
    for name, f, _, _ in LINKS:
        worst = max(
            max(second_diff(lambda z: math.log(f(z)), z), second_diff(lambda z: math.log(f(-z)), z)) for z in zs
        )
        print(f"links: {name}: max second derivative of log g, log(1−g) over [-8, 8] = {worst:+.4f}")
        assert (worst <= 1e-6) == (name != "Cauchy")
    # What that costs, concretely: a Cauchy-link log-likelihood that is convex
    # somewhere, so the argument ruling out local maxima no longer applies.
    xc, yc = [1.0, 2.0, -3.0], [1, 1, 1]

    def ll_c(t):
        """Log-likelihood of P(y=1) = cauchit(θx), one parameter."""
        return sum(math.log(cauchit(t * x)) if y else math.log(cauchit(-t * x)) for x, y in zip(xc, yc))

    curv = max((second_diff(ll_c, t / 10), t / 10) for t in range(-100, 101))
    print(f"links: Cauchy-link log-likelihood on x = {xc}, y = {yc} has positive curvature {curv[0]:.4f} at θ = {curv[1]:.1f}")
    assert curv[0] > 0

    # 6. Newton's method: quadratic convergence, against gradient ascent.
    nx, ny = newton_1d_data()
    path = extras["diagram_newton.svg"]
    star = newton_1d(nx, ny, NEWTON_START, 40)[-1]
    errs = [abs(t - star) for t in path]
    print("newton: errors " + ", ".join(f"{e:.1e}" for e in errs))
    for a, b in zip(errs[1:4], errs[2:5]):
        assert b < 1.5 * a * a + 1e-15
    # Gradient ascent with the safe step 1/L, where L = λmax(XᵀX)/4 bounds the
    # curvature of the log-likelihood (h(1 − h) ≤ 1/4), against Newton, both from 0.
    full = newton(xt, yt, iters=40)[-1]
    tol = 1e-6

    def first_within(path):
        """Index of the first iterate within tol of the optimum in every coordinate."""
        return next((k for k, t in enumerate(path) if max(abs(u - v) for u, v in zip(t, full)) < tol), None)

    nsteps = first_within(newton(xt, yt, iters=40))
    a, b, c = len(xt), sum(xt), sum(x * x for x in xt)
    lmax = (a + c) / 2 + math.hypot((a - c) / 2, b)
    alpha = 4 / lmax
    gsteps = first_within(gradient_ascent(xt, yt, alpha, 400000))
    print(f"newton vs ascent: optimum θ = ({full[0]:.2f}, {full[1]:.2f}); Newton {nsteps} iterations, gradient ascent at α = 1/L = {alpha:.4f} {gsteps}, to {tol:g}")
    assert nsteps < 15 and gsteps and gsteps > 1000 * nsteps
