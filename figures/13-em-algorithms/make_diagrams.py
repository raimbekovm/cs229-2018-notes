"""Figures for lecture 13: k-means, mixtures of Gaussians and the EM algorithm.

Four computed figures: k-means on a two-cluster data set, step by step; a
one-dimensional mixture of two Gaussians with the posterior probability that
EM's E-step computes, for shared and for separate variances; Jensen's
inequality as a chord above a convex function; and EM on a one-parameter
mixture drawn as a sequence of lower bounds under the log-likelihood.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: that the k-means cost never rises, and
that restarts find different local minima; that the squared and unsquared
distance give the same assignments but not the same centroid step; that the
E-step posterior is exactly a sigmoid of x when the variances are shared and
is not monotone when they differ; that EM's log-likelihood never decreases;
that the lower bound touches the log-likelihood at the current parameter with
the same slope and lies below it everywhere; that the M-step's closed form is
the bound's maximiser; that EM with a shared, shrinking variance turns into
k-means; and that the mixture likelihood is unbounded.

Plain Python on purpose, as in lectures 1-12.

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


def _off(pt, dx, dy):
    """Shift a pixel point."""
    return pt[0] + dx, pt[1] + dy


# ---------------------------------------------------------------- k-means


def d2(a, b):
    """Squared Euclidean distance."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def assign(X, mus, sq=True):
    """Index of the nearest centroid for each point, by squared or plain distance."""
    f = d2 if sq else (lambda a, b: math.sqrt(d2(a, b)))
    return [min(range(len(mus)), key=lambda j: f(x, mus[j])) for x in X]


def update(X, c, mus):
    """Move each centroid to the mean of its points (an empty cluster keeps its centroid)."""
    out = []
    for j, mu in enumerate(mus):
        pts = [x for x, cj in zip(X, c) if cj == j]
        out.append([sum(v) / len(pts) for v in zip(*pts)] if pts else mu)
    return out


def distortion(X, c, mus, sq=True):
    """The k-means cost: total squared (or plain) distance from each point to its centroid."""
    return sum(d2(x, mus[j]) if sq else math.sqrt(d2(x, mus[j])) for x, j in zip(X, c))


def kmeans(X, mus, iters=50):
    """Run k-means; return the final assignment, centroids and the cost after each half-step."""
    costs = []
    for _ in range(iters):
        c = assign(X, mus)
        costs.append(distortion(X, c, mus))
        new = update(X, c, mus)
        costs.append(distortion(X, c, new))
        if new == mus:
            break
        mus = new
    return c, mus, costs


def two_clusters():
    """The figure's data: two Gaussian blobs of 30 points."""
    rng = random.Random(1)
    X = [[rng.gauss(-1.4, 0.55), rng.gauss(0.7, 0.55)] for _ in range(30)]
    X += [[rng.gauss(1.4, 0.55), rng.gauss(-0.5, 0.55)] for _ in range(30)]
    return X


KM_INIT = [[-2.2, -1.6], [-1.6, 2.2]]


def fig_kmeans():
    """Four panels: initial centroids, first assignment, first update, converged."""
    X = two_clusters()
    mus0 = [m[:] for m in KM_INIT]
    c1 = assign(X, mus0)
    mus1 = update(X, c1, mus0)
    c_fin, mus_fin, _ = kmeans(X, mus0)
    panels = (
        ("1. centroids at random", None, mus0),
        ("2. assign to nearest", c1, mus0),
        ("3. move to the mean", c1, mus1),
        ("4. converged", c_fin, mus_fin),
    )
    cols = (PURPLE, TEAL)
    dom = (-3.2, 3.2)
    s = ""
    for k, (title, c, mus) in enumerate(panels):
        x0 = 15 + k * 170
        to_px = mapper(dom, dom, (x0, x0 + 150), (40, 190))
        s += rect(x0, 40, 150, 150, fill="none", so=0.5, sw=0.8)
        s += text(x0 + 75, 28, title, size=11.5)
        for i, x in enumerate(X):
            fill = "currentColor" if c is None else cols[c[i]]
            s += circle(*to_px(*x), 2.6, fill=fill, stroke="none", op=0.45 if c is None else 0.8)
        for j, mu in enumerate(mus):
            px, py = to_px(*mu)
            s += line(px - 6, py - 6, px + 6, py + 6, stroke=cols[j], sw=2.6)
            s += line(px - 6, py + 6, px + 6, py - 6, stroke=cols[j], sw=2.6)
    return wrap(
        700,
        205,
        "figKmeans",
        "k-means on two clusters",
        "Four panels of the same 60 unlabelled points. First, two centroids placed at random, both on the left. "
        "Second, every point coloured by the nearer centroid. Third, each centroid moved to the mean of its "
        "points. Fourth, after a few more rounds, the centroids sit in the middle of the two clusters and no "
        "assignment changes.",
        s,
    )


# ---------------------------------------------------------------- mixture of Gaussians, 1-D


def npdf(x, mu, s2):
    """Normal density."""
    return math.exp(-((x - mu) ** 2) / (2 * s2)) / math.sqrt(2 * math.pi * s2)


def log_npdf(x, mu, s2):
    """Log of the normal density, so tiny variances do not underflow."""
    return -((x - mu) ** 2) / (2 * s2) - 0.5 * math.log(2 * math.pi * s2)


def posterior1(x, phi, mus, s2s):
    """E-step weight: probability that x came from component 1 (index 0)."""
    la = math.log(phi) + log_npdf(x, mus[0], s2s[0])
    lb = math.log(1 - phi) + log_npdf(x, mus[1], s2s[1])
    d = lb - la
    return 1 / (1 + math.exp(d)) if d < 700 else 0.0


MIX_PHI, MIX_MU = 0.5, (-1.5, 1.5)
SHARED, SEPARATE = (0.8, 0.8), (0.4, 2.5)


def fig_mixture():
    """A two-component density and its E-step posterior, shared and separate variances."""
    xdom = (-5, 5)
    xs = [xdom[0] + i * 0.02 for i in range(501)]
    s = ""
    to_a = mapper(xdom, (0, 0.5), (60, 330), (40, 200))
    s += axes(to_a, xdom, (0, 0.5), [-4, -2, 0, 2, 4], [0, 0.25, 0.5], "x", "density")
    s += text(195, 26, "Density, separate variances", size=12.5, weight="bold")
    phi, mus, v = MIX_PHI, MIX_MU, SEPARATE
    s += curve([(x, phi * npdf(x, mus[0], v[0])) for x in xs], to_a, xdom, (0, 0.5), stroke=PURPLE, sw=1.2, dash="4 3")
    s += curve(
        [(x, (1 - phi) * npdf(x, mus[1], v[1])) for x in xs], to_a, xdom, (0, 0.5), stroke=TEAL, sw=1.2, dash="4 3"
    )
    s += curve(
        [(x, phi * npdf(x, mus[0], v[0]) + (1 - phi) * npdf(x, mus[1], v[1])) for x in xs],
        to_a,
        xdom,
        (0, 0.5),
        stroke=INDIGO,
        sw=2.0,
    )
    s += text(*_off(to_a(-1.5, 0.5 * npdf(-1.5, -1.5, 0.4)), 30, 6), "p(x)", size=11.5, fill=INDIGO, anchor="start")
    to_b = mapper(xdom, (0, 1), (400, 670), (40, 200))
    s += axes(to_b, xdom, (0, 1), [-4, -2, 0, 2, 4], [0, 0.5, 1], "x", "")
    s += text(535, 26, "Posterior p(z = 1 | x)", size=12.5, weight="bold")
    s += curve([(x, posterior1(x, phi, mus, SHARED)) for x in xs], to_b, xdom, (0, 1), stroke=PURPLE, sw=2.0)
    s += curve([(x, posterior1(x, phi, mus, SEPARATE)) for x in xs], to_b, xdom, (0, 1), stroke=TEAL, sw=2.0)
    s += text(*_off(to_b(-2.1, 0.5), 0, 0), "separate", size=11.5, fill=TEAL)
    s += text(*_off(to_b(2.6, 0.2), 0, 0), "shared:", size=11.5, fill=PURPLE)
    s += text(*_off(to_b(2.6, 0.2), 0, 14), "a sigmoid", size=11.5, fill=PURPLE)
    return wrap(
        700,
        250,
        "figMixture",
        "A mixture of two Gaussians and its E-step posterior",
        "Left, the density of a mixture of a narrow Gaussian at minus 1.5 and a wide one at plus 1.5, each with "
        "weight one half, drawn as two dashed components and their solid sum. Right, the probability that a point "
        "came from the left component. With a shared variance it is a sigmoid in x, falling from one to zero. With "
        "the separate variances of the left panel it is near zero on both sides and peaks near the narrow "
        "component, because far from both means the wide component is the likelier source.",
        s,
    )


# ---------------------------------------------------------------- Jensen


def f_convex(x):
    """The convex function of the Jensen figure."""
    return 0.18 * (x - 1.6) ** 2 + 0.4


def fig_jensen():
    """The chord between f(1) and f(5) lies above f at the mean."""
    xdom, ydom = (0, 6), (0, 4)
    to_px = mapper(xdom, ydom, (60, 400), (20, 230))
    s = axes(to_px, xdom, ydom, [1, 2, 3, 4, 5], [], "x", "f(x)")
    s += curve([(i / 50, f_convex(i / 50)) for i in range(301)], to_px, xdom, ydom, stroke=INDIGO, sw=2.0)
    a, b = (1, f_convex(1)), (5, f_convex(5))
    s += line(*to_px(*a), *to_px(*b), stroke=TEAL, sw=1.4)
    ef = (a[1] + b[1]) / 2
    for (x, y), lab in ((a, "f(1)"), (b, "f(5)")):
        s += circle(*to_px(x, y), 3.5, fill=TEAL, stroke="none")
        s += text(*_off(to_px(x, y), -10 if x < 3 else 10, -9), lab, size=11.5, anchor="end" if x < 3 else "start")
    s += circle(*to_px(3, ef), 4, fill=PURPLE, stroke="none")
    s += circle(*to_px(3, f_convex(3)), 4, fill=INDIGO, stroke="none")
    s += line(*to_px(3, f_convex(3)), *to_px(3, 0), sw=0.8, op=0.5, dash="2 3")
    s += text(*_off(to_px(3, ef), -2, -14), "E[f(X)]", size=11.5, fill=PURPLE, anchor="end")
    s += text(*_off(to_px(3, f_convex(3)), 10, 14), "f(E[X])", size=11.5, fill=INDIGO, anchor="start")
    return wrap(
        440,
        270,
        "figJensen",
        "Jensen's inequality as a chord",
        "A convex curve with the chord joining its values at x equals 1 and x equals 5. For X equal to 1 or 5 "
        "with probability one half each, the midpoint of the chord is the expected value of f of X, and it lies "
        "above the curve at 3, which is f of the expected value of X.",
        s,
    )


# ---------------------------------------------------------------- EM as lower-bound maximisation

LB_DATA_SEED = 3


def lb_data():
    """Data for the one-parameter mixture: half from N(0, 1), half from N(3, 1)."""
    rng = random.Random(LB_DATA_SEED)
    return [rng.gauss(0, 1) for _ in range(20)] + [rng.gauss(3, 1) for _ in range(20)]


def loglik(mu, X):
    """Log-likelihood of mu, the mean of the second component; the first is N(0, 1), phi = 1/2."""
    return sum(math.log(0.5 * npdf(x, 0.0, 1.0) + 0.5 * npdf(x, mu, 1.0)) for x in X)


def e_step(mu, X):
    """Q_i(z = 2): the posterior responsibility of the second component."""
    return [posterior1(x, 0.5, (mu, 0.0), (1.0, 1.0)) for x in X]


def bound(mu, Q, X):
    """The lower bound sum_i sum_z Q_i(z) log(p(x_i, z; mu) / Q_i(z))."""
    tot = 0.0
    for x, q in zip(X, Q):
        for qz, p in ((q, 0.5 * npdf(x, mu, 1.0)), (1 - q, 0.5 * npdf(x, 0.0, 1.0))):
            if qz > 0:
                tot += qz * math.log(p / qz)
    return tot


def m_step(Q, X):
    """Closed-form maximiser of the bound: the responsibility-weighted mean."""
    return sum(q * x for q, x in zip(Q, X)) / sum(Q)


LB_START = -1.0


def em_path(X, mu, steps):
    """EM iterates of the one-parameter mixture."""
    out = [mu]
    for _ in range(steps):
        mu = m_step(e_step(mu, X), X)
        out.append(mu)
    return out


def fig_em_bound():
    """The log-likelihood and the lower bounds EM builds at its first iterates."""
    X = lb_data()
    path = em_path(X, LB_START, 3)
    xdom = (-2, 5)
    ms = [xdom[0] + i * 0.02 for i in range(351)]
    ll = [(m, loglik(m, X)) for m in ms]
    ydom = (math.floor(min(v for _, v in ll) / 10) * 10 - 10, math.ceil(max(v for _, v in ll) / 10) * 10 + 5)
    to_px = mapper(xdom, ydom, (70, 610), (20, 250))
    yt = list(range(int(ydom[0]), int(ydom[1]) + 1, 20))
    s = axes(to_px, xdom, ydom, [-2, -1, 0, 1, 2, 3, 4, 5], yt, "θ (the mean of the second component)", "")
    s += curve(ll, to_px, xdom, ydom, stroke=INDIGO, sw=2.2)
    cols = (TEAL, PURPLE, TEAL)
    for k in range(3):
        Q = e_step(path[k], X)
        s += curve(
            [(m, bound(m, Q, X)) for m in ms], to_px, xdom, ydom, stroke=cols[k], sw=1.3, op=0.85 if k < 2 else 0.5
        )
        s += circle(*to_px(path[k], loglik(path[k], X)), 3.5, fill=cols[k], stroke="none")
        px, py = to_px(path[k + 1], bound(path[k + 1], Q, X))
        s += line(px, py, *to_px(path[k + 1], loglik(path[k + 1], X)), sw=1.0, op=0.6, dash="2 2")
        s += text(*_off(to_px(path[k], ydom[0]), 0, -6), f"θ{k}", size=11, op=0.85)
    s += text(*_off(to_px(4.9, loglik(4.9, X)), -2, -22), "ℓ(θ)", size=12, fill=INDIGO, anchor="end")
    s += text(
        *_off(to_px(-1.9, bound(-1.9, e_step(path[0], X), X)), 4, 16), "bound at θ0", size=11, fill=TEAL, anchor="start"
    )
    return wrap(
        650,
        290,
        "figBound",
        "EM as repeated maximisation of a lower bound",
        "The log-likelihood of a one-parameter mixture as a function of the second component's mean, with the "
        "lower bounds built by the E-step at the first three EM iterates. Each bound is a parabola that touches "
        "the log-likelihood at the current iterate and lies below it elsewhere; the M-step moves to the top of the "
        "parabola, where the log-likelihood is higher still.",
        s,
    )


FIGURES = {
    "diagram_kmeans.svg": fig_kmeans,
    "diagram_mixture.svg": fig_mixture,
    "diagram_jensen.svg": fig_jensen,
    "diagram_em_bound.svg": fig_em_bound,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    rng = random.Random(0)

    # 1. The k-means cost never rises, on the figure's data and on random restarts.
    X = two_clusters()
    _, mus, costs = kmeans(X, [m[:] for m in KM_INIT])
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))
    print(f"figure k-means: {len(costs) // 2} rounds, cost {costs[0]:.1f} -> {costs[-1]:.1f}")
    X3 = []
    for cx, cy in ((-2, 0), (2, 0), (0, 3)):
        X3 += [[rng.gauss(cx, 0.6), rng.gauss(cy, 0.6)] for _ in range(40)]
    finals = []
    for _ in range(100):
        _, m3, cs = kmeans(X3, [p[:] for p in rng.sample(X3, 3)])
        assert all(b <= a + 1e-9 for a, b in zip(cs, cs[1:]))
        finals.append(round(cs[-1], 3))
    distinct = sorted(set(finals))
    print(
        f"k = 3, 100 restarts from random examples: {len(distinct)} distinct final costs, best {distinct[0]}, worst {distinct[-1]}, best found {finals.count(distinct[0])} times"
    )
    assert len(distinct) > 1

    # 2. Squared and plain distance: same assignment, different centroid step.
    for _ in range(200):
        mus_r = [[rng.uniform(-3, 3), rng.uniform(-3, 3)] for _ in range(3)]
        assert assign(X3, mus_r, sq=True) == assign(X3, mus_r, sq=False)
    print("assignment by |x - mu| and by |x - mu|^2: identical on 200 random centroid sets")
    worse = 0
    for _ in range(2000):
        mus_r = [list(p) for p in rng.sample(X3, 3)]
        c = assign(X3, mus_r)
        new = update(X3, c, mus_r)
        if distortion(X3, c, new, sq=False) > distortion(X3, c, mus_r, sq=False) + 1e-9:
            worse += 1
    print(f"mean step raised the unsquared cost on {worse} of 2000 random starts")
    assert worse > 0

    # 3. The E-step posterior: a sigmoid with shared variance, non-monotone otherwise.
    s2 = SHARED[0]
    a = (MIX_MU[0] - MIX_MU[1]) / s2
    b = (MIX_MU[1] ** 2 - MIX_MU[0] ** 2) / (2 * s2) + math.log(MIX_PHI / (1 - MIX_PHI))
    for i in range(-50, 51):
        x = i / 10
        assert abs(posterior1(x, MIX_PHI, MIX_MU, SHARED) - 1 / (1 + math.exp(-(a * x + b)))) < 1e-12
    print(f"shared variance: posterior = sigmoid({a:.3f} x + {b:.3f}) exactly")
    sep = [posterior1(i / 10, MIX_PHI, MIX_MU, SEPARATE) for i in range(-50, 51)]
    xmax = -5 + sep.index(max(sep)) / 10
    print(f"separate variances: posterior peaks at x = {xmax:.1f} ({max(sep):.3f}), is {sep[0]:.3f} at x = -5")
    assert sep[0] < max(sep) - 0.1

    # 4. EM on the one-parameter mixture: monotone, tight, tangent, closed-form M-step.
    Xl = lb_data()
    path = em_path(Xl, LB_START, 60)
    lls = [loglik(m, Xl) for m in path]
    assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:]))
    print("EM iterates:", [round(m, 3) for m in path[:5]], "->", round(path[-1], 4))
    print("log-likelihood:", [round(v, 2) for v in lls[:5]], "->", round(lls[-1], 3))
    grid = [-3 + i * 0.01 for i in range(901)]
    for t in range(4):
        Q = e_step(path[t], Xl)
        assert abs(bound(path[t], Q, Xl) - lls[t]) < 1e-9
        h = 1e-5
        gb = (bound(path[t] + h, Q, Xl) - bound(path[t] - h, Q, Xl)) / (2 * h)
        gl = (loglik(path[t] + h, Xl) - loglik(path[t] - h, Xl)) / (2 * h)
        assert abs(gb - gl) < 1e-5
        assert all(bound(m, Q, Xl) <= loglik(m, Xl) + 1e-9 for m in grid)
        best = max(grid, key=lambda m: bound(m, Q, Xl))
        assert abs(best - path[t + 1]) < 0.006
    print(
        "bound: equal to the log-likelihood and to its slope at each iterate, below it on the grid, maximised by the M-step"
    )
    other = em_path(Xl, -1.8, 60)
    print(f"from theta0 = -1.8 EM ends at {other[-1]:.4f}, log-likelihood {loglik(other[-1], Xl):.3f}")

    # 5. EM with a shared, shrinking variance becomes k-means.
    X1 = list(Xl)
    mus1 = [0.0, 3.0]  # no data point within 0.08 of the midpoint
    hard = [min((0, 1), key=lambda j: (x - mus1[j]) ** 2) for x in X1]
    for s2 in (1.0, 0.1, 0.01, 0.001, 1e-4):
        w = [posterior1(x, 0.5, mus1, (s2, s2)) for x in X1]
        gap = max(abs(wi - (1.0 if h == 0 else 0.0)) for wi, h in zip(w, hard))
        print(f"shared variance {s2}: largest gap between soft and hard assignment {gap:.2e}")
    assert gap < 1e-3

    # 6. The mixture likelihood is unbounded: centre one component on a point and shrink it.
    prev = -math.inf
    for s2 in (1e-8, 1e-16, 1e-32, 1e-64, 1e-128):
        v = sum(math.log(0.5 * npdf(x, Xl[0], s2) + 0.5 * npdf(x, 1.5, 3.0)) for x in Xl)
        print(f"component 1 on x_1 with variance {s2:g}: log-likelihood {v:.2f}")
        assert v > prev
        prev = v
