"""Figures for lecture 18: continuous-state MDPs, discretization, models and fitted value iteration.

Two computed figures. The first puts the lecture's housing-price analogy on a
value function: a one-dimensional MDP (move left, stay, move right; reward
-s^2; Gaussian slip) whose V* is computed on a fine grid, beside the staircase
a six-cell discretization gives and the smooth curve fitted value iteration
gives with five polynomial features; a second panel counts the k^n states of
discretizing the lecture's three robots. The second figure is Tsitsiklis and
Van Roy's two-state example, in which least-squares fitted value iteration
diverges although the features can represent V* exactly.

`__main__` prints every number the prose quotes and asserts the claims by two
routes that must agree: fitting [A B] jointly against n separate regressions
and against the Gaussian maximum-likelihood fit with a correlated noise
covariance; k = 1 against k = 5 on a deterministic simulator; the divergence
ratio 6 gamma / 5 against the closed form; and, for the policy read-out, the
noise-free argmax V(f(s, a)) against the exact expectation, which must agree
for a quadratic V and need not for any other.

Plain Python, as in lectures 1-17. Run from the repository root.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _linalg import T, inv, mm, mv
from _svg import INDIGO, PURPLE, TEAL, axes, circle, curve, line, mapper, text, wrap

OUT = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------- the 1-D toy MDP

ACTS = (-1, 0, 1)
STEP, SIGMA, GAMMA = 0.25, 0.15, 0.9
LO, HI = -3.0, 3.0  # the region the chapter plots and fitted VI samples from


def reward(s):
    """State reward: stay near the origin."""
    return -s * s


def noise_nodes(sigma, n=11):
    """A discretized N(0, sigma^2): n equally spaced points on +-4 sigma with normalised weights."""
    if sigma == 0:
        return [(0.0, 1.0)]
    xs = [-4 * sigma + 8 * sigma * i / (n - 1) for i in range(n)]
    ws = [math.exp(-0.5 * (x / sigma) ** 2) for x in xs]
    z = sum(ws)
    return [(x, w / z) for x, w in zip(xs, ws)]


def reference_v(sigma, lo=-5.0, hi=5.0, n=401, tol=1e-10):
    """V* on a fine grid by value iteration with linear interpolation; returns an evaluator on [lo, hi]."""
    h = (hi - lo) / (n - 1)
    grid = [lo + i * h for i in range(n)]
    nodes = noise_nodes(sigma)
    plan = []  # per state, per action: list of (index, weight on index, weight on index+1)
    for s in grid:
        per_a = []
        for a in ACTS:
            terms = []
            for e, w in nodes:
                x = min(max(s + STEP * a + e, lo), hi)
                j = min(int((x - lo) / h), n - 2)
                f = (x - grid[j]) / h
                terms.append((j, w * (1 - f), w * f))
            per_a.append(terms)
        plan.append(per_a)
    V = [0.0] * n
    while True:
        new = [
            reward(s) + GAMMA * max(sum(w0 * V[j] + w1 * V[j + 1] for j, w0, w1 in terms) for terms in per_a)
            for s, per_a in zip(grid, plan)
        ]
        delta = max(abs(x - y) for x, y in zip(new, V))
        V = new
        if delta < tol:
            break

    def ev(s):
        j = min(int((s - lo) / h), n - 2)
        f = (s - grid[j]) / h
        return V[j] * (1 - f) + V[j + 1] * f

    return ev


def discretized_v(cells):
    """Solve the discretized MDP: equal cells on [LO, HI], transitions from points spread over each cell."""
    w = (HI - LO) / cells
    centres = [LO + (i + 0.5) * w for i in range(cells)]
    nodes = noise_nodes(SIGMA)
    P = []
    for c in centres:
        per_a = []
        for a in ACTS:
            p = [0.0] * cells
            for u in range(10):
                s = c - w / 2 + w * (u + 0.5) / 10
                for e, q in nodes:
                    x = min(max(s + STEP * a + e, LO), HI - 1e-9)
                    p[int((x - LO) / w)] += q / 10
            per_a.append(p)
        P.append(per_a)
    V = [0.0] * cells
    for _ in range(400):
        V = [reward(c) + GAMMA * max(sum(pi * v for pi, v in zip(p, V)) for p in per_a) for c, per_a in zip(centres, P)]
    return lambda s: V[min(int((s - LO) / w), cells - 1)]


def phi(s, degree):
    """Polynomial features 1, s, ..., s^degree."""
    return [s**d for d in range(degree + 1)]


def least_squares(X, y):
    """Theta = argmin ||X theta - y||^2 by the normal equations."""
    return mv(inv(mm(T(X), X)), mv(T(X), y))


def fitted_vi(degree, m=200, k=10, iters=150, sigma=SIGMA, seed=1):
    """Fitted value iteration as in the lecture; returns theta and the history of max |change in V| on the samples."""
    rng = random.Random(seed)
    states = [rng.uniform(LO, HI) for _ in range(m)]
    X = [phi(s, degree) for s in states]
    theta = [0.0] * (degree + 1)
    for _ in range(iters):
        y = []
        for s in states:
            q = []
            for a in ACTS:
                nxt = [s + STEP * a + rng.gauss(0, sigma) for _ in range(k)]
                q.append(sum(reward(s) + GAMMA * sum(t * v for t, v in zip(theta, phi(x, degree))) for x in nxt) / k)
            y.append(max(q))
        theta = least_squares(X, y)
    return theta


def v_of(theta):
    """The fitted value function as a callable."""
    return lambda s: sum(t * s**d for d, t in enumerate(theta))


def readout_disagreement(V, sigma, n=601):
    """Fraction of states where argmax_a V(f(s,a)) differs from argmax_a E[V(f(s,a) + eps)]."""
    nodes = noise_nodes(sigma, 41)
    bad = 0
    for i in range(n):
        s = LO + (HI - LO) * i / (n - 1)
        a0 = max(ACTS, key=lambda a: V(s + STEP * a))
        a1 = max(ACTS, key=lambda a: sum(w * V(s + STEP * a + e) for e, w in nodes))
        bad += a0 != a1
    return bad / n


# ---------------------------------------------------------------- learning a model


def fit_model(rng):
    """Fly a 2-D linear system with correlated noise under random actions; fit [A B] three ways."""
    A = [[1.0, 0.1], [-0.2, 0.95]]
    B = [[0.0], [0.1]]
    L = [[0.05, 0.0], [0.04, 0.03]]  # Sigma = L L^T, correlated
    Z, Y = [], []
    for _ in range(20):
        s = [rng.gauss(0, 1), rng.gauss(0, 1)]
        for _ in range(50):
            a = rng.uniform(-1, 1)
            g = [rng.gauss(0, 1), rng.gauss(0, 1)]
            nxt = [A[i][0] * s[0] + A[i][1] * s[1] + B[i][0] * a + L[i][0] * g[0] + L[i][1] * g[1] for i in range(2)]
            Z.append([s[0], s[1], a])
            Y.append(nxt)
            s = nxt
    # Jointly: W = Y^T Z (Z^T Z)^-1, i.e. every output at once.
    G = inv(mm(T(Z), Z))
    W_joint = T(mm(G, mm(T(Z), Y)))
    # Row by row: one ordinary regression per state variable.
    W_rows = [least_squares(Z, [y[i] for y in Y]) for i in range(2)]
    # Maximum likelihood with the true covariance: gradient ascent on sum -(y-Wz)^T Sigma^-1 (y-Wz).
    Sigma = mm(L, T(L))
    Si = inv(Sigma)
    W = [[0.0] * 3 for _ in range(2)]
    lr = 0.5 / len(Z)
    for _ in range(6000):
        grad = [[0.0] * 3 for _ in range(2)]
        for z, y in zip(Z, Y):
            r = [y[i] - sum(W[i][j] * z[j] for j in range(3)) for i in range(2)]
            sr = mv(Si, r)
            for i in range(2):
                for j in range(3):
                    grad[i][j] += sr[i] * z[j]
        # Precondition by Sigma so the step size does not depend on its conditioning; the fixed point is unchanged.
        step = mm(Sigma, grad)
        W = [[W[i][j] + lr * step[i][j] for j in range(3)] for i in range(2)]
    truth = [A[0] + B[0], A[1] + B[1]]
    return W_joint, W_rows, W, truth


# ---------------------------------------------------------------- divergence


def tvr_iterates(gamma, theta0, iters):
    """Least-squares fitted VI on Tsitsiklis and Van Roy's two-state chain: phi = 1, 2; 1 -> 2 -> 2; zero reward."""
    feats, nxt = [1.0, 2.0], [1, 1]
    th, out = theta0, [theta0]
    for _ in range(iters):
        y = [0.0 + gamma * th * feats[nxt[i]] for i in range(2)]
        th = sum(f * v for f, v in zip(feats, y)) / sum(f * f for f in feats)
        out.append(th)
    return out


# ---------------------------------------------------------------- figures


def fig_staircase(vref, vdisc, vfit, samples):
    """Left: V*, a six-cell staircase and the fitted-VI curve. Right: k^n against n."""
    xdom, ydom = (LO, HI), (-40.0, 5.0)
    to_px = mapper(xdom, ydom, (80, 400), (30, 250))
    s = axes(to_px, xdom, ydom, (-3, -2, -1, 0, 1, 2, 3), (-40, -30, -20, -10, 0), "state s", "value", fmt="{:g}")
    xs = [LO + (HI - LO) * i / 600 for i in range(601)]
    s += curve([(x, vdisc(x)) for x in xs], to_px, xdom, ydom, strict=True, stroke=TEAL, sw=2.2)
    s += curve([(x, vref(x)) for x in xs], to_px, xdom, ydom, strict=True, stroke=PURPLE, sw=2.0)
    s += curve([(x, vfit(x)) for x in xs], to_px, xdom, ydom, strict=True, stroke=INDIGO, sw=1.6, dash="6 4")
    for x, y in samples:
        px, py = to_px(x, y)
        s += circle(px, py, 1.8, fill=INDIGO, stroke="none", op=0.45)
    for k, (col, dash, lab) in enumerate(
        (
            (PURPLE, None, "V* (fine grid)"),
            (TEAL, None, "6-cell discretization"),
            (INDIGO, "6 4", "fitted VI, 5 features"),
        )
    ):
        y = 196 + 17 * k
        s += line(176, y, 204, y, stroke=col, sw=2.0, dash=dash) + text(210, y + 4, lab, size=11.5, anchor="start")

    # Right panel: log10 of the number of cells.
    xd2, yd2 = (0, 12), (0, 24)
    to2 = mapper(xd2, yd2, (510, 730), (30, 250))
    s += axes(to2, xd2, yd2, (0, 4, 8, 12), (0, 6, 12, 18, 24), "state dimension n", "log₁₀ cells kⁿ")
    for kk, col, lab in ((10, TEAL, "k = 10"), (100, PURPLE, "k = 100")):
        s += curve([(n, n * math.log10(kk)) for n in range(0, 13)], to2, xd2, yd2, strict=True, stroke=col, sw=2.0)
        px, py = to2(5.2 if kk == 100 else 9.0, (5.2 if kk == 100 else 9.0) * math.log10(kk))
        s += text(px - 8, py - 8, lab, size=11.5, anchor="end", fill=col)
    for n, lab in ((4, "pendulum"), (6, "car"), (12, "helicopter")):
        px, py = to2(n, n)
        s += circle(px, py, 3.5, fill=INDIGO, stroke="none")
        s += text(px + (-6 if n == 12 else 6), py + 16, lab, size=11, anchor="end" if n == 12 else "start")
    return wrap(
        760,
        300,
        "figStaircase",
        "A discretized value function against a fitted one, and the cost of discretizing",
        "Left: the optimal value function of a one-dimensional MDP, a smooth bowl from minus 32.5 at the edges "
        "to minus 0.26 at the origin, drawn with the six-step staircase that a six-cell discretization produces, which "
        "misses it by up to 15, and the curve fitted value iteration produces with five polynomial features, which "
        "follows it to within 1.8, its worst miss being a slight overshoot at the origin; faint dots are the fitted targets on the last iteration. Right: the base-ten "
        "logarithm of the number of cells, k to the n, against the dimension n, for k equal to 10 and 100, with the "
        "inverted pendulum at n equal to 4, the car at 6 and the helicopter at 12 marked on the k equal to 10 line.",
        s,
    )


def fig_divergence(runs):
    """Log10 |theta| against iteration on the two-state chain."""
    xdom, ydom = (0, 100), (-5.0, 4.0)
    to_px = mapper(xdom, ydom, (80, 520), (30, 230))
    s = axes(to_px, xdom, ydom, (0, 20, 40, 60, 80, 100), (-4, -2, 0, 2, 4), "iteration", "log₁₀ |θ|")
    for (gamma, col, dash), its in runs.items():
        s += curve(
            [(t, math.log10(abs(v))) for t, v in enumerate(its)],
            to_px,
            xdom,
            ydom,
            strict=True,
            stroke=col,
            sw=2.0,
            dash=dash,
        )
    for k, (col, dash, lab) in enumerate(
        (
            (PURPLE, None, "γ = 0.9: ×1.08 per iteration"),
            (TEAL, None, "γ = 0.8: ×0.96"),
            (INDIGO, "6 4", "γ = 0.9, ordinary VI"),
        )
    ):
        y = 42 + 16 * k
        s += line(100, y, 128, y, stroke=col, sw=2.0, dash=dash) + text(134, y + 4, lab, size=11.5, anchor="start")
    return wrap(
        560,
        280,
        "figDivergence",
        "Fitted value iteration diverging on a two-state chain",
        "Base-ten logarithm of the size of the single parameter theta against the iteration, starting from theta "
        "equal to one, for Tsitsiklis and Van Roy's two-state example, whose true value function is zero and is "
        "represented exactly by theta equal to zero. At discount 0.9 the parameter grows steadily, passing a "
        "thousand after 90 iterations; at 0.8 it shrinks slowly; ordinary value iteration on the same two "
        "states at 0.9, started from the same values, falls by a factor of 0.9 per sweep.",
        s,
    )


FIGURES = {}

if __name__ == "__main__":
    # 1. Reference V*, the staircase, fitted VI.
    vref = reference_v(SIGMA)
    print(f"V*(0) = {vref(0):.3f}, V*(+-3) = {vref(3):.2f}, {vref(-3):.2f}")
    staircase = {}
    for cells in (6, 12, 24, 48):
        vd = discretized_v(cells)
        err = max(abs(vd(LO + 6 * i / 600) - vref(LO + 6 * i / 600)) for i in range(601))
        staircase[cells] = (vd, err)
        print(f"discretization, {cells} cells: max error on [-3, 3] {err:.2f}")
    fits = {}
    for degree in (2, 4):
        th = fitted_vi(degree)
        vf = v_of(th)
        err = max(abs(vf(LO + 6 * i / 600) - vref(LO + 6 * i / 600)) for i in range(601))
        fits[degree] = (th, err)
        print(
            f"fitted VI, degree {degree} ({degree + 1} features): max error {err:.2f}; theta =",
            [round(t, 3) for t in th],
        )
    assert fits[4][1] < staircase[6][1] / 3
    # The last iteration's targets, for the figure's dots.
    rng = random.Random(5)
    vf4 = v_of(fits[4][0])
    samples = []
    for _ in range(60):
        s0 = rng.uniform(LO, HI)
        q = [sum(reward(s0) + GAMMA * vf4(s0 + STEP * a + rng.gauss(0, SIGMA)) for _ in range(10)) / 10 for a in ACTS]
        samples.append((s0, max(q)))
    print("10^n cells for n = 4, 6, 12:", [10**n for n in (4, 6, 12)], "; notes' example 100^10 =", f"{100**10:.0e}")

    # 2. A deterministic simulator needs one sample: k = 1 and k = 5 give the same theta exactly.
    t1 = fitted_vi(2, k=1, iters=40, sigma=0.0)
    t5 = fitted_vi(2, k=5, iters=40, sigma=0.0)
    assert max(abs(a - b) for a, b in zip(t1, t5)) < 1e-9
    print("deterministic simulator: k = 1 and k = 5 agree to", max(abs(a - b) for a, b in zip(t1, t5)))

    # 3. Read-out with the noise switched off.
    d2 = readout_disagreement(v_of(fits[2][0]), SIGMA)
    d4 = readout_disagreement(v_of(fits[4][0]), SIGMA)
    d4big = readout_disagreement(v_of(fits[4][0]), 0.6)
    assert d2 == 0.0
    print(
        f"argmax V(f(s,a)) vs argmax E V(f(s,a)+eps): quadratic V {d2:.4f}; quartic V {d4:.4f} at sigma {SIGMA}, {d4big:.4f} at 0.6"
    )

    # 4. Learning a model: joint, row-wise and maximum-likelihood fits coincide.
    Wj, Wr, Wml, truth = fit_model(random.Random(3))
    dj = max(abs(Wj[i][j] - Wr[i][j]) for i in range(2) for j in range(3))
    dml = max(abs(Wj[i][j] - Wml[i][j]) for i in range(2) for j in range(3))
    derr = max(abs(Wj[i][j] - truth[i][j]) for i in range(2) for j in range(3))
    assert dj < 1e-9 and dml < 1e-6
    print(
        f"model fit: joint vs row-wise {dj:.1e}, joint vs Gaussian ML with correlated Sigma {dml:.1e}, max error vs truth {derr:.3f}"
    )

    # 5. Divergence.
    runs = {}
    for gamma, col, dash in ((0.9, PURPLE, None), (0.8, TEAL, None)):
        its = tvr_iterates(gamma, 1.0, 100)
        assert all(abs(its[t + 1] / its[t] - 6 * gamma / 5) < 1e-12 for t in range(100))
        runs[(gamma, col, dash)] = its
        print(f"two-state chain, gamma {gamma}: theta after 100 iterations {its[-1]:.3g}")
    assert tvr_iterates(0.9, 0.0, 40)[-1] == 0.0
    first = next(t for t, v in enumerate(runs[(0.9, PURPLE, None)]) if v > 1000)
    print("gamma 0.9 passes 1000 at iteration", first)
    # Exact VI on the two states from V = (1, 2): V <- 0 + gamma V(2); both go to zero.
    v = [1.0, 2.0]
    exact = [abs(v[1])]
    for _ in range(100):
        v = [0.9 * v[1], 0.9 * v[1]]
        exact.append(max(abs(x) for x in v))
    runs[(0.9, INDIGO, "6 4")] = exact
    print("exact VI, gamma 0.9, after 100 sweeps:", f"{exact[-1]:.3g}")

    FIGURES.update(
        {
            "diagram_staircase.svg": fig_staircase(vref, staircase[6][0], vf4, samples),
            "diagram_divergence.svg": fig_divergence(runs),
        }
    )
    for name, svg in FIGURES.items():
        (OUT / name).write_text(svg)
        print("wrote", name)
