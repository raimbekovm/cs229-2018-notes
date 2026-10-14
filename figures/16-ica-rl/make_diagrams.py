"""Figures for lecture 16: ICA by maximum likelihood, and the gridworld MDP.

Four computed figures: contours of the joint density of two independent unit-
variance sources under a Gaussian, a logistic (the density whose CDF is the
sigmoid) and a Laplace prior; the change-of-variables example and the tails of
those three densities on a log scale; ICA run by gradient ascent on mixed
Laplace sources; and the lecture's 11-state gridworld with its transition model
and its optimal policy.

`__main__` prints every number the prose quotes and asserts the claims by two
routes that must agree: that the ICA log-likelihood under a Gaussian prior is
unchanged when W is rotated, while under the logistic prior it is not; that the
density of x = 2s integrates to one only with the |det W| factor; that the
analytic gradient of the log-likelihood matches finite differences; that
gradient ascent recovers the sources up to permutation and scale; that with more
microphones than sources the square-W likelihood is unbounded; and, for the
gridworld, that value iteration reproduces the values Ng read off the board,
that the policy he showed is optimal for his reward, and how much the reward can
change before the decision at (3, 1) flips.

Plain Python, as in lectures 1-15. Run from the repository root.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _gridworld import (
    ACTIONS,
    STATES,
    TERMINAL,
    draw_grid,
    greedy,
    q_future,
    value_iteration,
)
from _svg import (
    INDIGO,
    PURPLE,
    TEAL,
    arrow,
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

# ---------------------------------------------------------------- densities

LOGISTIC_B = math.sqrt(3) / math.pi  # scale giving the logistic unit variance
LAPLACE_B = 1 / math.sqrt(2)  # scale giving the Laplace unit variance


def log_gauss(s):
    """Log density of N(0, 1)."""
    return -0.5 * s * s - 0.5 * math.log(2 * math.pi)


def log_logistic(s, b=LOGISTIC_B):
    """Log density of the logistic with scale b: the derivative of the sigmoid of s / b."""
    z = abs(s) / b
    return -z - 2 * math.log1p(math.exp(-z)) - math.log(b)


def log_laplace(s, b=LAPLACE_B):
    """Log density of the Laplace with scale b."""
    return -abs(s) / b - math.log(2 * b)


DENSITIES = (("Gaussian", log_gauss), ("logistic (sigmoid CDF)", log_logistic), ("Laplace", log_laplace))


def contour_radius(logp, theta, drop):
    """Distance along direction theta at which log p(s1) + log p(s2) falls `drop` below its peak."""
    peak = 2 * logp(0.0)
    c, s = math.cos(theta), math.sin(theta)
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if peak - (logp(mid * c) + logp(mid * s)) < drop:
            lo = mid
        else:
            hi = mid
    return lo


LEVELS = (0.5, 1.5, 3.0, 4.5)


def fig_source_densities():
    """Three panels: contours of the joint source density under each prior."""
    s = ""
    dom = (-3.5, 3.5)
    for k, (name, logp) in enumerate(DENSITIES):
        x0 = 30 + k * 240
        to_px = mapper(dom, dom, (x0, x0 + 200), (40, 240))
        s += rect(x0, 40, 200, 200, so=0.4, sw=0.8)
        s += line(*to_px(-3.5, 0), *to_px(3.5, 0), op=0.3) + line(*to_px(0, -3.5), *to_px(0, 3.5), op=0.3)
        for j, drop in enumerate(LEVELS):
            pts = []
            for i in range(361):
                t = 2 * math.pi * i / 360
                r = contour_radius(logp, t, drop)
                pts.append(to_px(r * math.cos(t), r * math.sin(t)))
            s += polyline(pts, stroke=(PURPLE, INDIGO, TEAL, TEAL)[j], sw=1.5, op=(1, 0.9, 0.8, 0.6)[j])
        s += text(x0 + 100, 28, name, size=12.5)
        s += text(x0 + 100, 258, "s₁", size=11.5, style="italic")
        s += text(x0 + 10, 56, "s₂", size=11.5, style="italic", anchor="start")
    return wrap(
        740,
        270,
        "figSrcDens",
        "Joint densities of two independent sources under three priors",
        "Three contour plots of the density of two independent unit-variance sources. Under the Gaussian the "
        "contours are circles, so every rotation of the axes looks the same. Under the logistic density, whose "
        "cumulative distribution is the sigmoid, the inner contours are nearly round and the outer ones become "
        "diamonds pointing along the axes. Under the Laplace density every contour is a diamond.",
        s,
    )


def fig_change_of_variables():
    """Left: densities of s ~ U[0, 1] and x = 2s. Right: log densities of the three priors."""
    s = ""
    xdom, ydom = (-0.5, 2.5), (0.0, 1.25)
    to_px = mapper(xdom, ydom, (70, 330), (40, 220))
    s += axes(to_px, xdom, ydom, (0, 1, 2), (0, 0.5, 1), "value", "density")
    s += polyline([to_px(0, 0), to_px(0, 1), to_px(1, 1), to_px(1, 0)], stroke=TEAL, sw=2.2)
    s += polyline([to_px(0, 0), to_px(0, 0.5), to_px(2, 0.5), to_px(2, 0)], stroke=PURPLE, sw=2.2)
    s += polyline([to_px(1, 1.0), to_px(2, 1.0), to_px(2, 0.5)], stroke=PURPLE, sw=1.4, dash="5 4")
    s += text(*to_px(0.5, 1.08), "pₛ(s), s ~ U[0, 1]", size=11.5, fill=TEAL)
    s += text(*to_px(1.0, 0.39), "pₓ(x) = ½ · pₛ(x/2)", size=11.5, fill=PURPLE)
    s += text(*to_px(1.75, 1.08), "pₛ(x/2): area 2", size=11.5, fill=PURPLE)

    xdom, ydom = (-5.0, 5.0), (-14.0, 0.0)
    to_px = mapper(xdom, ydom, (450, 730), (40, 220))
    s += axes(to_px, xdom, ydom, (-4, -2, 0, 2, 4), (-12, -8, -4, 0), "s (unit variance)", "log density")
    grid = [-5 + 10 * i / 400 for i in range(401)]
    for (name, logp), col, lab in zip(DENSITIES, (PURPLE, TEAL, INDIGO), ("Gaussian", "logistic", "Laplace")):
        s += curve([(v, logp(v)) for v in grid], to_px, xdom, ydom, stroke=col, sw=2.0)
    for j, (lab, col) in enumerate((("Gaussian", PURPLE), ("logistic", TEAL), ("Laplace", INDIGO))):
        lx, ly = to_px(-1.4, -7.0 - 2.0 * j)
        s += line(lx, ly - 4, lx + 22, ly - 4, stroke=col, sw=2.0)
        s += text(lx + 28, ly, lab, size=11.5, fill=col, anchor="start")
    return wrap(
        760,
        270,
        "figCov",
        "Densities under a linear map, and the tails of three source priors",
        "Left: the uniform density on zero to one has height one. Doubling the variable spreads it over zero to "
        "two, and the density halves to keep area one; plugging x over two into the original density without the "
        "determinant factor gives height one over a width of two, area two. Right: log densities of unit-variance "
        "Gaussian, logistic and Laplace distributions. The Gaussian falls as a parabola; the logistic and Laplace "
        "fall linearly, so far from zero they are orders of magnitude more likely.",
        s,
    )


# ---------------------------------------------------------------- ICA by maximum likelihood


def sigmoid(z):
    """The logistic function g."""
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def det(W):
    """Determinant by elimination (small matrices)."""
    n = len(W)
    M = [r[:] for r in W]
    d = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            return 0.0
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return d


def matvec(W, x):
    """W x."""
    return [sum(a * b for a, b in zip(row, x)) for row in W]


def loglik(W, X, logp):
    """ICA log-likelihood: sum_i [ sum_j log p_s(w_j^T x_i) + log |det W| ]."""
    ld = math.log(abs(det(W)))
    return sum(sum(logp(v) for v in matvec(W, x)) + ld for x in X)


def log_sigmoid_density(v):
    """Log g'(v): the source prior the lecture uses (logistic, unit scale)."""
    return log_logistic(v, 1.0)


def inv(W):
    """Inverse by Gauss-Jordan."""
    n = len(W)
    M = [[*r, *(1.0 if i == j else 0.0 for j in range(n))] for i, r in enumerate(W)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [r[n:] for r in M]


def grad(W, X):
    """Average gradient: mean_i (1 - 2 g(W x_i)) x_i^T + (W^T)^{-1}."""
    n = len(W)
    G = [[0.0] * n for _ in range(n)]
    for x in X:
        y = matvec(W, x)
        for j in range(n):
            c = 1 - 2 * sigmoid(y[j])
            for k in range(n):
                G[j][k] += c * x[k]
    Wi = inv(W)
    m = len(X)
    return [[G[j][k] / m + Wi[k][j] for k in range(n)] for j in range(n)]


MIX = [[1.0, 0.6], [0.45, 1.0]]


def ica_data(m=1500):
    """Independent Laplace sources (scale 1) and their mixture by MIX."""
    rng = random.Random(16)
    S = [[rng.choice((-1, 1)) * rng.expovariate(1.0) for _ in range(2)] for _ in range(m)]
    return S, [matvec(MIX, v) for v in S]


def run_ica(X, steps=400, alpha=0.05):
    """Batch gradient ascent on the average log-likelihood from W = I; returns W and the trace."""
    W = [[1.0, 0.0], [0.0, 1.0]]
    trace = [loglik(W, X, log_sigmoid_density) / len(X)]
    for _ in range(steps):
        G = grad(W, X)
        W = [[W[j][k] + alpha * G[j][k] for k in range(2)] for j in range(2)]
        trace.append(loglik(W, X, log_sigmoid_density) / len(X))
    return W, trace


def fig_ica_ml(X, W, trace):
    """Mixtures, recovered sources, and the log-likelihood over iterations."""
    s = ""
    dom = (-6.0, 6.0)
    Y = [matvec(W, x) for x in X]
    for k, (pts, title, lx, ly, col) in enumerate(
        ((X, "microphones x = As", "x₁", "x₂", INDIGO), (Y, "recovered Wx after ascent", "(Wx)₁", "(Wx)₂", TEAL))
    ):
        x0 = 40 + k * 230
        to_px = mapper(dom, dom, (x0, x0 + 180), (40, 220))
        s += rect(x0, 40, 180, 180, so=0.4, sw=0.8)
        s += line(*to_px(-6, 0), *to_px(6, 0), op=0.3) + line(*to_px(0, -6), *to_px(0, 6), op=0.3)
        for p in pts[:700]:
            if dom[0] < p[0] < dom[1] and dom[0] < p[1] < dom[1]:
                s += circle(*to_px(*p), 1.4, fill=col, stroke="none", op=0.55)
        s += text(x0 + 90, 28, title, size=12)
        s += text(x0 + 90, 238, lx, size=11.5, style="italic")
        s += text(x0 - 5, 50, ly, size=11.5, style="italic", anchor="end")
    xdom, ydom = (0, 400), (-3.8, -3.0)
    assert all(ydom[0] < v < ydom[1] for v in trace)
    to_px = mapper(xdom, ydom, (560, 740), (40, 220))
    s += axes(to_px, xdom, ydom, (0, 200, 400), (-3.8, -3.4, -3.0), "iteration", "ℓ(W) / m", fmt="{:g}")
    s += curve([(i, v) for i, v in enumerate(trace)], to_px, xdom, ydom, stroke=PURPLE, sw=2.0)
    return wrap(
        770,
        275,
        "figIcaMl",
        "ICA by gradient ascent on two mixed Laplace sources",
        "Left: 700 of 1,500 microphone readings, a sheared cloud with its long arms along two oblique directions. "
        "Centre: the same points multiplied by the unmixing matrix found by gradient ascent; the arms now lie along "
        "the axes, so each coordinate is one source. Right: the average log-likelihood rising over 400 iterations "
        "of batch gradient ascent from the identity matrix and levelling off.",
        s,
    )


# ---------------------------------------------------------------- the gridworld


def fig_gridworld(pi):
    """Left: rewards and the transition model from (3, 1). Right: the optimal policy."""
    cell = 64
    s = ""
    x0, y0 = 30, 40
    s += draw_grid(x0, y0, cell, mark=(3, 1))
    for gx, gy in STATES:
        if (gx, gy) not in TERMINAL and (gx, gy) != (3, 1):
            s += text(x0 + (gx - 0.5) * cell, y0 + (3.5 - gy) * cell + 5, "−0.02", size=11.5, op=0.75)
    cx, cy = x0 + 2.5 * cell, y0 + 2.5 * cell
    s += arrow(cx, cy, cx, cy - cell / 2 + 1, sw=2.2, op=0.9, head=7, stroke=PURPLE)
    s += text(cx + 5, cy - 14, "0.8", size=11, fill=PURPLE, anchor="start")
    s += arrow(cx - 3, cy + 12, cx - cell / 2 + 1, cy + 12, sw=1.4, op=0.85, head=6, stroke=TEAL)
    s += text(cx - 16, cy + 27, "0.1", size=11, fill=TEAL)
    s += arrow(cx + 3, cy + 12, cx + cell / 2 - 1, cy + 12, sw=1.4, op=0.85, head=6, stroke=TEAL)
    s += text(cx + 16, cy + 27, "0.1", size=11, fill=TEAL)
    s += text(x0 + 2 * cell, 26, "command North from (3, 1)", size=12)
    for gx in (1, 2, 3, 4):
        s += text(x0 + (gx - 0.5) * cell, y0 + 3 * cell + 18, str(gx), size=11, op=0.7)
    for gy in (1, 2, 3):
        s += text(x0 - 10, y0 + (3.5 - gy) * cell + 4, str(gy), size=11, op=0.7)

    x1 = 360
    s += draw_grid(x1, y0, cell, policy=pi, mark=(3, 1))
    s += text(x1 + 2 * cell, 26, "optimal policy, R = −0.02, γ = 0.99", size=12)
    for gx in (1, 2, 3, 4):
        s += text(x1 + (gx - 0.5) * cell, y0 + 3 * cell + 18, str(gx), size=11, op=0.7)
    for gy in (1, 2, 3):
        s += text(x1 - 10, y0 + (3.5 - gy) * cell + 4, str(gy), size=11, op=0.7)
    return wrap(
        640,
        250,
        "figGrid",
        "The 11-state gridworld and its optimal policy",
        "Left: a four-by-three grid with a wall at column two, row two, a plus-one cell at the top right and a "
        "minus-one cell below it; every other cell carries a reward of minus 0.02. From cell (3, 1), a command to go "
        "North reaches (3, 2) with probability 0.8 and slips West or East with probability 0.1 each. Right: the "
        "optimal policy. From the bottom row the robot goes West and round by the top, rather than North past the "
        "minus-one cell.",
        s,
    )


FIGURES = {}

if __name__ == "__main__":
    # 1. Contour shapes: ratio of the radius along the diagonal to the radius along an axis.
    for name, logp in DENSITIES:
        rs = [contour_radius(logp, math.pi / 4, d) / contour_radius(logp, 0.0, d) for d in LEVELS]
        print(f"{name}: diagonal/axis radius at levels {LEVELS}: " + ", ".join(f"{r:.3f}" for r in rs))
        assert max(contour_radius(logp, t / 10, LEVELS[-1]) for t in range(63)) < 3.5
    assert all(
        abs(contour_radius(log_gauss, math.pi / 4, d) / contour_radius(log_gauss, 0, d) - 1) < 1e-9 for d in LEVELS
    )

    # 2. Tails: excess kurtosis by numerical integration, and density at 4 standard deviations.
    def moment(logp, k):
        """Numerical E[s^k] on [-40, 40]."""
        h = 0.001
        return sum((i * h) ** k * math.exp(logp(i * h)) * h for i in range(-40000, 40001))

    for name, logp in DENSITIES:
        print(
            f"{name}: var {moment(logp, 2):.4f}, excess kurtosis {moment(logp, 4) - 3:.3f}, "
            f"p(4)/p_gauss(4) = {math.exp(logp(4) - log_gauss(4)):.1f}"
        )

    # 3. Change of variables in one dimension: x = 2s, s ~ U[0, 1], W = 1/2.
    h = 1e-4
    with_det = sum(0.5 * (0 <= 0.5 * (i * h) <= 1) * h for i in range(0, 30000))
    without = sum(1.0 * (0 <= 0.5 * (i * h) <= 1) * h for i in range(0, 30000))
    print(f"integral of |W| p_s(Wx): {with_det:.4f}; of p_s(Wx) alone: {without:.4f}")
    assert abs(with_det - 1) < 1e-3 and abs(without - 2) < 2e-3

    # 4. Rotational ambiguity: under a Gaussian prior l(RW) = l(W) for every rotation R; under the logistic it is not.
    S, X = ica_data()
    W0 = inv(MIX)
    for deg in (10, 30, 45, 60):
        t = math.radians(deg)
        R = [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
        RW = [[sum(R[i][k] * W0[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        dg = loglik(RW, X, log_gauss) - loglik(W0, X, log_gauss)
        dl = (loglik(RW, X, log_sigmoid_density) - loglik(W0, X, log_sigmoid_density)) / len(X)
        assert abs(dg) < 1e-7 * len(X)
        if deg == 45:
            print(f"rotating A^-1 by 45 deg: Gaussian l changes by {dg:.1e}; logistic l/m changes by {dl:+.4f}")
        assert dl < 0

    # 5. The gradient formula against central differences.
    Wt = [[0.9, -0.3], [0.2, 1.1]]
    G = grad(Wt, X[:200])
    eps = 1e-6
    worst = 0.0
    for j in range(2):
        for k in range(2):
            Wp = [r[:] for r in Wt]
            Wm = [r[:] for r in Wt]
            Wp[j][k] += eps
            Wm[j][k] -= eps
            fd = (loglik(Wp, X[:200], log_sigmoid_density) - loglik(Wm, X[:200], log_sigmoid_density)) / (2 * eps * 200)
            worst = max(worst, abs(fd - G[j][k]))
    print(f"gradient vs finite differences: max abs difference {worst:.1e}")
    assert worst < 1e-6

    # 6. Gradient ascent recovers the sources up to permutation and scale.
    W, trace = run_ica(X)
    P = [[sum(W[i][k] * MIX[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    cross = max(min(abs(P[i][0]), abs(P[i][1])) / max(abs(P[i][0]), abs(P[i][1])) for i in range(2))
    print(f"ICA: W A = [[{P[0][0]:.3f}, {P[0][1]:.3f}], [{P[1][0]:.3f}, {P[1][1]:.3f}]]; worst cross-talk {cross:.3f}")
    print(
        f"log-likelihood per example: {trace[0]:.4f} -> {trace[-1]:.4f}; change over the last 100 steps {trace[-1] - trace[-101]:.1e}"
    )
    assert cross < 0.05
    corr = []
    for i in range(2):
        y = [matvec(W, x)[i] for x in X]
        best = max(
            abs(sum(a * b[j] for a, b in zip(y, S))) / math.sqrt(sum(a * a for a in y) * sum(b[j] ** 2 for b in S))
            for j in range(2)
        )
        corr.append(best)
    print(f"|correlation| of each recovered signal with its best-matching source: {corr[0]:.4f}, {corr[1]:.4f}")

    # 7. More microphones than sources: three microphones, two sources. A row orthogonal to the
    #    column space of A outputs zero for every example, and scaling it by c adds m log c to l.
    A3 = [[1.0, 0.5], [0.3, 1.0], [0.7, 0.8]]
    X3 = [matvec(A3, v) for v in S[:300]]
    n3 = [
        A3[1][0] * A3[2][1] - A3[2][0] * A3[1][1],
        A3[2][0] * A3[0][1] - A3[0][0] * A3[2][1],
        A3[0][0] * A3[1][1] - A3[1][0] * A3[0][1],
    ]
    base = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], n3]
    l1 = loglik(base, X3, log_sigmoid_density)
    for c in (10.0, 1000.0):
        Wc = [base[0], base[1], [c * v for v in n3]]
        gain = loglik(Wc, X3, log_sigmoid_density) - l1
        assert abs(gain - len(X3) * math.log(c)) < 1e-6 * len(X3)
    print(f"3 mics, 2 sources: scaling the silent row by c raises l by exactly m log c (m = {len(X3)}); unbounded")

    # 8. The gridworld: Ng's board values, the decision at (3, 1), and how much reward it takes to flip it.
    V = value_iteration(-0.02, 0.99)
    pi = greedy(V)
    print("V*: " + ", ".join(f"{s}: {V[s]:.3f}" for s in STATES))
    q31 = {a: q_future(V, (3, 1), a) for a in ACTIONS}
    print("at (3,1), sum P V*: " + ", ".join(f"{a} {q:.3f}" for a, q in q31.items()))
    rounded = {(2, 1): 0.75, (3, 2): 0.69, (3, 1): 0.71, (4, 1): 0.49}
    assert all(abs(V[s] - v) < 0.005 for s, v in rounded.items())
    assert pi[(3, 1)] == "W"
    for s in ((3, 2), (4, 1)):
        qs = sorted((q_future(V, s, a), a) for a in ACTIONS)
        print(f"at {s}: best {qs[-1][1]} {qs[-1][0]:.4f}, runner-up {qs[-2][1]} {qs[-2][0]:.4f}")

    def flip(state, lo, hi):
        """Bisect on the step reward for where pi*(state) stops being West."""
        for _ in range(40):
            mid = (lo + hi) / 2
            if greedy(value_iteration(mid, 0.99, 1e-11))[state] == "W":
                lo = mid
            else:
                hi = mid
        return lo

    print(f"pi*(3,1) stays West for step rewards down to {flip((3, 1), -0.02, -0.06):.4f} (gamma 0.99)")
    print(f"pi*(3,2) is North below a step reward of {flip((3, 2), 0.0, -0.02):.4f}")
    print(f"policies: 4^9 = {4**9} over the nine non-absorbing states (4^11 = {4**11})")
    print(f"payoff bound for a step reward of -0.02 at gamma 0.99: {-0.02 / (1 - 0.99):.1f}")

    FIGURES.update(
        {
            "diagram_source_densities.svg": fig_source_densities(),
            "diagram_change_of_variables.svg": fig_change_of_variables(),
            "diagram_ica_ml.svg": fig_ica_ml(X, W, trace),
            "diagram_gridworld.svg": fig_gridworld(pi),
        }
    )
    for name, svg in FIGURES.items():
        (OUT / name).write_text(svg)
        print("wrote", name)
