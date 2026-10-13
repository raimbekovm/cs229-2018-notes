"""Figures for lecture 7: the dual, kernels, and the soft margin.

Every figure is computed by one small SVM solver — sequential minimal
optimisation on the dual, choosing the maximally violating pair at each step —
run on synthetic data: a one-dimensional set made separable by the lift
x ↦ (x, x²), two rings separated by a Gaussian-kernel SVM, and a separable set
with one outlier, solved with a hard and with a soft margin.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: the dual solution against lecture 6's
independent search over directions on the same data; the dual objective
against ½‖w‖² (strong duality); stochastic gradient descent's iterate against
its projection onto the span of the training inputs; each polynomial kernel
against the inner product of its explicit feature vectors, and the number of
those features against an enumeration; the Gaussian kernel against its
truncated feature expansion; the eigenvalues of several kernel matrices,
including one "similarity" that is not a kernel; the spectrum kernel computed
from k-mer counts against the explicit 160,000-dimensional inner product; and
the soft-margin slacks against the hinge loss.

Plain Python on purpose, as in lectures 1–6.

Run from the repository root.
"""

import importlib.util
import itertools
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

# Lecture 6's generator holds the data set, the independent max-margin search
# and the contour tracer; load it under its own name rather than copying them.
_spec = importlib.util.spec_from_file_location(
    "lecture06", OUT.parent / "06-support-vector-machines" / "make_diagrams.py"
)
L06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L06)
dot, contour = L06.dot, L06.contour


# ---------------------------------------------------------------- the solver


def linear_kernel(x, z):
    """K(x, z) = xᵀz."""
    return dot(x, z)


def gaussian_kernel(sigma):
    """K(x, z) = exp(−‖x − z‖² / 2σ²)."""
    return lambda x, z: math.exp(-sum((a - b) ** 2 for a, b in zip(x, z)) / (2 * sigma**2))


def smo(xs, ys, kernel, c=1e6, tol=1e-9, max_iter=200000):
    """Solve the SVM dual max Σα − ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ), 0 ≤ α ≤ C, Σ αᵢyᵢ = 0.

    Works on the equivalent minimisation f(α) = ½αᵀQα − Σα with Qᵢⱼ = yᵢyⱼKᵢⱼ, whose gradient is G. Each step moves the
    pair that most violates the optimality conditions along the one direction that keeps Σ αᵢyᵢ fixed.
    """
    m = len(xs)
    k = [[kernel(a, b) for b in xs] for a in xs]
    alpha, g = [0.0] * m, [-1.0] * m
    for _ in range(max_iter):
        up = [t for t in range(m) if (ys[t] == 1 and alpha[t] < c) or (ys[t] == -1 and alpha[t] > 0)]
        low = [t for t in range(m) if (ys[t] == 1 and alpha[t] > 0) or (ys[t] == -1 and alpha[t] < c)]
        i = max(up, key=lambda t: -ys[t] * g[t])
        j = min(low, key=lambda t: -ys[t] * g[t])
        if -ys[i] * g[i] + ys[j] * g[j] < tol:
            break
        slope = ys[i] * g[i] - ys[j] * g[j]
        curv = max(k[i][i] + k[j][j] - 2 * k[i][j], 1e-12)
        t = -slope / curv
        t = min(t, c - alpha[i] if ys[i] == 1 else alpha[i], alpha[j] if ys[j] == 1 else c - alpha[j])
        alpha[i] += t * ys[i]
        alpha[j] -= t * ys[j]
        for s in range(m):
            g[s] += t * ys[s] * (k[s][i] - k[s][j])
    free = [t for t in range(m) if 1e-8 < alpha[t] < c - 1e-8]
    if free:
        b = sum(-ys[t] * g[t] for t in free) / len(free)
    else:
        b = (max(-ys[t] * g[t] for t in up) + min(-ys[t] * g[t] for t in low)) / 2
    return alpha, b, k


def decision(alpha, b, xs, ys, kernel):
    """f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b, using only the examples with αᵢ > 0."""
    sv = [(a, y, x) for a, y, x in zip(alpha, ys, xs) if a > 1e-10]
    return lambda z: sum(a * y * kernel(x, z) for a, y, x in sv) + b


def primal_w(alpha, xs, ys):
    """W = Σ αᵢ yᵢ xᵢ, for the linear kernel."""
    return [sum(a * y * x[d] for a, y, x in zip(alpha, ys, xs)) for d in range(len(xs[0]))]


def dual_objective(alpha, ys, k):
    """W(α) = Σ αᵢ − ½ ΣΣ αᵢαⱼyᵢyⱼKᵢⱼ."""
    m = len(alpha)
    quad = sum(alpha[i] * alpha[j] * ys[i] * ys[j] * k[i][j] for i in range(m) for j in range(m))
    return sum(alpha) - quad / 2


# ---------------------------------------------------------------- linear algebra


def eigvals_sym(a, sweeps=60):
    """Eigenvalues of a symmetric matrix by cyclic Jacobi rotations."""
    n = len(a)
    a = [row[:] for row in a]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-22:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta * theta + 1))
                cs = 1 / math.sqrt(t * t + 1)
                sn = t * cs
                for r in range(n):
                    arp, arq = a[r][p], a[r][q]
                    a[r][p], a[r][q] = cs * arp - sn * arq, sn * arp + cs * arq
                for r in range(n):
                    apr, aqr = a[p][r], a[q][r]
                    a[p][r], a[q][r] = cs * apr - sn * aqr, sn * apr + cs * aqr
    return sorted(a[i][i] for i in range(n))


def gram(points, kernel):
    """The kernel matrix Kᵢⱼ = K(xᵢ, xⱼ)."""
    return [[kernel(a, b) for b in points] for a in points]


# ---------------------------------------------------------------- feature maps


def phi_square(x):
    """All n² ordered products xᵢxⱼ: the features of K(x, z) = (xᵀz)²."""
    return [a * b for a in x for b in x]


def phi_square_c(x, c):
    """The features of (xᵀz + c)²: the products xᵢxⱼ, then √(2c)·xᵢ, then c."""
    return phi_square(x) + [math.sqrt(2 * c) * a for a in x] + [c]


def phi_poly(x, c, d):
    """Explicit features of (xᵀz + c)ᵈ: one per monomial of degree ≤ d, weighted by a multinomial coefficient."""
    n = len(x)
    feats = []
    for powers in itertools.product(range(d + 1), repeat=n):
        k = sum(powers)
        if k > d:
            continue
        coef = math.factorial(d) / (math.factorial(d - k) * math.prod(math.factorial(p) for p in powers))
        feats.append(math.sqrt(coef * c ** (d - k)) * math.prod(a**p for a, p in zip(x, powers)))
    return feats


def phi_gauss_1d(x, sigma, terms):
    """The first `terms` features of the one-dimensional Gaussian kernel: e^(−x²/2σ²) xᵏ / (σᵏ √k!)."""
    return [math.exp(-(x**2) / (2 * sigma**2)) * x**k / (sigma**k * math.sqrt(math.factorial(k))) for k in range(terms)]


AMINO = "ACDEFGHIKLMNPQRSTVWY"


def kmer_counts(seq, k=4):
    """Number of occurrences of each length-k substring."""
    out = {}
    for i in range(len(seq) - k + 1):
        out[seq[i : i + k]] = out.get(seq[i : i + k], 0) + 1
    return out


def spectrum_kernel(s, t, k=4):
    """K(s, t) = Σ over k-mers of count_s · count_t, touching only the k-mers that occur."""
    cs, ct = kmer_counts(s, k), kmer_counts(t, k)
    return sum(v * ct.get(w, 0) for w, v in cs.items())


def spectrum_explicit(s, k=4):
    """The explicit 20ᵏ-dimensional count vector, one entry per possible k-mer in alphabetical order."""
    index = {a: i for i, a in enumerate(AMINO)}
    vec = [0] * len(AMINO) ** k
    for i in range(len(s) - k + 1):
        pos = 0
        for ch in s[i : i + k]:
            pos = pos * len(AMINO) + index[ch]
        vec[pos] += 1
    return vec


# ---------------------------------------------------------------- data


def lift_data():
    """One-dimensional points: the positive class sits between the negatives on both sides."""
    xs = [-2.6, -2.2, -1.9, -1.6, -0.9, -0.6, -0.2, 0.3, 0.7, 1.0, 1.7, 2.0, 2.4, 2.8]
    ys = [-1 if abs(x) > 1.3 else 1 for x in xs]
    return xs, ys


def rings(seed=4, m=110):
    """An inner disc of positives inside an annulus of negatives, with a little label noise."""
    rng = random.Random(seed)
    xs, ys = [], []
    for i in range(m):
        y = 1 if i % 2 else -1
        r = abs(rng.gauss(0.9, 0.45)) if y == 1 else rng.gauss(2.3, 0.4)
        a = rng.uniform(0, 2 * math.pi)
        xs.append((r * math.cos(a) * 1.15, r * math.sin(a)))
        ys.append(y)
    return xs, ys


OUTLIER = (1.15, 1.25)


def outlier_data():
    """Lecture 6's separable set with one positive example placed close to the negatives."""
    xs, ys = L06.separable_data()
    return [*xs, OUTLIER], [*ys, 1]


# ---------------------------------------------------------------- figures


def fig_lift():
    """A one-dimensional set no threshold separates, lifted to (x, x²) where a line does."""
    xs, ys = lift_data()
    feats = [(x, x * x) for x in xs]
    alpha, b, _ = smo(feats, ys, linear_kernel)
    w = primal_w(alpha, feats, ys)
    roots = sorted(r for r in ((-w[0] + sg * math.sqrt(w[0] ** 2 - 4 * w[1] * b)) / (2 * w[1]) for sg in (1, -1)))
    s = ""
    to_px = mapper((-3, 3), (0, 1), (40, 300), (200, 120))
    y0 = to_px(0, 0.5)[1]
    s += line(40, y0, 300, y0, op=0.7)
    for t in (-2, -1, 0, 1, 2):
        px = to_px(t, 0)[0]
        s += line(px, y0, px, y0 + 4, op=0.7) + text(px, y0 + 18, f"{t:g}", size=11, op=0.8)
    s += text(170, y0 + 40, "x", size=12)
    for x, y in zip(xs, ys):
        px = to_px(x, 0)[0]
        s += (
            circle(px, y0, 4, fill=TEAL, stroke=TEAL)
            if y == 1
            else rect(px - 3.6, y0 - 3.6, 7.2, 7.2, stroke=PURPLE, sw=1.4)
        )
    for r in roots:
        px = to_px(r, 0)[0]
        s += line(px, y0 - 22, px, y0 + 6, stroke=INDIGO, sw=2.2)
    s += text(170, 70, "one feature: no single threshold", size=13, weight="600")
    s += text(170, 88, "separates the classes", size=13, weight="600")
    s += text(170, y0 - 30, f"boundary pulled back: x = {roots[0]:.2f} and {roots[1]:.2f}".replace("-", "−"), size=11, fill=INDIGO)

    xdom, ydom = (-3.0, 3.0), (-0.5, 8.5)
    to_px = mapper(xdom, ydom, (420, 690), (40, 300))
    s += axes(to_px, xdom, ydom, [-2, 0, 2], [0, 2, 4, 6, 8], "φ₁(x) = x", "φ₂(x) = x²")
    grid = [xdom[0] + (xdom[1] - xdom[0]) * i / 200 for i in range(201)]
    s += curve([(t, t * t) for t in grid], to_px, xdom, ydom, stroke="currentColor", sw=1.0, op=0.35)
    for off, dash, sw in ((0, None, 2.2), (1, "5 4", 1.2), (-1, "5 4", 1.2)):
        s += curve([(t, (off - b - w[0] * t) / w[1]) for t in grid], to_px, xdom, ydom, stroke=INDIGO, sw=sw, dash=dash)
    for (u, v), y, a in zip(feats, ys, alpha):
        px, py = to_px(u, v)
        s += (
            circle(px, py, 4, fill=TEAL, stroke=TEAL)
            if y == 1
            else rect(px - 3.6, py - 3.6, 7.2, 7.2, stroke=PURPLE, sw=1.4)
        )
        if a > 1e-8:
            s += circle(px, py, 9, sw=1.2)
    s += text(555, 22, "lifted to (x, x²): a straight line separates", size=13, weight="600")
    return wrap(
        720,
        350,
        "figLift",
        "Lifting one-dimensional data into two dimensions",
        "Left: points on a line, positives in the middle and negatives on both sides, so no single threshold separates them. Right: the same points plotted at (x, x squared) lie on a parabola, and a straight maximum-margin line now separates the middle points from the outer ones; the support vectors are circled. Pulled back to the line, the straight boundary becomes two thresholds, marked on the left panel.",
        s,
    )


SIGMA, C_RBF = 0.8, 10.0


def fig_rbf_boundary():
    """A Gaussian-kernel SVM on two rings: its boundary, margins and support vectors."""
    xs, ys = rings()
    kern = gaussian_kernel(SIGMA)
    alpha, b, _ = smo(xs, ys, kern, c=C_RBF)
    f = decision(alpha, b, xs, ys, kern)
    dom = (-3.6, 3.6)
    to_px = mapper(dom, dom, (70, 380), (30, 340))
    s = axes(to_px, dom, dom, [-3, 0, 3], [-3, 0, 3], "x₁", "x₂")
    for off, dash, sw in ((0, None, 2.2), (1, "5 4", 1.1), (-1, "5 4", 1.1)):
        for run in contour(lambda a, c, off=off: f((a, c)) - off, dom, dom, n=90):
            s += curve(run, to_px, dom, dom, stroke=INDIGO, sw=sw, dash=dash)
    for x, y, a in zip(xs, ys, alpha):
        px, py = to_px(*x)
        s += (
            circle(px, py, 3.6, fill=TEAL, stroke=TEAL)
            if y == 1
            else rect(px - 3.2, py - 3.2, 6.4, 6.4, stroke=PURPLE, sw=1.3)
        )
        if a > 1e-8:
            s += circle(px, py, 8, sw=1.0, op=0.8)
    nsv = sum(a > 1e-8 for a in alpha)
    lx = 440
    s += line(lx, 110, lx + 28, 110, stroke=INDIGO, sw=2.2) + text(lx + 36, 114, "f(x) = 0", size=12, anchor="start")
    s += line(lx, 134, lx + 28, 134, stroke=INDIGO, sw=1.1, dash="5 4") + text(
        lx + 36, 138, "f(x) = ±1", size=12, anchor="start"
    )
    s += circle(lx + 14, 158, 8, sw=1.0) + text(
        lx + 36, 162, f"support vectors: {nsv} of {len(xs)}", size=12, anchor="start"
    )
    s += text(lx, 206, "f(x) = Σᵢ αᵢ y⁽ⁱ⁾ K(x⁽ⁱ⁾, x) + b", size=12, anchor="start")
    s += text(lx, 228, f"Gaussian kernel, σ = {SIGMA:g}, C = {C_RBF:g}", size=12, anchor="start", op=0.85)
    return wrap(
        720,
        380,
        "figRbfBoundary",
        "A Gaussian-kernel SVM separating two rings",
        "A disc of positive points surrounded by a ring of negative points. A closed curve, the SVM's decision boundary, runs between them, with dashed curves on either side where the decision function equals plus and minus one. The circled points, the support vectors, are the ones on or inside those dashed curves.",
        s,
    )


C_SOFT = 0.1


def fig_soft_margin():
    """One outlier swings the hard-margin separator; a soft margin with small C barely moves."""
    xs, ys = outlier_data()
    x0, y0 = L06.separable_data()
    s = ""
    dom = (-1.0, 4.0)
    grid = [dom[0] + (dom[1] - dom[0]) * i / 300 for i in range(301)]
    for k, (title, c) in enumerate((("hard margin", 1e6), (f"soft margin, C = {C_SOFT:g}", C_SOFT))):
        alpha, b, _ = smo(xs, ys, linear_kernel, c=c)
        w = primal_w(alpha, xs, ys)
        a0, b0, _ = smo(x0, y0, linear_kernel, c=c)
        w0 = primal_w(a0, x0, y0)
        ox = 50 + k * 360
        to_px = mapper(dom, dom, (ox, ox + 280), (40, 320))
        s += axes(to_px, dom, dom, [0, 2, 4], [0, 2, 4], "x₁", "x₂")
        s += curve([(t, (-b0 - w0[0] * t) / w0[1]) for t in grid], to_px, dom, dom, stroke=GREEN, sw=2.0, dash="2 3")
        for off, dash, sw in ((0, None, 2.2), (1, "5 4", 1.1), (-1, "5 4", 1.1)):
            s += curve([(t, (off - b - w[0] * t) / w[1]) for t in grid], to_px, dom, dom, stroke=INDIGO, sw=sw, dash=dash)
        for x, y, a in zip(xs, ys, alpha):
            px, py = to_px(*x)
            s += circle(px, py, 4, fill=TEAL, stroke=TEAL) if y == 1 else rect(px - 3.6, py - 3.6, 7.2, 7.2, stroke=PURPLE, sw=1.4)
            if a > 1e-8:
                s += circle(px, py, 9, sw=1.1, op=0.85)
        s += text(ox + 140, 24, title, size=13, weight="600")
        s += text(ox + 140, 44, f"outlier turns the boundary {abs(angle_deg(w) - angle_deg(w0)):.0f}°", size=11, op=0.85)
    px, py = mapper(dom, dom, (50, 330), (40, 320))(*OUTLIER)
    s += text(px - 12, py - 10, "outlier", size=11, anchor="end")
    s += line(160, 392, 188, 392, stroke=INDIGO, sw=2.2) + text(194, 396, "with the outlier", size=11, anchor="start")
    s += line(360, 392, 388, 392, stroke=GREEN, sw=2.0, dash="2 3") + text(394, 396, "same C, outlier removed", size=11, anchor="start")
    return wrap(
        720,
        410,
        "figSoftMargin",
        "Hard-margin and soft-margin SVMs on data with one outlier",
        "The same separable two-class data with one extra positive point placed close to the negatives. Left: the hard-margin separator tilts sharply to squeeze between the outlier and the negatives, leaving a narrow margin, far from where it sits without the outlier, drawn dotted. Right: with a soft margin and C equal to 0.1 the separator lies almost on top of its outlier-free position, and the outlier simply sits inside the margin.",
        s,
    )


FIGURES = {
    "diagram_lift.svg": fig_lift,
    "diagram_rbf_boundary.svg": fig_rbf_boundary,
    "diagram_soft_margin.svg": fig_soft_margin,
}


def angle_deg(w):
    """Direction of a two-dimensional normal vector, in degrees."""
    return math.degrees(math.atan2(w[1], w[0]))


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    # 1. The dual solved by SMO agrees with lecture 6's search over directions.
    xs, ys = L06.separable_data()
    alpha, b, k = smo(xs, ys, linear_kernel)
    w = primal_w(alpha, xs, ys)
    w6, b6, gam6 = L06.max_margin(xs, ys)
    assert max(abs(p - q) for p, q in zip(w, w6)) < 1e-5 and abs(b - b6) < 1e-5
    fm = [L06.functional_margin(w, b, x, y) for x, y in zip(xs, ys)]
    sv = [i for i, a in enumerate(alpha) if a > 1e-8]
    assert all(abs(fm[i] - 1) < 1e-6 for i in sv) and min(fm[i] for i in range(len(xs)) if i not in sv) > 1 + 1e-6
    half_norm = dot(w, w) / 2
    assert abs(dual_objective(alpha, ys, k) - half_norm) < 1e-6
    assert abs(sum(a * y for a, y in zip(alpha, ys))) < 1e-9
    print(
        f"dual: w = ({w[0]:.4f}, {w[1]:.4f}), b = {b:.4f}, matching the direction search; γ = 1/‖w‖ = {1 / math.sqrt(2 * half_norm):.4f}; "
        f"{len(sv)} of {len(xs)} α nonzero, all on examples with functional margin 1; W(α) = ½‖w‖² = {half_norm:.4f}"
    )

    # 2. Representer intuitions. SGD from zero stays in the span of the inputs.
    rng = random.Random(9)
    pts = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(3)]
    lab = [1, 0, 1]
    th = [0.0] * 5
    for step in range(300):
        i = step % 3
        p = L06.sigmoid(dot(th, pts[i]))
        th = [t + 0.1 * (lab[i] - p) * xj for t, xj in zip(th, pts[i])]
    gm = [[dot(a, c) for c in pts] for a in pts]
    coef = L06.solve(gm, [dot(th, a) for a in pts])
    resid = math.dist(th, [sum(cf * a[d] for cf, a in zip(coef, pts)) for d in range(5)])
    assert resid < 1e-10
    print(f"representer: after 300 SGD steps in R⁵ on 3 examples, distance from θ to span(x⁽ⁱ⁾) = {resid:.1e}")
    # ... and the x₃ = 0 argument is a property of the minimum-norm solution, not of every separator.
    xs3 = [(*x, 0.0) for x in xs]
    a3, b3, _ = smo(xs3, ys, linear_kernel)
    w3 = primal_w(a3, xs3, ys)
    assert w3[2] == 0.0
    tilted = [w6[0], w6[1], 5.0]
    assert all(y * (dot(tilted, x) + b6) >= 1 - 1e-9 for x, y in zip(xs3, ys))
    print(
        f"representer: with x₃ = 0 the SVM gives w₃ = {w3[2]}; w = (w₁, w₂, 5) separates equally well but has larger norm"
    )

    # 3. Polynomial kernels are inner products of explicit features.
    prng = random.Random(1)
    for _ in range(200):
        x = [prng.gauss(0, 1) for _ in range(4)]
        z = [prng.gauss(0, 1) for _ in range(4)]
        c = prng.uniform(0, 3)
        assert abs(dot(x, z) ** 2 - dot(phi_square(x), phi_square(z))) < 1e-9
        assert abs((dot(x, z) + c) ** 2 - dot(phi_square_c(x, c), phi_square_c(z, c))) < 1e-9
        assert abs((dot(x, z) + c) ** 3 - dot(phi_poly(x, c, 3), phi_poly(z, c, 3))) < 1e-8
    for n, d in ((3, 2), (4, 3), (3, 5)):
        assert len(phi_poly([1.0] * n, 1.0, d)) == math.comb(n + d, d)
    big = math.comb(10005, 5)
    print(
        f"poly: (xᵀz)², (xᵀz + c)² and (xᵀz + c)³ match explicit features on 200 draws; feature count = C(n+d, d) by enumeration; "
        f"n = 10⁴, d = 5: C = {big:.3e}, (n+d)ᵈ = {10005**5:.3e}, ratio {10005**5 / big:.1f} (≈ d! = 120)"
    )

    # 4. Which functions are kernels.
    gpts = [[prng.uniform(-2, 2), prng.uniform(-2, 2)] for _ in range(25)]
    ev_gauss = eigvals_sym(gram(gpts, gaussian_kernel(1.0)))
    assert ev_gauss[0] > -1e-10
    ball = eigvals_sym([[1.0 if abs(a - b) < 1.5 else 0.0 for b in (0, 1, 2)] for a in (0, 1, 2)])
    assert abs(ball[0] - (1 - math.sqrt(2))) < 1e-12
    vz = [1.0, -math.sqrt(2), 1.0]
    kb = [[1.0 if abs(a - b) < 1.5 else 0.0 for b in (0, 1, 2)] for a in (0, 1, 2)]
    quad_form = sum(vz[i] * kb[i][j] * vz[j] for i in range(3) for j in range(3))
    hists = [[prng.randint(0, 9) for _ in range(6)] for _ in range(20)]
    ev_hist = eigvals_sym(gram(hists, lambda h, g: sum(min(a, b) for a, b in zip(h, g))))
    assert ev_hist[0] > -1e-9
    print(
        f"mercer: Gaussian Gram on 25 points, smallest eigenvalue {ev_gauss[0]:.2e} ≥ 0; histogram intersection on 20 histograms {ev_hist[0]:.3f} ≥ 0; "
        f"'within 1.5' similarity at x = 0, 1, 2 has eigenvalues {[round(e, 4) for e in ball]}, zᵀKz = {quad_form:.4f} < 0 for z = (1, −√2, 1)"
    )
    for terms in (3, 6, 12, 20):
        err = max(
            abs(dot(phi_gauss_1d(x, 1.0, terms), phi_gauss_1d(z, 1.0, terms)) - gaussian_kernel(1.0)([x], [z]))
            for x in (-1.5, -0.4, 0.3, 1.2)
            for z in (-1.0, 0.0, 0.8, 1.5)
        )
        print(f"gaussian: first {terms} features reproduce the kernel to within {err:.1e}")
    assert err < 1e-10

    # 5. Soft margin: the outlier, the slacks and the box constraint.
    xs_o, ys_o = outlier_data()
    ah, bh, _ = smo(xs_o, ys_o, linear_kernel, c=1e6)
    wh = primal_w(ah, xs_o, ys_o)
    assert min(L06.functional_margin(wh, bh, x, y) for x, y in zip(xs_o, ys_o)) > 1 - 1e-6
    a_s, b_s, _ = smo(xs_o, ys_o, linear_kernel, c=C_SOFT)
    ws = primal_w(a_s, xs_o, ys_o)
    assert all(-1e-12 <= a <= C_SOFT + 1e-12 for a in a_s)
    xi = [max(0.0, 1 - L06.functional_margin(ws, b_s, x, y)) for x, y in zip(xs_o, ys_o)]
    for a, x, y, e in zip(a_s, xs_o, ys_o, xi):
        f_m = L06.functional_margin(ws, b_s, x, y)
        if 1e-6 < a < C_SOFT - 1e-6:
            assert abs(f_m - 1) < 1e-5
        if e > 1e-6:
            assert a > C_SOFT - 1e-6
    print(
        f"soft: no outlier, normal at {angle_deg(w6):.1f}°, γ = {gam6:.3f}; hard margin with outlier {angle_deg(wh):.1f}°, γ = {1 / math.hypot(*wh):.3f}; "
        f"soft (C = {C_SOFT:g}) {angle_deg(ws):.1f}°, 1/‖w‖ = {1 / math.hypot(*ws):.3f}; slacks ξ > 0 on {sum(e > 1e-6 for e in xi)} examples, "
        f"outlier ξ = {xi[-1]:.3f}; all 0 ≤ α ≤ C"
    )

    for c in (1.0, 0.3, C_SOFT):
        turns = []
        for dx, dy in ((xs, ys), (xs_o, ys_o)):
            a_, b_, _ = smo(dx, dy, linear_kernel, c=c)
            turns.append(angle_deg(primal_w(a_, dx, dy)))
        print(f"soft: C = {c:g}: outlier turns the normal from {turns[0]:.1f}° to {turns[1]:.1f}° ({turns[1] - turns[0]:+.1f}°)")

    # 6. The lift, the rings.
    lx, ly = lift_data()
    la, lb, _ = smo([(x, x * x) for x in lx], ly, linear_kernel)
    lw = primal_w(la, [(x, x * x) for x in lx], ly)
    print(f"lift: w = ({lw[0]:.3f}, {lw[1]:.3f}), b = {lb:.3f}; {sum(a > 1e-8 for a in la)} support vectors")
    rx, ry = rings()
    kern = gaussian_kernel(SIGMA)
    ra, rb, _ = smo(rx, ry, kern, c=C_RBF)
    f = decision(ra, rb, rx, ry, kern)
    acc = sum((f(x) >= 0) == (y == 1) for x, y in zip(rx, ry)) / len(rx)
    print(f"rings: training accuracy {acc:.3f}; {sum(a > 1e-8 for a in ra)} of {len(rx)} support vectors")

    # 7. The spectrum kernel: count dictionaries against the explicit vector.
    srng = random.Random(3)
    s1 = "".join(srng.choice(AMINO) for _ in range(60))
    s2 = s1[10:40] + "".join(srng.choice(AMINO) for _ in range(30))
    explicit = dot(spectrum_explicit(s1), spectrum_explicit(s2))
    assert spectrum_kernel(s1, s2) == explicit
    print(
        f"spectrum: 20⁴ = {20**4:,} features (26⁴ = {26**4:,}); K computed from k-mer counts = explicit inner product = {explicit}"
    )
