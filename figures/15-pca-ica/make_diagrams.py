"""Figures for lecture 15: principal components analysis and the ICA problem.

Four computed figures: projections of standardised two-dimensional data onto the
principal direction and onto the worst direction; what PCA finds when the data
are not centred and when they are not scaled; how stable the first principal
direction and the first two-dimensional principal subspace are across two
samples from the same distribution, as the top two eigenvalues approach each
other; and the ICA picture: independent uniform sources, their linear mixture,
and a recovery that is correct up to order and sign.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: that for every unit direction the
projected variance plus the mean squared projection distance is constant (so
maximising one minimises the other); that the stationary points of u^T Sigma u
on the unit circle are the eigenvectors and the values there are the
eigenvalues; that the mean squared reconstruction error of a k-dimensional
PCA equals the sum of the discarded eigenvalues; that an uncentred PCA points at
the mean and an unscaled one at the larger-unit feature; that a perturbation in
the fifth decimal place moves the first eigenvector only when the top two
eigenvalues are nearly tied; and that permuted, sign-flipped and rescaled
unmixing matrices give sources that are exactly as independent.

Plain Python on purpose, as in lectures 1-14; eigenvectors come from a small
Jacobi routine below.

Run from the repository root.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _linalg import dot, eigh, mean
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
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------- linear algebra


def centre(X):
    """Subtract the column means."""
    mu = mean(X)
    return [[a - b for a, b in zip(x, mu)] for x in X]


def standardise(X):
    """Centre, then divide each feature by its standard deviation."""
    Xc = centre(X)
    sd = [math.sqrt(sum(v * v for v in c) / len(Xc)) for c in zip(*Xc)]
    return [[a / s for a, s in zip(x, sd)] for x in Xc]


def second_moment(X):
    """(1/m) sum x x^T; the covariance when X is centred."""
    n, m = len(X[0]), len(X)
    return [[sum(x[i] * x[j] for x in X) / m for j in range(n)] for i in range(n)]


def angle_deg(u, v):
    """Angle between two lines (sign-free), in degrees."""
    c = abs(dot(u, v)) / math.sqrt(dot(u, u) * dot(v, v))
    return math.degrees(math.acos(min(1.0, c)))


def subspace_angle_deg(U, V):
    """Largest principal angle between the spans of two orthonormal sets, in degrees."""
    M = [[dot(u, v) for v in V] for u in U]
    MtM = [[sum(M[k][i] * M[k][j] for k in range(len(M))) for j in range(len(V))] for i in range(len(V))]
    vals, _ = eigh(MtM)
    return math.degrees(math.acos(min(1.0, math.sqrt(max(min(vals), 0.0)))))


# ---------------------------------------------------------------- data


def heights():
    """Children's heights in centimetres and inches, each rounded to a whole unit."""
    rng = random.Random(2)
    out = []
    for _ in range(60):
        h = rng.gauss(125, 12)
        out.append([round(h), round(h / 2.54)])
    return out


def polygon_data(gap):
    """A regular 12-gon in the x1-x2 plane plus two off-plane points; eigenvalues 0.5 (1 + gap) and 0.5."""
    pts = [[math.sqrt(1 + gap) * math.cos(math.pi * k / 6), math.sin(math.pi * k / 6), 0.0] for k in range(12)]
    return pts + [[0.0, 0.0, 0.3], [0.0, 0.0, -0.3]]


def plane_data(ratio, m, rng, n=3):
    """M points in R^3 with variances 1, ratio and 0.01 along fixed orthogonal axes."""
    sd = [1.0, math.sqrt(ratio), 0.1]
    return [[rng.gauss(0, s) for s in sd] for _ in range(m)]


# ---------------------------------------------------------------- figures


def cross(px, py, col, s=3.5, sw=1.6):
    """An x-shaped marker."""
    return line(px - s, py - s, px + s, py + s, stroke=col, sw=sw) + line(
        px - s, py + s, px + s, py - s, stroke=col, sw=sw
    )


def fig_projection():
    """Two panels: projections onto the principal direction, and onto the orthogonal one."""
    Z = standardise(heights())
    _, vecs = eigh(second_moment(Z))
    s = ""
    r = math.ceil(max(abs(v) for z in Z for v in z) * 10 + 2) / 10
    dom = (-r, r)
    for k, (u, col, title) in enumerate(
        (
            (vecs[0], TEAL, "onto u₁: spread large, distances small"),
            (vecs[1], PURPLE, "onto u₂: spread small, distances large"),
        )
    ):
        x0 = 40 + k * 340
        to_px = mapper(dom, dom, (x0, x0 + 260), (40, 300))
        s += rect(x0, 40, 260, 260, so=0.4, sw=0.8)
        s += text(x0 + 130, 28, title, size=11.5)
        s += curve([(t * u[0] / 50, t * u[1] / 50) for t in range(-200, 201)], to_px, dom, dom, stroke=col, sw=1.6)
        for x in Z:
            c = dot(x, u)
            p = (c * u[0], c * u[1])
            s += line(*to_px(*x), *to_px(*p), stroke=col, sw=0.8, op=0.55)
        for x in Z:
            c = dot(x, u)
            s += circle(*to_px(c * u[0], c * u[1]), 2.4, fill=col, stroke="none")
            s += cross(*to_px(*x), INDIGO)
    return wrap(
        680,
        315,
        "figPcaProj",
        "Projecting standardised data onto two directions",
        "Sixty children's heights, measured in centimetres and in inches and standardised, lie close to the "
        "45-degree line. Left: projected orthogonally onto that line, the projections spread widely and the "
        "projection segments are short. Right: projected onto the perpendicular line, the projections bunch "
        "near the origin and the segments are long.",
        s,
    )


def fig_preprocessing():
    """Two panels: uncentred data and unscaled data, with the direction PCA returns for each."""
    s = ""
    # panel 1: not centred
    rng = random.Random(9)
    X = []
    for _ in range(50):
        t = rng.gauss(0, 1.0)
        X.append([4.0 + 0.8 * t + rng.gauss(0, 0.15), 1.5 - 0.8 * t + rng.gauss(0, 0.15)])
    _, raw = eigh(second_moment(X))
    _, cen = eigh(second_moment(centre(X)))
    mu = mean(X)
    lo = math.floor(min(min(x) for x in X)) - 1.0
    hi = math.ceil(max(max(x) for x in X)) + 0.5
    dom = (min(lo, -1.0), hi)
    to_px = mapper(dom, dom, (60, 300), (40, 280))
    s += axes(to_px, dom, dom, (0, 2, 4, 6), (0, 2, 4, 6), "x₁", "x₂")
    for x in X:
        s += cross(*to_px(*x), INDIGO, s=2.8, sw=1.3)
    u = raw[0] if raw[0][0] > 0 else [-v for v in raw[0]]
    s += curve([(t * u[0] / 20, t * u[1] / 20) for t in range(0, 141)], to_px, dom, dom, stroke=PURPLE, sw=2.0)
    v = cen[0] if cen[0][0] > 0 else [-w for w in cen[0]]
    s += curve(
        [(mu[0] + t * v[0] / 20, mu[1] + t * v[1] / 20) for t in range(-50, 51)], to_px, dom, dom, stroke=TEAL, sw=2.0
    )
    s += text(180, 28, "not centred", size=11.5)
    px, py = to_px(0.3, 0.3 * u[1] / u[0])
    s += text(px, py + 22, "without centring", size=11, fill=PURPLE, anchor="start")
    px, py = to_px(mu[0] - 2.4 * v[0], mu[1] - 2.4 * v[1])
    s += text(px - 8, py, "centred", size=11, fill=TEAL, anchor="end")
    # panel 2: not scaled
    H = heights()
    _, raw2 = eigh(second_moment(centre(H)))
    _, std2 = eigh(second_moment(standardise(H)))
    Hc = centre(H)
    dom2 = (-40.0, 40.0)
    to2 = mapper(dom2, dom2, (420, 660), (40, 280))
    s += axes(to2, dom2, dom2, (-40, 0, 40), (-40, 0, 40), "height (cm), centred", "height (in), centred")
    for x in Hc:
        s += cross(*to2(*x), INDIGO, s=2.8, sw=1.3)
    for u, col in ((raw2[0], PURPLE), (std2[0], TEAL)):
        s += curve([(t * u[0] / 2, t * u[1] / 2) for t in range(-100, 101)], to2, dom2, dom2, stroke=col, sw=2.0)
    s += text(540, 28, "not scaled (equal axes)", size=11.5)
    a_raw = math.degrees(math.atan2(abs(raw2[0][1]), abs(raw2[0][0])))
    s += text(660, 262, f"unscaled u₁: {a_raw:.0f}°", size=11, fill=PURPLE, anchor="end")
    s += text(430, 60, "standardised u₁: 45°", size=11, fill=TEAL, anchor="start")
    return wrap(
        700,
        330,
        "figPcaPre",
        "What PCA finds without centring or scaling",
        "Left: points clustered away from the origin along a line of slope minus one. The first eigenvector of "
        "the uncentred second-moment matrix points from the origin towards the cluster, while PCA on centred "
        "data follows the cluster's own direction. Right: heights in centimetres against inches on equal axes; "
        "without standardising, the first principal direction tilts towards the centimetre axis, at about 21 "
        "degrees, instead of the 45 degrees it has after standardising.",
        s,
    )


def stability_curve(ratios, pairs=200):
    """Median angle between the first eigenvectors, and between the top-2 subspaces, of two samples."""
    rng = random.Random(21)
    out = []
    for r in ratios:
        a1, a2 = [], []
        for _ in range(pairs):
            V1 = eigh(second_moment(plane_data(r, 200, rng)))[1]
            V2 = eigh(second_moment(plane_data(r, 200, rng)))[1]
            a1.append(angle_deg(V1[0], V2[0]))
            a2.append(subspace_angle_deg(V1[:2], V2[:2]))
        out.append((r, sorted(a1)[pairs // 2], sorted(a2)[pairs // 2]))
    return out


def fig_stability(curve_pts):
    """Median angles against lambda_2 / lambda_1."""
    xdom, ydom = (0.5, 1.0), (0.0, 50.0)
    to_px = mapper(xdom, ydom, (80, 520), (30, 230))
    s = axes(
        to_px, xdom, ydom, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), (0, 10, 20, 30, 40, 50), "λ₂ / λ₁", "median angle (degrees)"
    )
    s += curve([(r, a) for r, a, _ in curve_pts], to_px, xdom, ydom, stroke=PURPLE, sw=2.0)
    s += curve([(r, b) for r, _, b in curve_pts], to_px, xdom, ydom, stroke=TEAL, sw=2.0)
    for r, a, b in curve_pts:
        s += circle(*to_px(r, a), 2.6, fill=PURPLE, stroke="none")
        s += circle(*to_px(r, b), 2.6, fill=TEAL, stroke="none")
    px, py = to_px(0.86, curve_pts[-3][1])
    s += text(px - 10, py - 12, "first eigenvector u₁", size=11.5, fill=PURPLE, anchor="end")
    px, py = to_px(0.75, 0)
    s += text(px, py - 16, "subspace spanned by u₁, u₂", size=11.5, fill=TEAL)
    return wrap(
        560,
        285,
        "figPcaStab",
        "Stability of a principal direction against stability of a principal subspace",
        "Median angle, over two hundred pairs of samples of 200 points from the same three-dimensional distribution, "
        "between the two samples' first eigenvectors and between their two-dimensional principal subspaces, "
        "plotted against the ratio of the second to the first eigenvalue. The first eigenvector's angle stays "
        "small until the ratio nears one and then climbs steeply; the subspace angle stays near zero throughout.",
        s,
    )


ICA_A = [[1.0, 0.6], [0.45, 1.0]]


def ica_sources(m=400):
    """Independent sources, each uniform on [-1, 1]."""
    rng = random.Random(5)
    return [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(m)]


def mat2(A, v):
    """2-by-2 matrix times vector."""
    return [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]]


def inv2(A):
    """Inverse of a 2-by-2 matrix."""
    d = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]


def fig_ica():
    """Three panels: sources, mixtures, and sources recovered up to order and sign."""
    S = ica_sources()
    X = [mat2(ICA_A, s) for s in S]
    W = inv2(ICA_A)
    W_amb = [[-W[1][0], -W[1][1]], [W[0][0], W[0][1]]]  # swap the rows, flip one sign
    Y = [mat2(W_amb, x) for x in X]
    s = ""
    titles = ("sources s (uniform, independent)", "observed x = As", "recovered Wx, W = PDA⁻¹")
    labels = (("s₁", "s₂"), ("x₁", "x₂"), ("−s₂", "s₁"))
    for k, (pts, title, (lx, ly)) in enumerate(zip((S, X, Y), titles, labels)):
        x0 = 60 + k * 230
        dom = (-2.0, 2.0)
        to_px = mapper(dom, dom, (x0, x0 + 170), (40, 210))
        s += rect(x0, 40, 170, 170, so=0.4, sw=0.8)
        s += line(*to_px(-2, 0), *to_px(2, 0), op=0.35) + line(*to_px(0, -2), *to_px(0, 2), op=0.35)
        for p in pts:
            s += circle(*to_px(*p), 1.5, fill=(TEAL, INDIGO, PURPLE)[k], stroke="none", op=0.7)
        s += text(x0 + 85, 28, title, size=11.5)
        s += text(x0 + 85, 228, lx, size=11.5, style="italic")
        s += text(x0 + 8, 58, ly, size=11.5, style="italic", anchor="start")
    s += arrow(232, 125, 286, 125, op=0.6)
    s += text(259, 117, "A", size=12, style="italic")
    s += arrow(462, 125, 516, 125, op=0.6)
    s += text(489, 117, "W", size=12, style="italic")
    return wrap(
        760,
        245,
        "figIca",
        "The ICA problem for two uniform sources",
        "Three scatter plots. Four hundred pairs of independent sources, each uniform between minus one and one, "
        "fill a square. Multiplied by a mixing matrix A, they fill a parallelogram. Multiplied by an unmixing "
        "matrix that is the inverse of A with its rows swapped and one sign flipped, they fill a square again: "
        "the sources are recovered, but in the other order and with one negated.",
        s,
    )


FIGURES = {}

if __name__ == "__main__":
    # 1. Max variance and min reconstruction are the same problem.
    Z = standardise(heights())
    C = second_moment(Z)
    vals, vecs = eigh(C)
    tot = sum(dot(z, z) for z in Z) / len(Z)
    best_v, best_r = None, None
    for i in range(3600):
        t = math.pi * i / 3600
        u = (math.cos(t), math.sin(t))
        var = sum(dot(z, u) ** 2 for z in Z) / len(Z)
        rec = sum(dot(z, z) - dot(z, u) ** 2 for z in Z) / len(Z)
        assert abs(var + rec - tot) < 1e-12
        if best_v is None or var > best_v[0]:
            best_v = (var, t)
        if best_r is None or rec < best_r[0]:
            best_r = (rec, t)
    assert best_v[1] == best_r[1]
    assert angle_deg(vecs[0], (math.cos(best_v[1]), math.sin(best_v[1]))) < 0.06
    print(
        f"heights: correlation {C[0][1]:.4f}; eigenvalues {vals[0]:.4f}, {vals[1]:.4f}; u1 at {math.degrees(best_v[1]):.1f} deg"
    )
    print(
        f"projected variance onto u1 {best_v[0]:.4f}, onto u2 {vals[1]:.4f}; mean squared distance onto u1 {best_r[0]:.4f}"
    )

    # 2. Stationary points of u^T C u on the circle are the eigenvectors, with the eigenvalues as values.
    rng = random.Random(3)
    M = [[rng.gauss(0, 1) for _ in range(2)] for _ in range(2)]
    A = [[sum(M[i][k] * M[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
    va, ve = eigh(A)
    f = [
        (A[0][0] * math.cos(t) ** 2 + 2 * A[0][1] * math.cos(t) * math.sin(t) + A[1][1] * math.sin(t) ** 2, t)
        for t in (math.pi * i / 20000 for i in range(20000))
    ]
    ext = [f[i] for i in range(20000) if (f[i][0] - f[i - 1][0]) * (f[(i + 1) % 20000][0] - f[i][0]) <= 0]
    for val, t in ext:
        u = (math.cos(t), math.sin(t))
        j = min(range(2), key=lambda j: abs(val - va[j]))
        assert abs(val - va[j]) < 1e-6 and angle_deg(u, ve[j]) < 0.02
    print(f"stationary points of u^T A u on the circle: {len(ext)}, each an eigenvector with value its eigenvalue")

    # 3. k-dimensional PCA: mean squared reconstruction error = sum of discarded eigenvalues.
    rng = random.Random(4)
    B = [[rng.gauss(0, 1) for _ in range(10)] for _ in range(10)]
    Xr = centre(
        [
            [sum(B[i][j] * g for j, g in enumerate([rng.gauss(0, 1) * (0.6**j) for j in range(10)])) for i in range(10)]
            for _ in range(300)
        ]
    )
    vr, Vr = eigh(second_moment(Xr))
    for k in range(11):
        err = sum(dot(x, x) - sum(dot(x, Vr[j]) ** 2 for j in range(k)) for x in Xr) / len(Xr)
        assert abs(err - sum(vr[k:])) < 1e-9 * (1 + sum(vr))
    retained = [sum(vr[:k]) / sum(vr) for k in range(1, 11)]
    k90 = next(k for k, r in enumerate(retained, 1) if r >= 0.9)
    k99 = next(k for k, r in enumerate(retained, 1) if r >= 0.99)
    print(f"reconstruction error = sum of discarded eigenvalues for k = 0..10; 90% at k = {k90}, 99% at k = {k99}")

    # 4. Pre-processing.
    rng = random.Random(9)
    Xp = []
    for _ in range(50):
        t = rng.gauss(0, 1.0)
        Xp.append([4.0 + 0.8 * t + rng.gauss(0, 0.15), 1.5 - 0.8 * t + rng.gauss(0, 0.15)])
    raw = eigh(second_moment(Xp))[1][0]
    cen = eigh(second_moment(centre(Xp)))[1][0]
    print(
        f"uncentred u1 is {angle_deg(raw, mean(Xp)):.1f} deg from the mean; centred u1 is {angle_deg(cen, (1, -1)):.1f} deg from the cluster's axis"
    )
    assert angle_deg(raw, mean(Xp)) < 5 and angle_deg(cen, (1, -1)) < 5
    H = heights()
    u_raw = eigh(second_moment(centre(H)))[1][0]
    u_std = eigh(second_moment(standardise(H)))[1][0]
    a_raw = math.degrees(math.atan2(abs(u_raw[1]), abs(u_raw[0])))
    a_std = math.degrees(math.atan2(abs(u_std[1]), abs(u_std[0])))
    print(
        f"heights u1: unscaled {a_raw:.1f} deg (atan(1/2.54) = {math.degrees(math.atan(1 / 2.54)):.1f}), standardised {a_std:.1f} deg"
    )
    assert abs(a_std - 45) < 1e-6

    # 5. Eigenvector versus subspace stability.
    for gap in (1e-1, 1e-3, 1e-5, 1e-7):
        D = polygon_data(gap)
        V0 = eigh(second_moment(D))[1]
        D2 = [r[:] for r in D]
        D2[1][1] += 1e-5  # the fifth decimal place of one coordinate of one example
        V1 = eigh(second_moment(D2))[1]
        print(f"gap {gap:g}: 1e-5 change moves u1 by {angle_deg(V0[0], V1[0]):.4f} deg, the u1-u2 plane by {subspace_angle_deg(V0[:2], V1[:2]):.1e} deg")
        if gap == 1e-1:
            assert angle_deg(V0[0], V1[0]) < 1e-3
        if gap == 1e-7:
            assert angle_deg(V0[0], V1[0]) > 10
    ratios = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
    stab = stability_curve(ratios)
    for r, a, b in stab:
        print(f"lambda2/lambda1 = {r:.2f}: median u1 angle {a:5.1f} deg, subspace angle {b:.2f} deg")
    assert stab[0][1] < 10 and stab[-1][1] > 25 and max(b for _, _, b in stab) < 3

    # 6. ICA ambiguities: any PD W A^{-1} gives independent sources.
    S = ica_sources()
    X = [mat2(ICA_A, s) for s in S]
    W = inv2(ICA_A)
    for P in ([[1, 0], [0, 1]], [[0, 1], [1, 0]]):
        for D in ([1, 1], [-1, 1], [2.0, -0.5]):
            Wa = [[D[i] * sum(P[i][k] * W[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
            Y = [mat2(Wa, x) for x in X]
            for i in range(2):
                src = P[i].index(1)
                assert all(abs(y[i] - D[i] * s[src]) < 1e-12 for y, s in zip(Y, S))
    print("ICA: every W = P D A^{-1} returns the sources permuted and rescaled, exactly")

    FIGURES.update(
        {
            "diagram_pca_projection.svg": fig_projection(),
            "diagram_pca_preprocessing.svg": fig_preprocessing(),
            "diagram_pca_stability.svg": fig_stability(stab),
            "diagram_ica_mixing.svg": fig_ica(),
        }
    )
    for name, svg in FIGURES.items():
        (OUT / name).write_text(svg)
        print("wrote", name)
