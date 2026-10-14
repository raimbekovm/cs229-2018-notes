"""Figures for lecture 14: EM for the mixture of Gaussians, and factor analysis.

Three computed figures: maximum-likelihood Gaussians fitted to a small
two-dimensional sample under a full, a diagonal and a spherical covariance,
beside the degenerate full fit to two points; the factor-analysis generative
process for the lecture's own one-dimensional example (Lambda = (2, 1),
Psi = diag(1, 2)), stage by stage; and EM for factor analysis on 100-dimensional
data with 30 examples, run correctly and with the common E[z z^T] mistake,
together with the held-out likelihood of the alternatives.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: that the mixture M-step's closed forms
maximise the lower bound, and that the unconstrained derivative in phi has no
zero; that the lower bound equals the log-likelihood after every E-step and
neither ever falls; that the full covariance estimate has rank min(m - 1, n);
that the spherical estimate's variance is the mean of the diagonal estimate's;
that the joint covariance of (z, x) in factor analysis matches a Monte Carlo
estimate; that the conditional-Gaussian formula for p(z | x) matches Bayes'
rule evaluated on a grid; that EM for factor analysis never lowers the
log-likelihood; and how far the E[z z^T] = mu mu^T shortcut falls short.

Plain Python on purpose, as in lectures 1-13, so the small dense linear algebra
factor analysis needs is written out below.

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

# ---------------------------------------------------------------- linear algebra


def zeros(r, c):
    """An r-by-c zero matrix."""
    return [[0.0] * c for _ in range(r)]


def eye(n):
    """The n-by-n identity."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def T(A):
    """Transpose."""
    return [list(r) for r in zip(*A)]


def mm(A, B):
    """Matrix product."""
    Bt = T(B)
    return [[sum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def mv(A, v):
    """Matrix-vector product."""
    return [sum(a * b for a, b in zip(r, v)) for r in A]


def add(A, B, s=1.0):
    """A + s B."""
    return [[a + s * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def outer(u, v):
    """U v^T."""
    return [[a * b for b in v] for a in u]


def chol(A):
    """Lower Cholesky factor of a symmetric positive-definite matrix."""
    n = len(A)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def inv(A):
    """Inverse of a symmetric positive-definite matrix, through its Cholesky factor."""
    n = len(A)
    L = chol(A)
    Li = zeros(n, n)
    for c in range(n):
        for i in range(n):
            s = (1.0 if i == c else 0.0) - sum(L[i][k] * Li[k][c] for k in range(i))
            Li[i][c] = s / L[i][i]
    return mm(T(Li), Li)


def rank(A, tol=1e-9):
    """Numerical rank by Gaussian elimination with partial pivoting."""
    M = [r[:] for r in A]
    n, m = len(M), len(M[0])
    scale = max(abs(x) for r in M for x in r) or 1.0
    r = 0
    for c in range(m):
        p = max(range(r, n), key=lambda i: abs(M[i][c]), default=None)
        if p is None or abs(M[p][c]) < tol * scale:
            continue
        M[r], M[p] = M[p], M[r]
        for i in range(r + 1, n):
            f = M[i][c] / M[r][c]
            M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
        if r == n:
            break
    return r


def gauss_logpdf_batch(X, mu, S):
    """Log density of every row of X under N(mu, S)."""
    n = len(mu)
    L = chol(S)
    logdet = 2 * sum(math.log(L[i][i]) for i in range(n))
    out = []
    for x in X:
        d = [a - b for a, b in zip(x, mu)]
        y = []
        for i in range(n):
            y.append((d[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i])
        out.append(-0.5 * (n * math.log(2 * math.pi) + logdet + sum(v * v for v in y)))
    return out


def mean(X):
    """Column means."""
    return [sum(c) / len(X) for c in zip(*X)]


def cov_ml(X):
    """Maximum-likelihood covariance (divides by m)."""
    mu = mean(X)
    D = [[a - b for a, b in zip(x, mu)] for x in X]
    return [[v / len(X) for v in r] for r in mm(T(D), D)]


# ---------------------------------------------------------------- mixture M-step


def npdf(x, mu, s2):
    """Univariate normal density."""
    return math.exp(-((x - mu) ** 2) / (2 * s2)) / math.sqrt(2 * math.pi * s2)


def mog_data():
    """A one-dimensional sample from a mixture of three Gaussians."""
    rng = random.Random(4)
    spec = ((0.5, -2.0, 0.6), (0.3, 1.0, 0.4), (0.2, 3.5, 1.0))
    X = []
    for w, m, s in spec:
        X += [rng.gauss(m, s) for _ in range(int(200 * w))]
    return X


def mog_estep(X, phi, mu, s2):
    """Responsibilities w_ij = p(z_i = j | x_i)."""
    W = []
    for x in X:
        p = [phi[j] * npdf(x, mu[j], s2[j]) for j in range(len(phi))]
        t = sum(p)
        W.append([v / t for v in p])
    return W


def mog_mstep(X, W):
    """Closed-form M-step: phi, mu and the variances."""
    k, m = len(W[0]), len(X)
    nj = [sum(w[j] for w in W) for j in range(k)]
    phi = [v / m for v in nj]
    mu = [sum(w[j] * x for w, x in zip(W, X)) / nj[j] for j in range(k)]
    s2 = [sum(w[j] * (x - mu[j]) ** 2 for w, x in zip(W, X)) / nj[j] for j in range(k)]
    return phi, mu, s2


def mog_bound(X, W, phi, mu, s2):
    """The lower bound J(theta, Q) with Q given by the responsibilities W."""
    tot = 0.0
    for x, w in zip(X, W):
        for j, q in enumerate(w):
            if q > 0:
                tot += q * math.log(phi[j] * npdf(x, mu[j], s2[j]) / q)
    return tot


def mog_loglik(X, phi, mu, s2):
    """The mixture log-likelihood."""
    return sum(math.log(sum(p * npdf(x, m, v) for p, m, v in zip(phi, mu, s2))) for x in X)


# ---------------------------------------------------------------- factor analysis

NG_LAMBDA = [[2.0], [1.0]]
NG_PSI = [1.0, 2.0]


def fa_cov(Lam, psi):
    """Marginal covariance of x: Lambda Lambda^T + Psi."""
    C = mm(Lam, T(Lam))
    for i, p in enumerate(psi):
        C[i][i] += p
    return C


def fa_posterior(x, mu, Lam, psi):
    """Mean and covariance of z | x by the conditional-Gaussian formulas."""
    G = inv(fa_cov(Lam, psi))
    LtG = mm(T(Lam), G)
    m = mv(LtG, [a - b for a, b in zip(x, mu)])
    S = add(eye(len(Lam[0])), mm(LtG, Lam), -1.0)
    return m, S


def fa_estep(X, mu, Lam, psi):
    """Posterior moments for every example, using Woodbury so only d-by-d inverses appear."""
    d = len(Lam[0])
    PiL = [[v / p for v in row] for row, p in zip(Lam, psi)]  # Psi^{-1} Lambda
    S = inv(add(eye(d), mm(T(Lam), PiL)))  # (I + Lambda^T Psi^{-1} Lambda)^{-1} = Sigma_{z|x}
    B = mm(S, T(PiL))  # d x n: maps x - mu to the posterior mean
    M = [mv(B, [a - b for a, b in zip(x, mu)]) for x in X]
    return M, S


def fa_mstep(X, M, S, shortcut=False):
    """M-step for mu, Lambda and diagonal Psi; `shortcut` drops Sigma_{z|x} from E[z z^T]."""
    m, n, d = len(X), len(X[0]), len(S)
    mu = mean(X)
    Xc = [[a - b for a, b in zip(x, mu)] for x in X]
    A = mm(T(Xc), M)  # n x d: sum_i (x_i - mu) E[z_i]^T
    Ezz = mm(T(M), M)  # d x d: sum_i E[z_i] E[z_i]^T
    if not shortcut:
        Ezz = add(Ezz, [[v * m for v in r] for r in S])
    Lam = mm(A, inv(Ezz))
    # diag of (1/m) sum_i [ x x^T - x E[z]^T Lam^T - Lam E[z] x^T + Lam E[z z^T] Lam^T ]
    LE = mm(Lam, Ezz)
    psi = []
    for j in range(n):
        sxx = sum(x[j] ** 2 for x in Xc)
        sxz = sum(A[j][k] * Lam[j][k] for k in range(d))
        szz = sum(LE[j][k] * Lam[j][k] for k in range(d))
        psi.append(max((sxx - 2 * sxz + szz) / m, 1e-8))
    return mu, Lam, psi


def fa_loglik(X, mu, Lam, psi):
    """Log-likelihood of X under N(mu, Lambda Lambda^T + Psi)."""
    return sum(gauss_logpdf_batch(X, mu, fa_cov(Lam, psi)))


def fa_problem():
    """100-dimensional data from a three-factor model: 30 training and 300 test examples."""
    rng = random.Random(7)
    n, d = 100, 3
    Lam = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    psi = [rng.uniform(0.2, 1.0) for _ in range(n)]
    mu = [rng.gauss(0, 2) for _ in range(n)]

    def draw():
        z = [rng.gauss(0, 1) for _ in range(d)]
        return [mu[j] + sum(Lam[j][k] * z[k] for k in range(d)) + rng.gauss(0, math.sqrt(psi[j])) for j in range(n)]

    return [draw() for _ in range(30)], [draw() for _ in range(300)]


def fa_fit(X, d, iters, shortcut=False, seed=0):
    """Run EM for factor analysis; return the parameters and the log-likelihood path."""
    rng = random.Random(seed)
    n = len(X[0])
    mu = mean(X)
    psi = [max(v, 1e-3) for v in (cov_ml(X)[j][j] for j in range(n))]
    Lam = [[rng.gauss(0, 0.1) for _ in range(d)] for _ in range(n)]
    path = [fa_loglik(X, mu, Lam, psi)]
    for _ in range(iters):
        M, S = fa_estep(X, mu, Lam, psi)
        mu, Lam, psi = fa_mstep(X, M, S, shortcut)
        path.append(fa_loglik(X, mu, Lam, psi))
    return (mu, Lam, psi), path


# ---------------------------------------------------------------- figure helpers


def ellipse(mu, S, r, k=120):
    """Points on the r-sigma contour of a 2-D Gaussian."""
    a, b, c = S[0][0], S[0][1], S[1][1]
    tr, det = a + c, a * c - b * b
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 0.0)
    ang = math.atan2(l1 - a, b) if abs(b) > 1e-12 else (0.0 if a >= c else math.pi / 2)
    ca, sa = math.cos(ang), math.sin(ang)
    out = []
    for i in range(k + 1):
        t = 2 * math.pi * i / k
        u, v = r * math.sqrt(l1) * math.cos(t), r * math.sqrt(l2) * math.sin(t)
        out.append((mu[0] + ca * u - sa * v, mu[1] + sa * u + ca * v))
    return out


def cross(px, py, col, s=4.0, sw=1.8):
    """An x-shaped marker."""
    return line(px - s, py - s, px + s, py + s, stroke=col, sw=sw) + line(
        px - s, py + s, px + s, py - s, stroke=col, sw=sw
    )


def small_sample():
    """Twelve correlated two-dimensional points (the restriction panels)."""
    rng = random.Random(3)
    out = []
    for _ in range(12):
        z = rng.gauss(0, 1)
        out.append([1.3 * z + rng.gauss(0, 0.35), 0.9 * z + rng.gauss(0, 0.35)])
    return out


TWO_POINTS = [[-1.4, -0.9], [1.2, 1.0]]


def fig_covariance():
    """Four panels: the full fit to two points, then full, diagonal and spherical fits to twelve."""
    X = small_sample()
    mu = mean(X)
    full = cov_ml(X)
    diag = [[full[0][0], 0.0], [0.0, full[1][1]]]
    s2 = (full[0][0] + full[1][1]) / 2
    sph = [[s2, 0.0], [0.0, s2]]
    m2 = mean(TWO_POINTS)
    full2 = cov_ml(TWO_POINTS)
    panels = (
        ("m = 2, full Σ", TWO_POINTS, m2, full2),
        ("m = 12, full Σ", X, mu, full),
        ("diagonal Σ", X, mu, diag),
        ("Σ = σ²I", X, mu, sph),
    )
    dom = (-3.0, 3.0)
    s = ""
    for k, (title, pts, mk, Sk) in enumerate(panels):
        x0 = 15 + k * 170
        to_px = mapper(dom, dom, (x0, x0 + 150), (40, 190))
        s += rect(x0, 40, 150, 150, so=0.5, sw=0.8)
        s += text(x0 + 75, 28, title, size=11.5)
        for r, op in ((1.0, 0.95), (2.0, 0.55)):
            s += curve(ellipse(mk, Sk, r), to_px, dom, dom, stroke=INDIGO if k else PURPLE, sw=1.6, op=op)
        for p in pts:
            s += circle(*to_px(*p), 2.8, fill=TEAL, stroke="none", op=0.9)
    return wrap(
        700,
        205,
        "figCovRestrict",
        "Maximum-likelihood Gaussians under three covariance restrictions",
        "Four panels. First, two points in two dimensions and the full-covariance Gaussian fitted to them: its "
        "contours collapse onto the segment between the points. Then twelve correlated points fitted with a full "
        "covariance (tilted ellipses that follow the data), a diagonal covariance (axis-aligned ellipses that miss "
        "the tilt) and a spherical covariance (circles).",
        s,
    )


def ng_sample():
    """The lecture's seven-point example: z, Lambda z + mu, and x."""
    rng = random.Random(11)
    Z = sorted(rng.gauss(0, 1) for _ in range(7))
    mid = [[NG_LAMBDA[0][0] * z, NG_LAMBDA[1][0] * z] for z in Z]
    X = [[a + rng.gauss(0, math.sqrt(NG_PSI[0])), b + rng.gauss(0, math.sqrt(NG_PSI[1]))] for a, b in mid]
    return Z, mid, X


def fig_generative():
    """Three panels: z on a line, Lambda z + mu on a line in R^2, then the noise and the sample x."""
    Z, mid, X = ng_sample()
    s = ""
    # panel 1: z in R^1
    to1 = mapper((-3, 3), (-1, 1), (20, 200), (110, 170))
    y = to1(0, 0)[1]
    s += line(20, y, 200, y, op=0.7)
    for t in (-2, 0, 2):
        px = to1(t, 0)[0]
        s += line(px, y, px, y + 4, op=0.7) + text(px, y + 17, f"{t}", size=11, op=0.8)
    s += text(110, y + 36, "z ∈ ℝ", size=12, style="italic")
    for z in Z:
        s += circle(*to1(z, 0), 3.4, fill=TEAL, stroke="none")
    s += text(110, 28, "1. z ~ 𝒩(0, 1)", size=11.5)
    s += text(110, 46, "seven draws", size=11, op=0.7)
    # panels 2 and 3: R^2
    dom = (-6.0, 6.0)
    for k, x0 in enumerate((250, 480)):
        to_px = mapper(dom, dom, (x0, x0 + 200), (60, 260))
        s += axes(to_px, dom, dom, (-4, 0, 4), (-4, 0, 4), "x₁", "x₂")
        s += curve(
            [(t / 10, t / 20) for t in range(-80, 81)], to_px, dom, dom, stroke=INDIGO, sw=1.0, op=0.45, dash="4 3"
        )
        for p in mid:
            if k == 0:
                s += circle(*to_px(*p), 3.4, fill=TEAL, stroke="none")
            else:
                s += curve(
                    ellipse(p, [[NG_PSI[0], 0], [0, NG_PSI[1]]], 1.0, 60), to_px, dom, dom, stroke=TEAL, sw=1.0, op=0.6
                )
        if k == 1:
            for p, q in zip(mid, X):
                s += line(*to_px(*p), *to_px(*q), stroke=PURPLE, sw=0.8, op=0.4)
                s += cross(*to_px(*q), PURPLE)
    s += text(350, 28, "2. Λz + μ, with Λ = (2, 1)ᵀ, μ = 0", size=11.5)
    s += text(580, 28, "3. x = Λz + μ + ε, Ψ = diag(1, 2)", size=11.5)
    s += text(580, 46, "one contour of 𝒩(0, Ψ) per point; × = x", size=11, op=0.7)
    return wrap(
        720,
        315,
        "figFaGen",
        "The factor-analysis model as a generative process",
        "Three panels. Seven values of z drawn from a standard normal on a line; the same seven mapped by "
        "Lambda z + mu onto the line x2 = x1 / 2 in the plane; and then, around each mapped point, one contour "
        "of the axis-aligned noise Gaussian, taller than wide, with the observed x drawn as a cross near each. "
        "Only the crosses are observed.",
        s,
    )


def fig_fa_em(path_ok, path_bad, heldout):
    """Two panels: training log-likelihood per EM iteration, and held-out log-likelihood per model."""
    s = ""
    n_it = len(path_ok) - 1
    top = math.ceil(max(path_ok + path_bad) / 10) * 10 + 10
    ydom = (top - 70, top)  # the first few iterations climb from far below and are clipped
    xdom = (0, n_it)
    to_px = mapper(xdom, ydom, (80, 340), (40, 230))
    s += axes(
        to_px,
        xdom,
        ydom,
        (0, n_it // 2, n_it),
        (ydom[0], (ydom[0] + ydom[1]) / 2, ydom[1]),
        "EM iteration",
        "training ℓ",
        fmt="{:.0f}",
    )
    s += curve([(i, v) for i, v in enumerate(path_ok)], to_px, xdom, ydom, stroke=INDIGO, sw=2.0)
    s += curve([(i, v) for i, v in enumerate(path_bad)], to_px, xdom, ydom, stroke=PURPLE, sw=2.0, dash="5 3")
    s += text(335, to_px(n_it, path_ok[-1])[1] - 8, "correct E[zzᵀ]", size=11, anchor="end", fill=INDIGO)
    s += text(335, to_px(n_it, path_bad[-1])[1] + 18, "E[z]E[z]ᵀ only", size=11, anchor="end", fill=PURPLE)
    s += text(210, 24, "Training log-likelihood, n = 100, m = 30", size=11.5)
    # bars
    names = list(heldout)
    vals = [heldout[k] for k in names]
    lo2 = math.floor(min(vals) / 20) * 20 - 20
    hi2 = math.ceil(max(vals) / 20) * 20
    top, bot = 50, 230
    x0 = 430
    for i, (nm, v) in enumerate(zip(names, vals)):
        y0 = top + i * (bot - top) / len(names)
        w = (v - lo2) / (hi2 - lo2) * 200
        s += rect(x0, y0 + 6, w, 22, fill=TEAL if i == 0 else INDIGO, stroke="none", op=0.85 if i == 0 else 0.55)
        s += text(x0 + w + 6, y0 + 22, f"{v:.1f}", size=11, anchor="start")
        s += text(x0, y0 + 2, nm, size=11, anchor="start", op=0.85)
    s += line(x0, top, x0, bot, op=0.7)
    s += text(x0 - 4, bot + 16, f"{lo2:.0f}", size=11, anchor="start", op=0.7)
    s += text(540, 24, "Held-out log-likelihood per example", size=11.5)
    return wrap(
        720,
        280,
        "figFaEm",
        "EM for factor analysis on 100-dimensional data with 30 examples",
        "Left: training log-likelihood against EM iteration, rising monotonically for the correct M-step and "
        "levelling off lower when E[z z^T] is replaced by E[z] E[z]^T. Right: average held-out log-likelihood "
        "for factor analysis with three factors, a full Gaussian whose covariance is shrunk towards the diagonal "
        "with the best shrinkage found on the test set, and a diagonal Gaussian; factor analysis is highest.",
        s,
    )


def shrunk_gaussian_heldout(Xtr, Xte):
    """Best held-out score of N(mean, S + lam I) over a grid of lam (an oracle choice)."""
    mu, S = mean(Xtr), cov_ml(Xtr)
    best = None
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
        Sl = [row[:] for row in S]
        for i in range(len(S)):
            Sl[i][i] += lam
        v = sum(gauss_logpdf_batch(Xte, mu, Sl)) / len(Xte)
        if best is None or v > best[1]:
            best = (lam, v)
    return best


FIGURES = {}

if __name__ == "__main__":
    # 1. Mixture M-step: closed forms maximise the bound; the unconstrained phi derivative has no zero.
    X = mog_data()
    phi, mu, s2 = [1 / 3] * 3, [-1.0, 0.0, 1.0], [1.0] * 3
    lls = [mog_loglik(X, phi, mu, s2)]
    for it in range(40):
        W = mog_estep(X, phi, mu, s2)
        assert abs(mog_bound(X, W, phi, mu, s2) - lls[-1]) < 1e-8  # tight after E-step
        nphi, nmu, ns2 = mog_mstep(X, W)
        J = mog_bound(X, W, nphi, nmu, ns2)
        if it == 0:
            rng = random.Random(0)
            for _ in range(300):
                e = [rng.gauss(0, 0.02) for _ in range(3)]
                e = [v - sum(e) / 3 for v in e]  # stay on the simplex
                pp = [p + v for p, v in zip(nphi, e)]
                pm = [m + rng.gauss(0, 0.05) for m in nmu]
                ps = [v * math.exp(rng.gauss(0, 0.05)) for v in ns2]
                assert mog_bound(X, W, pp, pm, ps) <= J + 1e-9
            dphi = [sum(w[j] for w in W) / nphi[j] for j in range(3)]
            print("unconstrained dJ/dphi_j at the M-step:", [round(v, 1) for v in dphi], "(all equal m, never 0)")
            assert all(abs(v - len(X)) < 1e-6 for v in dphi)
        phi, mu, s2 = nphi, nmu, ns2
        lls.append(mog_loglik(X, phi, mu, s2))
        assert lls[-1] >= J - 1e-9 >= lls[-2] - 1e-9
    print("mixture: M-step closed forms beat 300 perturbations; J = l after each E-step; l monotone")
    print("mixture l:", [round(v, 2) for v in lls[:4]], "->", round(lls[-1], 3))

    # 2. The full covariance estimate has rank min(m - 1, n).
    rng = random.Random(1)
    for n in (2, 5, 10):
        for m in range(2, n + 3):
            Xr = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(m)]
            assert rank(cov_ml(Xr)) == min(m - 1, n)
    print("rank of the ML covariance = min(m - 1, n): singular exactly when m <= n")
    full2 = cov_ml(TWO_POINTS)
    print(f"two points in 2-D: det of full Sigma = {full2[0][0] * full2[1][1] - full2[0][1] ** 2:.2e}")

    # 3. Spherical sigma^2 is the mean of the diagonal estimates.
    Xs = small_sample()
    F = cov_ml(Xs)
    s2_direct = sum((x[j] - mean(Xs)[j]) ** 2 for x in Xs for j in range(2)) / (2 * len(Xs))
    assert abs(s2_direct - (F[0][0] + F[1][1]) / 2) < 1e-12
    print(f"sample correlation of the 12 points: {F[0][1] / math.sqrt(F[0][0] * F[1][1]):.2f}")

    # 4. Joint covariance of (z, x) by formula and by Monte Carlo.
    rng = random.Random(5)
    N = 200000
    acc = zeros(3, 3)
    for _ in range(N):
        z = rng.gauss(0, 1)
        v = [z, 2 * z + rng.gauss(0, 1), z + rng.gauss(0, math.sqrt(2))]
        for i in range(3):
            for j in range(3):
                acc[i][j] += v[i] * v[j] / N
    LLt = fa_cov(NG_LAMBDA, NG_PSI)
    formula = [[1, 2, 1], [2, LLt[0][0], LLt[0][1]], [1, LLt[1][0], LLt[1][1]]]
    err = max(abs(acc[i][j] - formula[i][j]) for i in range(3) for j in range(3))
    print("joint covariance of (z, x):", formula, f"Monte Carlo max error {err:.3f}")
    assert err < 0.05

    # 5. p(z | x): conditional-Gaussian formula against Bayes' rule on a grid.
    for x in ([1.0, 2.0], [-3.0, 0.5], [4.0, -1.0]):
        pm, pS = fa_posterior(x, [0, 0], NG_LAMBDA, NG_PSI)
        grid = [-6 + i * 0.001 for i in range(12001)]
        w = [npdf(z, 0, 1) * npdf(x[0], 2 * z, 1) * npdf(x[1], z, 2) for z in grid]
        t = sum(w)
        gm = sum(a * b for a, b in zip(grid, w)) / t
        gv = sum((a - gm) ** 2 * b for a, b in zip(grid, w)) / t
        assert abs(gm - pm[0]) < 1e-6 and abs(gv - pS[0][0]) < 1e-6
    print(f"posterior of z | x: formula = grid Bayes; Var(z | x) = {pS[0][0]:.4f} for every x")

    # 6. EM for factor analysis, n = 100, m = 30, d = 3.
    Xtr, Xte = fa_problem()
    assert rank(cov_ml(Xtr)) == 29
    ITERS = 200
    (fmu, fLam, fpsi), ok = fa_fit(Xtr, 3, ITERS)
    assert all(b >= a - 1e-6 for a, b in zip(ok, ok[1:]))
    _, bad = fa_fit(Xtr, 3, ITERS, shortcut=True)
    drops = sum(1 for a, b in zip(bad, bad[1:]) if b < a - 1e-6)
    print(f"FA EM training l: {ok[0]:.1f} -> {ok[10]:.1f} (10 it) -> {ok[60]:.1f} (60 it) -> {ok[-1]:.1f} ({ITERS} it), never falls")
    peak = max(range(len(bad)), key=lambda k: bad[k])
    print(f"shortcut E[z]E[z]^T: peaks at iteration {peak} ({bad[peak]:.1f}), ends at {bad[-1]:.1f}; falls on {drops} of {ITERS} iterations")
    assert drops > 0 and bad[-1] < ok[-1]
    cross_it = next(i for i in range(1, ITERS + 1) if ok[i] > bad[i] and i > peak)
    print(f"correct EM overtakes the shortcut at iteration {cross_it}; last 50 iterations gain {ok[-1] - ok[-51]:.2f}")
    fa_te = sum(gauss_logpdf_batch(Xte, fmu, fa_cov(fLam, fpsi))) / len(Xte)
    lam, sh_te = shrunk_gaussian_heldout(Xtr, Xte)
    F = cov_ml(Xtr)
    dg_te = sum(
        gauss_logpdf_batch(Xte, mean(Xtr), [[F[i][i] if i == j else 0.0 for j in range(100)] for i in range(100)])
    ) / len(Xte)
    print(f"held-out per example: FA {fa_te:.1f}; full + {lam} I (best on test) {sh_te:.1f}; diagonal {dg_te:.1f}")
    assert fa_te > sh_te and fa_te > dg_te

    FIGURES.update(
        {
            "diagram_covariance.svg": fig_covariance(),
            "diagram_fa_generative.svg": fig_generative(),
            "diagram_fa_em.svg": fig_fa_em(
                ok,
                bad,
                {"factor analysis, d = 3": fa_te, f"full Σ + λI, λ = {lam:g} (best on test)": sh_te, "diagonal Σ": dg_te},
            ),
        }
    )
    for name, svg in FIGURES.items():
        (OUT / name).write_text(svg)
        print("wrote", name)
