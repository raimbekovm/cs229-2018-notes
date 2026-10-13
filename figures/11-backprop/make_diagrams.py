"""Figures for lecture 11: backpropagation and the tricks that make a network train.

One structural figure, the backward pass of the lecture's 3-2-1 network drawn
against its forward pass, and four computed ones: the three activations and
their derivatives; gradient descent on a least-squares cost before and after
the inputs are standardised; the size of the activations through a deep ReLU
network under three initialisation scales; and gradient descent against
momentum on an elongated quadratic.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: the lecture's backpropagation formulas
for every parameter of the 3-2-1 sigmoid network against finite differences;
that a network with identity activations is one affine map, whatever its
depth; that a network whose weights start equal keeps every hidden neuron of
a layer identical through training; the derivative bounds of the sigmoid and
tanh; that weights with variance 1/n keep the pre-activation variance of a
linear layer at the input's, and that a ReLU layer needs 2/n for the same
(He et al., 2015) — with 1/n the signal halves at every layer; the variance
of a mini-batch gradient against 1/B; and step counts to convergence that
the prose quotes.

Plain Python on purpose, as in lectures 1-10.

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


# ---------------------------------------------------------------- small linear algebra


def matmul(A, B):
    """A B for list-of-rows matrices."""
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def T(A):
    """Transpose."""
    return [list(r) for r in zip(*A)]


def add(A, B):
    """Entrywise sum."""
    return [[a + b for a, b in zip(r, s)] for r, s in zip(A, B)]


def hadamard(A, B):
    """Entrywise product."""
    return [[a * b for a, b in zip(r, s)] for r, s in zip(A, B)]


def apply(f, A):
    """Apply f to every entry."""
    return [[f(a) for a in r] for r in A]


def col(v):
    """A list as a column vector."""
    return [[x] for x in v]


def sigmoid(z):
    """The logistic function."""
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------- the 3-2-1 network


def init(sizes, rng, scale=0.8):
    """Random W[l], b[l] for layer widths `sizes` (input first)."""
    return [
        ([[rng.gauss(0, scale) for _ in range(a)] for _ in range(b)], [[rng.gauss(0, 0.3)] for _ in range(b)])
        for a, b in zip(sizes, sizes[1:])
    ]


def forward(params, x, act=sigmoid):
    """Forward propagation, returning the cache [a0, (z1, a1), (z2, a2), ...]."""
    a = x
    cache = [a]
    for W, b in params:
        z = add(matmul(W, a), b)
        a = apply(act, z)
        cache.append((z, a))
    return cache


def loss(params, x, y):
    """Logistic loss of the network's output."""
    a3 = forward(params, x)[-1][1][0][0]
    return -(y * math.log(a3) + (1 - y) * math.log(1 - a3))


def backward(params, cache, y):
    """The lecture's backpropagation for a sigmoid network with logistic loss.

    δ[L] = a[L] − y; δ[l] = (W[l+1]ᵀ δ[l+1]) ∘ a[l](1 − a[l]); ∂L/∂W[l] = δ[l] a[l−1]ᵀ; ∂L/∂b[l] = δ[l].
    """
    grads = [None] * len(params)
    a_last = cache[-1][1]
    delta = [[a_last[0][0] - y]]
    for k in range(len(params) - 1, -1, -1):
        a_prev = cache[k] if k == 0 else cache[k][1]
        grads[k] = (matmul(delta, T(a_prev)), delta)
        if k:
            a = cache[k][1]
            delta = hadamard(matmul(T(params[k][0]), delta), apply(lambda v: v * (1 - v), a))
    return grads


def fd_check(params, x, y, h=1e-6):
    """Largest gap between backward() and central finite differences, over every parameter."""
    grads = backward(params, forward(params, x), y)
    worst = 0.0
    for k, (W, b) in enumerate(params):
        for M, G in ((W, grads[k][0]), (b, grads[k][1])):
            for i in range(len(M)):
                for j in range(len(M[0])):
                    old = M[i][j]
                    M[i][j] = old + h
                    up = loss(params, x, y)
                    M[i][j] = old - h
                    dn = loss(params, x, y)
                    M[i][j] = old
                    worst = max(worst, abs((up - dn) / (2 * h) - G[i][j]))
    return worst


# ---------------------------------------------------------------- figures


def fig_backprop():
    """The forward pass left to right and the backward pass right to left, with the reused δ's."""
    nodes = ["x", "z[1]", "a[1]", "z[2]", "a[2]", "z[3]", "a[3]", "L"]
    xs = [50 + 96 * i for i in range(len(nodes))]
    y0 = 80
    s = ""
    for i, (x, nm) in enumerate(zip(xs, nodes)):
        fill = PURPLE if nm == "L" else (INDIGO if nm == "x" else TEAL)
        s += rect(x - 26, y0 - 16, 52, 32, fill=fill, stroke=fill, op=0.15, rx=6)
        s += text(x, y0 + 5, nm, size=13)
        if i:
            s += arrow(xs[i - 1] + 26, y0, x - 27, y0, stroke=TEAL)
    fwd = {1: "W[1]x+b[1]", 2: "σ", 3: "W[2]a[1]+b[2]", 4: "σ", 5: "W[3]a[2]+b[3]", 6: "σ", 7: "loss"}
    for i, lab in fwd.items():
        s += text((xs[i - 1] + xs[i]) / 2, y0 - 26, lab, size=10, fill=TEAL)
    s += text(30, 30, "forward: compute and cache", size=12, fill=TEAL, anchor="start", weight="600")
    # Backward chain along the z's, right to left.
    yb = 176
    deltas = [(5, "δ[3] = a[3] − y"), (3, "δ[2] = W[3]ᵀδ[3] ∘ a[2](1−a[2])"), (1, "δ[1] = W[2]ᵀδ[2] ∘ a[1](1−a[1])")]
    for k, (i, lab) in enumerate(deltas):
        x = xs[i]
        s += line(x, y0 + 16, x, yb - 12, stroke=PURPLE, sw=1.0, op=0.5, dash="3 3")
        s += circle(x, yb, 10, fill=PURPLE, stroke=PURPLE, op=0.2)
        s += text(x, yb + 4, "δ", size=12)
        s += text(x, yb + 32, lab, size=11, fill=PURPLE)
        s += text(x, yb + 50, f"∂L/∂W[{3 - k}] = δ[{3 - k}] a[{2 - k}]ᵀ".replace("a[0]", "x"), size=11)
        if k:
            px = xs[deltas[k - 1][0]]
            s += arrow(px - 12, yb, x + 12, yb, stroke=PURPLE, op=0.85)
    s += arrow(xs[7], y0 + 16, xs[5] + 12, yb - 6, stroke=PURPLE, op=0.85)
    s += text(30, 262, "backward: reuse each δ for the layer below", size=12, fill=PURPLE, anchor="start", weight="600")
    return wrap(
        770,
        278,
        "figBackprop",
        "Forward and backward passes through the 3-2-1 network",
        "Top row, left to right: x, then the linear part and activation of each of the three layers, then the loss; each value is cached as it is computed. Bottom row, right to left: the error δ[3] = a[3] − y at the output, then δ[2] obtained from δ[3] by multiplying by the transpose of W[3] and entrywise by the sigmoid's derivative a[2](1 − a[2]), then δ[1] from δ[2] the same way. Each layer's weight gradient is its δ times the transpose of the activation that fed it.",
        s,
    )


ACTS = (
    ("sigmoid", sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z)), (-0.2, 1.1), [0, 0.5, 1]),
    ("tanh", math.tanh, lambda z: 1 - math.tanh(z) ** 2, (-1.15, 1.15), [-1, 0, 1]),
    ("ReLU", lambda z: max(0.0, z), lambda z: 1.0 if z > 0 else 0.0, (-0.6, 3.0), [0, 1, 2, 3]),
)


def fig_activations():
    """Sigmoid, tanh and ReLU (solid) with their derivatives (dashed)."""
    s = ""
    grid = [-4 + 8 * i / 400 for i in range(401)]
    for k, (name, f, df, ydom, yt) in enumerate(ACTS):
        x0 = 62 + 240 * k
        xdom = (-4.0, 4.0) if name != "ReLU" else (-3.0, 3.0)
        to_px = mapper(xdom, ydom, (x0, x0 + 180), (44, 214))
        s += axes(to_px, xdom, ydom, [xdom[0], 0, xdom[1]], yt, "z", "")
        s += curve([(z, f(z)) for z in grid], to_px, xdom, ydom, stroke=INDIGO, sw=2.0)
        if name == "ReLU":
            s += curve([(z, 0.0) for z in grid if z <= 0], to_px, xdom, ydom, stroke=PURPLE, sw=1.6, dash="5 3")
            s += curve([(z, 1.0) for z in grid if z > 0], to_px, xdom, ydom, stroke=PURPLE, sw=1.6, dash="5 3")
        else:
            s += curve([(z, df(z)) for z in grid], to_px, xdom, ydom, stroke=PURPLE, sw=1.6, dash="5 3")
        s += text(x0 + 90, 26, name, size=13, weight="600")
        peak = max(df(z) for z in grid)
        s += text(x0 + 90, 268, f"largest slope {peak:g}", size=12, fill=PURPLE)
    return wrap(
        760,
        282,
        "figActivations",
        "Three activation functions and their derivatives",
        "Solid curves are the activations, dashed curves their derivatives. The sigmoid's derivative peaks at 0.25 at z = 0 and vanishes for large |z|; tanh's peaks at 1 and also vanishes at both ends; ReLU's is 0 for negative z and exactly 1 for positive z, so it never saturates on the positive side.",
        s,
    )


def ls_problem(standardise, m=200, seed=4):
    """A two-feature least-squares problem; returns (Hessian, minimiser) of J(w) = (1/2m)||Xw - y||²."""
    rng = random.Random(seed)
    x1 = [rng.gauss(2.0, 1.0) for _ in range(m)]
    x2 = [rng.gauss(0.0, 0.4) for _ in range(m)]
    w_true = (1.0, -2.0)
    y = [w_true[0] * a + w_true[1] * b + rng.gauss(0, 0.1) for a, b in zip(x1, x2)]
    if standardise:
        cols = []
        for c in (x1, x2):
            mu = sum(c) / m
            sd = math.sqrt(sum((v - mu) ** 2 for v in c) / m)
            cols.append([(v - mu) / sd for v in c])
        x1, x2 = cols
        ym = sum(y) / m
        y = [v - ym for v in y]
    H = [
        [sum(a * a for a in x1) / m, sum(a * b for a, b in zip(x1, x2)) / m],
        [sum(a * b for a, b in zip(x1, x2)) / m, sum(b * b for b in x2) / m],
    ]
    g = [sum(a * t for a, t in zip(x1, y)) / m, sum(b * t for b, t in zip(x2, y)) / m]
    det = H[0][0] * H[1][1] - H[0][1] ** 2
    w_star = ((H[1][1] * g[0] - H[0][1] * g[1]) / det, (H[0][0] * g[1] - H[0][1] * g[0]) / det)
    return H, w_star


def eig2(H):
    """Eigenvalues (descending) and unit eigenvectors of a symmetric 2x2 matrix."""
    a, b, d = H[0][0], H[0][1], H[1][1]
    tr, disc = a + d, math.sqrt(((a - d) / 2) ** 2 + b * b)
    out = []
    for lam in (tr / 2 + disc, tr / 2 - disc):
        v = (b, lam - a) if abs(b) > 1e-12 else ((1.0, 0.0) if abs(lam - a) < abs(lam - d) else (0.0, 1.0))
        nrm = math.hypot(*v)
        out.append((lam, (v[0] / nrm, v[1] / nrm)))
    return out


def quad_gd(H, w_star, w0, alpha, beta=0.0, tol=1e-6, cap=5000):
    """Gradient descent (beta = 0) or momentum v = βv + (1−β)g on ½(w−w*)ᵀH(w−w*); path and steps to tol."""

    def f(w):
        """Excess cost."""
        d = (w[0] - w_star[0], w[1] - w_star[1])
        return 0.5 * (H[0][0] * d[0] ** 2 + 2 * H[0][1] * d[0] * d[1] + H[1][1] * d[1] ** 2)

    w, v = list(w0), [0.0, 0.0]
    path, f0 = [tuple(w)], f(w0)
    for t in range(1, cap + 1):
        d = (w[0] - w_star[0], w[1] - w_star[1])
        g = (H[0][0] * d[0] + H[0][1] * d[1], H[0][1] * d[0] + H[1][1] * d[1])
        v = [beta * vi + (1 - beta) * gi for vi, gi in zip(v, g)]
        w = [wi - alpha * vi for wi, vi in zip(w, v)]
        path.append(tuple(w))
        fw = f(w)
        if fw < tol * f0:
            return path, t
        if fw > 1e12 * f0:
            return path, cap
    return path, cap


def ellipse(H, c, w_star, n=240):
    """Points of the level set ½(w−w*)ᵀH(w−w*) = c."""
    (l1, v1), (l2, v2) = eig2(H)
    r1, r2 = math.sqrt(2 * c / l1), math.sqrt(2 * c / l2)
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        a, b = r1 * math.cos(t), r2 * math.sin(t)
        pts.append((w_star[0] + a * v1[0] + b * v2[0], w_star[1] + a * v1[1] + b * v2[1]))
    return pts


def normalise_runs():
    """GD on the raw and the standardised problem, each at its best fixed step 2/(λmin+λmax)."""
    out = []
    for std in (False, True):
        H, ws = ls_problem(std)
        (l1, _), (l2, _) = eig2(H)
        alpha = 2 / (l1 + l2)
        w0 = (ws[0] + 0.8, ws[1] + 3.0) if not std else (ws[0] + 0.8, ws[1] + 1.2)
        path, steps = quad_gd(H, ws, w0, alpha)
        out.append((std, H, ws, l1, l2, alpha, path, steps))
    return out


def fig_normalise():
    """Contours of the least-squares cost and the GD path, raw inputs against standardised ones."""
    s = ""
    for k, (std, H, ws, l1, l2, alpha, path, steps) in enumerate(normalise_runs()):
        x0 = 70 + 330 * k
        span = 3.6 if not std else 1.8
        xdom = (ws[0] - span, ws[0] + span)
        ydom = (ws[1] - span, ws[1] + span)
        to_px = mapper(xdom, ydom, (x0, x0 + 230), (46, 276))
        s += rect(x0, 46, 230, 230, fill="none", op=0.0, so=0.5, sw=0.8)
        f0 = 0.5 * sum(H[i][j] * (path[0][i] - ws[i]) * (path[0][j] - ws[j]) for i in range(2) for j in range(2))
        for frac in (0.02, 0.1, 0.3, 1.0, 2.5):
            s += curve(ellipse(H, f0 * frac, ws), to_px, xdom, ydom, stroke=INDIGO, sw=1.0, op=0.45)
        shown = path[:40]
        s += polyline([to_px(*p) for p in shown], stroke=PURPLE, sw=1.5)
        for p in shown[:12]:
            s += circle(*to_px(*p), 2.4, fill=PURPLE, stroke=PURPLE)
        s += circle(*to_px(*ws), 3.5, fill=TEAL, stroke=TEAL)
        title = "standardised inputs" if std else "raw inputs"
        s += text(x0 + 115, 28, title, size=13, weight="600")
        s += text(x0 + 115, 300, f"curvature ratio {l1 / l2:.0f}: {steps} steps", size=12, fill=PURPLE)
    return wrap(
        690,
        312,
        "figNormalise",
        "Gradient descent before and after standardising the inputs",
        "Contours of a least-squares cost in its two weights, with the path of gradient descent from the same relative start. With raw inputs, one feature offset from zero and the other on a small scale, the contours are long thin ellipses and descent zig-zags across the narrow direction while crawling along the long one. After subtracting each feature's mean and dividing by its standard deviation the contours are nearly circles and descent heads straight for the minimum. Each run uses its best fixed learning rate.",
        s,
    )


def relu_depth(c, depth=30, width=64, batch=32, seed=0):
    """RMS of the pre-activations at each layer of a ReLU net with weights N(0, c/n)."""
    rng = random.Random(seed)
    a = [[rng.gauss(0, 1) for _ in range(batch)] for _ in range(width)]
    rms = []
    for _ in range(depth):
        W = [[rng.gauss(0, math.sqrt(c / width)) for _ in range(width)] for _ in range(width)]
        z = matmul(W, a)
        rms.append(math.sqrt(sum(v * v for r in z for v in r) / (width * batch)))
        a = apply(lambda v: max(0.0, v), z)
    return rms


def fig_init():
    """RMS pre-activation against depth for a ReLU network under three weight variances."""
    depth = 30
    runs = [(1, PURPLE, "Var w = 1/n"), (2, TEAL, "Var w = 2/n"), (4, INDIGO, "Var w = 4/n")]
    xdom, ydom = (0, depth), (-16, 16)
    to_px = mapper(xdom, ydom, (90, 520), (40, 280))
    s = axes(to_px, xdom, ydom, [0, 10, 20, 30], [-15, -10, -5, 0, 5, 10, 15], "layer", "log₂ RMS of z")
    for c, colr, lab in runs:
        r = relu_depth(c, depth)
        pts = [(i + 1, math.log2(v)) for i, v in enumerate(r)]
        s += curve(pts, to_px, xdom, ydom, stroke=colr, sw=2.0)
        lx, ly = to_px(*pts[-1])
        s += text(lx + 8, ly + 4, lab, size=12, fill=colr, anchor="start")
    return wrap(
        650,
        326,
        "figInit",
        "Signal size through a deep ReLU network",
        "The root-mean-square of the pre-activations at each of 30 layers of a ReLU network 64 units wide, on a log base 2 scale, for weights drawn with variance 1/n, 2/n and 4/n. With 1/n the signal halves in mean square at every layer and falls by about fifteen powers of two in 30 layers; with 4/n it doubles and grows by as much; with 2/n, the He initialisation, it stays level.",
        s,
    )


MOMENTUM_H = [[1.0, 0.0], [0.0, 100.0]]
MOMENTUM_W0 = (-9.0, 0.5)
ALPHA_GD = 2 / (1 + 100)  # best fixed rate for GD on this quadratic, just under its limit 2/100
ALPHA_MOM = 0.06  # three times GD's limit; only the momentum run survives it


def momentum_runs():
    """Steps to converge for GD and for momentum (beta = 0.9) at the two learning rates the prose compares."""
    out = {}
    for name, alpha, beta in (
        ("gd", ALPHA_GD, 0.0),
        ("mom_same", ALPHA_GD, 0.9),
        ("mom_fast", ALPHA_MOM, 0.9),
        ("gd_fast", ALPHA_MOM, 0.0),
    ):
        out[name] = quad_gd(MOMENTUM_H, (0.0, 0.0), MOMENTUM_W0, alpha, beta, cap=20000)
    return out


def fig_momentum():
    """Paths of gradient descent and of momentum on an elongated quadratic."""
    runs = momentum_runs()
    p_gd, n_gd = runs["gd"]
    p_m, n_m = runs["mom_fast"]
    shown = 60
    pts = p_gd[:shown] + p_m[:shown]
    xdom = (min(p[0] for p in pts) - 0.5, max(p[0] for p in pts) + 0.5)
    ydom = (min(p[1] for p in pts) - 0.15, max(p[1] for p in pts) + 0.15)
    to_px = mapper(xdom, ydom, (40, 640), (40, 260))
    s = rect(40, 40, 600, 220, fill="none", so=0.5, sw=0.8)
    H, w0, ws = MOMENTUM_H, MOMENTUM_W0, (0.0, 0.0)
    f0 = 0.5 * (H[0][0] * w0[0] ** 2 + H[1][1] * w0[1] ** 2)
    for frac in (0.002, 0.02, 0.1, 0.35, 1.0):
        s += curve(ellipse(H, f0 * frac, ws), to_px, xdom, ydom, stroke=INDIGO, sw=1.0, op=0.4)
    s += polyline([to_px(*p) for p in p_gd[:shown]], stroke=PURPLE, sw=1.2)
    s += polyline([to_px(*p) for p in p_m[:shown]], stroke=TEAL, sw=2.0)
    s += circle(*to_px(*w0), 4, fill="currentColor", stroke="none")
    s += circle(*to_px(*ws), 3.5, fill=TEAL, stroke=TEAL)
    s += text(56, 284, f"gradient descent, α = {ALPHA_GD:.4f}: {n_gd} steps", size=12, fill=PURPLE, anchor="start")
    s += text(370, 284, f"momentum, β = 0.9, α = {ALPHA_MOM:g}: {n_m} steps", size=12, fill=TEAL, anchor="start")
    s += text(340, 304, f"first {shown} steps of each; steps counted to a millionth of the starting cost", size=11, op=0.7)
    return wrap(
        680,
        316,
        "figMomentum",
        "Gradient descent and momentum on an elongated bowl",
        "Contours of a quadratic 100 times steeper vertically than horizontally, with the first 60 steps of two methods from the same start. Gradient descent at its best fixed learning rate bounces between the walls of the valley and creeps along its floor. Momentum, stepping along an exponential average of past gradients, cancels the alternating vertical components, which lets it use a learning rate three times past the one at which gradient descent diverges, and it runs along the valley floor to the minimum.",
        s,
    )


FIGURES = {
    "diagram_backprop.svg": fig_backprop,
    "diagram_activations.svg": fig_activations,
    "diagram_normalise.svg": fig_normalise,
    "diagram_init.svg": fig_init,
    "diagram_momentum.svg": fig_momentum,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    # 1. Backprop formulas against finite differences on the 3-2-1 sigmoid network.
    rng = random.Random(0)
    worst = 0.0
    for _ in range(50):
        params = init([4, 3, 2, 1], rng)
        x = col([rng.gauss(0, 1) for _ in range(4)])
        worst = max(worst, fd_check(params, x, rng.randrange(2)))
    print(f"backprop vs finite differences, every parameter, 50 networks: worst error {worst:.1e}")
    assert worst < 1e-7

    # 2. Identity activations: the whole network is one affine map.
    params = init([4, 3, 2, 1], rng)
    (W1, b1), (W2, b2), (W3, b3) = params
    Weff = matmul(W3, matmul(W2, W1))
    beff = add(add(matmul(W3, matmul(W2, b1)), matmul(W3, b2)), b3)
    for _ in range(20):
        x = col([rng.gauss(0, 1) for _ in range(4)])
        deep = forward(params, x, act=lambda v: v)[-1][1][0][0]
        flat = add(matmul(Weff, x), beff)[0][0]
        assert abs(deep - flat) < 1e-12
    print("identity activations: 3-layer network = W3 W2 W1 x + (W3 W2 b1 + W3 b2 + b3)")

    # 3. Symmetry: equal initial weights stay equal within each layer.
    params = [([[0.5] * a for _ in range(b)], [[0.1] for _ in range(b)]) for a, b in zip([4, 3, 2], [3, 2, 1])]
    data = [(col([rng.gauss(0, 1) for _ in range(4)]), rng.randrange(2)) for _ in range(30)]
    for _ in range(200):
        for x, y in data:
            g = backward(params, forward(params, x), y)
            params = [
                (
                    [[w - 0.1 * d for w, d in zip(r, dr)] for r, dr in zip(W, gW)],
                    [[bb[0] - 0.1 * db[0]] for bb, db in zip(b, gb)],
                )
                for (W, b), (gW, gb) in zip(params, g)
            ]
    W1 = params[0][0]
    spread = max(abs(W1[i][j] - W1[0][j]) for i in range(3) for j in range(4))
    print(f"equal init, 6,000 SGD steps: largest difference between hidden neurons' weights {spread:.1e}")
    assert spread < 1e-12

    # 4. Derivative bounds.
    grid = [-6 + 12 * i / 2400 for i in range(2401)]
    print(f"max sigmoid' = {max(ACTS[0][2](z) for z in grid):.4f}; max tanh' = {max(ACTS[1][2](z) for z in grid):.4f}")
    print(
        f"sigmoid'(4) = {ACTS[0][2](4):.4f}; sigmoid' at most 0.25, so ten sigmoid layers scale a gradient by at most {0.25**10:.1e}"
    )

    # 5. Deep linear chain of 1.5 I or 0.5 I.
    print(f"10 layers of 1.5 I: {1.5**10:.1f}; of 0.5 I: {0.5**10:.2e}")

    # 6. Initialisation: Var(w) = 1/n keeps Var(z) = Var(x) for a linear layer; ReLU halves it, so needs 2/n.
    n, trials = 200, 3000
    vz = []
    for _ in range(trials):
        w = [rng.gauss(0, math.sqrt(1 / n)) for _ in range(n)]
        x = [rng.gauss(0, 1) for _ in range(n)]
        vz.append(sum(a * b for a, b in zip(w, x)))
    var = sum(v * v for v in vz) / trials
    print(f"linear layer, n = {n}, Var w = 1/n, unit inputs: Var z = {var:.3f} (theory 1)")
    assert abs(var - 1) < 0.08
    for c in (1, 2, 4):
        r = relu_depth(c)
        ratio = (r[-1] / r[9]) ** (2 / 20)
        print(
            f"ReLU, Var w = {c}/n: mean-square ratio per layer {ratio:.3f} (theory {c / 2:g}); RMS at layer 30 = 2^{math.log2(r[-1]):.1f}"
        )
        assert abs(ratio - c / 2) < 0.15 * c / 2
    print(
        f"Glorot normalised init for fan-in 784, fan-out 256: Var = 2/(784+256) = {2 / 1040:.5f}; 1/fan-in = {1 / 784:.5f}"
    )

    # 7. Standardising the inputs.
    for std, H, ws, l1, l2, alpha, path, steps in normalise_runs():
        print(
            f"{'standardised' if std else 'raw':>12}: eigenvalues {l1:.3f}, {l2:.3f}; ratio {l1 / l2:.1f}; GD steps to 1e-6 {steps}"
        )

    # 8. Momentum: at GD's own rate it barely helps; it pays by surviving a rate at which GD diverges.
    runs = momentum_runs()
    print(
        f"100:1 quadratic, steps to 1e-6: GD at alpha {ALPHA_GD:.4f}: {runs['gd'][1]}; momentum (beta 0.9) at the same alpha: "
        f"{runs['mom_same'][1]}; momentum at alpha {ALPHA_MOM:g}: {runs['mom_fast'][1]}; GD at alpha {ALPHA_MOM:g}: diverges"
    )
    print(f"  stability limits: GD 2/lambda_max = {2 / 100:.3f}; momentum 2(1+beta)/((1-beta) lambda_max) = {2 * 1.9 / (0.1 * 100):.2f}")
    assert runs["gd_fast"][1] == 20000 and runs["mom_fast"][1] * 3 < runs["gd"][1]

    # 9. Mini-batch gradients: variance falls as 1/B (sampling with replacement).
    xs = [rng.gauss(0, 1) for _ in range(1000)]
    ys = [2 * v + rng.gauss(0, 1) for v in xs]
    w = 0.0
    full = sum((w * a - t) * a for a, t in zip(xs, ys)) / len(xs)
    for B in (1, 10, 100):
        gs = []
        for _ in range(4000):
            idx = [rng.randrange(1000) for _ in range(B)]
            gs.append(sum((w * xs[i] - ys[i]) * xs[i] for i in idx) / B)
        v = sum((g - full) ** 2 for g in gs) / len(gs)
        print(f"  batch {B:>3}: gradient variance {v:.3f}")
