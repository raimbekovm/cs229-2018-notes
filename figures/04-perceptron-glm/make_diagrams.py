"""Figures for lecture 4: the perceptron, exponential families, GLMs, softmax regression.

Every curve is the output of the function the chapter defines: the perceptron
update applied to a misclassified point, the logistic function at increasing
slopes, data sampled from the two GLMs the lecture draws, and a softmax
regression fitted by gradient descent to three overlapping clusters.

`__main__` prints every number the prose quotes and asserts the claims the
chapter makes by two routes that must agree: the perceptron's indifference to
its learning rate, its mistake bound on separable data and its failure on XOR;
each exponential-family form against the textbook density; the log-partition
function against the normaliser it is defined to be; its derivatives against
the mean and variance computed by summation; the GLM gradient (y − h)x against
finite differences for three families — and against a probit model, where it
does not hold; and, for softmax, the gradient, the two-class reduction to the
logistic function, and the shift of every θ_k by one vector that changes no
prediction.

Plain Python on purpose, as in lectures 1–3.

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
    poly,
    polyline,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent


def dot(a, b):
    """Inner product of two equal-length sequences."""
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z):
    """The logistic function, written to avoid overflow for large |z|."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def step(z):
    """The perceptron's threshold: 1 when z ≥ 0, else 0."""
    return 1.0 if z >= 0 else 0.0


def normal_cdf(z):
    """Φ(z), the standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def boundary_pts(theta, dom, n=200):
    """Dense points on the line θ0 + θ1 x1 + θ2 x2 = 0 (or θ·x = 0 when len(θ) == 2)."""
    if len(theta) == 2:
        t0, t1, t2 = 0.0, theta[0], theta[1]
    else:
        t0, t1, t2 = theta
    pts = []
    for i in range(n + 1):
        u = dom[0] + (dom[1] - dom[0]) * i / n
        if abs(t2) >= abs(t1):
            pts.append((u, -(t0 + t1 * u) / t2))
        else:
            pts.append((-(t0 + t2 * u) / t1, u))
    return pts


# ---------------------------------------------------------------- perceptron


def perceptron(xs, ys, alpha, epochs, theta=None):
    """Run the perceptron in order over the data; return θ and the list of mistakes per epoch."""
    theta = list(theta) if theta else [0.0] * len(xs[0])
    per_epoch, seq = [], []
    for _ in range(epochs):
        m = 0
        for x, y in zip(xs, ys):
            err = y - step(dot(theta, x))
            if err:
                theta = [t + alpha * err * v for t, v in zip(theta, x)]
                m += 1
            seq.append(err)
        per_epoch.append(m)
        if m == 0:
            break
    return theta, per_epoch, seq


def separable_data():
    """Two clouds with a clear gap, labelled by a known separator; features are (1, x1, x2)."""
    rng = random.Random(4)
    xs, ys = [], []
    while len(xs) < 40:
        x1, x2 = rng.uniform(-3, 3), rng.uniform(-3, 3)
        s = 0.5 + x1 - 0.8 * x2
        if abs(s) < 0.6:
            continue
        xs.append((1.0, x1, x2))
        ys.append(1 if s > 0 else 0)
    return xs, ys, (0.5, 1.0, -0.8)


XOR = ([(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)], [0, 0, 1, 1])

# The update the figure draws: theta, a positive example it misclassifies, theta + alpha x.
P_THETA = (1.0, 2.0)
P_X = (1.5, -1.6)
P_ALPHA = 1.0
P_POS = [(0.3, 2.6), (1.8, 2.7), (2.7, -0.6)]
P_NEG = [(-1.5, -1.0), (-2.0, 0.3), (-0.6, -2.2)]


def fig_perceptron_update():
    """One perceptron step: θ is moved toward a misclassified positive x."""
    s = ""
    dom = (-3.0, 3.0)
    to_px = mapper(dom, dom, (150, 450), (30, 330))
    new = tuple(t + P_ALPHA * v for t, v in zip(P_THETA, P_X))
    ox, oy = to_px(0, 0)
    s += line(150, oy, 450, oy, op=0.25) + line(ox, 30, ox, 330, op=0.25)
    s += curve(boundary_pts(P_THETA, dom), to_px, dom, dom, stroke=INDIGO, sw=1.6, dash="6 4")
    s += curve(boundary_pts(new, dom), to_px, dom, dom, stroke=PURPLE, sw=2.0)
    for p in P_POS:
        s += circle(*to_px(*p), 6, fill=TEAL, stroke=TEAL)
    for p in P_NEG:
        cx, cy = to_px(*p)
        s += rect(cx - 5.5, cy - 5.5, 11, 11, fill="none", stroke=PURPLE, sw=1.6)
    tx, ty = to_px(*P_THETA)
    nx, ny = to_px(*new)
    xx, xy = to_px(*P_X)
    s += arrow(ox, oy, tx, ty, sw=2, op=1, stroke=INDIGO)
    s += arrow(ox, oy, nx, ny, sw=2, op=1, stroke=PURPLE)
    s += arrow(ox, oy, xx, xy, sw=1.4, op=0.8)
    s += arrow(tx, ty, nx, ny, sw=1.4, op=0.8, dash="4 3")
    s += circle(xx, xy, 7, fill=TEAL, stroke=INK_RING, sw=2)
    s += text(tx - 12, ty - 6, "θ", size=15, style="italic", fill=INDIGO)
    s += text(nx + 14, ny + 4, "θ′", size=15, style="italic", fill=PURPLE)
    s += text(xx + 16, xy + 16, "x", size=15, style="italic")
    s += text((tx + nx) / 2 + 22, (ty + ny) / 2 - 8, "αx", size=13, style="italic")
    # Legend, outside the plotted square.
    s += line(470, 60, 498, 60, stroke=INDIGO, sw=1.6, dash="6 4") + text(
        504, 64, "old boundary θᵀx = 0", size=11, anchor="start"
    )
    s += line(470, 82, 498, 82, stroke=PURPLE, sw=2) + text(504, 86, "new boundary θ′ᵀx = 0", size=11, anchor="start")
    s += circle(484, 106, 6, fill=TEAL, stroke=TEAL) + text(504, 110, "y = 1", size=11, anchor="start")
    s += rect(478.5, 122.5, 11, 11, stroke=PURPLE, sw=1.6) + text(504, 132, "y = 0", size=11, anchor="start")
    s += circle(484, 152, 7, fill=TEAL, stroke=INK_RING, sw=2) + text(
        504, 156, "the misclassified x", size=11, anchor="start"
    )
    return wrap(
        720,
        350,
        "figPerceptron",
        "One perceptron update",
        "A two-dimensional input space with three positive points and three negative points. The dashed boundary, perpendicular to the parameter vector theta, puts one positive point x on the negative side. Adding alpha times x to theta gives theta prime, which has rotated toward x; the new solid boundary, perpendicular to theta prime, puts x on the positive side and keeps the other six points where they were.",
        s,
    )


INK_RING = "#1b1b1b"

SLOPES = [(1, INDIGO, "4 3"), (3, TEAL, None), (12, GREEN, None)]


def fig_soft_hard():
    """The step function as the limit of g(kz) as the slope k grows."""
    s = ""
    zdom, ydom = (-6.0, 6.0), (0.0, 1.0)
    to_px = mapper(zdom, ydom, (80, 520), (30, 230))
    s += axes(to_px, zdom, ydom, [-6, -3, 0, 3, 6], [0, 0.5, 1], "z", "g(z)")
    grid = [zdom[0] + i * 0.02 for i in range(601)]
    for k, col, dash in SLOPES:
        s += polyline([to_px(z, sigmoid(k * z)) for z in grid], stroke=col, sw=2, dash=dash)
    s += polyline([to_px(-6, 0), to_px(0, 0), to_px(0, 1), to_px(6, 1)], stroke=PURPLE, sw=2.4)
    ly = 70
    for k, col, dash in SLOPES:
        s += line(550, ly, 576, ly, stroke=col, sw=2, dash=dash) + text(
            582, ly + 4, f"g({k}z)" if k > 1 else "g(z)", size=11, anchor="start", style="italic"
        )
        ly += 20
    s += line(550, ly, 576, ly, stroke=PURPLE, sw=2.4) + text(582, ly + 4, "step (perceptron)", size=11, anchor="start")
    return wrap(
        720,
        280,
        "figSoftHard",
        "The perceptron's step function as a hard sigmoid",
        "The logistic function of k times z for k equal to 1, 3 and 12, drawn with the step function that is 0 for negative z and 1 for z at least 0. As k grows the logistic curve steepens around zero and approaches the step.",
        s,
    )


# ---------------------------------------------------------------- exponential family


def bern_pdf(y, phi):
    """Bernoulli pmf in its textbook form."""
    return phi**y * (1 - phi) ** (1 - y)


def gauss_pdf(y, mu, var=1.0):
    """Gaussian density in its textbook form."""
    return math.exp(-((y - mu) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)


def pois_pdf(y, lam):
    """Poisson pmf in its textbook form."""
    return lam**y * math.exp(-lam) / math.factorial(y)


# (b(y), a(eta), and the closed forms of its first and second derivatives)
FAMILIES = {
    "bernoulli": (
        lambda y: 1.0,
        lambda eta: math.log1p(math.exp(eta)),
        sigmoid,
        lambda eta: sigmoid(eta) * (1 - sigmoid(eta)),
    ),
    "gaussian": (
        lambda y: math.exp(-y * y / 2) / math.sqrt(2 * math.pi),
        lambda eta: eta * eta / 2,
        lambda eta: eta,
        lambda eta: 1.0,
    ),
    "poisson": (
        lambda y: 1.0 / math.factorial(y),
        math.exp,
        math.exp,
        math.exp,
    ),
}


def expfam_pdf(fam, y, eta):
    """p(y; η) = b(y) exp(η y − a(η)) with T(y) = y."""
    b, a, _, _ = FAMILIES[fam]
    return b(y) * math.exp(eta * y - a(eta))


def support(fam):
    """The y values to sum over (a fine grid for the Gaussian), with their quadrature weight."""
    if fam == "bernoulli":
        return [(0, 1.0), (1, 1.0)]
    if fam == "poisson":
        return [(k, 1.0) for k in range(0, 120)]
    h = 0.001
    return [(-30 + i * h, h) for i in range(60001)]


def log_normaliser(fam, eta):
    """Log Σ_y b(y) e^{η y}: the a(η) that b and T force, computed by summation."""
    b, _, _, _ = FAMILIES[fam]
    return math.log(sum(w * b(y) * math.exp(eta * y) for y, w in support(fam)))


def moments(fam, eta):
    """Mean and variance of p(y; η), computed by summation over the support."""
    pts = [(y, w * expfam_pdf(fam, y, eta)) for y, w in support(fam)]
    m = sum(y * p for y, p in pts)
    v = sum((y - m) ** 2 * p for y, p in pts)
    return m, v


# ---------------------------------------------------------------- GLM gradient


def glm_data(fam, theta, m, seed):
    """Sample (x, y) pairs from the GLM with natural parameter η = θᵀx, x = (1, x1)."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(m):
        x = (1.0, rng.uniform(-2, 2))
        eta = dot(theta, x)
        if fam == "bernoulli":
            y = 1 if rng.random() < sigmoid(eta) else 0
        elif fam == "gaussian":
            y = rng.gauss(eta, 1.0)
        else:
            lam, k, p = math.exp(eta), 0, rng.random()
            c = math.exp(-lam)
            acc = c
            while p > acc:
                k += 1
                c *= lam / k
                acc += c
            y = k
        xs.append(x)
        ys.append(y)
    return xs, ys


def glm_loglik(fam, theta, xs, ys):
    """ℓ(θ) = Σ log b(y) + η y − a(η), η = θᵀx."""
    b, a, _, _ = FAMILIES[fam]
    return sum(math.log(b(y)) + dot(theta, x) * y - a(dot(theta, x)) for x, y in zip(xs, ys))


def glm_grad(fam, theta, xs, ys):
    """The lecture's update direction Σ (y − h(x)) x with h = a′(θᵀx)."""
    _, _, h, _ = FAMILIES[fam]
    g = [0.0] * len(theta)
    for x, y in zip(xs, ys):
        r = y - h(dot(theta, x))
        g = [gi + r * v for gi, v in zip(g, x)]
    return g


def num_grad(f, theta, eps=1e-6):
    """Central finite differences of a scalar function of θ."""
    out = []
    for j in range(len(theta)):
        up = list(theta)
        dn = list(theta)
        up[j] += eps
        dn[j] -= eps
        out.append((f(up) - f(dn)) / (2 * eps))
    return out


def probit_loglik(theta, xs, ys):
    """Bernoulli log-likelihood with the non-canonical link φ = Φ(θᵀx)."""
    return sum(
        y * math.log(normal_cdf(dot(theta, x))) + (1 - y) * math.log(1 - normal_cdf(dot(theta, x)))
        for x, y in zip(xs, ys)
    )


def poisson_fit(xs, ys, newton):
    """Fit Poisson regression by gradient ascent or Newton's method; return θ and the iteration count."""
    theta = [0.0, 0.0]
    for it in range(1, 200001):
        g = glm_grad("poisson", theta, xs, ys)
        if newton:
            h = [[0.0, 0.0], [0.0, 0.0]]
            for x in xs:
                w = math.exp(dot(theta, x))
                for i in range(2):
                    for j in range(2):
                        h[i][j] -= w * x[i] * x[j]
            det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
            assert det > 0 and h[0][0] + h[1][1] < 0, "Hessian must be negative definite"
            d0 = (h[1][1] * g[0] - h[0][1] * g[1]) / det
            d1 = (-h[1][0] * g[0] + h[0][0] * g[1]) / det
            theta = [theta[0] - d0, theta[1] - d1]
        else:
            theta = [t + 0.002 * gi for t, gi in zip(theta, g)]
        if max(abs(v) for v in g) < 1e-8:
            return theta, it
    return theta, it


# ---------------------------------------------------------------- GLM pictures

G_THETA = (1.0, 0.8)
B_THETA = (-0.5, 1.2)


def fig_glm_generative():
    """The two generative stories: a Gaussian around a line, a Bernoulli along a sigmoid."""
    s = ""
    xdom, ydom = (0.0, 10.0), (-1.0, 11.0)
    to_px = mapper(xdom, ydom, (70, 330), (40, 260))
    s += axes(to_px, xdom, ydom, [0, 5, 10], [0, 5, 10], "x", "y")
    grid = [i * 0.05 for i in range(201)]
    s += curve([(x, dot(G_THETA, (1, x))) for x in grid], to_px, xdom, ydom, stroke=INDIGO, sw=2)
    for x0 in (2.0, 5.0, 8.0):
        mu = dot(G_THETA, (1, x0))
        dens = [(x0 + 3.5 * gauss_pdf(y, mu), y) for y in [mu - 3 + i * 0.05 for i in range(121)]]
        s += curve(dens, to_px, xdom, ydom, stroke=TEAL, sw=1.4)
        s += line(*to_px(x0, mu - 3), *to_px(x0, mu + 3), stroke=TEAL, sw=1, op=0.6)
    rng = random.Random(2)
    for _ in range(24):
        x = rng.uniform(0.3, 9.7)
        y = rng.gauss(dot(G_THETA, (1, x)), 1.0)
        s += circle(*to_px(x, y), 3.2, fill=PURPLE, stroke=PURPLE, op=0.8)
    s += text(200, 22, "Gaussian: y ~ N(θᵀx, 1)", size=13, weight="600")

    xdom, ydom = (-5.0, 5.0), (-0.1, 1.1)
    to_px = mapper(xdom, ydom, (430, 690), (40, 260))
    s += axes(to_px, xdom, ydom, [-5, 0, 5], [0, 0.5, 1], "x", "y,  P(y = 1 | x)")
    grid = [-5 + i * 0.05 for i in range(201)]
    s += curve([(x, sigmoid(dot(B_THETA, (1, x)))) for x in grid], to_px, xdom, ydom, stroke=INDIGO, sw=2)
    rng = random.Random(5)
    for _ in range(30):
        x = rng.uniform(-4.8, 4.8)
        y = 1 if rng.random() < sigmoid(dot(B_THETA, (1, x))) else 0
        s += circle(*to_px(x, y), 3.2, fill=PURPLE, stroke=PURPLE, op=0.8)
    s += text(560, 22, "Bernoulli: y ~ Bern(g(θᵀx))", size=13, weight="600")
    return wrap(
        720,
        300,
        "figGlmGen",
        "The data-generating process a GLM assumes",
        "Left: a straight line theta transpose x, with a unit-variance Gaussian density drawn sideways at three values of x, and points scattered around the line as samples from those Gaussians. Right: an S-shaped curve giving the probability that y equals 1 at each x, with points at y equals 0 and y equals 1 that are mostly 0 on the left and mostly 1 on the right.",
        s,
    )


def fig_parametrisations():
    """The three parametrisations of a GLM and the maps between them."""
    s = ""
    boxes = [
        (30, "model parameters", "θ", "learned by maximum likelihood"),
        (275, "natural parameter", "η", "one per example"),
        (520, "canonical parameter", "φ,  μ,  λ", "Bernoulli, Gaussian, Poisson"),
    ]
    for x, head, sym, note in boxes:
        s += rect(x, 40, 170, 90, fill=TEAL if x == 30 else "none", op=0.12, stroke=INDIGO, sw=1.4, rx=8)
        s += text(x + 85, 62, head, size=12, weight="600")
        s += text(x + 85, 94, sym, size=18, style="italic")
        s += text(x + 85, 118, note, size=11, op=0.8)
    s += arrow(202, 85, 272, 85, sw=1.6, op=0.9)
    s += text(237, 74, "η = θᵀx", size=12, style="italic")
    s += text(237, 104, "linear", size=10, op=0.7)
    s += arrow(447, 70, 517, 70, sw=1.6, op=0.9)
    s += text(482, 60, "g(η) = a′(η)", size=12, style="italic")
    s += arrow(517, 100, 447, 100, sw=1.6, op=0.9)
    s += text(482, 120, "g⁻¹", size=12, style="italic")
    s += text(360, 160, "g is the canonical response function; g⁻¹ the canonical link", size=11, op=0.8)
    return wrap(
        720,
        175,
        "figParams",
        "Three parametrisations of a generalised linear model",
        "Three boxes in a row. Model parameters theta, the only quantity learned, map linearly to the natural parameter eta equals theta transpose x. Eta maps to the canonical parameter (phi for a Bernoulli, mu for a Gaussian, lambda for a Poisson) through g, the derivative of the log-partition function, and back through its inverse, the canonical link.",
        s,
    )


# ---------------------------------------------------------------- softmax


def softmax(z):
    """Exponentiate and normalise, shifting by the max for stability."""
    m = max(z)
    e = [math.exp(v - m) for v in z]
    t = sum(e)
    return [v / t for v in e]


CLASSES = [("circle", TEAL), ("triangle", INDIGO), ("square", PURPLE)]
CENTRES = [(-2.0, 0.5), (2.0, 2.0), (1.0, -2.0)]


def blob_data():
    """Three overlapping Gaussian clusters; features (1, x1, x2), labels 0..2."""
    rng = random.Random(9)
    xs, ys = [], []
    for k, (cx, cy) in enumerate(CENTRES):
        for _ in range(25):
            xs.append((1.0, rng.gauss(cx, 1.1), rng.gauss(cy, 1.1)))
            ys.append(k)
    return xs, ys


def ce_loss(thetas, xs, ys):
    """Mean cross-entropy −log p̂(y | x), which is the multinomial negative log-likelihood."""
    return -sum(math.log(softmax([dot(t, x) for t in thetas])[y]) for x, y in zip(xs, ys)) / len(xs)


def ce_grad(thetas, xs, ys):
    """∂/∂θ_k of the mean cross-entropy: Σ (p̂_k − 1{y = k}) x / m."""
    g = [[0.0] * len(thetas[0]) for _ in thetas]
    for x, y in zip(xs, ys):
        p = softmax([dot(t, x) for t in thetas])
        for k in range(len(thetas)):
            r = p[k] - (1.0 if y == k else 0.0)
            g[k] = [gi + r * v / len(xs) for gi, v in zip(g[k], x)]
    return g


def fit_softmax(xs, ys, iters=6000, lr=0.5):
    """Gradient descent on the cross-entropy from θ = 0."""
    thetas = [[0.0, 0.0, 0.0] for _ in CLASSES]
    for _ in range(iters):
        g = ce_grad(thetas, xs, ys)
        thetas = [[t - lr * gi for t, gi in zip(tk, gk)] for tk, gk in zip(thetas, g)]
    return thetas


SHIFT = (-4.0, 0.0, 2.0)
SOFTMAX_LOGITS = [-0.5, 0.8, 2.2]  # circle, triangle, square, for an example whose label is triangle
SOFTMAX_LABEL = 1


def glyph(k, cx, cy, r=6):
    """The marker for class k: circle, triangle or square."""
    name, col = CLASSES[k]
    if name == "circle":
        return circle(cx, cy, r, fill=col, stroke=col)
    if name == "triangle":
        return poly([(cx, cy - r - 1), (cx - r - 1, cy + r - 1), (cx + r + 1, cy + r - 1)], fill=col, stroke=col)
    return rect(cx - r + 0.5, cy - r + 0.5, 2 * r - 1, 2 * r - 1, fill=col, stroke=col)


def fig_softmax_pipeline():
    """Logits → exponentials → normalised probabilities, against the one-hot label."""
    s = ""
    e = [math.exp(v) for v in SOFTMAX_LOGITS]
    p = softmax(SOFTMAX_LOGITS)
    onehot = [1.0 if k == SOFTMAX_LABEL else 0.0 for k in range(3)]
    panels = [
        ("logits θₖᵀx", SOFTMAX_LOGITS, (-1.0, 3.0), [-1, 0, 1, 2, 3]),
        ("exp(θₖᵀx)", e, (0.0, 10.0), [0, 5, 10]),
        ("p̂(y | x)", p, (0.0, 1.15), [0, 0.5, 1]),
        ("label p(y)", onehot, (0.0, 1.15), [0, 0.5, 1]),
    ]
    for i, (title, vals, dom, ticks) in enumerate(panels):
        x0 = 48 + i * 178
        to_px = mapper((0.0, 3.0), dom, (x0, x0 + 110), (40, 210))
        s += axes(to_px, (0.0, 3.0), dom, [], ticks, "", "")
        _, zy = to_px(0, 0)
        if dom[0] < 0:
            s += line(x0, zy, x0 + 110, zy, op=0.5)
        for k, v in enumerate(vals):
            bx0, top = to_px(k + 0.2, max(v, 0))
            bx1, bot = to_px(k + 0.8, min(v, 0))
            s += rect(bx0, top, bx1 - bx0, max(bot - top, 0.5), fill=CLASSES[k][1], stroke=CLASSES[k][1], op=0.85)
            s += text((bx0 + bx1) / 2, top - 5 if v >= 0 else bot + 13, f"{v:.2f}", size=10)
            s += glyph(k, (bx0 + bx1) / 2, 236, r=5)
        s += text(x0 + 55, 24, title, size=12, weight="600")
        if i < 3:
            s += arrow(x0 + 118, 125, x0 + 140, 125, sw=1.4, op=0.7)
    loss = -math.log(p[SOFTMAX_LABEL])
    s += text(360, 268, f"cross-entropy = −log p̂(triangle | x) = {loss:.2f}", size=12)
    return wrap(
        720,
        285,
        "figSoftmax",
        "Softmax regression for one example",
        "Four bar charts over the classes circle, triangle and square. The logits are minus 0.5, 0.8 and 2.2; exponentiating makes all three positive; dividing by their sum gives probabilities that add to one, with the square largest. The label is triangle, shown as a one-hot distribution, and the loss is minus the log of the probability assigned to triangle.",
        s,
    )


def argmax(v):
    """Index of the largest entry."""
    return max(range(len(v)), key=lambda i: v[i])


def fig_softmax_shift(thetas):
    """The same decision regions under θ_k and θ_k + c: the lines θ_kᵀx = 0 move, the regions do not."""
    s = ""
    dom = (-5.0, 5.0)
    xs, ys = blob_data()
    shifted = [[t + c for t, c in zip(tk, SHIFT)] for tk in thetas]
    for panel, (x0, th, title) in enumerate([(70, thetas, "fitted θₖ"), (430, shifted, "θₖ + c, same c for every k")]):
        to_px = mapper(dom, dom, (x0, x0 + 240), (40, 280))
        n = 40
        cell = 240 / n
        for i in range(n):
            for j in range(n):
                u = dom[0] + (i + 0.5) * 10 / n
                v = dom[0] + (j + 0.5) * 10 / n
                k = argmax([dot(t, (1, u, v)) for t in th])
                px, py = to_px(u, v)
                s += rect(px - cell / 2, py - cell / 2, cell, cell, fill=CLASSES[k][1], stroke="none", op=0.16)
        s += axes(to_px, dom, dom, [-5, 0, 5], [-5, 0, 5], "x₁", "x₂")
        for x, y in zip(xs, ys):
            if not (dom[0] < x[1] < dom[1] and dom[0] < x[2] < dom[1]):
                continue
            s += glyph(y, *to_px(x[1], x[2]), r=3)
        for k, t in enumerate(th):
            s += curve(boundary_pts(t, dom), to_px, dom, dom, stroke=CLASSES[k][1], sw=2, dash="6 3")
        s += text(x0 + 120, 24, title, size=13, weight="600")
    return wrap(
        720,
        330,
        "figSoftmaxShift",
        "Softmax regions are unchanged when every parameter vector is shifted by the same amount",
        "Two panels of the same three overlapping clusters, shaded by the class the softmax model predicts. Dashed lines mark where each class score theta k transpose x equals zero. On the right every theta k has had the same vector added: the dashed lines have moved to quite different places, but the shaded regions are identical, because only differences between class scores decide the prediction.",
        s,
    )


FIGURES = {
    "diagram_perceptron_update.svg": fig_perceptron_update,
    "diagram_soft_hard.svg": fig_soft_hard,
    "diagram_glm_generative.svg": fig_glm_generative,
    "diagram_parametrisations.svg": fig_parametrisations,
    "diagram_softmax_pipeline.svg": fig_softmax_pipeline,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)
    bxs, bys = blob_data()
    THETAS = fit_softmax(bxs, bys)
    (OUT / "diagram_softmax_shift.svg").write_text(fig_softmax_shift(THETAS))
    print("wrote diagram_softmax_shift.svg")

    # 1. The figure's update: x is misclassified before, correct after, and the
    # other six points keep their side.
    new = [t + P_ALPHA * v for t, v in zip(P_THETA, P_X)]
    assert dot(P_THETA, P_X) < 0 < dot(new, P_X)
    assert all(dot(P_THETA, p) > 0 and dot(new, p) > 0 for p in P_POS)
    assert all(dot(P_THETA, p) < 0 and dot(new, p) < 0 for p in P_NEG)
    print(f"perceptron fig: θᵀx = {dot(P_THETA, P_X):.2f} → θ′ᵀx = {dot(new, P_X):.2f}")

    # 2. From theta = 0 the learning rate only rescales theta: identical mistakes for any rate.
    xs, ys, u = separable_data()
    t1, e1, s1 = perceptron(xs, ys, 1.0, 100)
    t2, e2, s2 = perceptron(xs, ys, 0.001, 100)
    assert s1 == s2 and all(abs(a - 1000 * b) < 1e-9 for a, b in zip(t1, t2))
    print(f"perceptron: α = 1 and α = 0.001 make the same {sum(e1)} mistakes; θ differs by the factor 1000")

    # 3. Novikoff: on separable data, mistakes <= (R / margin)^2 for any unit separator u.
    norm = math.sqrt(dot(u, u))
    gamma = min((1 if y else -1) * dot(u, x) / norm for x, y in zip(xs, ys))
    R = max(math.sqrt(dot(x, x)) for x in xs)
    bound = (R / gamma) ** 2
    assert gamma > 0 and sum(e1) <= bound and e1[-1] == 0
    print(
        f"perceptron: separable, {len(xs)} points: {sum(e1)} mistakes over {len(e1)} epochs; bound (R/γ)² = {bound:.0f}"
    )

    # 4. XOR never settles: some example is wrong in every epoch, at any rate.
    _, ex, _ = perceptron(*XOR, 1.0, 1000)
    _, ex2, _ = perceptron(*XOR, 0.01, 1000)
    assert len(ex) == 1000 and min(ex) > 0 and min(ex2) > 0
    print(f"perceptron: XOR after 1000 epochs still makes {ex[-1]} mistakes per epoch")

    # 5. The step is the limit of g(kz).
    for k in (1, 3, 12, 100):
        gap = max(
            abs(sigmoid(k * z) - step(z))
            for z in [0.5 + i * 0.01 for i in range(600)] + [-0.5 - i * 0.01 for i in range(600)]
        )
        print(f"soft-hard: k = {k}: max |g(kz) − step(z)| for |z| ≥ 0.5 is {gap:.4f}")

    # 6. Each exponential-family form reproduces the textbook density.
    for phi in (0.1, 0.5, 0.83):
        eta = math.log(phi / (1 - phi))
        for y in (0, 1):
            assert abs(expfam_pdf("bernoulli", y, eta) - bern_pdf(y, phi)) < 1e-12
    for mu in (-1.3, 0.0, 2.7):
        for y in (-2.0, 0.4, 3.1):
            assert abs(expfam_pdf("gaussian", y, mu) - gauss_pdf(y, mu)) < 1e-12
    for lam in (0.5, 3.0, 9.0):
        for y in (0, 2, 7):
            assert abs(expfam_pdf("poisson", y, math.log(lam)) - pois_pdf(y, lam)) < 1e-12
    print("expfam: Bernoulli, Gaussian (σ² = 1) and Poisson forms match their densities")

    # 7. a(η) is not a free choice: it is the log of the normaliser that b and T fix.
    for fam in FAMILIES:
        for eta in (-1.5, 0.3, 2.0):
            assert abs(log_normaliser(fam, eta) - FAMILIES[fam][1](eta)) < 1e-6, (fam, eta)
    print("expfam: a(η) = log Σ b(y) e^{ηy} for all three families")

    # 8. The first derivative of a is the mean and the second the variance:
    # finite differences of a against summation.
    h = 1e-4
    for fam in FAMILIES:
        a = FAMILIES[fam][1]
        for eta in (-1.0, 0.5, 1.7):
            d1 = (a(eta + h) - a(eta - h)) / (2 * h)
            d2 = (a(eta + h) - 2 * a(eta) + a(eta - h)) / h**2
            m, v = moments(fam, eta)
            assert abs(d1 - m) < 1e-5 and abs(d2 - v) < 1e-4, (fam, eta, d1, m, d2, v)
    print("expfam: a′(η) = E[y] and a″(η) = Var[y] for all three families")

    # 9. The GLM gradient is sum (y - h(x)) x for every canonical family ...
    truth = {"bernoulli": (-0.3, 1.5), "gaussian": (1.0, 0.8), "poisson": (0.4, 0.6)}
    for fam, tt in truth.items():
        gx, gy = glm_data(fam, tt, 60, 3)
        at = [0.2, -0.4]
        an = glm_grad(fam, at, gx, gy)
        nu = num_grad(lambda t: glm_loglik(fam, t, gx, gy), at)
        assert max(abs(p - q) for p, q in zip(an, nu)) < 1e-5, fam
    print("glm: (y − h(x))x matches the numerical gradient for Bernoulli, Gaussian, Poisson")

    # ... and is not the gradient once the link is not canonical (probit).
    gx, gy = glm_data("bernoulli", truth["bernoulli"], 60, 3)
    at = [0.2, -0.4]
    fake = [0.0, 0.0]
    for x, y in zip(gx, gy):
        r = y - normal_cdf(dot(at, x))
        fake = [f + r * v for f, v in zip(fake, x)]
    real = num_grad(lambda t: probit_loglik(t, gx, gy), at)
    cos = dot(fake, real) / math.sqrt(dot(fake, fake) * dot(real, real))
    rel = math.sqrt(sum((p - q) ** 2 for p, q in zip(fake, real)) / dot(real, real))
    assert rel > 0.1

    def ascend(grad, theta, lr=0.01, iters=200000):
        """Plain gradient ascent until the step vanishes."""
        for _ in range(iters):
            g = grad(theta)
            theta = [t + lr * v for t, v in zip(theta, g)]
            if max(abs(v) for v in g) < 1e-10:
                break
        return theta

    def fake_grad(t):
        """The canonical-looking update applied to the probit model."""
        out = [0.0, 0.0]
        for x, y in zip(gx, gy):
            r = y - normal_cdf(dot(t, x))
            out = [o + r * v for o, v in zip(out, x)]
        return out

    mle = ascend(lambda t: num_grad(lambda q: probit_loglik(q, gx, gy), t, 1e-5), [0.0, 0.0])
    fp = ascend(fake_grad, [0.0, 0.0])
    gap_ll = probit_loglik(mle, gx, gy) - probit_loglik(fp, gx, gy)
    assert max(abs(p - q) for p, q in zip(mle, fp)) > 1e-3 and gap_ll > 0
    print(f"glm: probit link at (0.2, -0.4): (y - Phi)x vs true gradient: relative error {rel:.2f}, cosine {cos:.3f}")
    print(f"glm: probit MLE ({mle[0]:.4f}, {mle[1]:.4f}); fixed point of the look-alike update ({fp[0]:.4f}, {fp[1]:.4f}); log-lik gap {gap_ll:.2e}")

    # 10. Poisson regression: the generic update and Newton reach the same θ.
    px, py = glm_data("poisson", truth["poisson"], 200, 8)
    tg, ig = poisson_fit(px, py, newton=False)
    tn, inn = poisson_fit(px, py, newton=True)
    assert max(abs(p - q) for p, q in zip(tg, tn)) < 1e-6
    print(
        f"glm: Poisson θ = ({tn[0]:.3f}, {tn[1]:.3f}) from truth {truth['poisson']}: gradient ascent {ig} iterations, Newton {inn}"
    )

    # 11. Softmax: the gradient is (p_hat - y) x ...
    an = ce_grad(THETAS, bxs, bys)
    for k in range(3):

        def f(t, k=k):
            """Cross-entropy as a function of the k-th parameter vector alone."""
            th = [list(v) for v in THETAS]
            th[k] = t
            return ce_loss(th, bxs, bys)

        assert max(abs(p - q) for p, q in zip(an[k], num_grad(f, THETAS[k]))) < 1e-7
    print("softmax: (p̂ − y)x matches the numerical gradient of the cross-entropy")

    # ... two classes reduce to the logistic function of (theta1 - theta2)^T x ...
    rng = random.Random(1)
    for _ in range(100):
        a_, b_, x_ = ([rng.gauss(0, 2) for _ in range(3)] for _ in range(3))
        assert abs(softmax([dot(a_, x_), dot(b_, x_)])[0] - sigmoid(dot([p - q for p, q in zip(a_, b_)], x_))) < 1e-12
    print("softmax: K = 2 is logistic regression with θ = θ₁ − θ₂")

    # ... and a common shift of every θ_k changes no probability, so the lines
    # θ_kᵀx = 0 that the lecture draws carry no meaning on their own.
    shifted = [[t + c for t, c in zip(tk, SHIFT)] for tk in THETAS]
    grid = [(1.0, -5 + i * 0.25, -5 + j * 0.25) for i in range(41) for j in range(41)]
    maxdiff = max(
        abs(p - q)
        for x in grid
        for p, q in zip(softmax([dot(t, x) for t in THETAS]), softmax([dot(t, x) for t in shifted]))
    )
    assert maxdiff < 1e-12

    def lecture_picture(th):
        """Fraction of the grid where exactly one score is positive and it is the predicted class."""
        ok = 0
        for x in grid:
            sc = [dot(t, x) for t in th]
            pos = [k for k, v in enumerate(sc) if v > 0]
            ok += len(pos) == 1 and pos[0] == argmax(sc)
        return ok / len(grid)

    acc = sum(argmax([dot(t, x) for t in THETAS]) == y for x, y in zip(bxs, bys)) / len(bxs)
    print(
        f"softmax: fitted θ = {[[round(v, 2) for v in t] for t in THETAS]}, training accuracy {acc:.2f}, loss {ce_loss(THETAS, bxs, bys):.3f}"
    )
    print(f"softmax: shift by c = {SHIFT}: max change in any probability {maxdiff:.1e}")
    print(
        f"softmax: 'exactly one positive score' holds on {lecture_picture(THETAS):.0%} of the plane before the shift, {lecture_picture(shifted):.0%} after"
    )
    p = softmax(SOFTMAX_LOGITS)
    print(
        f"softmax fig: logits {SOFTMAX_LOGITS} → p̂ = {[round(v, 3) for v in p]}, loss {-math.log(p[SOFTMAX_LABEL]):.3f}"
    )
