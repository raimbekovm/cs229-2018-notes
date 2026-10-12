"""Figures for lecture 2: linear regression fitted by gradient descent.

Every figure is computed. The housing data are synthetic — 47 houses, the size
of the Portland set in the official notes, drawn from a seeded generator around
the line the notes report fitting to the real data — so the files are
reproducible and nothing is traced from the lecture's slides.

The figures are drawn by running the algorithms the chapter derives: batch
gradient descent and stochastic gradient descent on the least-squares cost, and
the normal equations for the point they head towards. `__main__` prints every
number the prose quotes and asserts the identities the chapter states, each by
two routes that must agree: the gradient formula against finite differences,
gradient descent against the normal equations, the matrix-derivative rules
against numerical derivatives, and one epoch of stochastic gradient descent
against one batch step.

Plain Python on purpose, as in lecture 1: nothing here needs numpy.

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
    circle,
    line,
    mapper,
    polyline,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent

# The fit the official notes report on the real 47-house Portland data
# (cs229-notes1, section 1), with size in square feet and price in $1000s.
NOTES_THETA = (71.27, 0.1345)


# ---------------------------------------------------------------- linear algebra


def matmul(a, b):
    """Product of two matrices stored as lists of rows."""
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    """Transpose of a matrix stored as a list of rows."""
    return [list(r) for r in zip(*a)]


def trace(a):
    """Sum of the diagonal entries of a square matrix."""
    return sum(a[i][i] for i in range(len(a)))


def solve(a, b):
    """Solve a small dense system a x = b by Gaussian elimination with pivoting."""
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        if abs(m[c][c]) < 1e-12:
            raise ZeroDivisionError("singular system")
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in reversed(range(n)):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def eig2(h):
    """Eigenvalues (ascending) and unit eigenvectors of a symmetric 2×2 matrix."""
    a, b, d = h[0][0], h[0][1], h[1][1]
    mid, rad = (a + d) / 2, math.hypot((a - d) / 2, b)
    out = []
    for lam in (mid - rad, mid + rad):
        v = (b, lam - a) if abs(b) > 1e-12 else ((1.0, 0.0) if abs(lam - a) < 1e-12 else (0.0, 1.0))
        n = math.hypot(*v)
        out.append((lam, (v[0] / n, v[1] / n)))
    return out


# ---------------------------------------------------------------- the model


def housing_data(n=47, seed=2):
    """Synthetic house sizes (1000 ft²) and prices ($1000s), shaped like the Portland set."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.uniform(0.85, 4.5)
        y = NOTES_THETA[0] + NOTES_THETA[1] * 1000 * x + rng.gauss(0, 55)
        xs.append(round(x, 3))
        ys.append(round(max(y, 120.0), 1))
    return xs, ys


def design(xs):
    """Design matrix with the intercept feature x₀ = 1 in the first column."""
    return [[1.0, x] for x in xs]


def cost(theta, rows, ys):
    """J(θ) = ½ Σ (θᵀx⁽ⁱ⁾ − y⁽ⁱ⁾)²."""
    return 0.5 * sum((sum(t * v for t, v in zip(theta, r)) - y) ** 2 for r, y in zip(rows, ys))


def grad(theta, rows, ys):
    """∂J/∂θⱼ = Σ (h(x⁽ⁱ⁾) − y⁽ⁱ⁾) xⱼ⁽ⁱ⁾ — the LMS gradient, summed over the batch."""
    g = [0.0] * len(theta)
    for r, y in zip(rows, ys):
        e = sum(t * v for t, v in zip(theta, r)) - y
        for j, v in enumerate(r):
            g[j] += e * v
    return g


def normal_equations(rows, ys):
    """Θ = (XᵀX)⁻¹Xᵀy, solved rather than inverted."""
    xtx = matmul(transpose(rows), rows)
    xty = [sum(r[j] * y for r, y in zip(rows, ys)) for j in range(len(rows[0]))]
    return solve(xtx, xty)


def batch_gd(rows, ys, alpha, steps, theta=None):
    """Batch gradient descent; returns every iterate including the start."""
    theta = list(theta or [0.0] * len(rows[0]))
    path = [theta[:]]
    for _ in range(steps):
        g = grad(theta, rows, ys)
        theta = [t - alpha * gj for t, gj in zip(theta, g)]
        path.append(theta[:])
    return path


def sgd(rows, ys, alpha, epochs, theta=None, seed=0, decay=False):
    """Stochastic gradient descent, one example per update, shuffled each epoch."""
    rng = random.Random(seed)
    theta = list(theta or [0.0] * len(rows[0]))
    path = [theta[:]]
    order = list(range(len(rows)))
    t = 0
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            a = alpha / (1 + t / len(rows)) if decay else alpha
            e = sum(th * v for th, v in zip(theta, rows[i])) - ys[i]
            theta = [th - a * e * v for th, v in zip(theta, rows[i])]
            path.append(theta[:])
            t += 1
    return path


def hessian(rows):
    """XᵀX, the (constant) Hessian of J."""
    return matmul(transpose(rows), rows)


def ellipse(center, h, level, n=120):
    """Points of the contour J = J(θ*) + level of a quadratic with Hessian h."""
    (l1, v1), (l2, v2) = eig2(h)
    r1, r2 = math.sqrt(2 * level / l1), math.sqrt(2 * level / l2)
    pts = []
    for k in range(n + 1):
        a = 2 * math.pi * k / n
        c, s = r1 * math.cos(a), r2 * math.sin(a)
        pts.append((center[0] + c * v1[0] + s * v2[0], center[1] + c * v1[1] + s * v2[1]))
    return pts


# ---------------------------------------------------------------- drawing helpers


def axes(to_px, xdom, ydom, xticks, yticks, xlabel, ylabel, fmt="{:g}"):
    """Axis lines, ticks and labels for a mapped box."""
    x0, y0 = to_px(xdom[0], ydom[0])
    x1, _ = to_px(xdom[1], ydom[0])
    _, y1 = to_px(xdom[0], ydom[1])
    s = line(x0, y0, x1, y0, op=0.7) + line(x0, y0, x0, y1, op=0.7)
    for t in xticks:
        px, _ = to_px(t, ydom[0])
        s += line(px, y0, px, y0 + 4, op=0.7) + text(px, y0 + 17, fmt.format(t), size=11, op=0.8)
    for t in yticks:
        _, py = to_px(xdom[0], t)
        s += line(x0 - 4, py, x0, py, op=0.7) + text(x0 - 8, py + 4, fmt.format(t), size=11, anchor="end", op=0.8)
    s += text((x0 + x1) / 2, y0 + 36, xlabel, size=12)
    s += f'  <text x="{x0 - 40:.1f}" y="{(y0 + y1) / 2:.1f}" font-size="12" fill="currentColor" text-anchor="middle" transform="rotate(-90 {x0 - 40:.1f} {(y0 + y1) / 2:.1f})">{ylabel}</text>\n'
    return s


def clip_box(pts, to_px, xdom, ydom):
    """Split a curve into the runs that lie inside the plotted domain, in pixels."""
    runs, cur = [], []
    for x, y in pts:
        if xdom[0] <= x <= xdom[1] and ydom[0] <= y <= ydom[1]:
            cur.append(to_px(x, y))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def curve(pts, to_px, xdom, ydom, **kw):
    """A polyline for each in-domain run of a curve, so nothing crosses the axes."""
    return "".join(polyline(r, **kw) for r in clip_box(pts, to_px, xdom, ydom) if len(r) > 1)


# ---------------------------------------------------------------- figures

# The batch learning rate used in the figures, as a fraction of the stability
# limit 2/lambda_max. Half the limit damps the steep direction in one step.
GD_ALPHA_FRACTION = 0.5


def gd_setup():
    """Data, design matrix, the optimum, and the batch step size used in the figures."""
    xs, ys = housing_data()
    rows = design(xs)
    star = normal_equations(rows, ys)
    lmax = eig2(hessian(rows))[1][0]
    return xs, ys, rows, star, GD_ALPHA_FRACTION * 2 / lmax


LINE_STEPS = (0, 1, 2, 3)
SLOW_FRACTION = 0.3


def fig_gd_lines():
    """The hypothesis after successive steps of batch gradient descent, drawn over the data."""
    xs, ys, rows, star, alpha = gd_setup()
    path = batch_gd(rows, ys, alpha * SLOW_FRACTION / GD_ALPHA_FRACTION, max(LINE_STEPS))
    xdom, ydom = (0.0, 5.0), (0.0, 800.0)
    to_px = mapper(xdom, ydom, (80, 610), (30, 300))
    s = axes(to_px, xdom, ydom, [0, 1, 2, 3, 4, 5], [0, 200, 400, 600, 800], "size of house (1000 ft²)", "price ($1000s)")
    for x, y in zip(xs, ys):
        s += circle(*to_px(x, y), 3.2, fill="currentColor", stroke="none", op=0.5)
    cols = [PURPLE, PURPLE, INDIGO, INDIGO]
    for k, col in zip(LINE_STEPS, cols):
        t = path[k]
        p0, p1 = to_px(0.2, t[0] + t[1] * 0.2), to_px(4.8, t[0] + t[1] * 4.8)
        s += line(*p0, *p1, stroke=col, sw=1.8, op=0.8, dash="6 4")
        s += text(p1[0] + 6, p1[1] + (4 if k else -6), f"step {k}", size=11, anchor="start", fill=col)
    p0, p1 = to_px(0.2, star[0] + star[1] * 0.2), to_px(4.8, star[0] + star[1] * 4.8)
    s += line(*p0, *p1, stroke=TEAL, sw=2.2)
    s += text(p1[0] + 6, p1[1] - 8, "converged", size=11, anchor="start", fill=TEAL)
    return wrap(
        690,
        350,
        "figGdLines",
        "Gradient descent seen in the data",
        "Scatter of house size against price. A horizontal line at zero is the hypothesis at initialisation; the lines after one, two and three steps of gradient descent rotate up towards the data, and the converged line runs through it.",
        s,
    ), path


SGD_RATE, SGD_PASSES = 0.01, 40


def fig_contours():
    """Contours of J with the paths of batch and stochastic gradient descent from θ = 0."""
    _, ys, rows, star, alpha = gd_setup()
    h = hessian(rows)
    start = [0.0, 0.0]
    bpath = batch_gd(rows, ys, alpha, 400, theta=start)
    spath = sgd(rows, ys, SGD_RATE, SGD_PASSES, theta=start, seed=3)
    jmin = cost(star, rows, ys)
    top = cost(start, rows, ys) - jmin
    levels = [top * f for f in (1.0, 0.3, 0.08, 0.02, 0.005, 0.0012)]
    xdom, ydom = (-20.0, 160.0), (-10.0, 170.0)

    def panel(ox, title, path, col, dots):
        to_px = mapper(xdom, ydom, (ox + 60, ox + 320), (40, 290))
        s = axes(to_px, xdom, ydom, [0, 50, 100, 150], [0, 50, 100, 150], "θ₀ (intercept)", "θ₁ (per 1000 ft²)")
        for lv in levels:
            s += curve(ellipse(star, h, lv, n=400), to_px, xdom, ydom, stroke="currentColor", sw=1, op=0.35)
        pts = [to_px(*p) for p in path]
        s += polyline(pts, stroke=col, sw=1.6 if dots else 0.9, op=0.95)
        if dots:
            for p in pts[:6]:
                s += circle(*p, 2.6, fill=col, stroke="none")
        sx, sy = to_px(*start)
        s += line(sx - 5, sy - 5, sx + 5, sy + 5, stroke=PURPLE, sw=2) + line(sx - 5, sy + 5, sx + 5, sy - 5, stroke=PURPLE, sw=2)
        s += text(sx + 9, sy + 4, "θ = 0", size=11, anchor="start")
        cx, cy = to_px(*star)
        s += circle(cx, cy, 4, fill=PURPLE, stroke="none")
        s += text(ox + 190, 22, title, size=13, weight="600")
        return s

    s = panel(0, f"batch, {len(bpath) - 1} steps", bpath, INDIGO, True)
    s += panel(360, f"stochastic, {SGD_PASSES} passes", spath, TEAL, False)
    return wrap(
        720,
        350,
        "figContours",
        "Batch and stochastic gradient descent on the contours of J",
        "Two copies of the same long, thin elliptical contours of the least-squares cost, both starting from theta equals zero. On the left batch gradient descent drops straight across the contours into the valley in one step and then slides slowly along its floor towards the minimum. On the right stochastic gradient descent takes the same route noisily and keeps jittering near the floor.",
        s,
    ), (bpath, spath)



def fig_learning_rate():
    """Gradient descent on J(θ) = θ² for a small, a good and a too-large learning rate."""
    cases = [(0.05, "α = 0.05: slow"), (0.35, "α = 0.35: fast"), (1.05, "α = 1.05: diverges")]
    s = ""
    xdom, ydom = (-2.6, 2.6), (0.0, 6.5)
    for k, (a, title) in enumerate(cases):
        ox = 20 + k * 230
        to_px = mapper(xdom, ydom, (ox + 20, ox + 210), (40, 230))
        s += line(*to_px(xdom[0], 0), *to_px(xdom[1], 0), op=0.6)
        grid = [xdom[0] + i * (xdom[1] - xdom[0]) / 120 for i in range(121)]
        s += curve([(x, x * x) for x in grid], to_px, xdom, ydom, stroke="currentColor", sw=1.4, op=0.6)
        th, pts = 1.8 * (1 if k != 2 else 0.9), []
        for _ in range(6 if k == 2 else 8):
            pts.append((th, th * th))
            th -= a * 2 * th
        pts = [p for p in pts if p[1] <= ydom[1]]
        px = [to_px(*p) for p in pts]
        for p, q in zip(px, px[1:]):
            s += arrow(*p, *q, sw=1.3, op=0.9, head=5, stroke=INDIGO)
        for p in px:
            s += circle(*p, 2.8, fill=PURPLE, stroke="none")
        s += text(ox + 115, 22, title, size=12, weight="600")
        s += text(ox + 115, 254, "θ", size=12, style="italic")
    return wrap(
        710,
        270,
        "figLearningRate",
        "The learning rate",
        "Three copies of the parabola J of theta equals theta squared. With a small learning rate the iterates creep down one side; with a moderate one they reach the bottom in a few steps; with one above the stability limit each step overshoots to the other side, higher than before.",
        s,
    )


def bumpy(t):
    """A one-parameter cost with two minima, standing in for Ng's hilly surface."""
    return 0.12 * t**4 - 1.0 * t**2 + 0.35 * t + 3.0


def bumpy_grad(t):
    """Derivative of `bumpy`."""
    return 0.48 * t**3 - 2.0 * t + 0.35


def fig_local_minima():
    """Gradient descent from two starting points reaching two different minima."""
    xdom, ydom = (-3.4, 3.4), (0.0, 6.0)
    to_px = mapper(xdom, ydom, (60, 620), (30, 250))
    s = line(*to_px(xdom[0], 0), *to_px(xdom[1], 0), op=0.6)
    grid = [xdom[0] + i * (xdom[1] - xdom[0]) / 200 for i in range(201)]
    s += curve([(x, bumpy(x)) for x in grid], to_px, xdom, ydom, stroke="currentColor", sw=1.6, op=0.65)
    ends = []
    for t0, col in ((-0.35, INDIGO), (0.45, TEAL)):
        t, pts = t0, []
        for _ in range(60):
            pts.append((t, bumpy(t)))
            t -= 0.12 * bumpy_grad(t)
        ends.append(t)
        px = [to_px(*p) for p in pts]
        for p, q in zip(px[:6], px[1:7]):
            s += arrow(*p, *q, sw=1.4, op=0.9, head=5, stroke=col)
        s += circle(*px[0], 4, fill=PURPLE, stroke="none")
        s += circle(*px[-1], 4, fill=col, stroke="none")
    s += text(to_px(-0.35, 0)[0], to_px(0, bumpy(-0.35))[1] - 12, "two nearby starts", size=11, anchor="end")
    s += text(to_px(ends[0], 0)[0], to_px(0, bumpy(ends[0]))[1] + 22, "global minimum", size=11, fill=INDIGO)
    s += text(to_px(ends[1], 0)[0], to_px(0, bumpy(ends[1]))[1] + 22, "local minimum", size=11, fill=TEAL)
    s += text(340, 272, "θ", size=12, style="italic")
    return wrap(
        680,
        285,
        "figLocalMinima",
        "Where gradient descent ends depends on where it starts",
        "A curve with two valleys. Gradient descent started just left of the central hump runs down into the deeper valley on the left; started just right of it, it runs into the shallower valley on the right and stops there.",
        s,
    ), ends


FIGURES = {
    "diagram_gd_lines.svg": fig_gd_lines,
    "diagram_contours.svg": fig_contours,
    "diagram_learning_rate.svg": fig_learning_rate,
    "diagram_local_minima.svg": fig_local_minima,
}


# ---------------------------------------------------------------- checks


def num_grad_matrix(f, a, eps=1e-6):
    """Central-difference gradient of a scalar function of a matrix."""
    g = [[0.0] * len(a[0]) for _ in a]
    for i in range(len(a)):
        for j in range(len(a[0])):
            hi = [r[:] for r in a]
            lo = [r[:] for r in a]
            hi[i][j] += eps
            lo[i][j] -= eps
            g[i][j] = (f(hi) - f(lo)) / (2 * eps)
    return g


def max_abs_diff(a, b):
    """Largest entrywise difference between two matrices."""
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


if __name__ == "__main__":
    extras = {}
    for name, fn in FIGURES.items():
        out = fn()
        if isinstance(out, tuple):
            out, extras[name] = out
        (OUT / name).write_text(out)
        print("wrote", name)

    xs, ys, rows, star, alpha = gd_setup()
    m = len(rows)
    print(f"data: m = {m}; normal equations θ = ({star[0]:.2f}, {star[1]:.2f}) per 1000 ft²")

    # 1. The LMS gradient against central differences of J.
    th = [40.0, 90.0]
    g = grad(th, rows, ys)
    for j in range(2):
        e = [0.0, 0.0]
        e[j] = 1e-4
        fd = (cost([a + b for a, b in zip(th, e)], rows, ys) - cost([a - b for a, b in zip(th, e)], rows, ys)) / 2e-4
        assert abs(fd - g[j]) < 1e-4 * max(1.0, abs(g[j])), (fd, g[j])
    print("gradient: Σ(h−y)xⱼ matches finite differences")

    # 2. Gradient descent lands where the normal equations say.
    final = batch_gd(rows, ys, alpha, 3000)[-1]
    assert max(abs(a - b) for a, b in zip(final, star)) < 1e-6
    print(f"batch GD (α = {alpha:.5f}) converges to the normal-equation θ")

    # 3. The stability limit a < 2/lambda_max, checked on both sides.
    lam = eig2(hessian(rows))
    lmax = lam[1][0]
    for f, should in ((0.99, True), (1.01, False)):
        p = batch_gd(rows, ys, f * 2 / lmax, 4000)
        ok = cost(p[-1], rows, ys) < cost(p[0], rows, ys)
        assert ok == should
    print(f"stability: λmax(XᵀX) = {lmax:.0f}, limit 2/λmax = {2 / lmax:.5f}; condition number {lmax / lam[0][0]:.0f}")
    # The same data with size in square feet rather than thousands: the limit
    # collapses by roughly a factor of a million, which is why the 0.01 rule of
    # thumb comes with "if the features are scaled".
    raw = [[1.0, x * 1000] for x in xs]
    lraw = eig2(hessian(raw))[1][0]
    scaled_x = [(x - sum(xs) / m) / (max(xs) - min(xs)) * 2 for x in xs]
    lsc = eig2(hessian(design(scaled_x)))[1][0]
    print(f"stability: raw ft² limit 2/λmax = {2 / lraw:.2e}; features scaled to about ±1: {2 / lsc:.4f} (m = {m})")
    assert 2 / lsc > 0.01 > 2 / lraw

    # 4. Gradient orthogonal to the contour, at one point of one ellipse.
    h = hessian(rows)
    ell = ellipse(star, h, 5e4, n=400)
    p, q = ell[37], ell[38]
    gp = grad(list(p), rows, ys)
    tang = (q[0] - p[0], q[1] - p[1])
    cosang = (gp[0] * tang[0] + gp[1] * tang[1]) / (math.hypot(*gp) * math.hypot(*tang))
    assert abs(cosang) < 0.02
    print(f"contours: cos(gradient, contour tangent) = {cosang:.4f}")

    # 5. The student question about the SGD learning rate. Because J is a sum,
    # an epoch of per-example updates at rate a adds up, to first order in a,
    # to one batch step at the same a: the gap between them shrinks like a².
    # So dividing a by m would make a pass of SGD m times slower than a batch
    # step. What SGD tolerates instead is a far larger a: one update is stable
    # for a < 2/|x_i|², against 2/lambda_max(XtX) for the whole batch.
    start = [0.0, 0.0]
    gaps = []
    for a in (alpha, alpha / 10):
        b = batch_gd(rows, ys, a, 1, theta=start)[-1]
        s_ = start[:]
        for r, y in zip(rows, ys):
            e = sum(t * v for t, v in zip(s_, r)) - y
            s_ = [t - a * e * v for t, v in zip(s_, r)]
        gaps.append(max(abs(u - v) for u, v in zip(b, s_)) / max(abs(v) for v in b))
    assert gaps[1] < gaps[0] / 5
    print(f"sgd vs batch: one epoch at a vs one batch step at a, relative gap {gaps[0]:.3f} at a = {alpha:.5f}, {gaps[1]:.4f} at a/10")
    per_example = min(2 / sum(v * v for v in r) for r in rows)
    print(f"sgd vs batch: per-update stability limit >= {per_example:.4f}, batch {2 / lmax:.5f}: ratio {per_example * lmax / 2:.0f} (m = {m})")
    # 6. Matrix derivatives. Ng's example f(A) = A₁₁ + A₁₂².
    a = [[5.0, 6.0], [7.0, 8.0]]
    f = lambda A: A[0][0] + A[0][1] ** 2  # noqa: E731
    assert f(a) == 41.0
    assert max_abs_diff(num_grad_matrix(f, a), [[1.0, 12.0], [0.0, 0.0]]) < 1e-6
    print("matrix derivative: f(A) = A11 + A12² at [[5,6],[7,8]] is 41, gradient [[1,12],[0,0]]")
    rng = random.Random(7)
    A = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(2)]
    B = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(3)]
    C = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
    assert max_abs_diff(num_grad_matrix(lambda X: trace(matmul(X, B)), A), transpose(B)) < 1e-6
    assert abs(trace(matmul(matmul(A, B), C)) - trace(matmul(matmul(C, A), B))) < 1e-12
    print("trace: ∇A tr(AB) = Bᵀ and tr(ABC) = tr(CAB) hold numerically")
    # The lecture's "matrix version of d/da a²c = 2ac": ∇A tr(A Aᵀ C). The true
    # gradient is (C + Cᵀ)A, which is 2CA only when C is symmetric.
    sq = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
    f2 = lambda X: trace(matmul(matmul(X, transpose(X)), sq))  # noqa: E731
    num = num_grad_matrix(f2, A)
    true = matmul([[sq[i][j] + sq[j][i] for j in range(2)] for i in range(2)], A)
    naive = [[2 * v for v in r] for r in matmul(sq, A)]
    assert max_abs_diff(num, true) < 1e-6
    gap = max_abs_diff(num, naive)
    assert gap > 0.05
    sym = [[(sq[i][j] + sq[j][i]) / 2 for j in range(2)] for i in range(2)]
    f3 = lambda X: trace(matmul(matmul(X, transpose(X)), sym))  # noqa: E731
    assert max_abs_diff(num_grad_matrix(f3, A), [[2 * v for v in r] for r in matmul(sym, A)]) < 1e-6
    print(f"trace: ∇A tr(AAᵀC) = (C+Cᵀ)A; 2CA is off by {gap:.3f} for a non-symmetric C, exact for symmetric C")

    # 7. The normal equations in the matrix form against J written as a sum.
    v = [a - b for a, b in zip([sum(t * x for t, x in zip(th, r)) for r in rows], ys)]
    assert abs(0.5 * sum(e * e for e in v) - cost(th, rows, ys)) < 1e-6
    # XtX theta - Xt y is the gradient: zero at θ*.
    assert max(abs(x) for x in grad(star, rows, ys)) < 1e-6

    # 8. A duplicated feature: XᵀX is singular, yet the predictions are fixed.
    dup = [[1.0, x, x] for x in xs]
    try:
        normal_equations(dup, ys)
        singular = False
    except ZeroDivisionError:
        singular = True
    assert singular
    # Every θ with θ₁ + θ₂ equal to the single-feature slope fits identically.
    for split in (0.0, 0.5, 3.0):
        t = [star[0], star[1] * split, star[1] * (1 - split)]
        assert abs(cost(t, dup, ys) - cost(star, rows, ys)) < 1e-6
    print("duplicate feature: XᵀX singular; every split of the slope gives the same J")

    lpath = extras["diagram_gd_lines.svg"]
    print("lines: " + ", ".join(f"step {k}: θ = ({t[0]:.0f}, {t[1]:.0f})" for k, t in enumerate(lpath)))
    bpath, spath = extras["diagram_contours.svg"]
    print(f"contours: batch after 1 step θ = ({bpath[1][0]:.1f}, {bpath[1][1]:.1f}), after 400 ({bpath[-1][0]:.1f}, {bpath[-1][1]:.1f}); SGD ends ({spath[-1][0]:.1f}, {spath[-1][1]:.1f})")
    n_to = next(k for k, t in enumerate(batch_gd(rows, ys, alpha, 20000)) if abs(t[0] - star[0]) < 0.5)
    print(f"contours: batch at half the limit needs {n_to} steps to bring θ0 within 0.5 of the optimum")
    print(
        f"contours: batch path {len(bpath) - 1} steps; SGD path {len(spath) - 1} updates ({(len(spath) - 1) // m} passes)"
    )
    ends = extras["diagram_local_minima.svg"]
    print(f"local minima: starts -0.35 and 0.45 end at {ends[0]:.2f} and {ends[1]:.2f}")
    assert ends[0] < 0 < ends[1]
    print(f"notes' fit on the real data: θ0 = {NOTES_THETA[0]}, θ1 = {NOTES_THETA[1]} per ft²")
