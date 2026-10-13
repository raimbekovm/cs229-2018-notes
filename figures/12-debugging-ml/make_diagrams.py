"""Figures for lecture 12: diagnosing a learning algorithm that does not work.

Two computed figures and two structural ones. The learning curves are the
exact expected training and test errors of least squares, in closed form, for
a model with too many features and one with too few. The second computed
figure is the optimisation-versus-objective diagnostic run on a toy spam
problem: the logistic-regression objective along the ray through the SVM's
parameters, against the values reached by logistic regression. The other two
redraw the helicopter diagnostics as a flow and the lecture's face-recognition
pipeline with its error-analysis numbers.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: the closed-form learning curves against
a Monte Carlo fit of least squares; that the excess test error of least
squares falls like 1/m, not 1/sqrt(m); that weighted accuracy has zero gradient
almost everywhere; that the converged logistic regression beats every
rescaling of the SVM on its own objective; that the diagnostic's verdict can
change with the scale the SVM happened to return; the arithmetic of the
lecture's error-analysis and ablation tables; and, on a Gaussian toy with two
redundant features, that leave-one-out and cumulative ablation attribute the
same accuracy differently.

Plain Python on purpose, as in lectures 1-11.

Run from the repository root.
"""

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
    arrow,
    axes,
    curve,
    line,
    mapper,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------- small linear algebra


def solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [[*row, bb] for row, bb in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def dot(a, b):
    """Inner product."""
    return sum(x * y for x, y in zip(a, b))


def Phi(z):
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------- learning curves

SIGMA2 = 1.0
TARGET = 1.15  # desired test error, in units of the noise variance
VAR_P = 20  # high variance: 20 features fitted, of which 5 matter
BIAS_P = 2  # high bias: 2 features fitted
BIAS_OMITTED = 0.6  # variance the 2-feature model cannot explain


def ls_train(m, p, s2):
    """Expected training MSE of least squares with p Gaussian features, m examples."""
    return 0.0 if m <= p else s2 * (1 - p / m)


def ls_test(m, p, s2):
    """Expected test MSE of least squares with p Gaussian features, m > p + 1 examples."""
    return s2 * (1 + p / (m - p - 1))


def mc_learning_curve(m, p, p_true, omitted, trials, rng):
    """Monte Carlo train and test MSE of least squares, for the closed form to agree with."""
    tr = te = 0.0
    w_true = [1.0] * p_true
    sd_om = math.sqrt(omitted)
    for _ in range(trials):
        X = [[rng.gauss(0, 1) for _ in range(p)] for _ in range(m)]
        y = [dot(x[:p_true], w_true) + sd_om * rng.gauss(0, 1) + rng.gauss(0, 1) for x in X]
        A = [[sum(X[i][a] * X[i][b] for i in range(m)) for b in range(p)] for a in range(p)]
        w = solve(A, [sum(X[i][a] * y[i] for i in range(m)) for a in range(p)])
        tr += sum((dot(x, w) - t) ** 2 for x, t in zip(X, y)) / m
        # Test error in expectation over a fresh x: noise plus the parameter error.
        err = [wi - (1.0 if i < p_true else 0.0) for i, wi in enumerate(w)]
        te += SIGMA2 + omitted + dot(err, err)
    return tr / trials, te / trials


def fig_learning_curves():
    """Expected training and test error against training-set size, two regimes."""
    panels = (
        ("High variance: 20 features, 5 relevant", VAR_P, SIGMA2),
        ("High bias: 2 features, the rest omitted", BIAS_P, SIGMA2 + BIAS_OMITTED),
    )
    xdom, ydom = (0, 400), (0, 2.4)
    s = ""
    for k, (title, p, s2) in enumerate(panels):
        x0 = 60 + k * 330
        to_px = mapper(xdom, ydom, (x0, x0 + 260), (50, 250))
        s += axes(
            to_px, xdom, ydom, [0, 100, 200, 300, 400], [0, 1, 2], "training-set size m", "error" if k == 0 else ""
        )
        s += text(x0 + 130, 34, title, size=12.5, weight="bold")
        ms = [i / 2 for i in range(2, 801)]
        s += curve([(m, ls_train(m, p, s2)) for m in ms], to_px, xdom, ydom, stroke=TEAL, sw=2.0)
        s += curve([(m, ls_test(m, p, s2)) for m in ms if m > p + 1.5], to_px, xdom, ydom, stroke=PURPLE, sw=2.0)
        s += curve([(m, TARGET) for m in ms], to_px, xdom, ydom, stroke="currentColor", sw=1.0, dash="5 4", op=0.6)
        s += curve([(m, s2) for m in ms], to_px, xdom, ydom, stroke=INDIGO, sw=0.9, dash="2 3", op=0.7)
    # Labels placed by hand at points the curves do not pass through.
    to_l = mapper(xdom, ydom, (60, 320), (50, 250))
    to_r = mapper(xdom, ydom, (390, 650), (50, 250))
    s += text(*_off(to_l(300, ls_test(300, VAR_P, SIGMA2)), 0, -10), "test", size=11.5, fill=PURPLE)
    s += text(*_off(to_l(300, ls_train(300, VAR_P, SIGMA2)), 0, 16), "training", size=11.5, fill=TEAL)
    s += text(*_off(to_l(60, TARGET), 0, -8), "desired", size=11, op=0.75)
    s += text(*_off(to_r(300, ls_test(300, BIAS_P, SIGMA2 + BIAS_OMITTED)), 0, -10), "test", size=11.5, fill=PURPLE)
    s += text(*_off(to_r(300, ls_train(300, BIAS_P, SIGMA2 + BIAS_OMITTED)), 0, 18), "training", size=11.5, fill=TEAL)
    s += text(*_off(to_r(300, TARGET), 0, 16), "desired", size=11, op=0.75)
    s += text(
        355,
        296,
        "dotted: the error neither curve can go below (noise, plus the omitted signal on the right)",
        size=11,
        op=0.7,
    )
    return wrap(
        690,
        322,
        "figCurves",
        "Learning curves for high variance and for high bias",
        "Expected training and test error of least squares against the number of training examples, computed exactly. "
        "Left, a model with 20 features of which 5 matter: training error starts at zero and rises, test error falls, "
        "and the large gap between them closes as data are added, crossing the desired level at about 155 examples. "
        "Right, a model with 2 features when more are needed: both curves flatten out at the same level, above the "
        "desired error, so more data cannot reach it.",
        s,
    )


def _off(pt, dx, dy):
    """Shift a pixel point."""
    return pt[0] + dx, pt[1] + dy


# ---------------------------------------------------------------- optimisation vs objective

LAMBDA = 0.05
W_HAM = 10.0  # weight on getting a non-spam email right


def spam_data(rng):
    """Two overlapping Gaussian classes in the plane; y = 1 is spam. Rows end in a constant 1."""
    data = []
    for _ in range(150):
        data.append(([rng.gauss(1.0, 1.0), rng.gauss(0.6, 1.0), 1.0], 1))
    for _ in range(150):
        data.append(([rng.gauss(-1.0, 1.0), rng.gauss(-0.6, 1.0), 1.0], 0))
    return data


def J(theta, data):
    """Regularised log-likelihood that Bayesian logistic regression maximises (bias unpenalised)."""
    ll = 0.0
    for x, y in data:
        z = dot(theta, x)
        ll += -math.log1p(math.exp(-z)) if y else -math.log1p(math.exp(z))
    return ll - LAMBDA * (theta[0] ** 2 + theta[1] ** 2)


def grad_J(theta, data):
    """Gradient of J."""
    g = [0.0, 0.0, 0.0]
    for x, y in data:
        r = y - 1 / (1 + math.exp(-dot(theta, x)))
        for k in range(3):
            g[k] += r * x[k]
    g[0] -= 2 * LAMBDA * theta[0]
    g[1] -= 2 * LAMBDA * theta[1]
    return g


def A(theta, data):
    """Weighted accuracy: a non-spam email counts W_HAM times as much as a spam one."""
    num = den = 0.0
    for x, y in data:
        w = 1.0 if y else W_HAM
        num += w * ((dot(theta, x) >= 0) == bool(y))
        den += w
    return num / den


def fit_blr(data, steps, lr=0.002):
    """Gradient ascent on J from zero; returns every iterate, the start included."""
    th = [0.0, 0.0, 0.0]
    path = [th]
    for _ in range(steps):
        g = grad_J(th, data)
        th = [t + lr * gg for t, gg in zip(th, g)]
        path.append(th)
    return path


def fit_svm(data, C=0.02, steps=4000):
    """Class-weighted linear soft-margin SVM by subgradient descent: errors on non-spam cost W_HAM times more."""
    th = [0.0, 0.0, 0.0]
    for t in range(1, steps + 1):
        g = [th[0], th[1], 0.0]
        for x, y in data:
            s = 1 if y else -1
            if s * dot(th, x) < 1:
                c = C * (1.0 if y else W_HAM)
                for k in range(3):
                    g[k] -= c * s * x[k]
        th = [a - (1.0 / (t + 10)) * b for a, b in zip(th, g)]
    return th


def best_scale(theta, data):
    """Max over c > 0 of J(c theta), by golden-section search (J along a ray is concave)."""
    lo, hi = 1e-3, 50.0
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if J([a * t for t in theta], data) < J([b * t for t in theta], data):
            lo = a
        else:
            hi = b
    c = (lo + hi) / 2
    return c, J([c * t for t in theta], data)


CONVERGED_STEPS = 3000


def spam_runs():
    """The toy's data, the SVM's parameters and the gradient-ascent path of logistic regression."""
    rng = random.Random(4)
    data = spam_data(rng)
    return data, fit_svm(data), fit_blr(data, CONVERGED_STEPS)


def first_above(path, data, level):
    """The first gradient-ascent step whose objective exceeds `level`."""
    return next(t for t, th in enumerate(path) if J(th, data) > level)


def fig_diagnostic():
    """J along the gradient-ascent path of logistic regression, against J at the SVM's parameters."""
    data, th_svm, path = spam_runs()
    j_svm, j_star = J(th_svm, data), J(path[-1], data)
    t_cross = first_above(path, data, j_svm)
    shown = 200
    pts = [(math.log10(t), J(path[t], data)) for t in range(1, shown + 1)]
    xdom, ydom = (0, math.log10(shown)), (-180, -60)
    to_px = mapper(xdom, ydom, (80, 610), (30, 240))
    s = axes(to_px, xdom, ydom, [], [-180, -150, -120, -90, -60], "gradient-ascent steps (log scale)", "J(θ)")
    for k in range(3):
        px, py = to_px(k, ydom[0])
        s += line(px, py, px, py + 4, op=0.7) + text(px, py + 17, f"{10**k}", size=11, op=0.8)
    s += curve(pts, to_px, xdom, ydom, stroke=PURPLE, sw=2.0)
    s += curve([(x / 20, j_svm) for x in range(0, 47)], to_px, xdom, ydom, stroke=TEAL, sw=1.2, dash="5 4")
    s += text(*_off(to_px(xdom[1], j_svm), -4, -6), "J(θ_SVM)", size=11.5, fill=TEAL, anchor="end")
    s += text(*_off(to_px(xdom[1], j_star), -4, -10), "J(θ_BLR), converged", size=11.5, fill=PURPLE, anchor="end")
    cx, cy = to_px(math.log10(t_cross), j_svm)
    s += line(cx, cy, cx, to_px(0, ydom[0])[1], sw=0.8, op=0.5, dash="2 3")
    t99 = next(t for t, th in enumerate(path) if j_star - J(th, data) < 0.01 * (j_star - J(path[0], data)))
    del cx, cy
    s += text(*_off(to_px(0, j_svm), 10, 16), f"after {t_cross} step the test already blames the objective", size=11, anchor="start", op=0.85)
    qx, qy = to_px(math.log10(t99), J(path[t99], data))
    s += line(qx, qy, qx, to_px(0, ydom[0])[1], sw=0.8, op=0.5, dash="2 3")
    s += text(qx + 6, qy + 30, f"step {t99}: within 1% of the optimum", size=11, anchor="start", op=0.85)
    return wrap(
        650,
        290,
        "figDiagnostic",
        "The optimisation-versus-objective test along a training run",
        "The regularised log-likelihood of a toy spam problem along gradient ascent, on a logarithmic step axis, "
        "with a dashed line at the value the SVM's parameters reach. The SVM wins on weighted accuracy throughout. "
        "After a single step logistic regression already scores above the SVM on its own objective, so the test "
        "reports that the objective is at fault, although the optimiser needs about thirty more steps to come within "
        "one per cent of its optimum.",
        s,
    )


# ---------------------------------------------------------------- structural figures


def box(x, y, w, h, label, fill, sub=None, op=0.22):
    """A labelled rounded box."""
    s = rect(x, y, w, h, fill=fill, op=op, rx=6)
    if sub is None:
        s += text(x + w / 2, y + h / 2 + 4, label, size=12)
    else:
        s += text(x + w / 2, y + h / 2 - 3, label, size=12)
        s += text(x + w / 2, y + h / 2 + 13, sub, size=10.5, op=0.75)
    return s


def fig_helicopter():
    """The helicopter diagnostics as a decision flow."""
    s = box(20, 120, 150, 50, "θ_RL flies badly", INDIGO, "on the real helicopter")
    s += box(220, 120, 170, 50, "Flies well in simulation?", TEAL)
    s += box(220, 230, 170, 50, "Simulator", GREEN, "fix the simulator", op=0.35)
    s += box(440, 120, 180, 50, "J(θ_human) &lt; J(θ_RL)?", TEAL)
    s += box(440, 20, 180, 50, "RL algorithm", GREEN, "fix the optimiser", op=0.35)
    s += box(440, 230, 180, 50, "Cost function", GREEN, "fix J", op=0.35)
    s += arrow(170, 145, 218, 145)
    s += arrow(305, 170, 305, 228)
    s += text(312, 204, "yes", size=11, anchor="start", op=0.8)
    s += arrow(390, 145, 438, 145)
    s += text(414, 138, "no", size=11, op=0.8)
    s += arrow(530, 120, 530, 72)
    s += text(537, 100, "yes", size=11, anchor="start", op=0.8)
    s += arrow(530, 170, 530, 228)
    s += text(537, 204, "no", size=11, anchor="start", op=0.8)
    return wrap(
        640,
        300,
        "figHeli",
        "Diagnosing the helicopter controller",
        "A decision flow. A controller that flies badly on the real helicopter is first tried in simulation; if it "
        "flies well there, the simulator is at fault. If not, the human pilot's cost under the squared-error function "
        "is compared with the controller's: if the human scores lower, the reinforcement-learning algorithm failed to "
        "minimise the cost; if the human scores higher while flying better, the cost function is wrong.",
        s,
    )


PIPELINE = (
    ("Background removal", 85.1),
    ("Face detection", 91.0),
    ("Eyes", 95.0),
    ("Nose", 96.0),
    ("Mouth", 97.0),
    ("Logistic regression", 100.0),
)
OVERALL = 85.0


def fig_pipeline():
    """The face-recognition pipeline with the accuracy after each component is made perfect."""
    s = ""
    xs = [20, 130, 250, 370, 370, 370, 490]
    s += box(20, 70, 90, 44, "Camera", INDIGO, "image")
    s += box(130, 70, 100, 44, "Background", PURPLE, "removal")
    s += box(250, 70, 100, 44, "Face", PURPLE, "detection")
    for k, nm in enumerate(("Eyes", "Nose", "Mouth")):
        s += box(370, 20 + 50 * k, 90, 40, nm, PURPLE, "segmentation")
    s += box(480, 70, 100, 44, "Logistic", PURPLE, "regression")
    s += arrow(110, 92, 128, 92)
    s += arrow(230, 92, 248, 92)
    for k in range(3):
        s += arrow(350, 92, 368, 40 + 50 * k)
        s += arrow(460, 40 + 50 * k, 478, 92)
    s += text(600, 96, "label", size=12, anchor="start")
    s += arrow(580, 92, 598, 92)
    del xs
    # Bars: accuracy after making each stage, and every stage before it, perfect.
    to_px = mapper((0, 7), (80, 100), (60, 640), (190, 330))
    s += axes(to_px, (0, 7), (80, 100), [], [80, 85, 90, 95, 100], "", "accuracy (%)")
    prev = OVERALL
    rows = [("Overall", OVERALL), *PIPELINE]
    for k, (nm, acc) in enumerate(rows):
        x0, y0 = to_px(k + 0.15, acc)
        x1, yb = to_px(k + 0.85, 80)
        fill = TEAL if acc - prev >= 3 else INDIGO
        s += rect(x0, y0, x1 - x0, yb - y0, fill=fill, op=0.55 if k else 0.3, stroke="none")
        s += text((x0 + x1) / 2, y0 - 6, f"{acc:g}", size=11)
        lab = nm if len(nm) < 12 else nm.split()[0]
        s += text((x0 + x1) / 2, yb + 16, lab, size=10.5)
        if k:
            s += text(
                (x0 + x1) / 2,
                yb + 30,
                f"+{acc - prev:.1f}",
                size=10.5,
                fill=TEAL if acc - prev >= 3 else "currentColor",
                op=0.9,
            )
        prev = acc
    return wrap(
        660,
        375,
        "figPipeline",
        "Error analysis of a face-recognition pipeline",
        "Top, the pipeline: camera image, background removal, face detection, separate eye, nose and mouth "
        "segmentation, and logistic regression producing the label. Bottom, the overall accuracy of 85 per cent and "
        "the accuracy after replacing each component, cumulatively from the left, with ground truth: 85.1, 91, 95, "
        "96, 97 and 100. Face detection and eye segmentation account for the largest gains.",
        s,
    )


# ---------------------------------------------------------------- ablation on a Gaussian toy

FEATURES = ("sender host", "email header", "text parser", "javascript")
MU = (1.0, 1.0, 1.2, 0.3)  # class-mean difference per feature, in units of its sd
RHO_12 = 0.95  # sender host and header carry nearly the same signal


def sigma():
    """Shared covariance: unit variances, features 1 and 2 almost duplicates."""
    S = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    S[0][1] = S[1][0] = RHO_12
    return S


def bayes_acc(subset):
    """Accuracy of the Bayes rule using only `subset` of the features: Phi(Delta/2)."""
    if not subset:
        return 0.5
    S = sigma()
    Ss = [[S[i][j] for j in subset] for i in subset]
    m = [MU[i] for i in subset]
    return Phi(math.sqrt(dot(m, solve(Ss, m))) / 2)


def mc_acc(subset, n, rng):
    """Monte Carlo accuracy of the same linear rule, sampled."""
    S = sigma()
    Ss = [[S[i][j] for j in subset] for i in subset]
    w = solve(Ss, [MU[i] for i in subset])
    # Cholesky of the full covariance for sampling.
    L = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i + 1):
            v = S[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(v) if i == j else v / L[j][j]
    ok = 0
    for _ in range(n):
        y = rng.random() < 0.5
        z = [rng.gauss(0, 1) for _ in range(4)]
        x = [(0.5 if y else -0.5) * MU[i] + sum(L[i][k] * z[k] for k in range(4)) for i in range(4)]
        ok += (dot(w, [x[i] for i in subset]) > 0) == y
    return ok / n


def shapley(full):
    """Shapley attribution of accuracy to each feature: the drop averaged over every removal order."""
    out = {i: 0.0 for i in full}
    perms = list(itertools.permutations(full))
    for order in perms:
        cur = list(full)
        for i in order:
            before = bayes_acc(cur)
            cur.remove(i)
            out[i] += before - bayes_acc(cur)
    return {i: v / len(perms) for i, v in out.items()}


FIGURES = {
    "diagram_learning_curves.svg": fig_learning_curves,
    "diagram_diagnostic.svg": fig_diagnostic,
    "diagram_helicopter.svg": fig_helicopter,
    "diagram_pipeline.svg": fig_pipeline,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    # 1. Learning curves: closed form against Monte Carlo least squares.
    rng = random.Random(0)
    for p, p_true, om in ((VAR_P, 5, 0.0), (BIAS_P, 2, BIAS_OMITTED)):
        for m in (40, 120):
            tr, te = mc_learning_curve(m, p, p_true, om, 300, rng)
            ftr, fte = ls_train(m, p, SIGMA2 + om), ls_test(m, p, SIGMA2 + om)
            print(f"p={p:2d} m={m:3d}: train {tr:.3f} (closed {ftr:.3f}), test {te:.3f} (closed {fte:.3f})")
            assert abs(tr - ftr) < 0.05 * (SIGMA2 + om) and abs(te - fte) < 0.06 * fte
    cross = next(m for m in range(VAR_P + 2, 2000) if ls_test(m, VAR_P, SIGMA2) <= TARGET)
    print(f"high variance: test error reaches the target {TARGET} at m = {cross}")
    print(f"high bias: both curves tend to {SIGMA2 + BIAS_OMITTED}, above the target {TARGET}")

    # 2. The excess test error falls like 1/m, not 1/sqrt(m).
    e200, e400 = ls_test(200, VAR_P, 1) - 1, ls_test(400, VAR_P, 1) - 1
    print(
        f"excess test error, m=200: {e200:.4f}, m=400: {e400:.4f}, ratio {e400 / e200:.3f} (1/sqrt(2) = {1 / math.sqrt(2):.3f})"
    )
    assert abs(e400 / e200 - 0.5) < 0.03

    # 3. Weighted accuracy is piecewise constant: zero gradient at a random point.
    data, th_svm, path = spam_runs()
    th_blr = path[-1]
    for _ in range(20):
        th = [rng.gauss(0, 1) for _ in range(3)]
        for k in range(3):
            e = [1e-7 if i == k else 0.0 for i in range(3)]
            assert A([a + b for a, b in zip(th, e)], data) == A(th, data)
    print("weighted accuracy: finite-difference gradient exactly zero at 20 random points")

    # 4. The diagnostic on the toy.
    j_svm, j_blr = J(th_svm, data), J(th_blr, data)
    c_best, j_best = best_scale(th_svm, data)
    t_cross = first_above(path, data, j_svm)
    j_cross = J(path[t_cross], data)
    print(f"A: SVM {A(th_svm, data):.4f}, BLR converged {A(th_blr, data):.4f}")
    print(f"J: SVM {j_svm:.1f} (best rescaling c={c_best:.2f}: {j_best:.1f}), BLR converged {j_blr:.1f}")
    print(f"BLR passes J(SVM) at step {t_cross}, J there {j_cross:.1f}; gap to optimum {j_blr - j_cross:.1f}")
    t99 = next(t for t, th in enumerate(path) if j_blr - J(th, data) < 0.01 * (j_blr - J(path[0], data)))
    print(f"BLR within 1% of the total improvement at step {t99}")
    assert max(A(th, data) for th in path[1:]) < A(th_svm, data)  # SVM wins on A at every step
    assert j_blr >= j_best - 1e-6  # converged BLR beats every rescaling of the SVM
    assert t99 > 10 * t_cross  # case 2 fires long before the optimiser is done
    g = grad_J(th_blr, data)
    print(f"converged BLR gradient norm {math.sqrt(dot(g, g)):.1e}")
    assert math.sqrt(dot(g, g)) < 1e-4

    # 5. The lecture's tables.
    gains = [b - a for a, b in zip([OVERALL] + [a for _, a in PIPELINE], [a for _, a in PIPELINE])]
    print("error analysis gains:", [f"{g:.1f}" for g in gains], "sum", f"{sum(gains):.1f}")
    assert abs(sum(gains) - (100 - OVERALL)) < 1e-9
    ablation = [99.9, 99.0, 98.9, 98.9, 95.0, 94.5, 94.0]
    drops = [a - b for a, b in zip(ablation, ablation[1:])]
    print("ablation drops:", [f"{d:.1f}" for d in drops], "sum", f"{sum(drops):.1f}")
    assert abs(sum(drops) - 5.9) < 1e-9

    # 6. Redundant features: leave-one-out, cumulative and Shapley disagree.
    full = [0, 1, 2, 3]
    acc_full = bayes_acc(full)
    mc = mc_acc(full, 40000, rng)
    print(f"toy, all features: {acc_full:.4f} (Monte Carlo {mc:.4f}); none: 0.5")
    assert abs(mc - acc_full) < 0.01
    print(
        "leave-one-out drop:", {FEATURES[i]: round(acc_full - bayes_acc([j for j in full if j != i]), 4) for i in full}
    )
    both = acc_full - bayes_acc([2, 3])
    print(f"remove sender host and header together: drop {both:.4f}")
    for order in ((0, 1, 2, 3), (1, 0, 2, 3)):
        cur, out = list(full), {}
        for i in order:
            before = bayes_acc(cur)
            cur.remove(i)
            out[FEATURES[i]] = round(before - bayes_acc(cur), 4)
        print("cumulative, order", [FEATURES[i] for i in order], out)
    sh = shapley(full)
    print("Shapley:", {FEATURES[i]: round(v, 4) for i, v in sh.items()}, "sum", round(sum(sh.values()), 4))
    assert abs(sum(sh.values()) - (acc_full - 0.5)) < 1e-9
    loo = [acc_full - bayes_acc([j for j in full if j != i]) for i in (0, 1)]
    assert max(loo) < 0.25 * both  # each looks nearly useless alone, the pair is not
