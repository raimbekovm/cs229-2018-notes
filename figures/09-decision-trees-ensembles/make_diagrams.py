"""Figures for lecture 9: decision trees, bagging, random forests and boosting.

Everything is computed by one small CART implementation written here — greedy
axis-aligned splits chosen by size-weighted cross-entropy for classification
and squared error for regression — run on synthetic data: the lecture's ski
example (month and latitude), a diagonal boundary that trees can only
approximate by a staircase, a regression problem with one dominant feature
for bagging and random forests, and the same diagonal for AdaBoost with
decision stumps.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: the gains of the lecture's two
900/100 splits under misclassification, cross-entropy (in two log bases) and
Gini; that a strictly concave impurity never loses on a split while
misclassification can register no gain; the q − 1 sorted splits of a
categorical feature against all 2^(q−1) − 1 subsets; zero gain at the root of
an XOR problem a depth-2 tree solves; the variance of an average of correlated
variables against ρσ² + (1 − ρ)σ²/n; the distinct fraction of a bootstrap
sample against 1 − 1/e; the correlation between trees under bagging and under
random feature selection; and AdaBoost as written in the notes against
forward stagewise fitting of the exponential loss.

Plain Python on purpose, as in lectures 1–8.

Run from the repository root.
"""

import itertools
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import (
    INDIGO,
    PURPLE,
    TEAL,
    YELLOW,
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


# --- impurities ---------------------------------------------------------------


def misclassification(p):
    """1 − max_c p̂_c."""
    return 1 - max(p)


def cross_entropy(p, base=2.0):
    """−Σ p̂_c log p̂_c."""
    return -sum(q * math.log(q, base) for q in p if q > 0)


def gini(p):
    """Σ p̂_c (1 − p̂_c)."""
    return sum(q * (1 - q) for q in p)


def props(counts):
    """Class proportions from counts."""
    n = sum(counts)
    return [c / n for c in counts]


def children_loss(children, impurity):
    """Size-weighted impurity of the children, |R1|L(R1) + |R2|L(R2) over |R1| + |R2|."""
    n = sum(sum(c) for c in children)
    return sum(sum(c) * impurity(props(c)) for c in children) / n


def gain(parent, children, impurity):
    """L(R_parent) minus the size-weighted loss of the children."""
    return impurity(props(parent)) - children_loss(children, impurity)


# --- a small CART ---------------------------------------------------------------


def node_value(ys, task):
    """Leaf prediction: majority class, or mean."""
    if task == "class":
        return int(2 * sum(ys) >= len(ys))
    return sum(ys) / len(ys)


def node_loss(ys, task):
    """Total loss of a node: n × binary entropy, or the sum of squared deviations."""
    n = len(ys)
    if task == "class":
        p = sum(ys) / n
        return n * cross_entropy([p, 1 - p])
    mu = sum(ys) / n
    return sum((y - mu) ** 2 for y in ys)


def best_split(xs, ys, task, features):
    """The (loss, feature, threshold) of the best split, scanning sorted values once per feature."""
    n = len(ys)
    best = (node_loss(ys, task), None, None)
    for j in features:
        order = sorted(range(n), key=lambda i: xs[i][j])
        if task == "class":
            tot1 = sum(ys)
            left1 = 0
        else:
            tot_s, tot_q = sum(ys), sum(y * y for y in ys)
            ls = lq = 0.0
        for k in range(1, n):
            y = ys[order[k - 1]]
            if task == "class":
                left1 += y
            else:
                ls += y
                lq += y * y
            a, b = xs[order[k - 1]][j], xs[order[k]][j]
            if a == b:
                continue
            if task == "class":
                pl, pr = left1 / k, (tot1 - left1) / (n - k)
                loss = k * cross_entropy([pl, 1 - pl]) + (n - k) * cross_entropy([pr, 1 - pr])
            else:
                rs, rq = tot_s - ls, tot_q - lq
                loss = (lq - ls * ls / k) + (rq - rs * rs / (n - k))
            if loss < best[0] - 1e-12:
                best = (loss, j, (a + b) / 2)
    return best


def grow(xs, ys, task="class", max_depth=30, min_leaf=1, mtry=None, rng=None, depth=0):
    """Grow a tree greedily; mtry < number of features gives a random-forest tree."""
    f = len(xs[0])
    if depth >= max_depth or len(ys) < 2 * min_leaf or len(set(ys)) == 1:
        return ("leaf", node_value(ys, task))
    features = rng.sample(range(f), mtry) if mtry else range(f)
    _, j, t = best_split(xs, ys, task, features)
    if j is None:
        return ("leaf", node_value(ys, task))
    left = [i for i in range(len(ys)) if xs[i][j] < t]
    right = [i for i in range(len(ys)) if xs[i][j] >= t]
    if len(left) < min_leaf or len(right) < min_leaf:
        return ("leaf", node_value(ys, task))
    return (
        "split",
        j,
        t,
        grow([xs[i] for i in left], [ys[i] for i in left], task, max_depth, min_leaf, mtry, rng, depth + 1),
        grow([xs[i] for i in right], [ys[i] for i in right], task, max_depth, min_leaf, mtry, rng, depth + 1),
    )


def predict(tree, x):
    """Route x to its leaf."""
    while tree[0] == "split":
        tree = tree[3] if x[tree[1]] < tree[2] else tree[4]
    return tree[1]


def regions(tree, box):
    """Leaf rectangles (x0, x1, y0, y1, value) of a two-feature tree inside box."""
    if tree[0] == "leaf":
        return [(*box, tree[1])]
    _, j, t, lo, hi = tree
    x0, x1, y0, y1 = box
    if j == 0:
        return regions(lo, (x0, min(t, x1), y0, y1)) + regions(hi, (max(t, x0), x1, y0, y1))
    return regions(lo, (x0, x1, y0, min(t, y1))) + regions(hi, (x0, x1, max(t, y0), y1))


def depth_of(tree):
    """Depth of a tree."""
    return 0 if tree[0] == "leaf" else 1 + max(depth_of(tree[3]), depth_of(tree[4]))


def leaves(tree):
    """Number of leaves."""
    return 1 if tree[0] == "leaf" else leaves(tree[3]) + leaves(tree[4])


# --- data ---------------------------------------------------------------------


def can_ski(month, lat):
    """Northern winter (Dec–Mar) above 30°, southern winter (Jun–Sep) below −30°."""
    return int((lat > 30 and (month < 3.5 or month >= 12)) or (lat < -30 and 5.5 <= month < 9.5))


def ski_data(n=160, seed=1):
    """Months in [0.5, 12.5), latitudes in [−90, 90)."""
    rng = random.Random(seed)
    xs = [(0.5 + 12 * rng.random(), -90 + 180 * rng.random()) for _ in range(n)]
    return xs, [can_ski(m, lat) for m, lat in xs]


def diagonal_data(n, seed):
    """Points in the unit square labelled 1 above the line x₂ = x₁."""
    rng = random.Random(seed)
    xs = [(rng.random(), rng.random()) for _ in range(n)]
    return xs, [int(b > a) for a, b in xs]


def dominant_truth(x):
    """A regression function with one dominant feature, x₁, and weaker ones."""
    return 2.0 * math.sin(3 * x[0]) + 0.8 * x[1] + 0.6 * x[2] * x[3] + 0.4 * x[4]


def dominant_data(n, rng):
    """n examples with six features (the sixth is noise) and N(0, 0.5²) error."""
    xs = [tuple(rng.uniform(-1, 1) for _ in range(6)) for _ in range(n)]
    return xs, [dominant_truth(x) + rng.gauss(0, 0.5) for x in xs]


def forest_data(n, rng):
    """Eight noisy measurements: five of a latent z, three of a latent w; y = 2 sin 3z + w + N(0, 0.5²)."""
    xs, ys = [], []
    for _ in range(n):
        z, w = rng.uniform(-1, 1), rng.uniform(-1, 1)
        xs.append(tuple([z + rng.gauss(0, 0.3) for _ in range(5)] + [w + rng.gauss(0, 0.3) for _ in range(3)]))
        ys.append(2 * math.sin(3 * z) + w + rng.gauss(0, 0.5))
    return xs, ys


# --- ensembles ------------------------------------------------------------------


def bootstrap(xs, ys, rng):
    """N draws with replacement."""
    idx = [rng.randrange(len(ys)) for _ in range(len(ys))]
    return [xs[i] for i in idx], [ys[i] for i in idx]


def forest(xs, ys, m_trees, mtry, rng, min_leaf=3):
    """Bagged regression trees; mtry = None is plain bagging, a smaller mtry a random forest."""
    return [grow(*bootstrap(xs, ys, rng), task="reg", min_leaf=min_leaf, mtry=mtry, rng=rng) for _ in range(m_trees)]


def stump(xs, ys, w):
    """The weighted-error-minimising decision stump on ±1 labels: (feature, threshold, sign, error)."""
    best = (math.inf, 0, 0.0, 1)
    tot = sum(w)
    for j in range(len(xs[0])):
        order = sorted(range(len(ys)), key=lambda i: xs[i][j])
        # err_plus(k): predict +1 for x_j ≥ threshold. Start with everything predicted +1.
        err = sum(wi for wi, y in zip(w, ys) if y != 1)
        cands = [(err, xs[order[0]][j] - 1e-9)]
        for k in range(len(order)):
            i = order[k]
            err += w[i] if ys[i] == 1 else -w[i]
            nxt = xs[order[k + 1]][j] if k + 1 < len(order) else xs[i][j] + 1e-9
            cands.append((err, (xs[i][j] + nxt) / 2))
        for e, t in cands:
            for s, ee in ((1, e), (-1, tot - e)):
                if ee < best[0] - 1e-15:
                    best = (ee, j, t, s)
    e, j, t, s = best
    return j, t, s, e / tot


def stump_predict(st, x):
    """±1 prediction of a stump."""
    j, t, s, _ = st
    return s if x[j] >= t else -s


def adaboost_m1(xs, ys, rounds):
    """AdaBoost as in the notes: α = log((1 − err)/err), w ← w·exp(α·1{wrong})."""
    w = [1 / len(ys)] * len(ys)
    model, history = [], []
    for _ in range(rounds):
        st = stump(xs, ys, w)
        err = max(st[3], 1e-12)
        alpha = math.log((1 - err) / err)
        model.append((alpha, st))
        history.append((st, err, w[:]))
        w = [wi * math.exp(alpha * (stump_predict(st, x) != y)) for wi, x, y in zip(w, xs, ys)]
        s = sum(w)
        w = [wi / s for wi in w]
    return model, history


def stagewise_exp(xs, ys, rounds):
    """Forward stagewise additive modelling with the exponential loss: β = ½ log((1 − err)/err)."""
    f = [0.0] * len(ys)
    model = []
    for _ in range(rounds):
        w = [math.exp(-y * fi) for y, fi in zip(ys, f)]
        st = stump(xs, ys, w)
        err = max(st[3], 1e-12)
        beta = 0.5 * math.log((1 - err) / err)
        model.append((beta, st))
        f = [fi + beta * stump_predict(st, x) for fi, x in zip(f, xs)]
    return model


def ensemble_predict(model, x):
    """Σ α_m G_m(x)."""
    return sum(a * stump_predict(st, x) for a, st in model)


# --- figures --------------------------------------------------------------------


def shade(regs, to_px, colour_of):
    """Fill leaf rectangles."""
    s = ""
    for x0, x1, y0, y1, v in regs:
        px0, py0 = to_px(x0, y1)
        px1, py1 = to_px(x1, y0)
        c, op = colour_of(v)
        s += rect(px0, py0, px1 - px0, py1 - py0, fill=c, stroke="currentColor", sw=0.6, op=op, so=0.35)
    return s


def mark(px, py, y):
    """Filled circle for a positive, open square for a negative."""
    return (
        circle(px, py, 3.6, fill=TEAL, stroke=TEAL)
        if y == 1
        else rect(px - 3.2, py - 3.2, 6.4, 6.4, stroke=PURPLE, sw=1.3)
    )


def fig_ski():
    """The ski data and the regions a fully grown tree carves out."""
    xs, ys = ski_data()
    tree = grow(xs, ys)
    xdom, ydom = (0.5, 12.5), (-90.0, 90.0)
    to_px = mapper(xdom, ydom, (80, 640), (40, 300))
    s = shade(regions(tree, (*xdom, *ydom)), to_px, lambda v: (YELLOW, 0.45) if v == 1 else ("#ffffff", 0.0))
    s += axes(
        to_px, xdom, ydom, [1, 3, 5, 7, 9, 11], [-90, -30, 0, 30, 90], "month (1 = January)", "latitude (degrees)"
    )
    for x, y in zip(xs, ys):
        s += mark(*to_px(*x), y)
    s += circle(110, 358, 3.6, fill=TEAL, stroke=TEAL) + text(120, 362, "can ski", size=11, anchor="start")
    s += rect(206.8, 354.8, 6.4, 6.4, stroke=PURPLE, sw=1.3) + text(220, 362, "cannot", size=11, anchor="start")
    s += rect(300, 352, 14, 12, fill=YELLOW, stroke="none", op=0.45) + text(
        320,
        362,
        f"regions the tree predicts 'ski' ({leaves(tree)} leaves, depth {depth_of(tree)})",
        size=11,
        anchor="start",
    )
    return wrap(
        680,
        376,
        "figSki",
        "Recursive partitioning of the ski data",
        "Synthetic examples of month against latitude: one can ski in the northern winter above 30 degrees and in the southern winter below minus 30. A greedily grown decision tree splits the plane into axis-aligned rectangles that isolate the two skiing regions.",
        s,
    )


def fig_impurity():
    """Misclassification, cross-entropy and Gini against p̂, with the 900/100 parent split into 400/100 and 500/0."""
    s = ""
    curves = (
        ("misclassification", lambda p: misclassification([p, 1 - p]), 0.5, [0, 0.25, 0.5]),
        ("cross-entropy (bits)", lambda p: cross_entropy([p, 1 - p]), 1.0, [0, 0.5, 1]),
        ("Gini", lambda p: gini([p, 1 - p]), 0.5, [0, 0.25, 0.5]),
    )
    grid = [i / 400 for i in range(401)]
    for k, (name, fn, top, yt) in enumerate(curves):
        x0 = 60 + 235 * k
        xdom, ydom = (0.0, 1.0), (0.0, top * 1.08)
        to_px = mapper(xdom, ydom, (x0, x0 + 180), (46, 216))
        s += axes(to_px, xdom, ydom, [0, 0.5, 1], yt, "p̂ (share positive)", "")
        s += curve([(p, fn(p)) for p in grid], to_px, xdom, ydom, stroke=INDIGO, sw=2.0)
        p1, p2 = 0.8, 1.0
        a, b = to_px(p1, fn(p1)), to_px(p2, fn(p2))
        s += line(*a, *b, stroke=PURPLE, sw=1.4, dash="4 3")
        mid = to_px(0.9, (fn(p1) + fn(p2)) / 2)
        par = to_px(0.9, fn(0.9))
        s += circle(*a, 3.5, fill=PURPLE, stroke=PURPLE) + circle(*b, 3.5, fill=PURPLE, stroke=PURPLE)
        s += circle(*par, 4, fill=TEAL, stroke=TEAL) + circle(*mid, 3, fill="#ffffff", stroke=PURPLE, sw=1.4)
        s += line(par[0], par[1], mid[0], mid[1], stroke=TEAL, sw=2.0)
        g = fn(0.9) - (fn(p1) + fn(p2)) / 2
        s += text(x0 + 90, 26, name, size=13, weight="600")
        s += text(x0 + 90, 274, f"gain {g:.3f}", size=12, fill=TEAL)
    return wrap(
        760,
        286,
        "figImpurity",
        "Three impurity functions and one split",
        "Each panel plots an impurity against the share of positives in a region. The parent with 900 positives and 100 negatives sits at 0.9; splitting it into 400/100 and 500/0 gives two equal children at 0.8 and 1.0, whose average loss is the midpoint of the dashed chord. For misclassification the chord lies on the curve and the gain is zero; for the strictly concave cross-entropy and Gini it lies below, and the gain is positive.",
        s,
    )


def fig_staircase():
    """Trees of growing depth approximating the diagonal boundary x₂ = x₁."""
    xs, ys = diagonal_data(400, 2)
    tx, ty = diagonal_data(4000, 3)
    s = ""
    for k, d in enumerate((1, 3, 30)):
        tree = grow(xs, ys, max_depth=d)
        acc = sum(predict(tree, x) == y for x, y in zip(tx, ty)) / len(ty)
        x0 = 60 + 255 * k
        xdom, ydom = (0.0, 1.0), (0.0, 1.0)
        to_px = mapper(xdom, ydom, (x0, x0 + 180), (46, 236))
        s += shade(regions(tree, (0, 1, 0, 1)), to_px, lambda v: (TEAL, 0.25) if v == 1 else ("#ffffff", 0.0))
        s += axes(to_px, xdom, ydom, [0, 1], [0, 1], "x₁", "x₂")
        s += curve(
            [(t, t) for t in [i / 100 for i in range(101)]], to_px, xdom, ydom, stroke=PURPLE, sw=1.6, dash="5 4"
        )
        s += text(x0 + 90, 26, f"depth {depth_of(tree)}: {leaves(tree)} leaves" + (", pure" if d == 30 else ""), size=13, weight="600")
        s += text(x0 + 90, 290, f"test accuracy {100 * acc:.1f} %", size=12)
    return wrap(
        780,
        302,
        "figStaircase",
        "A tree approximating a diagonal boundary",
        "Points above the dashed diagonal are positive. Trees of depth 1 and 3, and one grown until every leaf is pure, trained on 400 points, shade the region they call positive; the boundary is a staircase that needs ever more leaves to follow a line a linear classifier represents exactly.",
        s,
    )


def var_of_mean(rho, n):
    """Variance of the average of n variables with variance 1 and pairwise correlation ρ."""
    return rho + (1 - rho) / n


def fig_correlation():
    """Variance of an average against the number of members, for three correlations."""
    xdom, ydom = (1, 50), (0.0, 1.05)
    to_px = mapper(xdom, ydom, (80, 560), (40, 250))
    s = axes(
        to_px,
        xdom,
        ydom,
        [1, 10, 20, 30, 40, 50],
        [0, 0.25, 0.5, 0.75, 1],
        "number of models averaged, n",
        "Var(average) / σ²",
    )
    rng = random.Random(7)
    for rho, colour in ((0.0, TEAL), (0.3, INDIGO), (0.6, PURPLE)):
        s += curve(
            [(n, var_of_mean(rho, n)) for n in [1 + i * 49 / 300 for i in range(301)]],
            to_px,
            xdom,
            ydom,
            stroke=colour,
            sw=2.0,
        )
        for n in (2, 5, 10, 25, 50):
            s += circle(*to_px(n, simulate_var(rho, n, 4000, rng)), 3.2, fill="#ffffff", stroke=colour, sw=1.4)
        s += line(*to_px(1, rho), *to_px(50, rho), stroke=colour, sw=0.8, dash="2 3")
        s += text(
            to_px(50, 0)[0] + 8,
            to_px(50, var_of_mean(rho, 50))[1] + 4,
            f"ρ = {rho:g}",
            size=12,
            anchor="start",
            fill=colour,
        )
    s += text(320, 22, "ρσ² + (1 − ρ)σ²/n, with simulated averages (rings)", size=13, weight="600")
    return wrap(
        640,
        300,
        "figCorrelation",
        "Averaging correlated models",
        "The variance of an average of n equally correlated variables, relative to one variable's, falls like 1/n when they are independent but levels off at ρ when they are correlated, shown by the dotted lines. Rings are simulated averages.",
        s,
    )


def simulate_var(rho, n, reps, rng):
    """Monte Carlo variance of the mean of n unit-variance variables with pairwise correlation ρ."""
    vals = []
    for _ in range(reps):
        z = rng.gauss(0, 1)
        vals.append(sum(math.sqrt(rho) * z + math.sqrt(1 - rho) * rng.gauss(0, 1) for _ in range(n)) / n)
    mu = sum(vals) / reps
    return sum((v - mu) ** 2 for v in vals) / (reps - 1)


FOREST = None


def forest_experiment():
    """Test error against the number of trees for bagging and a random forest (cached)."""
    global FOREST
    if FOREST is None:
        rng = random.Random(11)
        xs, ys = forest_data(200, rng)
        tx, ty = forest_data(600, rng)
        single = grow(xs, ys, task="reg", min_leaf=3)
        e_single = sum((predict(single, x) - y) ** 2 for x, y in zip(tx, ty)) / len(ty)
        curves = {}
        for name, mtry in (("bagging", None), ("random forest", 3)):
            trees = forest(xs, ys, 100, mtry, rng)
            preds = [[predict(t, x) for x in tx] for t in trees]
            errs = []
            acc = [0.0] * len(tx)
            for m, p in enumerate(preds, 1):
                acc = [a + v for a, v in zip(acc, p)]
                errs.append(sum((a / m - y) ** 2 for a, y in zip(acc, ty)) / len(ty))
            curves[name] = errs
        FOREST = (e_single, curves)
    return FOREST


def fig_forest():
    """Test error of bagged trees and of a random forest as trees are added."""
    e_single, curves = forest_experiment()
    xdom, ydom = (1, 100), (0.2, 1.7)
    to_px = mapper(xdom, ydom, (80, 560), (40, 250))
    s = axes(
        to_px,
        xdom,
        ydom,
        [1, 20, 40, 60, 80, 100],
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        "number of trees M",
        "test squared error",
    )
    s += line(*to_px(1, e_single), *to_px(100, e_single), stroke="currentColor", sw=1.2, dash="5 4")
    s += text(to_px(100, 0)[0], to_px(100, e_single)[1] - 8, "one full tree", size=11, anchor="end")
    s += line(*to_px(1, 0.25), *to_px(100, 0.25), stroke="currentColor", sw=0.8, dash="2 3", op=0.6)
    s += text(to_px(100, 0)[0], to_px(100, 0.25)[1] - 6, "noise σ² = 0.25", size=11, anchor="end", op=0.8)
    for name, colour in (("bagging", INDIGO), ("random forest", TEAL)):
        errs = curves[name]
        s += curve(list(zip(range(1, 101), errs)), to_px, xdom, ydom, stroke=colour, sw=2.0)
        at = 12 if colour == INDIGO else 40
        s += text(to_px(at, 0)[0] + 6, to_px(at, errs[at - 1])[1] - 10, name, size=12, anchor="start", fill=colour)
    return wrap(
        620,
        300,
        "figForest",
        "Bagging and random forests as trees are added",
        "Test error of an average of M regression trees, each grown on a bootstrap sample, for plain bagging and for a random forest that considers three of eight redundant noisy features at each split. Both fall steeply over the first few trees and then flatten; neither rises as trees are added, and the forest ends a little lower. A single fully grown tree is far worse.",
        s,
    )


def fig_adaboost():
    """AdaBoost with stumps on the diagonal: round 1, the reweighted round 2, and 40 rounds."""
    xs, ys01 = diagonal_data(60, 5)
    ys = [1 if y else -1 for y in ys01]
    model, hist = adaboost_m1(xs, ys, 40)
    s = ""
    xdom, ydom = (0.0, 1.0), (0.0, 1.0)
    titles = ("round 1: one stump", "round 2: mistakes up-weighted", "40 stumps combined")
    for k in range(3):
        x0 = 60 + 255 * k
        to_px = mapper(xdom, ydom, (x0, x0 + 180), (46, 236))
        if k == 2:
            n = 60
            for a in range(n):
                for b in range(n):
                    u, v = (a + 0.5) / n, (b + 0.5) / n
                    if ensemble_predict(model, (u, v)) > 0:
                        px, py = to_px(u - 0.5 / n, v + 0.5 / n)
                        s += rect(px, py, 180 / n + 0.3, 180 / n + 0.3, fill=TEAL, stroke="none", op=0.22)
        s += axes(to_px, xdom, ydom, [0, 1], [0, 1], "x₁", "x₂")
        s += curve(
            [(t, t) for t in [i / 100 for i in range(101)]], to_px, xdom, ydom, stroke=PURPLE, sw=1.2, dash="5 4"
        )
        st, _, w = hist[min(k, 1)]
        if k < 2:
            j, t, _, _ = st
            if j == 0:
                s += line(*to_px(t, 0), *to_px(t, 1), stroke=INDIGO, sw=2.2)
            else:
                s += line(*to_px(0, t), *to_px(1, t), stroke=INDIGO, sw=2.2)
        for x, y, wi in zip(xs, ys, w if k < 2 else [1 / 60] * 60):
            r = 2.2 + 30 * math.sqrt(wi) if k < 2 else 3.2
            px, py = to_px(*x)
            s += (
                circle(px, py, r, fill=TEAL, stroke=TEAL, op=0.7)
                if y == 1
                else circle(px, py, r, fill="none", stroke=PURPLE, sw=1.3)
            )
            if k == 0 and stump_predict(st, x) != y:
                s += circle(px, py, r + 4, stroke=PURPLE, sw=1.0, so=0.8)
        s += text(x0 + 90, 26, titles[k], size=13, weight="600")
    tr = sum((ensemble_predict(model, x) > 0) != (y == 1) for x, y in zip(xs, ys)) / len(ys)
    s += text(660, 290, f"training error after 40 rounds: {100 * tr:.0f} %", size=12)
    s += text(50, 290, f"stump 1 weighted error {hist[0][1]:.2f}; stump 2 {hist[1][1]:.2f}", size=12, anchor="start")
    return wrap(
        780,
        302,
        "figAdaboost",
        "AdaBoost with decision stumps",
        "Sixty points labelled by the diagonal. Left: the first stump, with its mistakes ringed. Middle: the same points with area proportional to their AdaBoost weight before round 2, the mistakes now heavy, and the second stump. Right: the region the weighted vote of forty stumps calls positive, an additive staircase along the diagonal.",
        s,
    )


FIGURES = {
    "diagram_ski.svg": fig_ski,
    "diagram_impurity.svg": fig_impurity,
    "diagram_staircase.svg": fig_staircase,
    "diagram_correlation.svg": fig_correlation,
    "diagram_forest.svg": fig_forest,
    "diagram_adaboost.svg": fig_adaboost,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)

    # 1. The lecture's two splits of 900 positives / 100 negatives.
    parent = (900, 100)
    split_a = [(700, 100), (200, 0)]
    split_b = [(400, 100), (500, 0)]
    for name, imp in (("misclassification", misclassification), ("cross-entropy", cross_entropy), ("Gini", gini)):
        print(
            f"{name:>18}: parent {imp(props(parent)):.4f}; gain A {gain(parent, split_a, imp):.4f}; gain B {gain(parent, split_b, imp):.4f}"
        )
    print("  misclassified counts: parent 100, A", sum(min(c) for c in split_a), ", B", sum(min(c) for c in split_b))
    assert abs(gain(parent, split_a, misclassification)) < 1e-12 and abs(gain(parent, split_b, misclassification)) < 1e-12
    assert gain(parent, split_b, cross_entropy) > gain(parent, split_a, cross_entropy) > 0
    # Base of the log rescales every gain by the same factor, so the ranking cannot change.
    ratio = gain(parent, split_b, lambda p: cross_entropy(p, math.e)) / gain(parent, split_b, cross_entropy)
    ratio_a = gain(parent, split_a, lambda p: cross_entropy(p, math.e)) / gain(parent, split_a, cross_entropy)
    print(f"  natural-log gains / bit gains: {ratio:.6f} and {ratio_a:.6f} (ln 2 = {math.log(2):.6f})")
    assert abs(ratio - math.log(2)) < 1e-12 and abs(ratio_a - math.log(2)) < 1e-12
    # Unweighted sum of children's proportional losses ranks splits by something else.
    unweighted = lambda ch: sum(cross_entropy(props(c)) for c in ch)  # noqa: E731
    print(
        f"  unweighted L(R1)+L(R2) with proportions: A {unweighted(split_a):.4f}, B {unweighted(split_b):.4f}; parent {cross_entropy(props(parent)):.4f}"
    )
    tiny = [(899, 99), (1, 1)]
    print(
        f"  e.g. peeling off one positive and one negative: unweighted children {unweighted(tiny):.4f} > parent; weighted gain {gain(parent, tiny, cross_entropy):.6f}"
    )

    # 2. Concave impurities never lose; misclassification registers no gain when both children keep the parent's majority.
    rng = random.Random(1)
    zero_mis = 0
    trials = 20000
    dis = 0
    for _ in range(trials):
        n1p, n1n = rng.randint(0, 50), rng.randint(0, 50)
        n2p, n2n = rng.randint(0, 50), rng.randint(0, 50)
        if min(n1p + n1n, n2p + n2n) == 0:
            continue
        par = (n1p + n2p, n1n + n2n)
        ch = [(n1p, n1n), (n2p, n2n)]
        assert gain(par, ch, cross_entropy) >= -1e-12 and gain(par, ch, gini) >= -1e-12
        if abs(gain(par, ch, misclassification)) < 1e-12 and gain(par, ch, cross_entropy) > 1e-6:
            zero_mis += 1
    print(
        f"random splits: concave gains never negative; misclassification gain 0 while cross-entropy > 0 in {100 * zero_mis / trials:.1f} %"
    )
    # Do cross-entropy and Gini pick the same split? Compare argmax over 10 random candidate splits of one parent.
    for _ in range(2000):
        cands = []
        tp, tn = 60, 40
        for _ in range(10):
            a, b = rng.randint(0, tp), rng.randint(0, tn)
            if 0 < a + b < tp + tn:
                cands.append([(a, b), (tp - a, tn - b)])
        if cands:
            be = max(range(len(cands)), key=lambda i: gain((tp, tn), cands[i], cross_entropy))
            bg = max(range(len(cands)), key=lambda i: gain((tp, tn), cands[i], gini))
            dis += be != bg
    print(f"cross-entropy and Gini choose different splits in {100 * dis / 2000:.1f} % of 2,000 random candidate sets")

    # 3. Categorical splits: q - 1 sorted splits find the best of all 2^(q-1) - 1.
    for q in (3, 5, 8):
        subsets = [set(c) for r in range(1, q) for c in itertools.combinations(range(q), r) if 0 in c]
        nontrivial = [s for s in subsets if len(s) < q]
        print(
            f"  q={q}: distinct non-trivial binary splits {len(nontrivial)} = 2^(q-1) - 1 = {2 ** (q - 1) - 1}; sorted candidates {q - 1}"
        )
        assert len(nontrivial) == 2 ** (q - 1) - 1
    agree = 0
    for _ in range(500):
        q = 6
        cnt = [(rng.randint(0, 30), rng.randint(1, 30)) for _ in range(q)]
        tot = (sum(c[0] for c in cnt), sum(c[1] for c in cnt))

        def split_gain(sub, cnt=cnt, tot=tot):
            """Cross-entropy gain of sending the categories in sub left."""
            left = (sum(cnt[i][0] for i in sub), sum(cnt[i][1] for i in sub))
            return gain(tot, [left, (tot[0] - left[0], tot[1] - left[1])], cross_entropy)

        brute = max(split_gain(c) for r in range(1, q) for c in itertools.combinations(range(q), r))
        order = sorted(range(q), key=lambda i: cnt[i][0] / (cnt[i][0] + cnt[i][1]))
        fast = max(split_gain(order[:k]) for k in range(1, q))
        agree += abs(brute - fast) < 1e-12
    print(f"sorted-category shortcut matches brute force in {agree}/500 random problems")
    assert agree == 500

    # 4. XOR: no single split gains anything, a depth-2 tree is perfect.
    xor_x = [(a + 0.25 * i, b + 0.25 * k) for a in (0, 1) for b in (0, 1) for i in range(4) for k in range(4)]
    xor_y = [int((x >= 1) != (y >= 1)) for x, y in xor_x]
    loss0 = node_loss(xor_y, "class")
    root = best_split(xor_x, xor_y, "class", [0, 1])
    print(
        f"XOR, {len(xor_y)} points: root entropy {loss0:.1f} bits total; best single split leaves {root[0]:.1f} (gain {loss0 - root[0]:.2e})"
    )
    t_greedy = grow(xor_x, xor_y, max_depth=2)
    greedy_mist = sum(predict(t_greedy, x) != y for x, y in zip(xor_x, xor_y))
    print(f"  a tree that splits only on positive gain stops at the root: {leaves(t_greedy)} leaf, {greedy_mist} mistakes")
    left = [i for i, x in enumerate(xor_x) if x[0] < 1]
    right = [i for i, x in enumerate(xor_x) if x[0] >= 1]
    t_forced = (
        "split",
        0,
        1.0,
        grow([xor_x[i] for i in left], [xor_y[i] for i in left]),
        grow([xor_x[i] for i in right], [xor_y[i] for i in right]),
    )
    mist = sum(predict(t_forced, x) != y for x, y in zip(xor_x, xor_y))
    print(f"  taking the zero-gain split x1 < 1 anyway, the children split perfectly: depth {depth_of(t_forced)}, {mist} mistakes")
    assert loss0 - root[0] < 1e-9 and leaves(t_greedy) == 1 and mist == 0 and depth_of(t_forced) == 2

    # 5. Staircase: leaves against accuracy on the diagonal.
    xs, ys = diagonal_data(400, 2)
    tx, ty = diagonal_data(4000, 3)
    for d in (1, 2, 4, 6, 8, 12):
        tree = grow(xs, ys, max_depth=d)
        acc = sum(predict(tree, x) == y for x, y in zip(tx, ty)) / len(ty)
        print(f"  diagonal, depth {d:>2}: {leaves(tree):>3} leaves, test accuracy {100 * acc:.1f} %")

    # 6. Variance of a correlated average.
    rng = random.Random(3)
    for rho in (0.0, 0.3, 0.6):
        for n in (10, 50):
            sim = simulate_var(rho, n, 20000, rng)
            print(f"  ρ={rho}, n={n}: formula {var_of_mean(rho, n):.4f}, simulated {sim:.4f}")
            assert abs(sim - var_of_mean(rho, n)) < 0.03

    # 7. Bootstrap: distinct fraction -> 1 - 1/e.
    rng = random.Random(4)
    for n in (10, 100, 10000):
        frac = sum(len({rng.randrange(n) for _ in range(n)}) / n for _ in range(200)) / 200
        print(
            f"  bootstrap n={n}: distinct fraction {frac:.4f}; 1 − (1 − 1/n)^n = {1 - (1 - 1 / n) ** n:.4f}; 1 − 1/e = {1 - 1 / math.e:.4f}"
        )

    # 8. Bagging and random forests.
    e_single, curves_ = forest_experiment()
    for name, errs in curves_.items():
        print(
            f"{name}: M=1 {errs[0]:.3f}, M=10 {errs[9]:.3f}, M=100 {errs[99]:.3f}; largest rise between consecutive M {max(b - a for a, b in zip(errs, errs[1:])):.4f}"
        )
    print(f"single full tree on all data: {e_single:.3f}")
    for label, maker in (("redundant", forest_data), ("dominant", dominant_data)):
        rng = random.Random(21)
        xs, ys = maker(200, rng)
        tx, ty = maker(600, rng)
        row = []
        for mtry in (None, 4, 3, 2, 1):
            trees = forest(xs, ys, 100, mtry, rng)
            row.append((mtry, sum((sum(predict(t, x) for t in trees) / 100 - y) ** 2 for x, y in zip(tx, ty)) / len(ty)))
        print(f"  {label} problem, 100 trees, test error by features per split:", ", ".join(f"{m or 'all'}: {e:.3f}" for m, e in row))
    rng = random.Random(12)
    # Correlation between two trees across training sets, and each tree's variance.
    rng = random.Random(12)
    for name, mtry, maker in (
        ("bagging, redundant features", None, forest_data),
        ("random forest mtry=3, redundant", 3, forest_data),
        ("bagging, dominant feature", None, dominant_data),
        ("random forest mtry=2, dominant", 2, dominant_data),
    ):
        tx = [maker(1, rng)[0][0] for _ in range(150)]
        pairs = []
        for _ in range(40):
            xs, ys = maker(200, rng)
            t1, t2 = forest(xs, ys, 2, mtry, rng)
            pairs.append(([predict(t1, x) for x in tx], [predict(t2, x) for x in tx]))
        rhos, sig = [], []
        for i in range(len(tx)):
            a = [p[0][i] for p in pairs]
            b = [p[1][i] for p in pairs]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            va = sum((u - ma) ** 2 for u in a) / len(a)
            vb = sum((u - mb) ** 2 for u in b) / len(b)
            cov = sum((u - ma) * (v - mb) for u, v in zip(a, b)) / len(a)
            rhos.append(cov / math.sqrt(va * vb))
            sig.append((va + vb) / 2)
        print(
            f"  {name}: mean correlation between two trees ρ ≈ {sum(rhos) / len(rhos):.3f}; single-tree variance σ² ≈ {sum(sig) / len(sig):.3f}; ρσ² ≈ {sum(r * s for r, s in zip(rhos, sig)) / len(sig):.3f}"
        )

    # 9. AdaBoost as in the notes against forward stagewise exponential loss.
    xs, ys01 = diagonal_data(60, 5)
    ys = [1 if y else -1 for y in ys01]
    m1, hist = adaboost_m1(xs, ys, 40)
    fs = stagewise_exp(xs, ys, 40)
    same_stumps = all(a[1][:3] == b[1][:3] for a, b in zip(m1, fs))
    ratios = [a[0] / b[0] for a, b in zip(m1, fs)]
    grid = [(i / 50, k / 50) for i in range(51) for k in range(51)]
    same_sign = all((ensemble_predict(m1, g) > 0) == (ensemble_predict(fs, g) > 0) for g in grid)
    print(
        f"AdaBoost.M1 vs stagewise exp-loss: same stumps {same_stumps}; α/β ∈ [{min(ratios):.6f}, {max(ratios):.6f}]; same predictions on a 51×51 grid {same_sign}"
    )
    assert same_stumps and same_sign and max(abs(r - 2) for r in ratios) < 1e-9
    bound = 1.0
    for m, (_, err, _) in enumerate(hist, 1):
        bound *= 2 * math.sqrt(err * (1 - err))
        tr = sum((ensemble_predict(m1[:m], x) > 0) != (y == 1) for x, y in zip(xs, ys)) / len(ys)
        assert tr <= bound + 1e-12
        if m in (1, 2, 5, 10, 20, 40):
            print(
                f"  round {m:>2}: weighted error of this stump {err:.3f}; training error {100 * tr:.1f} %; bound Π 2√(ε(1−ε)) = {bound:.3f}"
            )
