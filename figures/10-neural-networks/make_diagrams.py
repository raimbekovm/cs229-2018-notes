"""Figures for lecture 10: logistic regression as a neuron, softmax, and the first network.

Three of the four figures are structural — the cat image flattened into one
neuron, the house-price network wired by hand against fully connected, and the
3-2-1 network with every shape written on it. The fourth is computed: one set
of logits pushed through three independent sigmoids and through a softmax.

`__main__` prints every number the prose quotes and asserts the chapter's
claims by two routes that must agree: the parameter counts of the lecture's
three architectures against a count of edges plus nodes; the gradient of the
softmax cross-entropy with respect to the logits against finite differences
(it is ŷ − y, exactly as for one sigmoid neuron); that the iguana neuron of the
three-sigmoid network receives a non-zero gradient on every image, including
those with no iguana in them; that a softmax trained on the multi-hot label
(1, 1, 0) drifts to (1/2, 1/2, 0) and can never fit it; that the broadcast
batch forward pass equals the per-example loop; and that the gradient of the
averaged cost is the average of the per-example gradients.

Plain Python on purpose, as in lectures 1–9.

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
    YELLOW,
    arrow,
    circle,
    line,
    mapper,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------- the maths


def sigmoid(z):
    """The logistic function."""
    return 1.0 / (1.0 + math.exp(-z))


def softmax(z):
    """Softmax of a list of logits, shifted by the max for stability."""
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def softmax_ce(z, y):
    """Cross-entropy −Σ y_k log ŷ_k of a softmax output against a target y."""
    p = softmax(z)
    return -sum(yk * math.log(pk) for yk, pk in zip(y, p) if yk)


def sigmoids_bce(z, y):
    """The three-neuron loss: a sum of independent logistic losses."""
    return -sum(yk * math.log(sigmoid(v)) + (1 - yk) * math.log(1 - sigmoid(v)) for v, yk in zip(z, y))


def num_grad(f, z, h=1e-6):
    """Central finite-difference gradient of f at the list z."""
    g = []
    for i in range(len(z)):
        zp, zm = list(z), list(z)
        zp[i] += h
        zm[i] -= h
        g.append((f(zp) - f(zm)) / (2 * h))
    return g


def matvec(W, x):
    """W x for a list-of-rows matrix and a list vector."""
    return [sum(w * v for w, v in zip(row, x)) for row in W]


def params_dense(sizes):
    """Weights plus biases of a fully connected network with these layer widths (input first)."""
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def edges_plus_nodes(sizes):
    """The lecture's shortcut: one weight per edge, one bias per neuron."""
    edges = sum(a * b for a, b in zip(sizes, sizes[1:]))
    neurons = sum(sizes[1:])
    return edges + neurons


# ---------------------------------------------------------------- figures


def neuron_node(cx, cy, r=26, fill=TEAL):
    """A neuron drawn as a circle split into its linear part and its activation."""
    s = circle(cx, cy, r, fill=fill, stroke=fill, op=0.18, sw=1.6)
    s += line(cx, cy - r, cx, cy + r, stroke=fill, sw=1.0, op=0.6)
    return s


def fig_neuron():
    """The cat image flattened into a vector and fed to one neuron: logistic regression."""
    s = ""
    # The image: three stacked 64x64 channels.
    for k, col in enumerate((PURPLE, GREEN, INDIGO)):
        x, y = 40 + 9 * (2 - k), 64 + 9 * k
        s += rect(x, y, 92, 92, fill=col, stroke=col, op=0.28, sw=1.2)
    s += text(97, 186, "64 × 64 × 3 image", size=12)
    s += text(97, 202, "R, G, B channels", size=11, op=0.7)
    s += arrow(160, 112, 214, 112)
    s += text(187, 100, "flatten", size=11, op=0.8)
    # The vector x.
    xs = 236
    cells = [(48 + 22 * i) for i in range(7)]
    labels = ["x₁", "x₂", "x₃", "⋮", "", "", "xₙ"]
    for i, y in enumerate(cells):
        if labels[i] == "⋮":
            s += text(xs + 13, y + 15, "⋮", size=14)
            continue
        if not labels[i]:
            continue
        s += rect(xs, y, 26, 20, fill=INDIGO, stroke=INDIGO, op=0.12)
        s += text(xs + 13, y + 14, labels[i], size=11)
    s += text(xs + 13, 222, "n = 12,288", size=11, op=0.8)
    # Edges to the neuron.
    cx, cy = 420, 112
    for i, y in enumerate(cells):
        if labels[i] in ("x₁", "x₂", "x₃", "xₙ"):
            s += line(xs + 26, y + 10, cx - 26, cy, op=0.45)
    s += text(330, 214, "one weight per edge", size=11, op=0.7)
    s += neuron_node(cx, cy, 30)
    s += text(cx - 14, cy + 4, "z", size=13, style="italic")
    s += text(cx + 14, cy + 4, "σ", size=13)
    s += text(cx, cy + 50, "z = wx + b", size=12)
    s += text(cx, cy + 66, "a = σ(z)", size=12)
    s += text(cx, cy - 42, "linear | activation", size=11, op=0.75)
    s += arrow(cx + 30, cy, cx + 86, cy)
    s += text(cx + 112, cy + 5, "ŷ = a", size=13)
    s += text(cx + 112, cy + 23, "P(cat)", size=11, op=0.7)
    return wrap(
        580,
        236,
        "figNeuron",
        "Logistic regression drawn as one neuron",
        "A 64 by 64 colour image has three channels, so it is 12,288 numbers. Flattened into a column vector x, every entry is connected by its own weight to a single neuron, which computes a linear part z = wx + b and then the sigmoid activation a = σ(z). The output a is the predicted probability that the image contains a cat.",
        s,
    )


def fig_sigmoid_softmax():
    """One set of logits through three independent sigmoids and through a softmax."""
    z = [2.0, 1.5, -1.0]
    names = ["cat", "lion", "iguana"]
    panels = (
        ("three sigmoids", [sigmoid(v) for v in z], PURPLE),
        ("softmax", softmax(z), TEAL),
    )
    s = ""
    for k, (title, p, col) in enumerate(panels):
        x0 = 70 + 300 * k
        to_px = mapper((0, 3), (0, 1), (x0, x0 + 210), (50, 210))
        s += line(x0, 210, x0 + 210, 210, op=0.7) + line(x0, 210, x0, 50, op=0.7)
        for t in (0, 0.5, 1):
            _, py = to_px(0, t)
            s += line(x0 - 4, py, x0, py, op=0.7) + text(x0 - 8, py + 4, f"{t:g}", size=11, anchor="end", op=0.8)
            if t:
                s += line(x0, py, x0 + 210, py, op=0.12)
        for i, (v, nm) in enumerate(zip(p, names)):
            bx, by = to_px(i + 0.2, v)
            _, b0 = to_px(0, 0)
            s += rect(bx, by, 0.6 * 70, b0 - by, fill=col, stroke=col, op=0.55)
            s += text(bx + 21, by - 6, f"{v:.2f}", size=11)
            s += text(bx + 21, 228, nm, size=12)
        s += text(x0 + 105, 30, title, size=13, weight="600")
        s += text(x0 + 105, 256, f"outputs sum to {sum(p):.2f}", size=12, fill=col)
    return wrap(
        600,
        270,
        "figSoftmax",
        "Independent sigmoids against a softmax",
        "The same three logits, 2.0 for cat, 1.5 for lion and −1.0 for iguana, are passed through three independent sigmoids on the left and through a softmax on the right. The sigmoids give 0.88, 0.82 and 0.27, which sum to 1.97 and can say cat and lion at once; the softmax gives 0.60, 0.37 and 0.03, a distribution over the three classes that sums to one.",
        s,
    )


def column(x, n, top, gap):
    """Centres of n nodes stacked in a column."""
    h = (n - 1) * gap
    return [(x, top - h / 2 + i * gap) for i in range(n)]


def fig_house():
    """The house-price network wired by hand from domain knowledge, against fully connected."""
    inputs = ["size", "bedrooms", "zip code", "wealth"]
    hidden = ["family size", "walkability", "school quality"]
    hand = {0: [0, 1], 1: [2], 2: [2, 3]}
    s = ""
    for k, title in enumerate(("wired by hand", "fully connected")):
        x0 = 30 + 340 * k
        ins = column(x0 + 70, 4, 140, 50)
        hid = column(x0 + 190, 3, 140, 62)
        out = (x0 + 290, 140)
        for j, (hx, hy) in enumerate(hid):
            src = hand[j] if k == 0 else range(4)
            for i in src:
                s += line(ins[i][0] + 9, ins[i][1], hx - 15, hy, op=0.5, stroke=INDIGO if k == 0 else "currentColor")
            s += line(hx + 15, hy, out[0] - 15, out[1], op=0.5)
        for i, (ix, iy) in enumerate(ins):
            s += circle(ix, iy, 9, fill=INDIGO, stroke=INDIGO, op=0.2)
            s += text(ix - 16, iy + 4, inputs[i], size=11, anchor="end")
        for j, (hx, hy) in enumerate(hid):
            s += circle(hx, hy, 15, fill=TEAL, stroke=TEAL, op=0.2, sw=1.4)
            if k == 0:
                s += text(hx, hy - 22, hidden[j], size=11, fill=INDIGO)
            else:
                s += text(hx, hy + 5, "?", size=13)
        s += circle(*out, 15, fill=PURPLE, stroke=PURPLE, op=0.2, sw=1.4)
        s += text(out[0], out[1] + 34, "price", size=12)
        s += text(x0 + 165, 24, title, size=13, weight="600")
    return wrap(
        690,
        262,
        "figHouse",
        "A house-price network wired by hand and fully connected",
        "Four inputs, size, bedrooms, zip code and wealth of the neighbourhood, feed three hidden neurons and one output, the price. On the left the edges are chosen by hand so that each hidden neuron computes a named quantity: family size from size and bedrooms, walkability from the zip code, school quality from zip code and wealth. On the right every input is connected to every hidden neuron and the hidden neurons are unnamed: what each computes is left to training.",
        s,
    )


def fig_shapes():
    """The 3-2-1 network with the shape of every weight matrix and activation."""
    s = ""
    ins = column(80, 4, 150, 48)
    l1 = column(240, 3, 150, 66)
    l2 = column(390, 2, 150, 80)
    l3 = column(520, 1, 150, 0)
    labels = ["x₁", "x₂", "⋮", "xₙ"]
    for a, b in ((ins, l1), (l1, l2), (l2, l3)):
        for p in a:
            if a is ins and p == ins[2]:
                continue
            for q in b:
                s += line(p[0] + (12 if a is ins else 16), p[1], q[0] - 16, q[1], op=0.4)
    for i, (x, y) in enumerate(ins):
        if labels[i] == "⋮":
            s += text(x, y + 6, "⋮", size=15)
        else:
            s += circle(x, y, 12, fill=INDIGO, stroke=INDIGO, op=0.2)
            s += text(x, y + 4, labels[i], size=11)
    for col, layer in ((TEAL, l1), (TEAL, l2), (PURPLE, l3)):
        for x, y in layer:
            s += neuron_node(x, y, 16, fill=col)
    s += arrow(536, 150, 586, 150)
    s += text(606, 155, "ŷ", size=14)
    rows = (
        (80, "x", "n × 1", ""),
        (240, "W[1], b[1]", "3 × n, 3 × 1", "z[1], a[1]: 3 × 1"),
        (390, "W[2], b[2]", "2 × 3, 2 × 1", "z[2], a[2]: 2 × 1"),
        (520, "W[3], b[3]", "1 × 2, 1 × 1", "z[3], a[3]: 1 × 1"),
    )
    for x, a, b, c in rows:
        s += text(x, 276, a, size=12, weight="600")
        s += text(x, 293, b, size=11, font="JetBrains Mono, SFMono-Regular, Menlo, monospace")
        if c:
            s += text(x, 310, c, size=11, op=0.75)
    s += text(240, 34, "layer 1", size=12, op=0.75)
    s += text(390, 34, "layer 2", size=12, op=0.75)
    s += text(520, 34, "layer 3", size=12, op=0.75)
    return wrap(
        640,
        322,
        "figShapes",
        "The three-layer network and its shapes",
        "An n-dimensional input feeds a layer of three neurons, then a layer of two, then one output neuron. Under each layer are its parameters and their shapes: W[1] is 3 by n and b[1] is 3 by 1; W[2] is 2 by 3 and b[2] is 2 by 1; W[3] is 1 by 2 and b[3] is 1 by 1. The linear parts z and activations a of each layer are column vectors with one entry per neuron.",
        s,
    )


FIGURES = {
    "diagram_neuron.svg": fig_neuron,
    "diagram_softmax.svg": fig_sigmoid_softmax,
    "diagram_house.svg": fig_house,
    "diagram_shapes.svg": fig_shapes,
}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn())
        print("wrote", name)
    assert YELLOW  # imported for palette parity with other chapters; never used as a line

    # 1. Parameter counts, by formula and by the lecture's edges-plus-nodes shortcut.
    n = 64 * 64 * 3
    for name, sizes in (("one neuron", [n, 1]), ("three neurons", [n, 3]), ("3-2-1 network", [n, 3, 2, 1])):
        a, b = params_dense(sizes), edges_plus_nodes(sizes)
        assert a == b
        print(f"{name:>14}: {a:,} parameters")
    assert params_dense([n, 3, 2, 1]) == 3 * n + 14
    print(f"  share of 3-2-1 parameters in the first layer: {100 * (3 * n + 3) / (3 * n + 14):.3f} %")

    # 2. Softmax cross-entropy: gradient wrt the logits is yhat - y; the same holds for sigmoid + logistic loss.
    rng = random.Random(0)
    worst = 0.0
    for _ in range(500):
        z = [rng.gauss(0, 2) for _ in range(3)]
        y = [0, 0, 0]
        y[rng.randrange(3)] = 1
        g = num_grad(lambda v, y=y: softmax_ce(v, y), z)
        ana = [p - t for p, t in zip(softmax(z), y)]
        worst = max(worst, max(abs(a - b) for a, b in zip(g, ana)))
        yb = [rng.randrange(2) for _ in range(3)]
        g2 = num_grad(lambda v, yb=yb: sigmoids_bce(v, yb), z)
        ana2 = [sigmoid(v) - t for v, t in zip(z, yb)]
        worst = max(worst, max(abs(a - b) for a, b in zip(g2, ana2)))
    print(f"softmax-CE and sigmoid-BCE gradients vs finite differences: worst error {worst:.1e}")
    assert worst < 1e-6

    # 3. The figure's logits.
    z = [2.0, 1.5, -1.0]
    sg, sm = [sigmoid(v) for v in z], softmax(z)
    print("sigmoids", [f"{v:.3f}" for v in sg], f"sum {sum(sg):.3f}; softmax", [f"{v:.3f}" for v in sm])

    # 4. The rare-class neuron is not starved of gradient: on an image with no iguana its
    #    logit still gets sigmoid(z) - 0 > 0, pushing it down. What is scarce is positives.
    g_neg = sigmoid(0.0) - 0
    print(f"iguana neuron on a no-iguana image at z = 0: gradient {g_neg:.2f} (non-zero on every image)")
    assert g_neg > 0

    # 5. Softmax trained on a multi-hot target (1, 1, 0) cannot fit it and drifts to (1/2, 1/2, 0).
    y = [1, 1, 0]
    zz = [0.3, -0.2, 0.1]
    for _ in range(20000):
        p = softmax(zz)
        zz = [v - 0.5 * (pk * sum(y) - yk) for v, pk, yk in zip(zz, p, y)]
    p = softmax(zz)
    print("softmax on target (1,1,0) after 20,000 steps:", [f"{v:.4f}" for v in p])
    assert abs(p[0] - 0.5) < 1e-3 and abs(p[1] - 0.5) < 1e-3 and p[2] < 2e-3
    # yhat - y never vanishes: sum(yhat) = 1 but sum(y) = 2.
    assert abs(sum(a - b for a, b in zip(p, y)) + 1) < 1e-12

    # 6. Broadcasting: W X + b (b added to every column) equals the per-example loop.
    W = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(3)]
    b = [rng.gauss(0, 1) for _ in range(3)]
    X = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(5)]  # 5 x m, m = 4
    WX = [[sum(W[i][k] * X[k][j] for k in range(5)) for j in range(4)] for i in range(3)]
    Z = [[WX[i][j] + b[i] for j in range(4)] for i in range(3)]
    for j in range(4):
        col = matvec(W, [X[k][j] for k in range(5)])
        assert all(abs(Z[i][j] - (col[i] + b[i])) < 1e-12 for i in range(3))
    print("broadcast batch forward pass equals the per-example loop (3 × 4 batch)")

    # 7. dJ/dw for J = (1/m) Σ L_i is the mean of the per-example gradients (one logistic neuron).
    m, d = 6, 4
    xs = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(m)]
    ys = [rng.randrange(2) for _ in range(m)]
    w = [rng.gauss(0, 0.5) for _ in range(d)]

    def cost(w):
        """Mean logistic loss over the toy batch."""
        out = 0.0
        for x, t in zip(xs, ys):
            a = sigmoid(sum(wi * xi for wi, xi in zip(w, x)))
            out -= t * math.log(a) + (1 - t) * math.log(1 - a)
        return out / m

    g = num_grad(cost, w)
    mean_grad = [
        sum((sigmoid(sum(wi * xi for wi, xi in zip(w, x))) - t) * x[k] for x, t in zip(xs, ys)) / m for k in range(d)
    ]
    assert max(abs(a - b) for a, b in zip(g, mean_grad)) < 1e-8
    print("gradient of the averaged cost = average of per-example gradients")
