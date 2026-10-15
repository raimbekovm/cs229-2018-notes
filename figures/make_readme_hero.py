"""Generate the site's signature images: the course in six panels, and gradient descent on a quadratic bowl.

The picture is the one lecture 2 is built around — the contours of a cost
J(θ) over two parameters, and the path batch gradient descent takes across
them. It is computed, not drawn: the bowl is

    J(θ) = ½ (λ₁ a² + λ₂ b²),   (a, b) = θ − θ* in the bowl's principal axes,

and every dot on the path is one update θ := θ − α ∇J(θ). With λ₁ ≫ λ₂ and a
step size near 2/λ₁ the iterates overshoot across the narrow axis and creep
along the long one — the zig-zag every student draws once and remembers. The
numbers printed by `__main__` are what was actually drawn.

The README banner is the whole course at a glance: one small panel for each
of its six parts, each computed the same way the chapters' figures are — a
least-squares line, a maximum-margin separator, a network, the learning
curves of lecture 12, a Gaussian mixture fitted by EM, and Ng's grid world
solved by value iteration (`banner()`, and the functions above it).

The other two outputs share one geometry:

- `figures/social-card.svg`, rasterised by hand to `figures/social-card.png`
  (the command is printed at the end), which Open Graph and Twitter use.
- The animated hero on the landing page, written into `index.qmd` between the
  `hero:start` and `hero:end` markers. Its motion lives in theme-common.scss;
  this file only supplies the path length the animation needs.

Two constraints come from GitHub rather than from the site, and they are why
the first two do not look like the other figure scripts:

- **No `currentColor`.** A README image is rendered in isolation inside an
  `<img>`, where `currentColor` falls back to black and vanishes on GitHub's
  dark theme. So ink is set explicitly here; `__main__` asserts it.
- **An explicit background.** The plate is `#eef1f5`, the value
  `theme-dark.scss` gives `$cs-figure-plate`, so the banner and the in-page
  figures are the same object.

Run from the repository root.
"""

import math
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _gridworld import STATES, TERMINAL, greedy, value_iteration
from _svg import INK, MONO, TEAL, circle, line, poly, polyline, rect, text, wrap

PLATE = "#eef1f5"
VIRIDIS = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")

# The bowl. Condition number λ₁/λ₂ ≈ 6.7; the long axis tilts up to the right.
LAM1, LAM2 = 1.0, 0.15
TILT = -28.0  # degrees, screen convention (y down), so negative is anticlockwise
ALPHA = 1.7  # step size; ALPHA * LAM1 = 1.7 < 2, so it converges, but by overshooting
START = (1.1, -2.4)  # (a, b) in bowl units: on the outer slope, along the long axis
LEVELS = 7
STOP = 5e-3  # stop once J has fallen to this fraction of where it started


def viridis(t):
    """Sample the viridis ramp at t, clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    s = t * (len(VIRIDIS) - 1)
    i = min(int(s), len(VIRIDIS) - 2)
    f = s - i
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    return "#" + "".join(
        f"{round(int(a[k : k + 2], 16) + (int(b[k : k + 2], 16) - int(a[k : k + 2], 16)) * f):02x}"
        for k in (1, 3, 5)
    )


def cost(a, b):
    """J at a point given in the bowl's principal axes."""
    return 0.5 * (LAM1 * a * a + LAM2 * b * b)


def descent():
    """The gradient-descent iterates, from START until J has fallen by STOP."""
    a, b = START
    j0 = cost(a, b)
    path = [(a, b)]
    while cost(a, b) > STOP * j0:
        # ∇J = (λ₁ a, λ₂ b) in these axes, so each coordinate shrinks by its own factor.
        a, b = a - ALPHA * LAM1 * a, b - ALPHA * LAM2 * b
        path.append((a, b))
    return path


def levels():
    """Contour values, geometric from just outside the start down to the floor of the bowl."""
    top = cost(*START) * 1.12
    return [top * 0.5**k for k in range(LEVELS)]


def scene(cx, cy, scale):
    """Contour ellipses and the path, both in screen coordinates.

    Returns (ellipses, points): ellipses as (rx, ry, t) outermost first, with t the colour position on the ramp; points
    as (x, y) along the descent.
    """
    phi = math.radians(TILT)

    def to_screen(a, b):
        # Local x is the long (b) axis, local y the short (a) axis, then rotate by TILT.
        return (
            cx + scale * (b * math.cos(phi) - a * math.sin(phi)),
            cy + scale * (b * math.sin(phi) + a * math.cos(phi)),
        )

    cs = levels()
    ellipses = [
        (
            scale * math.sqrt(2 * c / LAM2),
            scale * math.sqrt(2 * c / LAM1),
            i / (len(cs) - 1),
        )
        for i, c in enumerate(cs)
    ]
    return ellipses, [to_screen(a, b) for a, b in descent()]


def contours_svg(cx, cy, ellipses, extra=""):
    """Filled contour bands: purple on the rim to yellow at the minimum."""
    out = f'  <g transform="rotate({TILT} {cx:.1f} {cy:.1f})"{extra}>\n'
    for rx, ry, t in ellipses:
        out += f'    <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{viridis(t)}"/>\n'
    return out + "  </g>\n"


def path_svg(points, halo=True, cls=""):
    """The descent path: a white line with dots at each iterate, on a dark halo so it reads on every band."""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    out = ""
    if halo:
        out += (
            f'  <polyline points="{pts}" fill="none" stroke="{INK}" stroke-opacity="0.55"'
            ' stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>\n'
        )
    c = f' class="{cls}"' if cls else ""
    out += (
        f'  <polyline{c} points="{pts}" fill="none" stroke="#ffffff"'
        ' stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>\n'
    )
    for x, y in points:
        out += f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#ffffff" stroke="{INK}" stroke-width="1.2"/>\n'
    return out


def ink(x, y, s, size=13, op=1.0, anchor="start", weight=None, font=None, style=""):
    """Text with explicit ink, since currentColor is unavailable here."""
    return text(
        x,
        y,
        s,
        size=size,
        fill=INK,
        anchor=anchor,
        op=op,
        weight=weight,
        font=font,
        style=style,
    )


DESC = (
    "Filled contour lines of a quadratic cost over two parameters, purple at the rim and yellow at the "
    "minimum, elongated and tilted. A white path of dots starts on the outer slope and zig-zags across the "
    "narrow axis of the bowl while creeping along the long one until it reaches the centre: batch gradient "
    "descent with a step size close to the largest stable one."
)


# ---------------------------------------------------------------- the README banner

PANEL = 170  # side of one panel, px
PANEL_GAP = 32
PANEL_TOP = 150
PARTS = (
    ("Supervised learning", "lectures 1–7"),
    ("Generalization", "lecture 8, theory"),
    ("Trees and neural nets", "lectures 9–11"),
    ("Applying ML", "lecture 12"),
    ("Unsupervised learning", "lectures 13–15"),
    ("Reinforcement learning", "lectures 16–20"),
)


def regression_data():
    """Fourteen noisy points on a line, and their least-squares fit (θ₀, θ₁)."""
    rng = random.Random(2)
    pts = [(u / 13, 0.25 + 0.55 * u / 13 + rng.gauss(0, 0.07)) for u in range(14)]
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    t1 = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return pts, (my - t1 * mx, t1)


def svm_data():
    """Two separable classes of 22 points each."""
    rng = random.Random(7)
    pos, neg = [], []
    while len(pos) < 22:
        p = (rng.gauss(1.3, 0.75), rng.gauss(1.1, 0.75))
        if p[0] + p[1] > 1.5:
            pos.append(p)
    while len(neg) < 22:
        p = (rng.gauss(-1.2, 0.75), rng.gauss(-1.0, 0.75))
        if p[0] + p[1] < -0.9:
            neg.append(p)
    return pos, neg


def max_margin(pos, neg, steps=7200):
    """The hard-margin separator by search over the direction of w: returns (w, c, margin) for wᵀx = c.

    In two dimensions the margin of the best separator with unit normal w is half the gap between the classes'
    projections onto w, so a fine sweep of the angle finds the optimum to within its resolution.
    """
    best = None
    for k in range(steps):
        a = 2 * math.pi * k / steps
        w = (math.cos(a), math.sin(a))
        lo = min(w[0] * x + w[1] * y for x, y in pos)
        hi = max(w[0] * x + w[1] * y for x, y in neg)
        if lo > hi and (best is None or lo - hi > best[0]):
            best = (lo - hi, w, (lo + hi) / 2)
    gap, w, c = best
    return w, c, gap / 2


def mixture_data():
    """210 points from three correlated Gaussians."""
    rng = random.Random(3)
    comps = (
        ((-1.6, 0.6), (0.55, 0.35), 0.5),
        ((1.2, 1.1), (0.45, 0.6), -0.4),
        ((0.3, -1.3), (0.7, 0.3), 0.1),
    )
    pts = []
    for (mx, my), (sx, sy), r in comps:
        for _ in range(70):
            u, v = rng.gauss(0, 1), rng.gauss(0, 1)
            pts.append((mx + sx * u, my + sy * (r * u + math.sqrt(1 - r * r) * v)))
    return pts


def gaussian(p, mu, cov):
    """The bivariate normal density at p."""
    dx, dy = p[0] - mu[0], p[1] - mu[1]
    (a, b), (_, d) = cov
    det = a * d - b * b
    q = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    return math.exp(-q / 2) / (2 * math.pi * math.sqrt(det))


def em(pts, iters=60):
    """EM for a mixture of three Gaussians: returns the means, covariances and every point's responsibilities."""
    mus = [(-2.0, -1.0), (2.0, -0.5), (0.0, 2.0)]
    covs = [((1.0, 0.0), (0.0, 1.0))] * 3
    phis = [1 / 3] * 3
    for _ in range(iters):
        resp = []
        for p in pts:
            ws = [phis[j] * gaussian(p, mus[j], covs[j]) for j in range(3)]
            resp.append([w / sum(ws) for w in ws])
        for j in range(3):
            n = sum(r[j] for r in resp)
            phis[j] = n / len(pts)
            mx = sum(r[j] * x for r, (x, _) in zip(resp, pts)) / n
            my = sum(r[j] * y for r, (_, y) in zip(resp, pts)) / n
            sxx = sum(r[j] * (x - mx) ** 2 for r, (x, _) in zip(resp, pts)) / n
            syy = sum(r[j] * (y - my) ** 2 for r, (_, y) in zip(resp, pts)) / n
            sxy = sum(r[j] * (x - mx) * (y - my) for r, (x, y) in zip(resp, pts)) / n
            mus[j], covs[j] = (mx, my), ((sxx, sxy), (sxy, syy))
    return mus, covs, resp


def ellipse(cov):
    """Semi-axes (standard deviations) and the angle in degrees of a covariance's principal axes."""
    (a, b), (_, c) = cov
    mid, rad = (a + c) / 2, math.sqrt(((a - c) / 2) ** 2 + b * b)
    angle = (
        math.degrees(math.atan2(mid + rad - a, b))
        if abs(b) > 1e-12
        else (0.0 if a >= c else 90.0)
    )
    return math.sqrt(mid + rad), math.sqrt(mid - rad), angle


def mix(colours, weights):
    """A colour blended from hex colours by weights summing to one."""
    rgb = [
        sum(w * int(c[k : k + 2], 16) for c, w in zip(colours, weights))
        for k in (1, 3, 5)
    ]
    return "#" + "".join(f"{round(v):02x}" for v in rgb)


def panel(i):
    """Panel i's frame and caption; returns (svg, x, y) with (x, y) the frame's top left."""
    x, y = 60 + i * (PANEL + PANEL_GAP), PANEL_TOP
    name, lectures = PARTS[i]
    out = rect(x, y, PANEL, PANEL, fill="#ffffff", stroke=INK, sw=1.0, so=0.15, rx=10)
    out += ink(x, y + PANEL + 26, name, size=15, weight="600")
    out += ink(x, y + PANEL + 46, lectures, size=12.5, op=0.6, font=MONO)
    return out, x, y


def inset(x, y, pad=18):
    """Map unit-square coordinates (y up) into a panel at (x, y)."""
    return lambda u, v: (
        x + pad + u * (PANEL - 2 * pad),
        y + PANEL - pad - v * (PANEL - 2 * pad),
    )


def clipped(i, x, y, body):
    """Body clipped to panel i's frame, for lines that are meant to run off its edge."""
    return (
        f'  <clipPath id="panel{i}"><rect x="{x}" y="{y}" width="{PANEL}" height="{PANEL}" rx="10"/></clipPath>\n'
        f'  <g clip-path="url(#panel{i})">\n{body}  </g>\n'
    )


def banner_panels():
    """The six panels, in course order."""
    out = ""

    # Supervised learning: the least-squares line through the points.
    frame, x, y = panel(0)
    pts, (t0, t1) = regression_data()
    to = inset(x, y)
    out += frame + line(*to(0, t0), *to(1, t0 + t1), stroke=TEAL, sw=2.6)
    out += "".join(
        circle(*to(*p), 4, fill=VIRIDIS[0], stroke="#ffffff", sw=0.8) for p in pts
    )

    # Generalization: the maximum-margin separator and its margin.
    frame, x, y = panel(1)
    pos, neg = svm_data()
    w, c, m = max_margin(pos, neg)
    cx, cy, scale = x + PANEL / 2, y + PANEL / 2, 22

    def to(u, v):
        return cx + scale * u, cy - scale * v

    along = (-w[1], w[0])
    inner = ""
    for off, sw, op, dash in (
        (m, 1.0, 0.5, "4 4"),
        (-m, 1.0, 0.5, "4 4"),
        (0.0, 2.2, 0.9, None),
    ):
        bx, by = w[0] * (c + off), w[1] * (c + off)
        a = to(bx - 6 * along[0], by - 6 * along[1])
        b = to(bx + 6 * along[0], by + 6 * along[1])
        inner += line(*a, *b, stroke=INK, sw=sw, op=op, dash=dash)
    inner += "".join(
        circle(*to(*p), 3.4, fill=VIRIDIS[4], stroke=INK, sw=0.7) for p in pos
    )
    inner += "".join(
        circle(*to(*p), 3.4, fill=VIRIDIS[0], stroke="#ffffff", sw=0.7) for p in neg
    )
    out += frame + clipped(1, x, y, inner)

    # Trees and neural nets: a fully connected 3-4-4-1 network.
    frame, x, y = panel(2)
    layers = (3, 4, 4, 1)
    nodes = [
        [
            (x + 24 + li * (PANEL - 48) / 3, y + PANEL / 2 + (k - (n - 1) / 2) * 34)
            for k in range(n)
        ]
        for li, n in enumerate(layers)
    ]
    out += frame
    for left, right in zip(nodes, nodes[1:]):
        out += "".join(
            line(*a, *b, stroke=INK, sw=0.9, op=0.28) for a in left for b in right
        )
    for li, layer in enumerate(nodes):
        out += "".join(
            circle(*p, 8.5, fill=viridis(li / 3), stroke=INK, sw=0.9) for p in layer
        )

    # Applying ML: lecture 12's high-variance learning curves -- training error rising and dev error falling
    # towards the desired performance (dashed) as the training set grows, with a gap still between them.
    frame, x, y = panel(3)
    to = inset(x, y)
    train = [to(u / 40, 0.12 + 0.30 * (1 - math.exp(-0.1 * u))) for u in range(1, 41)]
    dev = [to(u / 40, 0.58 + 0.36 * math.exp(-0.08 * u)) for u in range(1, 41)]
    out += frame + line(
        *to(0, 0.5), *to(1, 0.5), stroke=INK, sw=1.0, op=0.5, dash="4 4"
    )
    out += polyline(train, stroke=TEAL, sw=2.4) + polyline(
        dev, stroke=VIRIDIS[0], sw=2.4
    )

    # Unsupervised learning: the fitted mixture, points coloured by their responsibilities.
    frame, x, y = panel(4)
    pts = mixture_data()
    mus, covs, resp = em(pts)
    colours = [None] * 3
    for colour, j in zip(
        (VIRIDIS[0], VIRIDIS[2], VIRIDIS[4]), sorted(range(3), key=lambda j: mus[j][0])
    ):
        colours[j] = colour
    cx, cy, scale = x + PANEL / 2, y + PANEL / 2 + 4, 26
    out += frame
    for j in range(3):
        a, b, angle = ellipse(covs[j])
        ex, ey = cx + scale * mus[j][0], cy - scale * mus[j][1]
        out += (
            f'  <ellipse cx="{ex:.1f}" cy="{ey:.1f}" rx="{2 * a * scale:.1f}" ry="{2 * b * scale:.1f}"'
            f' transform="rotate({-angle:.1f} {ex:.1f} {ey:.1f})" fill="{colours[j]}" fill-opacity="0.15"'
            f' stroke="{colours[j]}" stroke-width="1.4"/>\n'
        )
    out += "".join(
        circle(
            cx + scale * px, cy - scale * py, 2.2, fill=mix(colours, r), stroke="none"
        )
        for (px, py), r in list(zip(pts, resp))[::2]
    )

    # Reinforcement learning: V* and the greedy policy on the grid world.
    frame, x, y = panel(5)
    V = value_iteration()
    policy = greedy(V)
    lo, hi = min(V.values()), max(V.values())
    cell, gx0, gy0 = 38, x + 11, y + 28
    out += frame
    for gx in (1, 2, 3, 4):
        for gy in (1, 2, 3):
            px, py = gx0 + (gx - 1) * cell, gy0 + (3 - gy) * cell
            if (gx, gy) not in STATES:
                out += rect(
                    px, py, cell - 4, cell - 4, fill="#9aa3ad", stroke="none", rx=6
                )
                continue
            t = (V[(gx, gy)] - lo) / (hi - lo)
            out += rect(
                px, py, cell - 4, cell - 4, fill=viridis(t), stroke="none", rx=6
            )
            if (gx, gy) in TERMINAL:
                continue
            fg = INK if t > 0.55 else "#ffffff"
            dx, dy = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}[
                policy[(gx, gy)]
            ]
            mx, my, half, head = px + (cell - 4) / 2, py + (cell - 4) / 2, 8.4, 3.0
            tip = (mx + dx * half, my + dy * half)
            out += line(mx - dx * half, my - dy * half, *tip, stroke=fg, sw=2.0)
            out += poly(
                [
                    (tip[0] + dx * head, tip[1] + dy * head),
                    (tip[0] - dx * head - dy * head, tip[1] - dy * head + dx * head),
                    (tip[0] - dx * head + dy * head, tip[1] - dy * head - dx * head),
                ],
                fill=fg,
                stroke="none",
            )
    return out


BANNER_DESC = (
    "Six small panels, one for each part of the course: a least-squares line through noisy points, a "
    "maximum-margin separator between two classes, a fully connected neural network, the learning curves of a "
    "model with high variance, a mixture of three Gaussians fitted by EM with each point coloured by its soft "
    "assignment, and the four-by-three grid world solved by value iteration, coloured by value with the optimal "
    "action in each cell."
)


def banner():
    """The README banner, 1280 x 420: the course in six panels."""
    w, h = 1280, 420
    body = rect(0, 0, w, h, fill=PLATE, stroke="none")
    body += rect(0, 0, w, 5, fill=TEAL, stroke="none", op=0.9)
    body += ink(60, 82, "CS229 2018 Lecture Notes", size=44, weight="700")
    body += ink(
        60,
        116,
        "Stanford Machine Learning, written from Andrew Ng’s Autumn 2018 lectures",
        size=18,
        op=0.74,
    )
    body += banner_panels()
    return wrap(w, h, "readmeHero", "CS229 2018 Lecture Notes", BANNER_DESC, body)


# ---------------------------------------------------------------- the social card and the landing-page hero


def social_card():
    """The Open Graph card, 1200 x 630."""
    w, h = 1200, 630
    cx, cy = 870, 330
    ellipses, points = scene(cx, cy, 68)
    body = rect(0, 0, w, h, fill=PLATE, stroke="none")
    body += rect(0, 0, w, 8, fill=TEAL, stroke="none", op=0.9)
    body += contours_svg(cx, cy, ellipses)
    body += path_svg(points)
    x = 72
    body += ink(x, 250, "CS229", size=92, weight="700")
    body += ink(x, 320, "2018 Lecture Notes", size=46, weight="700")
    body += ink(x, 378, "Stanford Machine Learning", size=26, op=0.74)
    body += ink(x, 414, "written from the Autumn 2018 lectures", size=26, op=0.74)
    body += ink(
        x, 560, "raimbekovm.github.io/cs229-2018-notes", size=20, op=0.6, font=MONO
    )
    return wrap(w, h, "socialCard", "CS229 2018 Lecture Notes", DESC, body)


def hero():
    """The landing-page hero: same scene on the dark field, the path drawn by CSS."""
    w, h = 560, 320
    cx, cy = 280, 160
    ellipses, points = scene(cx, cy, 58)
    length = sum(math.dist(p, q) for p, q in zip(points, points[1:]))
    out = (
        f'<svg class="cs-descent__svg" viewBox="0 0 {w} {h}" role="img" aria-label="'
        "Gradient descent zig-zagging down the contours of a quadratic cost toward its minimum"
        f'" style="--cs-descent-len: {length:.0f}">\n'
    )
    out += contours_svg(cx, cy, ellipses, extra=' class="cs-descent__bowl"')
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    out += f'  <polyline class="cs-descent__path" points="{pts}"/>\n'
    n = len(points)
    for i, (x, y) in enumerate(points):
        # Each dot lands when the line reaches it; the delay is its share of the draw.
        out += f'  <circle class="cs-descent__step" cx="{x:.1f}" cy="{y:.1f}" r="3.2" style="--i: {i / (n - 1):.3f}"/>\n'
    return out + "</svg>"


def write_hero(markup):
    """Replace the block between the hero markers in index.qmd."""
    index = pathlib.Path("index.qmd")
    src = index.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- hero:start -->).*?([ \t]*<!-- hero:end -->)", re.S)
    if not pattern.search(src):
        raise SystemExit("index.qmd has no hero:start / hero:end markers")
    index.write_text(
        pattern.sub(lambda m: m.group(1) + "\n" + markup + "\n" + m.group(2), src),
        encoding="utf-8",
    )


if __name__ == "__main__":
    for name, svg in (("readme-hero", banner()), ("social-card", social_card())):
        assert "currentColor" not in svg, (
            "a README or card image cannot inherit its ink"
        )
        out = pathlib.Path(f"figures/{name}.svg")
        out.write_text(svg, encoding="utf-8")
        print(f"wrote {out} ({len(svg) / 1024:.0f} KB)")
    write_hero(hero())
    print("wrote the hero into index.qmd")

    path = descent()
    j0, jn = cost(*path[0]), cost(*path[-1])
    print()
    print(
        f"bowl: λ₁ = {LAM1}, λ₂ = {LAM2} (condition number {LAM1 / LAM2:.1f}), tilt {TILT}°"
    )
    print(
        f"step: α = {ALPHA}; per-step factors {1 - ALPHA * LAM1:+.3f} (short axis), {1 - ALPHA * LAM2:+.3f} (long axis)"
    )
    print(f"path: {len(path) - 1} updates, J {j0:.3f} → {jn:.5f}")
    # The picture only says what it is meant to say if both of these hold.
    assert ALPHA * LAM1 < 2, "past 2/λ₁ the iterates diverge"
    assert 1 - ALPHA * LAM1 < 0, "the short axis must overshoot, or there is no zig-zag"
    assert cost(*START) < levels()[0], "the start must lie inside the outermost contour"
    print()
    print("rasterise the card with:")
    print(
        '  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu'
        " --hide-scrollbars --window-size=1200,630 --screenshot=figures/social-card.png"
        ' "file://$PWD/figures/social-card.svg"'
    )
