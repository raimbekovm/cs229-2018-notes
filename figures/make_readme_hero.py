"""Generate the site's three signature images: gradient descent on a quadratic bowl.

The picture is the one lecture 2 is built around — the contours of a cost
J(θ) over two parameters, and the path batch gradient descent takes across
them. It is computed, not drawn: the bowl is

    J(θ) = ½ (λ₁ a² + λ₂ b²),   (a, b) = θ − θ* in the bowl's principal axes,

and every dot on the path is one update θ := θ − α ∇J(θ). With λ₁ ≫ λ₂ and a
step size near 2/λ₁ the iterates overshoot across the narrow axis and creep
along the long one — the zig-zag every student draws once and remembers. The
numbers printed by `__main__` are what was actually drawn.

Three outputs, one geometry:

- `figures/readme-hero.svg`, the README banner.
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
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _svg import INK, MONO, TEAL, rect, text, wrap

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
        f"{round(int(a[k : k + 2], 16) + (int(b[k : k + 2], 16) - int(a[k : k + 2], 16)) * f):02x}" for k in (1, 3, 5)
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
        (scale * math.sqrt(2 * c / LAM2), scale * math.sqrt(2 * c / LAM1), i / (len(cs) - 1)) for i, c in enumerate(cs)
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
    return text(x, y, s, size=size, fill=INK, anchor=anchor, op=op, weight=weight, font=font, style=style)


DESC = (
    "Filled contour lines of a quadratic cost over two parameters, purple at the rim and yellow at the "
    "minimum, elongated and tilted. A white path of dots starts on the outer slope and zig-zags across the "
    "narrow axis of the bowl while creeping along the long one until it reaches the centre: batch gradient "
    "descent with a step size close to the largest stable one."
)


def banner():
    """The README banner, 1280 x 420."""
    w, h = 1280, 420
    cx, cy = 300, 214
    ellipses, points = scene(cx, cy, 58)
    body = rect(0, 0, w, h, fill=PLATE, stroke="none")
    body += rect(0, 0, w, 5, fill=TEAL, stroke="none", op=0.9)
    body += contours_svg(cx, cy, ellipses)
    body += path_svg(points)
    x = 620
    body += ink(x, 150, "CS229 2018 Lecture Notes", size=46, weight="700")
    body += ink(x, 190, "Stanford Machine Learning, written from the Autumn 2018 lectures", size=19, op=0.74)
    body += ink(x, 240, "θ := θ − α ∇J(θ)", size=17, op=0.7, font=MONO)
    body += ink(x, 330, "Every diagram in these notes is drawn by a script committed", size=15, op=0.6, style="italic")
    body += ink(x, 352, "beside it. This one included.", size=15, op=0.6, style="italic")
    return wrap(w, h, "readmeHero", "CS229 2018 Lecture Notes", DESC, body)


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
    body += ink(x, 560, "raimbekovm.github.io/cs229-2018-notes", size=20, op=0.6, font=MONO)
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
        out += (
            f'  <circle class="cs-descent__step" cx="{x:.1f}" cy="{y:.1f}" r="3.2" style="--i: {i / (n - 1):.3f}"/>\n'
        )
    return out + "</svg>"


def write_hero(markup):
    """Replace the block between the hero markers in index.qmd."""
    index = pathlib.Path("index.qmd")
    src = index.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- hero:start -->).*?([ \t]*<!-- hero:end -->)", re.S)
    if not pattern.search(src):
        raise SystemExit("index.qmd has no hero:start / hero:end markers")
    index.write_text(pattern.sub(lambda m: m.group(1) + "\n" + markup + "\n" + m.group(2), src), encoding="utf-8")


if __name__ == "__main__":
    for name, svg in (("readme-hero", banner()), ("social-card", social_card())):
        assert "currentColor" not in svg, "a README or card image cannot inherit its ink"
        out = pathlib.Path(f"figures/{name}.svg")
        out.write_text(svg, encoding="utf-8")
        print(f"wrote {out} ({len(svg) / 1024:.0f} KB)")
    write_hero(hero())
    print("wrote the hero into index.qmd")

    path = descent()
    j0, jn = cost(*path[0]), cost(*path[-1])
    print()
    print(f"bowl: λ₁ = {LAM1}, λ₂ = {LAM2} (condition number {LAM1 / LAM2:.1f}), tilt {TILT}°")
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
