"""Figures for lecture 19: state-action rewards, finite-horizon MDPs, linearization and LQR.

Three computed figures. A corridor with a +10 at one end and a +1 nearer the
other, solved by the finite-horizon dynamic programme, showing the optimal
action at every cell for every number of steps left: the non-stationary policy
the lecture describes. An inverted pendulum, its gravity term beside the
tangent line that linearization replaces it with, and the closed loop under the
LQR gain designed on that line. The LQR gains L_t of a cart against the time
step, for two noise levels, flat far from the horizon and identical for both.

`__main__` prints every number the prose quotes and asserts the claims by two
routes that must agree: the dynamic programme against brute force over every
action sequence; the closed-form action L_t s against a numerical maximisation
of the Bellman right-hand side; the predicted value s'Phi s + Psi against Monte
Carlo rollouts, which is what settles the sign of the trace term (the 2018
notes and the board print it with the opposite sign); the gains for zero noise
against large noise; and the LQR policy against hundreds of perturbed policies,
linear and not.

Plain Python, as in lectures 1-18. Run from the repository root.
"""

import itertools
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _linalg import T, add, mm, mv
from _svg import (
    GREEN,
    INDIGO,
    PURPLE,
    TEAL,
    axes,
    curve,
    line,
    mapper,
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------- finite-horizon corridor

CELLS = 11
GOALS = {0: 10.0, 8: 1.0}  # reaching a goal pays once and ends the episode
MOVE_COST = -0.1  # a state-action reward: moving costs, staying is free
MOVES = {"L": -1, "S": 0, "R": 1}


def step(s, a):
    """Deterministic move, clamped to the corridor; returns (next state, reward)."""
    nxt = min(max(s + MOVES[a], 0), CELLS - 1)
    return nxt, (MOVE_COST if a != "S" else 0.0) + GOALS.get(nxt, 0.0)


def finite_horizon(H):
    """V[h][s] and pi[h][s] with h steps left, by the backward recursion; goals are absorbing with value 0."""
    V = [[0.0] * CELLS]
    pi = [[None] * CELLS]
    for h in range(1, H + 1):
        v, p = [], []
        for s in range(CELLS):
            if s in GOALS:
                v.append(0.0)
                p.append(None)
                continue
            q = {a: step(s, a)[1] + V[h - 1][step(s, a)[0]] for a in MOVES}
            best = max(q.values())
            # Undiscounted and with waiting free, moving now and moving later often tie; show the move,
            # so "stay" in the figure means staying is strictly better.
            a = next(a for a in ("L", "R", "S") if abs(q[a] - best) < 1e-12)
            v.append(best)
            p.append(a)
        V.append(v)
        pi.append(p)
    return V, pi


def brute_force(s, h):
    """Best total reward over every action sequence of length h."""
    best = -1e9
    for seq in itertools.product(MOVES, repeat=h):
        x, tot = s, 0.0
        for a in seq:
            if x in GOALS:
                break
            x, r = step(x, a)
            tot += r
        best = max(best, tot)
    return best


# ---------------------------------------------------------------- matrices


def neg(M):
    """-M."""
    return [[-x for x in r] for r in M]


def inv_any(M):
    """Inverse of a small square matrix by Gauss-Jordan elimination."""
    n = len(M)
    A = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [x / piv for x in A[c]]
        for r in range(n):
            if r != c:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [r[n:] for r in A]


def trace(M):
    """Sum of the diagonal."""
    return sum(M[i][i] for i in range(len(M)))


def quad(s, M):
    """s^T M s."""
    return sum(s[i] * M[i][j] * s[j] for i in range(len(s)) for j in range(len(s)))


def riccati(A, B, U, V, Sigma, H, psi_sign=1.0):
    """Backward LQR recursion for V_t(s) = s'Phi_t s + Psi_t, R = -s'Us - a'Va, horizon H.

    Returns lists Phi[t], Psi[t], L[t] for t = 0..H (L[H] is the zero gain). `psi_sign` = -1 reproduces the sign printed
    in the notes, so the Monte Carlo check can tell the two apart.
    """
    Phi, Psi, L = [None] * (H + 1), [0.0] * (H + 1), [None] * (H + 1)
    Phi[H] = neg(U)
    L[H] = [[0.0] * len(A) for _ in range(len(B[0]))]
    for t in range(H - 1, -1, -1):
        P = Phi[t + 1]
        M = add(mm(mm(T(B), P), B), V, -1.0)  # B'PB - V, negative definite
        Mi = inv_any(M)
        L[t] = neg(mm(Mi, mm(mm(T(B), P), A)))  # a* = -(B'PB - V)^-1 B'PA s
        inner = add(P, mm(mm(mm(P, B), Mi), mm(T(B), P)), -1.0)
        Phi[t] = add(mm(mm(T(A), inner), A), U, -1.0)
        Psi[t] = Psi[t + 1] + psi_sign * trace(mm(Sigma, P))
    return Phi, Psi, L


def chol2(S):
    """Cholesky factor of a 2x2 covariance (allows zeros)."""
    a = math.sqrt(S[0][0])
    b = S[1][0] / a if a else 0.0
    c = math.sqrt(max(S[1][1] - b * b, 0.0))
    return [[a, 0.0], [b, c]]


def rollout(A, B, U, V, Sigma, policy, s0, H, rng):
    """Total reward of one episode of the linear-Gaussian system under policy(t, s)."""
    Lc = chol2(Sigma)
    s, tot = list(s0), 0.0
    for t in range(H + 1):
        a = policy(t, s)
        tot += -quad(s, U) - quad(a, V)
        if t == H:
            break
        g = [rng.gauss(0, 1), rng.gauss(0, 1)]
        w = mv(Lc, g)
        s = [x + y + z for x, y, z in zip(mv(A, s), mv(B, a), w)]
    return tot


# The cart: position and velocity at 10 Hz, force as the action.
DT = 0.1
A_CART = [[1.0, DT], [0.0, 1.0]]
B_CART = [[0.5 * DT * DT], [DT]]
U_CART = [[1.0, 0.0], [0.0, 0.1]]
V_CART = [[0.1]]
H_CART = 60

# ---------------------------------------------------------------- the pendulum

G_L, DAMP, DTP = 9.8, 0.5, 0.05  # g / l, damping, time step; torque enters unscaled


def pend_f(s, u):
    """One Euler step of the inverted pendulum, theta measured from upright."""
    th, om = s
    return [th + DTP * om, om + DTP * (G_L * math.sin(th) - DAMP * om + u)]


A_PEND = [[1.0, DTP], [DTP * G_L, 1.0 - DTP * DAMP]]  # Jacobian of pend_f at (0, 0)
B_PEND = [[0.0], [DTP]]


def numeric_jacobian(f, s, u, h=1e-6):
    """Finite-difference Jacobians of f with respect to s and u."""
    f0 = f(s, u)
    As = [[(f([s[0] + h * (j == 0), s[1] + h * (j == 1)], u)[i] - f0[i]) / h for j in range(2)] for i in range(2)]
    Bs = [[(f(s, u + h)[i] - f0[i]) / h] for i in range(2)]
    return As, Bs


# ---------------------------------------------------------------- figures


def fig_horizon(V, pi, H):
    """Rows: steps left; columns: cells; fill: optimal action; label: optimal value."""
    cw, ch, x0, y0 = 46, 22, 110, 40
    fills = {"L": TEAL, "R": PURPLE, "S": None}
    s = ""
    for c in range(CELLS):
        s += text(x0 + c * cw + cw / 2, y0 - 8, f"{c}", size=11, op=0.8)
    for h in range(1, H + 1):
        y = y0 + (H - h) * ch
        s += text(x0 - 10, y + 15, f"{h}", size=11, anchor="end", op=0.8)
        for c in range(CELLS):
            x = x0 + c * cw
            if c in GOALS:
                s += rect(x, y, cw, ch, fill=GREEN, stroke="currentColor", sw=0.5, op=0.55, so=0.4)
                s += text(x + cw / 2, y + 15, f"+{GOALS[c]:g}", size=10.5)
                continue
            a = pi[h][c]
            if fills[a]:
                s += rect(x, y, cw, ch, fill=fills[a], stroke="currentColor", sw=0.5, op=0.85, so=0.4)
                s += text(x + cw / 2, y + 15, f"{V[h][c]:.1f}", size=10.5, fill="#ffffff")
            else:
                s += rect(x, y, cw, ch, fill="none", stroke="currentColor", sw=0.5, so=0.4)
                s += text(x + cw / 2, y + 15, f"{V[h][c]:.1f}", size=10.5, op=0.7)
    s += text(x0 + CELLS * cw / 2, y0 - 24, "cell", size=12)
    s += f'  <text x="{x0 - 40:.1f}" y="{y0 + H * ch / 2:.1f}" font-size="12" fill="currentColor" text-anchor="middle" transform="rotate(-90 {x0 - 40:.1f} {y0 + H * ch / 2:.1f})">steps left on the clock</text>\n'
    ly = y0 + H * ch + 22
    for k, (col, lab) in enumerate(((TEAL, "move left"), (PURPLE, "move right"), (None, "stay"))):
        lx = x0 + 150 * k
        s += rect(
            lx, ly - 10, 14, 14, fill=col or "none", stroke="currentColor", sw=0.5, op=0.85 if col else 1.0, so=0.4
        )
        s += text(lx + 20, ly + 1, lab, size=11.5, anchor="start")
    return wrap(
        640,
        y0 + H * ch + 40,
        "figHorizon",
        "A non-stationary optimal policy",
        "A grid with one column per corridor cell, 0 to 10, and one row per number of steps left, 1 at the bottom "
        "to 12 at the top. Cell 0 holds a plus-ten and cell 8 a plus-one; moving costs 0.1. Each square is coloured "
        "by the optimal action and shows the optimal value. With many steps left every cell from 1 to 7 heads left "
        "for the plus-ten; as the clock runs down, the cells nearest the plus-one switch to heading right, and cells "
        "that can reach neither goal in time stay still with value zero. Cells 9 and 10 always head left to the "
        "plus-one.",
        s,
    )


def fig_pendulum(runs):
    """Left: the gravity term and its tangent. Right: theta(t) under LQR, nonlinear against the linear model."""
    xdom, ydom = (-math.pi, math.pi), (-16.0, 16.0)
    to_px = mapper(xdom, ydom, (70, 330), (30, 240))
    s = axes(
        to_px,
        xdom,
        ydom,
        (-3, -2, -1, 0, 1, 2, 3),
        (-15, -10, -5, 0, 5, 10, 15),
        "angle θ (rad)",
        "gravity term (g/l) sin θ",
    )
    xs = [xdom[0] + (xdom[1] - xdom[0]) * i / 400 for i in range(401)]
    s += curve([(x, G_L * math.sin(x)) for x in xs], to_px, xdom, ydom, strict=True, stroke=PURPLE, sw=2.0)
    s += curve([(x, G_L * x) for x in xs], to_px, xdom, ydom, stroke=GREEN, sw=2.0)
    px, py = to_px(1.75, 13.2)
    s += text(px, py, "tangent at θ̄ = 0", size=11.5, anchor="start")
    px, py = to_px(0.5, -7.0)
    s += text(px, py, "(g/l) sin θ", size=11.5, anchor="start", fill=PURPLE)

    xd2, yd2 = (0.0, 4.0), (-0.25, 1.5)
    to2 = mapper(xd2, yd2, (440, 720), (30, 240))
    s += axes(to2, xd2, yd2, (0, 1, 2, 3, 4), (0, 0.5, 1, 1.5), "time (s)", "angle θ (rad)")
    for (th0, col), (nonlin, lin) in runs.items():
        s += curve(
            [(k * DTP, v) for k, v in enumerate(lin)], to2, xd2, yd2, strict=True, stroke=col, sw=1.4, dash="5 4"
        )
        s += curve([(k * DTP, v) for k, v in enumerate(nonlin)], to2, xd2, yd2, strict=True, stroke=col, sw=2.0)
    for k, (th0, col) in enumerate(runs):
        s += line(580, 46 + 16 * k, 608, 46 + 16 * k, stroke=col, sw=2.0)
        s += text(614, 50 + 16 * k, f"θ₀ = {th0:g}", size=11.5, anchor="start")
    s += line(580, 104, 608, 104, sw=2.0) + text(614, 108, "true pendulum", size=11.5, anchor="start")
    s += line(580, 120, 608, 120, sw=1.4, dash="5 4") + text(614, 124, "linear model", size=11.5, anchor="start")
    return wrap(
        760,
        290,
        "figPendulum",
        "Linearizing the inverted pendulum and controlling it with LQR",
        "Left: the gravity term of the inverted pendulum, g over l times sine theta, against the angle, with the "
        "straight line tangent to it at the upright position. The two agree closely within about half a radian and "
        "part company beyond it; at 1.5 radians the line overstates gravity by half. Right: the pole angle against "
        "time under the LQR controller designed on the linear model, from starting angles of 0.3, 1.0 and 1.4 "
        "radians, for the true nonlinear pendulum (solid) and for the linear model's own prediction (dashed). At "
        "0.3 the two are indistinguishable; at 1.4 they visibly differ, although the controller still brings the "
        "true pole upright.",
        s,
    )


def fig_gains(L_lo, L_hi):
    """The two entries of L_t against t, for zero and large noise."""
    xdom, ydom = (0, H_CART), (-5.0, 0.5)
    to_px = mapper(xdom, ydom, (80, 520), (30, 230))
    s = axes(
        to_px,
        xdom,
        ydom,
        (0, 10, 20, 30, 40, 50, 60),
        (-5, -4, -3, -2, -1, 0),
        "time step t (horizon T = 60)",
        "gain entry",
    )
    for j, col, lab in ((0, PURPLE, "gain on position"), (1, TEAL, "gain on velocity")):
        s += curve([(t, L_lo[t][0][j]) for t in range(H_CART + 1)], to_px, xdom, ydom, strict=True, stroke=col, sw=2.2)
        for t in range(0, H_CART + 1, 3):
            px, py = to_px(t, L_hi[t][0][j])
            s += f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="none" stroke="{col}" stroke-width="1.2"/>\n'
        px, py = to_px(18, L_lo[18][0][j])
        s += text(px, py - 8 if j else py + 18, lab, size=11.5, fill=col)
    s += line(110, 196, 138, 196, sw=2.2) + text(144, 200, "Σ_w = 0", size=11.5, anchor="start")
    s += '  <circle cx="124.0" cy="214.0" r="2.6" fill="none" stroke="currentColor" stroke-width="1.2"/>\n'
    s += text(144, 218, "Σ_w = 0.25 I (every third step)", size=11.5, anchor="start")
    return wrap(
        560,
        280,
        "figGains",
        "LQR gains over the horizon, with and without noise",
        "The two entries of the optimal gain L_t, which multiplies position and velocity to give the force, against "
        "the time step for a cart with a 60-step horizon. Both are flat, at about minus 2.76 and minus 2.51, for the "
        "first half of the horizon and relax to zero over the last 25 or so steps, reaching zero at the final step. Curves for zero "
        "noise and circles for large noise coincide exactly.",
        s,
    )


FIGURES = {}

if __name__ == "__main__":
    # 1. Finite horizon: DP against brute force, and the policy's dependence on the clock.
    HMAX = 12
    Vh, pih = finite_horizon(HMAX)
    for h in range(1, 8):
        for s0 in range(CELLS):
            if s0 not in GOALS:
                assert abs(Vh[h][s0] - brute_force(s0, h)) < 1e-9, (h, s0)
    print("finite-horizon DP agrees with brute force for h <= 7")
    for h in (2, 4, 12):
        print(f"{h} steps left: V =", [round(v, 1) for v in Vh[h]], "pi =", "".join(a or "." for a in pih[h]))
    print("cell 5: action by steps left", "".join(pih[h][5] for h in range(1, HMAX + 1)))

    # 2. Riccati: closed form against numerical maximisation of the Bellman right-hand side.
    Sig_lo = [[0.0, 0.0], [0.0, 0.0]]
    Sig_hi = [[0.25, 0.0], [0.0, 0.25]]
    Phi, Psi, L = riccati(A_CART, B_CART, U_CART, V_CART, Sig_hi, H_CART)
    rng = random.Random(1)
    for _ in range(20):
        t = rng.randrange(0, H_CART)
        s0 = [rng.uniform(-2, 2), rng.uniform(-2, 2)]

        def rhs(a, t=t, s0=s0):
            """Bellman right-hand side, expectation in closed form via E[w'Pw] = tr(Sigma P)."""
            m = [x + y for x, y in zip(mv(A_CART, s0), mv(B_CART, [a]))]
            return (
                -quad(s0, U_CART)
                - V_CART[0][0] * a * a
                + quad(m, Phi[t + 1])
                + trace(mm(Sig_hi, Phi[t + 1]))
                + Psi[t + 1]
            )

        lo, hi = -50.0, 50.0  # golden-section search on a concave quadratic
        for _ in range(200):
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            lo, hi = (lo, m2) if rhs(m1) > rhs(m2) else (m1, hi)
        a_num = (lo + hi) / 2
        a_cf = mv(L[t], s0)[0]
        assert abs(a_num - a_cf) < 1e-6, (a_num, a_cf)
        assert abs(rhs(a_num) - (quad(s0, Phi[t]) + Psi[t])) < 1e-6
    print("a* = L_t s and V_t = s'Phi_t s + Psi_t agree with numerical maximisation at 20 random (t, s)")

    # 3. The sign of the trace term: Monte Carlo of the closed loop against both signs.
    s0 = [1.0, 0.0]
    _, Psi_notes, _ = riccati(A_CART, B_CART, U_CART, V_CART, Sig_hi, H_CART, psi_sign=-1.0)
    rng = random.Random(2)
    N = 4000
    tots = [
        rollout(A_CART, B_CART, U_CART, V_CART, Sig_hi, lambda t, s: mv(L[t], s), s0, H_CART, rng) for _ in range(N)
    ]
    mc = sum(tots) / N
    se = math.sqrt(sum((x - mc) ** 2 for x in tots) / (N - 1) / N)
    pred, pred_notes = quad(s0, Phi[0]) + Psi[0], quad(s0, Phi[0]) + Psi_notes[0]
    print(
        f"V_0(s0): Monte Carlo {mc:.2f} +- {se:.2f}; s'Phi s + Psi with +tr {pred:.2f}; with the notes' -tr {pred_notes:.2f}"
    )
    assert abs(mc - pred) < 4 * se and abs(mc - pred_notes) > 20 * se
    # The notes' a* without the minus sign: the same magnitude pushing the wrong way.
    bad = [rollout(A_CART, B_CART, U_CART, V_CART, Sig_lo, lambda t, s: [-x for x in mv(L[t], s)], s0, H_CART, rng)]
    print(
        f"deterministic payoff from s0: with -(B'PB-V)^-1 B'PA {quad(s0, Phi[0]):.2f}; without the minus {bad[0]:.3g}"
    )

    # 4. Noise does not enter the gains.
    _, Psi0, L0 = riccati(A_CART, B_CART, U_CART, V_CART, Sig_lo, H_CART)
    dL = max(abs(L0[t][0][j] - L[t][0][j]) for t in range(H_CART + 1) for j in range(2))
    assert dL == 0.0
    print(f"max |L_t(Sigma=0) - L_t(Sigma=0.25 I)| = {dL}; Psi_0 = {Psi0[0]:.2f} vs {Psi[0]:.2f}")
    print(
        "L_0 =",
        [round(x, 3) for x in L[0][0]],
        "; L_50 =",
        [round(x, 3) for x in L[50][0]],
        "; L_59 =",
        [round(x, 3) for x in L[59][0]],
    )
    flat = next(t for t in range(H_CART, -1, -1) if max(abs(L[t][0][j] - L[0][0][j]) for j in range(2)) < 0.01)
    print("gains within 0.01 of L_0 up to t =", flat)

    # 5. No policy, linear or not, beats L_t (deterministic system, so one rollout is the exact payoff).
    best = quad(s0, Phi[0])
    rng = random.Random(3)
    worse = 0
    for k in range(400):
        c = [rng.gauss(0, 0.3), rng.gauss(0, 0.3), rng.gauss(0, 0.3)]

        def pol(t, s, c=c, nonlinear=k % 2):
            """L_t s plus a linear or nonlinear perturbation."""
            base = mv(L[t], s)[0]
            extra = (
                c[0] * s[0] + c[1] * s[1]
                if not nonlinear
                else c[0] * math.tanh(3 * s[0]) + c[1] * s[1] ** 2 + c[2] * math.sin(t)
            )
            return [base + extra]

        p = rollout(A_CART, B_CART, U_CART, V_CART, Sig_lo, pol, s0, H_CART, rng)
        worse += p < best - 1e-9
    assert worse == 400
    print(f"400 perturbed policies (half nonlinear): all worse than L_t, whose payoff is {best:.4f}")

    # 6. Linearization: Jacobian by hand against finite differences; LQR on the pendulum.
    Aj, Bj = numeric_jacobian(pend_f, [0.0, 0.0], 0.0)
    assert max(abs(Aj[i][j] - A_PEND[i][j]) for i in range(2) for j in range(2)) < 1e-5
    assert max(abs(Bj[i][0] - B_PEND[i][0]) for i in range(2)) < 1e-5
    for th in (0.25, 0.5, 1.0, 1.5):
        print(
            f"gravity term at {th}: sin {G_L * math.sin(th):.2f}, tangent {G_L * th:.2f}, ratio {th / math.sin(th):.3f}"
        )
    _, _, Lp = riccati(A_PEND, B_PEND, [[1.0, 0.0], [0.0, 0.1]], [[0.01]], Sig_lo, 400)
    Kp = Lp[0]
    print("pendulum stationary gain", [round(x, 2) for x in Kp[0]])
    runs = {}
    for th0, col in ((0.3, TEAL), (1.0, INDIGO), (1.4, PURPLE)):
        sn, sl = [th0, 0.0], [th0, 0.0]
        nonlin, lin = [th0], [th0]
        for _ in range(80):
            sn = pend_f(sn, mv(Kp, sn)[0])
            sl = [x + y for x, y in zip(mv(A_PEND, sl), mv(B_PEND, mv(Kp, sl)))]
            nonlin.append(sn[0])
            lin.append(sl[0])
        gap = max(abs(a - b) for a, b in zip(nonlin, lin))
        print(
            f"pendulum from {th0}: final angle {nonlin[-1]:.4f}, max gap to linear prediction {gap:.3f}, min angle {min(nonlin):.3f}"
        )
        runs[(th0, col)] = (nonlin, lin)
    for th0 in (1.6, 2.0, 2.5, 3.0):
        sn = [th0, 0.0]
        for _ in range(400):
            sn = pend_f(sn, mv(Kp, sn)[0])
        print(f"pendulum from {th0}: angle after 20 s {sn[0]:.3f}")

    FIGURES.update(
        {
            "diagram_horizon.svg": fig_horizon(Vh, pih, HMAX),
            "diagram_pendulum.svg": fig_pendulum(runs),
            "diagram_gains.svg": fig_gains(L0, L),
        }
    )
    for name, svg in FIGURES.items():
        (OUT / name).write_text(svg)
        print("wrote", name)
