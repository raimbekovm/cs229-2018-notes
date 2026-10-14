"""Figures for lecture 20: direct policy search and REINFORCE.

Two computed figures. The logistic policy the lecture writes for the inverted
pendulum, P(accelerate right) against the pole angle, for the two parameter
vectors on the board and for the same direction scaled up. And REINFORCE
learning curves on a small balancing MDP, with the update as written in the
lecture against the same update with each action weighted only by the rewards
that follow it, and with a baseline subtracted.

The MDP: the pole's tilt is an integer in -2..2, and it falls (absorbing, no
more reward) if it leaves that range. Gravity pushes a tilted pole further over
with probability 1/2; an upright pole wobbles either way with probability 1/4
each. Accelerating right moves the tilt one step left, accelerating left one
step right, each with probability 0.6 (otherwise it has no effect). Reward 1 for every step the pole is up. The policy is
P(right | s) = sigmoid(theta_0 + theta_1 s).

`__main__` prints every number the prose quotes and asserts the derivation by
routes that must agree: the expected update of the lecture's estimator,
computed exactly by enumerating every trajectory of the two-step problem the
board uses, against a finite-difference derivative of the expected payoff from
a dynamic programme; Monte Carlo means of all three estimators on the 20-step
problem against the same derivative; and the score function's zero mean, which
is why dropping the earlier rewards and subtracting a baseline leave the
expectation alone while cutting the variance.

Plain Python, as in lectures 1-19. Run from the repository root.
"""

import itertools
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _svg import INDIGO, PURPLE, TEAL, axes, curve, line, mapper, text, wrap

OUT = pathlib.Path(__file__).resolve().parent
STATES = (-2, -1, 0, 1, 2)
ACTIONS = (1, -1)  # accelerate right, accelerate left
H = 20
PUSH = 0.6  # the chance that an acceleration actually moves the tilt


def sigmoid(z):
    """The logistic function."""
    return 1.0 / (1.0 + math.exp(-z))


def pi(theta, s, a):
    """P(a | s) under the logistic policy."""
    p = sigmoid(theta[0] + theta[1] * s)
    return p if a == 1 else 1.0 - p


def grad_log_pi(theta, s, a):
    """The score: gradient of log P(a | s) with respect to theta."""
    p = sigmoid(theta[0] + theta[1] * s)
    c = (1.0 - p) if a == 1 else -p
    return (c, c * s)


def transition(s, a):
    """Distribution over next tilt as {s': prob}; None stands for fallen."""
    g = {-1: 0.25, 0: 0.5, 1: 0.25} if s == 0 else {(1 if s > 0 else -1): 0.5, 0: 0.5}
    out = {}
    for d, p in g.items():
        for u, q in ((a, PUSH), (0, 1.0 - PUSH)):
            n = s + d - u
            key = n if -2 <= n <= 2 else None
            out[key] = out.get(key, 0.0) + p * q
    return out


def reward(s):
    """One per step the pole is up."""
    return 0.0 if s is None else 1.0


def value_table(theta, steps):
    """V[k][s] = E[rewards from s with k transitions left], by a backward dynamic programme."""
    V = [{s: reward(s) for s in STATES} | {None: 0.0}]
    for _ in range(steps):
        prev = V[-1]
        V.append(
            {
                s: reward(s)
                + sum(pi(theta, s, a) * sum(p * prev[n] for n, p in transition(s, a).items()) for a in ACTIONS)
                for s in STATES
            }
            | {None: 0.0}
        )
    return V


def expected_payoff(theta, steps, s0=0):
    """E[R(s_0) + ... + R(s_steps)] under the policy."""
    return value_table(theta, steps)[steps][s0]


def fd_grad(theta, steps):
    """Central finite differences of the expected payoff."""
    h = 1e-5
    g = []
    for i in range(2):
        tp, tm = list(theta), list(theta)
        tp[i] += h
        tm[i] -= h
        g.append((expected_payoff(tp, steps) - expected_payoff(tm, steps)) / (2 * h))
    return g


def sample(rng, dist):
    """Draw a key from {key: prob}."""
    u, acc = rng.random(), 0.0
    for k, p in dist.items():
        acc += p
        if u < acc:
            return k
    return k


def rollout(theta, rng, steps=H, s0=0):
    """One trajectory: a list of (s_t, a_t, r_t) while the pole is up, plus the payoff."""
    s, traj = s0, []
    for t in range(steps + 1):
        if s is None:
            break
        a = 1 if rng.random() < pi(theta, s, 1) else -1
        traj.append((s, a, reward(s)))
        if t < steps:
            s = sample(rng, transition(s, a))
    return traj


def estimate(theta, traj, kind, baseline=None):
    """One REINFORCE gradient estimate from one trajectory.

    'lecture': every score term times the whole payoff, as on the board. 'to-go': the score at t times the rewards from
    t on. 'baseline':the score at t times (rewards from t on - b_t).
    """
    payoff = sum(r for _, _, r in traj)
    g = [0.0, 0.0]
    before = 0.0
    for t, (s, a, r) in enumerate(traj):
        if kind == "lecture":
            w = payoff
        else:
            w = payoff - before
            if kind == "baseline":
                w -= baseline[t]
        sc = grad_log_pi(theta, s, a)
        g[0] += sc[0] * w
        g[1] += sc[1] * w
        before += r
    return g


def baselines(theta):
    """b_t = expected rewards from step t on, given the pole is still up at t (by simulation-free DP)."""
    # Distribution of s_t, then expected reward-to-go; computed exactly.
    V = value_table(theta, H)
    dist = {0: 1.0}
    out = []
    for t in range(H + 1):
        up = sum(p for s, p in dist.items() if s is not None)
        out.append(
            sum(p * V[H - t][s] for s, p in dist.items() if s is not None) / up if up else 0.0
        )
        nxt = {}
        for s, p in dist.items():
            if s is None:
                nxt[None] = nxt.get(None, 0.0) + p
                continue
            for a in ACTIONS:
                for n, q in transition(s, a).items():
                    nxt[n] = nxt.get(n, 0.0) + p * pi(theta, s, a) * q
        dist = nxt
    return out


def exact_lecture_update(theta):
    """The lecture's two-decision estimator, averaged exactly over every trajectory (s_0 = 0 fixed)."""
    g = [0.0, 0.0]
    s0 = 0
    for a0 in ACTIONS:
        for s1, p1 in transition(s0, a0).items():
            if s1 is None:
                acts = [None]
            else:
                acts = ACTIONS
            for a1 in acts:
                prob = pi(theta, s0, a0) * p1 * (1.0 if a1 is None else pi(theta, s1, a1))
                payoff = reward(s0) + reward(s1)
                sc = list(grad_log_pi(theta, s0, a0))
                if a1 is not None:
                    sc1 = grad_log_pi(theta, s1, a1)
                    sc = [sc[0] + sc1[0], sc[1] + sc1[1]]
                g[0] += prob * sc[0] * payoff
                g[1] += prob * sc[1] * payoff
    return g


def train(kind, seed, iters, alpha):
    """Run REINFORCE; return the exact expected payoff after every 10 updates."""
    rng = random.Random(seed)
    theta = [0.0, 0.0]
    curve_ = [expected_payoff(theta, H)]
    b = baselines(theta) if kind == "baseline" else None
    for k in range(1, iters + 1):
        g = estimate(theta, rollout(theta, rng), kind, b)
        theta = [theta[0] + alpha * g[0], theta[1] + alpha * g[1]]
        if k % 10 == 0:
            curve_.append(expected_payoff(theta, H))
            if kind == "baseline":
                b = baselines(theta)
    return curve_, theta


def fig_policy():
    """P(accelerate right) against the pole angle for the lecture's parameter vectors."""
    xdom, ydom = (-0.5, 0.5), (0.0, 1.0)
    to_px = mapper(xdom, ydom, (80, 520), (30, 230))
    s = axes(
        to_px,
        xdom,
        ydom,
        (-0.4, -0.2, 0, 0.2, 0.4),
        (0, 0.25, 0.5, 0.75, 1),
        "pole angle φ (radians)",
        "P(accelerate right)",
    )
    grid = [i / 200 - 0.5 for i in range(201)]
    s += line(*to_px(-0.5, 0.5), *to_px(0.5, 0.5), op=0.3, dash="3 3")
    s += curve([(f, sigmoid(f)) for f in grid], to_px, xdom, ydom, strict=True, stroke=PURPLE, sw=2.2)
    s += curve([(f, sigmoid(f - 0.5)) for f in grid], to_px, xdom, ydom, strict=True, stroke=PURPLE, sw=2.0, dash="6 4")
    s += curve([(f, sigmoid(10 * f)) for f in grid], to_px, xdom, ydom, strict=True, stroke=TEAL, sw=2.2)
    s += curve(
        [(f, sigmoid(10 * f - 5)) for f in grid], to_px, xdom, ydom, strict=True, stroke=TEAL, sw=2.0, dash="6 4"
    )
    s += text(*(lambda p: (p[0], p[1] + 18))(to_px(-0.3, sigmoid(-0.3))), "weight 1 on φ", size=11.5, fill=PURPLE)
    s += text(*(lambda p: (p[0] + 8, p[1] + 4))(to_px(-0.15, sigmoid(-1.5))), "weight 10 on φ", size=11.5, fill=TEAL, anchor="start")
    s += line(96, 46, 124, 46, sw=2.0) + text(130, 50, "cart at x = 0", size=11.5, anchor="start")
    s += line(96, 66, 124, 66, sw=2.0, dash="6 4") + text(130, 70, "cart at x = 1, weight on x = −½ × weight on φ", size=11.5, anchor="start")
    return wrap(
        560,
        280,
        "figPolicy",
        "The logistic policy for the inverted pendulum",
        "Probability of accelerating right against the pole angle. With the lecture's weight of 1 on the angle the "
        "curve stays between 0.38 and 0.62 over half a radian either way, close to a coin flip. With a weight of 10 "
        "it is nearly a step. Moving the cart one unit right with a weight of minus a half on position shifts both "
        "curves to the right, so the policy needs a larger tilt before it pushes right.",
        s,
    )


def fig_learning(curves, jstar, iters, alphas):
    """Mean exact expected payoff over training, for the three estimators, at two step sizes."""
    ydom = (4.0, 14.0)
    s = ""
    styles = (
        ("lecture", PURPLE, "whole payoff"),
        ("to-go", INDIGO, "rewards that follow"),
        ("baseline", TEAL, "… minus a baseline"),
    )
    for k, alpha in enumerate(alphas):
        x0 = 80 + 330 * k
        xdom = (0, iters)
        to_px = mapper(xdom, ydom, (x0, x0 + 250), (40, 230))
        s += axes(to_px, xdom, ydom, (0, 100, 200, 300, 400, 500), (4, 6, 8, 10, 12, 14), "REINFORCE updates", "expected payoff J(θ)" if k == 0 else "")
        s += text(x0 + 125, 26, f"step size α = {alpha:g}", size=12.5, weight="600")
        s += line(*to_px(0, jstar), *to_px(iters, jstar), op=0.45, dash="3 3")
        for kind, col, _ in styles:
            pts = [(10 * i, v) for i, v in enumerate(curves[(kind, alpha)])]
            s += curve(pts, to_px, xdom, ydom, strict=True, stroke=col, sw=2.0)
    for j, (_, col, lab) in enumerate(styles):
        y = 160 + 18 * j
        s += line(530, y - 4, 556, y - 4, stroke=col, sw=2.2) + text(562, y, lab, size=11.5, anchor="start")
    s += line(530, 216, 556, 216, op=0.45, dash="3 3") + text(562, 220, "best policy", size=11.5, anchor="start")
    return wrap(
        700,
        290,
        "figLearning",
        "REINFORCE learning curves at two step sizes",
        "Exact expected payoff of the current policy against the number of updates, averaged over 20 runs, for "
        "three estimators of the same gradient: the whole payoff multiplying every score term, as in the lecture; "
        "each action's score multiplied only by the rewards from that step on; and the same with a baseline "
        "subtracted. At the smaller step size all three reach the best policy, the lecture's version last. At the "
        "larger one the two lower-variance versions converge within about a hundred updates, while the lecture's "
        "version dips and some of its runs stall on a poor policy.",
        s,
    )


FIGURES = {}

if __name__ == "__main__":
    # 1. The board's derivation: exact expected update on the two-decision problem = gradient of J.
    rng = random.Random(0)
    for _ in range(20):
        th = [rng.uniform(-2, 2), rng.uniform(-2, 2)]
        ex, fd = exact_lecture_update(th), fd_grad(th, 1)
        assert all(abs(x - y) < 1e-8 for x, y in zip(ex, fd)), (th, ex, fd)
    print("two-decision problem: exact E[lecture update] = dJ/dtheta at 20 random theta")

    # 2. Score function has zero mean in every state.
    for th, s in itertools.product([[0.3, -1.2], [-1, 2]], STATES):
        m = [sum(pi(th, s, a) * grad_log_pi(th, s, a)[i] for a in ACTIONS) for i in range(2)]
        assert max(abs(x) for x in m) < 1e-12
    print("E_a[grad log pi(a|s)] = 0 in every state")

    # 3. 20-step problem: Monte Carlo means of all three estimators vs finite differences; variances.
    th = [0.0, 0.0]
    fd = fd_grad(th, H)
    b = baselines(th)
    N = 20000
    rng = random.Random(1)
    trajs = [rollout(th, rng) for _ in range(N)]
    print(f"theta = 0: J = {expected_payoff(th, H):.3f}, dJ/dtheta = {[round(x, 3) for x in fd]}")
    var = {}
    for kind in ("lecture", "to-go", "baseline"):
        gs = [estimate(th, tr, kind, b) for tr in trajs]
        mean = [sum(g[i] for g in gs) / N for i in range(2)]
        v = [sum((g[i] - mean[i]) ** 2 for g in gs) / (N - 1) for i in range(2)]
        se = [math.sqrt(x / N) for x in v]
        assert all(abs(mean[i] - fd[i]) < 4 * se[i] for i in range(2)), (kind, mean, fd, se)
        var[kind] = v
        print(
            f"  {kind:9s} mean {[round(x, 3) for x in mean]} +- {[round(x, 3) for x in se]}, variance {[round(x, 1) for x in v]}"
        )
    for kind in ("to-go", "baseline"):
        print(f"  variance ratio lecture/{kind} on theta_1: {var['lecture'][1] / var[kind][1]:.1f}")

    # 4. Optimum: the best deterministic policy, and the logistic policy's supremum.
    best = 0.0
    for signs in itertools.product((1, -1), repeat=5):
        dpol = dict(zip(STATES, signs))
        V = {s: reward(s) for s in STATES} | {None: 0.0}
        for _ in range(H):
            V = {s: reward(s) + sum(p * V[n] for n, p in transition(s, dpol[s]).items()) for s in STATES} | {None: 0.0}
        if V[0] > best:
            best, best_pol = V[0], dpol
    jstar = best
    print(f"best deterministic policy from tilt 0: J* = {jstar:.3f}, actions {best_pol}")
    print(
        f"logistic policy theta = (0, 50): J = {expected_payoff([0, 50], H):.3f}; theta = 0: {expected_payoff([0, 0], H):.3f}"
    )

    # 5. Learning curves at two step sizes.
    ITERS, SEEDS, ALPHAS = 500, 20, (0.01, 0.03)
    curves = {}
    for alpha, kind in itertools.product(ALPHAS, ("lecture", "to-go", "baseline")):
        runs = [train(kind, seed, ITERS, alpha)[0] for seed in range(SEEDS)]
        c = [sum(r[i] for r in runs) / SEEDS for i in range(len(runs[0]))]
        curves[(kind, alpha)] = c
        reach = next((10 * i for i, v in enumerate(c) if v >= 0.9 * jstar), None)
        stuck = sum(r[-1] < 0.9 * jstar for r in runs)
        print(
            f"  alpha {alpha} {kind:9s} mean J at 50/100/500: {c[5]:.2f} / {c[10]:.2f} / {c[-1]:.2f}; "
            f"mean first >= 0.9 J* at {reach}; runs below 0.9 J* at the end: {stuck}/{SEEDS}"
        )

    print(f"policy figure: P(right) at phi = +-0.5 with theta_phi = 1: {sigmoid(-0.5):.3f}, {sigmoid(0.5):.3f}")

    FIGURES.update({"diagram_policy.svg": fig_policy(), "diagram_learning.svg": fig_learning(curves, jstar, ITERS, ALPHAS)})
    for name, svg in FIGURES.items():
        (OUT / name).write_text(svg)
        print("wrote", name)
