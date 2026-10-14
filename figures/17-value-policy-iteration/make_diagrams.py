"""Figures for lecture 17: value functions, value and policy iteration, learning a model, exploration.

Three computed figures: the value function of a poor policy beside the optimal
value function and policy of the lecture's gridworld; the error of value
iteration against the number of sweeps, for synchronous and asynchronous
updates at two discount factors, with the gamma^k bound; and a corridor MDP in
which a greedy model-based learner can settle on the near +1 and never find the
far +10, against epsilon-greedy learners.

`__main__` prints every number the prose quotes and asserts the claims by two
routes that must agree: V_pi from the linear system against iterated policy
evaluation; V* from value iteration against the linear solve of its own greedy
policy, and against thousands of random policies; the synchronous error bound
gamma^k; policy iteration's monotone improvement and finite termination at the
same policy; and a learned model that, with enough experience, gives the same
optimal policy with or without Laplace smoothing.

Plain Python, as in lectures 1-16. Run from the repository root.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _gridworld import (
    ACTIONS,
    STATES,
    TERMINAL,
    backup,
    draw_grid,
    greedy,
    policy_value,
    q_future,
    reward,
    trans,
    value_iteration,
)
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
    rect,
    text,
    wrap,
)

OUT = pathlib.Path(__file__).resolve().parent
STEP, GAMMA = -0.02, 0.99
NONTERM = [s for s in STATES if s not in TERMINAL]

# ---------------------------------------------------------------- value functions


def evaluate_iteratively(pi, sweeps=5000):
    """V_pi by repeated application of the linear Bellman operator."""
    V = {s: 0.0 for s in STATES}
    for _ in range(sweeps):
        V = {s: reward(s, STEP) + (0 if s in TERMINAL else GAMMA * q_future(V, s, pi[s])) for s in STATES}
    return V


def fig_values(pi_bad, v_bad, pi_star, v_star):
    """Left: a poor policy and its value. Right: the optimal policy and V*."""
    cell = 72
    s = draw_grid(30, 40, cell, values=v_bad, policy=pi_bad, fmt="{:.2f}")
    s += text(30 + 2 * cell, 26, "π = East everywhere, and its values", size=12)
    s += draw_grid(370, 40, cell, values=v_star, policy=pi_star, mark=(3, 1), fmt="{:.3f}")
    s += text(370 + 2 * cell, 26, "π* and V*", size=12)
    return wrap(
        690,
        270,
        "figValues",
        "The value of a poor policy and the optimal value function",
        "Two copies of the gridworld. Left: a policy that always commands East, with the value of every state "
        "under it: 0.59 to 0.76 along the top row, which leads into the plus-one, and minus 0.3 to minus 1.09 "
        "below, where the robot runs into the minus-one or stalls against the wall. Right: the optimal policy and "
        "the optimal values, from 0.49 at the bottom right to 0.93 next to the plus-one. Cell (3, 1) is "
        "highlighted: its optimal action is West.",
        s,
    )


# ---------------------------------------------------------------- value iteration


def vi_errors(gamma, vstar, sweeps, asynchronous):
    """Max-norm error after each sweep of value iteration from V = 0, and the sweep where greedy(V) becomes optimal."""
    V = {s: 0.0 for s in STATES}
    errs = [max(abs(V[s] - vstar[s]) for s in STATES)]
    pistar = greedy(vstar)
    first_opt = None
    for k in range(1, sweeps + 1):
        if asynchronous:
            for s in STATES:  # Gauss-Seidel: each state sees the values already updated in this sweep
                V[s] = reward(s, STEP) + (0 if s in TERMINAL else gamma * max(q_future(V, s, a) for a in ACTIONS))
        else:
            V = backup(V, STEP, gamma)
        errs.append(max(abs(V[s] - vstar[s]) for s in STATES))
        if first_opt is None and greedy(V) == pistar:
            first_opt = k
        elif first_opt is not None and greedy(V) != pistar:
            first_opt = None
    return errs, first_opt


def fig_vi(runs):
    """Log10 error against sweep for each run, with gamma^k bounds."""
    xdom, ydom = (0, 40), (-10.0, 0.5)
    to_px = mapper(xdom, ydom, (80, 560), (30, 250))
    s = axes(to_px, xdom, ydom, (0, 10, 20, 30, 40), (-10, -8, -6, -4, -2, 0), "sweeps k", "log₁₀ max-norm error", fmt="{:g}")
    for (gamma, asyn), (errs, _) in runs.items():
        col = PURPLE if gamma == 0.99 else TEAL
        pts = [(k, math.log10(e)) for k, e in enumerate(errs) if e > 0]
        s += curve(pts, to_px, xdom, ydom, stroke=col, sw=2.0, dash="6 4" if asyn else None)
        if not asyn:
            e0 = errs[0]
            s += curve(
                [(k, math.log10(e0) + k * math.log10(gamma)) for k in range(0, 41)],
                to_px,
                xdom,
                ydom,
                stroke=col,
                sw=1.0,
                op=0.5,
                dash="2 3",
            )
    legend = (
        (PURPLE, None, 2.0, "γ = 0.99, synchronous"),
        (PURPLE, "6 4", 2.0, "γ = 0.99, asynchronous"),
        (TEAL, None, 2.0, "γ = 0.9, synchronous"),
        (TEAL, "6 4", 2.0, "γ = 0.9, asynchronous"),
        (INDIGO, "2 3", 1.0, "bound γᵏ ‖V₀ − V*‖ (synchronous)"),
    )
    for k, (col, dash, sw, lab) in enumerate(legend):
        y = 160 + 18 * k
        s += line(100, y, 128, y, stroke=col, sw=sw, dash=dash) + text(134, y + 4, lab, size=11.5, anchor="start")
    return wrap(
        640,
        300,
        "figVi",
        "Value iteration converges geometrically",
        "Base-ten logarithm of the largest error in the value function against the number of sweeps, for the "
        "gridworld. Every curve falls below one in a hundred million within about 35 sweeps, at either discount "
        "factor; asynchronous sweeps, which use each new value at once, fall faster than synchronous ones. The "
        "dotted lines show the general guarantee, gamma to the k times the initial error, which at 0.99 has barely "
        "moved after 40 sweeps.",
        s,
    )


def policy_iteration(pi):
    """Policy iteration from pi; returns the final policy and the list of value functions visited."""
    history = []
    while True:
        V = policy_value(pi, STEP, GAMMA)
        history.append(V)
        new = {s: max(ACTIONS, key=lambda a, s=s: (q_future(V, s, a), a == pi[s])) for s in NONTERM}
        if new == pi:
            return pi, history
        pi = new


# ---------------------------------------------------------------- learning a model


def learned_policy(steps, laplace, rng):
    """Run a uniformly random policy for `steps` moves, estimate P_sa, and solve by value iteration."""
    cnt = {(s, a): {} for s in NONTERM for a in ACTIONS}
    s = (1, 1)
    for _ in range(steps):
        if s in TERMINAL:
            s = rng.choice(NONTERM)
        a = rng.choice("NSEW")
        u, acc = rng.random(), 0.0
        for t, p in trans(s, a).items():
            acc += p
            if u < acc:
                break
        cnt[(s, a)][t] = cnt[(s, a)].get(t, 0) + 1
        s = t
    n = len(STATES)

    def phat(s, a):
        """Estimated P_sa: counts, Laplace-smoothed counts, or 1/|S| when the action was never tried."""
        c = cnt[(s, a)]
        tot = sum(c.values())
        if laplace:
            return {t: (c.get(t, 0) + 1) / (tot + n) for t in STATES}
        if tot == 0:
            return {t: 1 / n for t in STATES}
        return {t: v / tot for t, v in c.items()}

    P = {(s, a): phat(s, a) for s in NONTERM for a in ACTIONS}
    V = {s: 0.0 for s in STATES}
    while True:
        N = {
            s: reward(s, STEP)
            + (0 if s in TERMINAL else GAMMA * max(sum(p * V[t] for t, p in P[(s, a)].items()) for a in ACTIONS))
            for s in STATES
        }
        if max(abs(N[s] - V[s]) for s in STATES) < 1e-10:
            break
        V = N
    return {s: max(ACTIONS, key=lambda a, s=s: sum(p * V[t] for t, p in P[(s, a)].items())) for s in NONTERM}


# ---------------------------------------------------------------- exploration

CORRIDOR = 8  # cells 0..7: +10 at 0, +1 at 7, start at 5
C_START, C_TERM, C_STEP, C_GAMMA = 5, {0: 10.0, 7: 1.0}, -0.02, 0.95


def corridor_move(s, d, rng):
    """Move d with probability 0.8, the other way with 0.1, stay with 0.1."""
    u = rng.random()
    d = d if u < 0.8 else (-d if u < 0.9 else 0)
    return min(max(s + d, 0), CORRIDOR - 1)


def corridor_plan(cnt, rsum, rn, V):
    """Value iteration on the estimated model (warm-started), then the greedy policy; None where undecided."""
    n = CORRIDOR

    def q(s, a):
        """Expected next value under the estimated P_sa, uniform 1/|S| if (s, a) was never tried."""
        tot = sum(cnt[s][a])
        return sum(c * V[t] for t, c in enumerate(cnt[s][a])) / tot if tot else sum(V) / n

    for _ in range(500):
        delta = 0.0
        for s in range(n):
            r = rsum[s] / rn[s] if rn[s] else 0.0  # rewards are learned too: unseen states look like 0
            nv = r if s in C_TERM else r + C_GAMMA * max(q(s, 0), q(s, 1))
            delta = max(delta, abs(nv - V[s]))
            V[s] = nv
        if delta < 1e-7:
            break
    out = []
    for s in range(n):
        a, b = q(s, 0), q(s, 1)
        out.append(0 if a > b else 1 if b > a else None)
    return out


def corridor_run(eps, seed, episodes):
    """One learner over many episodes; returns, per episode, whether it heads for +10."""
    rng = random.Random(seed)
    n = CORRIDOR
    cnt = [[[0] * n for _ in range(2)] for _ in range(n)]
    rsum, rn, V, pi = [0.0] * n, [0] * n, [0.0] * n, [None] * n
    out = []
    for _ in range(episodes):
        s = C_START
        for _ in range(200):
            rsum[s] += C_TERM.get(s, C_STEP)
            rn[s] += 1
            if s in C_TERM:
                break
            a = rng.randrange(2) if (pi[s] is None or rng.random() < eps) else pi[s]
            t = corridor_move(s, (-1, 1)[a], rng)
            cnt[s][a][t] += 1
            s = t
        pi = corridor_plan(cnt, rsum, rn, V)
        out.append(pi[C_START] == 0)
    return out


EPISODES, SEEDS, EPSILONS = 150, 200, (0.0, 0.1, 0.2)


def fig_exploration(curves):
    """Left: the corridor. Right: fraction of learners heading for +10, by episode."""
    s = ""
    w = 40
    for i in range(CORRIDOR):
        x = 30 + i * w
        fill = GREEN if i in C_TERM else "none"
        s += rect(x, 70, w, w, fill=fill, op=0.3 if i == 0 else 0.15 if i in C_TERM else 1.0, sw=0.8)
        lab = {0: "+10", 7: "+1"}.get(i, "")
        if lab:
            s += text(x + w / 2, 95, lab, size=13, weight="600")
    s += circle(30 + C_START * w + w / 2, 90, 9, fill=INDIGO, stroke="none", op=0.8)
    s += text(30 + C_START * w + w / 2, 135, "start", size=11.5)
    s += text(30 + 4 * w, 52, "−0.02 per step, γ = 0.95, moves slip 20%", size=11.5)
    s += arrow(30 + C_START * w + 4, 165, 30 + 1 * w, 165, stroke=PURPLE, sw=1.6, op=0.85)
    s += text(30 + 3 * w, 182, "optimal: 5 steps to +10", size=11.5, fill=PURPLE)
    s += arrow(30 + C_START * w + w - 4, 200, 30 + 7 * w, 200, stroke=TEAL, sw=1.6, op=0.85)
    s += text(30 + 6.5 * w, 217, "greedy trap: 2 steps to +1", size=11.5, fill=TEAL)

    xdom, ydom = (0, EPISODES), (0.0, 1.0)
    to_px = mapper(xdom, ydom, (420, 700), (30, 220))
    s += axes(to_px, xdom, ydom, (0, 50, 100, 150), (0, 0.5, 1), "episodes", "fraction heading for +10")
    for eps, col in zip(EPSILONS, (TEAL, INDIGO, PURPLE)):
        pts = [(k + 1, v) for k, v in enumerate(curves[eps])]
        s += curve(pts, to_px, xdom, ydom, stroke=col, sw=2.0)
        x, y = to_px(EPISODES, curves[eps][-1])
        s += text(x + 6, y + 4, f"ε = {eps:g}", size=11.5, fill=col, anchor="start")
    return wrap(
        780,
        275,
        "figExplore",
        "A greedy learner can settle for the nearer, smaller reward",
        "Left: an eight-cell corridor. The robot starts in cell five; a plus-ten reward sits five cells to the left "
        "and a plus-one two cells to the right, with a small cost per step. Right: over 200 independent learners, "
        "the fraction whose current policy heads for the plus-ten, by episode. A purely greedy learner reaches about "
        "six in ten after 150 episodes, because the runs that find the plus-one first rarely leave it; with epsilon "
        "0.2 nearly eight in ten do.",
        s,
    )


FIGURES = {}

if __name__ == "__main__":
    # 1. V_pi two ways, for the poor policy.
    pi_bad = {s: "E" for s in NONTERM}
    v_lin = policy_value(pi_bad, STEP, GAMMA)
    v_it = evaluate_iteratively(pi_bad)
    assert max(abs(v_lin[s] - v_it[s]) for s in STATES) < 1e-9
    print("V_pi (East everywhere): " + ", ".join(f"{s}: {v_lin[s]:.3f}" for s in STATES))

    # 2. V* is the value of its own greedy policy, and beats random policies everywhere.
    vstar = value_iteration(STEP, GAMMA)
    pistar = greedy(vstar)
    v_of_pistar = policy_value(pistar, STEP, GAMMA)
    assert max(abs(vstar[s] - v_of_pistar[s]) for s in STATES) < 1e-9
    rng = random.Random(17)
    for _ in range(3000):
        pi = {s: rng.choice("NSEW") for s in NONTERM}
        v = policy_value(pi, STEP, GAMMA)
        assert all(vstar[s] >= v[s] - 1e-12 for s in STATES)
    print("V* = V_{pi*}, and V* >= V_pi state by state for 3000 random policies")
    w31 = {a: q_future(vstar, (3, 1), a) for a in ACTIONS}
    rounded = {(2, 1): 0.75, (3, 2): 0.69, (3, 1): 0.71, (4, 1): 0.49}
    ng_w = 0.8 * rounded[(2, 1)] + 0.1 * rounded[(3, 2)] + 0.1 * rounded[(3, 1)]
    ng_n = 0.8 * rounded[(3, 2)] + 0.1 * rounded[(2, 1)] + 0.1 * rounded[(4, 1)]
    print(
        "(3,1): exact "
        + ", ".join(f"{a} {q:.4f}" for a, q in w31.items())
        + f"; from board-rounded values W {ng_w:.3f}, N {ng_n:.3f}"
    )

    # 3. Value iteration: error per sweep, bound, synchronous vs asynchronous.
    runs = {}
    for gamma in (0.99, 0.9):
        vs = value_iteration(STEP, gamma)
        for asyn in (False, True):
            errs, first = vi_errors(gamma, vs, 1500, asyn)
            runs[(gamma, asyn)] = (errs, first)
            if not asyn:
                assert all(e <= gamma**k * errs[0] + 1e-12 for k, e in enumerate(errs))
            k6 = next(k for k, e in enumerate(errs) if e < 1e-6)
            k3 = next(k for k, e in enumerate(errs) if e < 1e-3)
            ratio = errs[10] / errs[9]
            print(
                f"gamma {gamma}, {'async' if asyn else 'sync '}: error < 1e-3 after {k3}, < 1e-6 after {k6} sweeps;"
                f" error ratio at sweep 10 {ratio:.4f}; greedy policy optimal from sweep {first}"
            )
    for gamma in (0.99, 0.9):
        print(
            f"gamma {gamma}: bound gives 1e-6 after {math.ceil(math.log(1e-6) / math.log(gamma))} sweeps per unit of initial error"
        )

    # Why the bound is loose here: under pi* the robot is absorbed within a few steps.
    dist = {(1, 1): 1.0}
    for k in range(1, 41):
        nd = {}
        for st, p in dist.items():
            if st in TERMINAL:
                continue
            for t, q in trans(st, pistar[st]).items():
                nd[t] = nd.get(t, 0.0) + p * q
        dist = nd
        if k in (5, 10, 20, 40):
            alive = sum(p for st, p in dist.items() if st not in TERMINAL)
            print(f"under pi* from (1,1): P(not yet absorbed after {k} steps) = {alive:.4f}")

    # 4. Policy iteration: monotone, finite, same answer.
    for name, start in (("all North", {s: "N" for s in NONTERM}), ("East everywhere", pi_bad)):
        final, hist = policy_iteration(start)
        assert final == pistar
        for a, b in zip(hist, hist[1:]):
            assert all(b[s] >= a[s] - 1e-12 for s in STATES)
        print(f"policy iteration from {name}: {len(hist)} evaluations, ends at pi*")
    lens = []
    for k in range(200):
        r2 = random.Random(1000 + k)
        lens.append(len(policy_iteration({s: r2.choice("NSEW") for s in NONTERM})[1]))
    print(
        f"policy iteration from 200 random policies: {min(lens)}-{max(lens)} evaluations, mean {sum(lens) / len(lens):.1f}"
    )

    # 5. Learning the model from a random policy's experience, with and without Laplace smoothing.
    for steps in (100, 300, 1000, 3000, 10000):
        res = []
        for laplace in (False, True):
            hits, losses = 0, []
            for k in range(40):
                pi = learned_policy(steps, laplace, random.Random(7000 + k))
                hits += pi == pistar
                v = policy_value(pi, STEP, GAMMA)
                losses.append(max(vstar[s] - v[s] for s in STATES))
            res.append((hits, sum(losses) / len(losses), max(losses)))
        print(
            f"model from {steps} random moves: counts {res[0][0]}/40 exactly pi*, mean worst-state loss {res[0][1]:.4f}"
            f" (max {res[0][2]:.3f}); Laplace {res[1][0]}/40, mean loss {res[1][1]:.4f} (max {res[1][2]:.3f})"
        )

    # 6. Exploration in the corridor.
    curves = {}
    for eps in EPSILONS:
        runs_e = [corridor_run(eps, k, EPISODES) for k in range(SEEDS)]
        curves[eps] = [sum(r[e] for r in runs_e) / SEEDS for e in range(EPISODES)]
        print(
            f"corridor, eps {eps}: heading for +10 after 1/10/50/150 episodes: "
            + ", ".join(f"{curves[eps][e]:.2f}" for e in (0, 9, 49, 149))
        )
    assert curves[0.2][-1] > curves[0.0][-1] + 0.1
    opt_left = sum(C_STEP * C_GAMMA**k for k in range(5)) + 10 * C_GAMMA**5
    opt_right = sum(C_STEP * C_GAMMA**k for k in range(2)) + 1 * C_GAMMA**2
    print(f"corridor, deterministic payoffs from start: left {opt_left:.2f}, right {opt_right:.2f}")

    FIGURES.update(
        {
            "diagram_values.svg": fig_values(pi_bad, v_lin, pistar, vstar),
            "diagram_vi_convergence.svg": fig_vi(runs),
            "diagram_exploration.svg": fig_exploration(curves),
        }
    )
    for name, svg in FIGURES.items():
        (OUT / name).write_text(svg)
        print("wrote", name)
