"""The lecture's 11-state gridworld, its solvers, and a drawing of it.

Lectures 16 and 17 both work with the same MDP, taken by the lecture from
Russell and Norvig: a 4-by-3 grid with a wall at (2, 2), an absorbing +1 at
(4, 3) and an absorbing -1 at (4, 2), a step reward on every other cell, and
noisy moves that go where commanded with probability 0.8 and veer to either
side with probability 0.1 each. A move into the wall or off the grid leaves the
robot where it is.

States are (column, row) pairs counted from the bottom left, as on the board.
Rewards are received in a state, so the value of an absorbing state is its
reward: the world ends there and nothing follows.

Chapter scripts reach this module the same way they reach `_svg`.
"""

from _svg import GREEN, INDIGO, PURPLE, arrow, rect, text

STATES = [(x, y) for y in (1, 2, 3) for x in (1, 2, 3, 4) if (x, y) != (2, 2)]
TERMINAL = {(4, 3): 1.0, (4, 2): -1.0}
ACTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
SIDEWAYS = {"N": "EW", "S": "EW", "E": "NS", "W": "NS"}


def move(s, a):
    """Where a deterministic move in direction a would take the robot."""
    t = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])
    return t if t in STATES else s


def trans(s, a):
    """The distribution P_sa as a dict: 0.8 ahead, 0.1 to each side."""
    out = {}
    for b, p in ((a, 0.8), (SIDEWAYS[a][0], 0.1), (SIDEWAYS[a][1], 0.1)):
        t = move(s, b)
        out[t] = out.get(t, 0.0) + p
    return out


def reward(s, step):
    """R(s): the terminal payoff, or the step reward everywhere else."""
    return TERMINAL.get(s, step)


def q_future(V, s, a):
    """The expected next-state value sum_s' P_sa(s') V(s')."""
    return sum(p * V[t] for t, p in trans(s, a).items())


def backup(V, step, gamma):
    """One synchronous Bellman backup B(V)."""
    return {
        s: reward(s, step) if s in TERMINAL else reward(s, step) + gamma * max(q_future(V, s, a) for a in ACTIONS)
        for s in STATES
    }


def greedy(V):
    """The policy that is greedy with respect to V (ties broken in N, S, E, W order)."""
    return {s: max(ACTIONS, key=lambda a: q_future(V, s, a)) for s in STATES if s not in TERMINAL}


def value_iteration(step=-0.02, gamma=0.99, tol=1e-13):
    """V* by synchronous value iteration from V = 0."""
    V = {s: 0.0 for s in STATES}
    while True:
        N = backup(V, step, gamma)
        if max(abs(N[s] - V[s]) for s in STATES) < tol:
            return N
        V = N


def solve_linear(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [[*row, bi] for row, bi in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def policy_value(pi, step=-0.02, gamma=0.99):
    """V_pi by solving the 11 linear Bellman equations V = R + gamma P_pi V."""
    idx = {s: i for i, s in enumerate(STATES)}
    n = len(STATES)
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for s, i in idx.items():
        A[i][i] = 1.0
        b[i] = reward(s, step)
        if s in TERMINAL:
            continue
        for t, p in trans(s, pi[s]).items():
            A[i][idx[t]] -= gamma * p
    v = solve_linear(A, b)
    return {s: v[idx[s]] for s in STATES}


ARROW = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def draw_grid(x0, y0, cell, values=None, policy=None, mark=None, fmt="{:.3f}"):
    """The grid at (x0, y0), top left, with optional values, policy arrows and a marked cell."""
    s = ""
    for gx in (1, 2, 3, 4):
        for gy in (1, 2, 3):
            px, py = x0 + (gx - 1) * cell, y0 + (3 - gy) * cell
            if (gx, gy) == (2, 2):
                s += rect(px, py, cell, cell, fill="currentColor", op=0.28, sw=0.8)
                continue
            if (gx, gy) in TERMINAL:
                pos = TERMINAL[(gx, gy)] > 0
                s += rect(px, py, cell, cell, fill=GREEN if pos else PURPLE, op=0.28 if pos else 0.16, sw=0.8)
                s += text(px + cell / 2, py + cell / 2 + 6, "+1" if pos else "−1", size=17, weight="600")
                continue
            hl = mark == (gx, gy)
            s += rect(px, py, cell, cell, fill=INDIGO if hl else "none", op=0.12 if hl else 1.0, sw=0.8)
            cx, cy = px + cell / 2, py + cell / 2
            if policy:
                dx, dy = ARROW[policy[(gx, gy)]]
                ay = cy - (9 if values else 0)
                s += arrow(cx - dx * 13, ay - dy * 13, cx + dx * 13, ay + dy * 13, sw=1.8, op=0.85, head=7)
            if values:
                s += text(cx, cy + (17 if policy else 5), fmt.format(values[(gx, gy)]), size=12)
    return s
