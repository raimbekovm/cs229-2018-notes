# 17. MDPs and Value/Policy Iteration

> The value function of a policy and Bellman's equations, the optimal value function and how the optimal policy is read off it, value iteration with synchronous and asynchronous updates and how fast it converges, policy iteration, estimating the transition probabilities from experience, and exploration versus exploitation with epsilon-greedy and Boltzmann exploration.

Web version: https://raimbekovm.github.io/cs229-2018-notes/lectures/17-value-policy-iteration.html

Lecture video: [Lecture 17: MDPs and Value/Policy Iteration](https://www.youtube.com/watch?v=d5gaWTo6kDM)

*Based on the lecture by Andrew Ng — [CS229](https://cs229.stanford.edu/), Stanford
University, Autumn 2018.*

Lecture 16 defined a Markov decision process and claimed an optimal policy for the 11-state
gridworld without saying where it came from. This lecture supplies the method. It defines
three objects: $V^\pi$, the value of following a fixed policy; $V^*$, the best value any
policy can achieve; and $\pi^*$, the policy that achieves it. Each satisfies a version of
Bellman's equation, and the two standard algorithms, value iteration and policy iteration,
are two ways of solving those equations. The lecture then drops the assumption that the
transition probabilities are known. It estimates them from the robot's own experience, and
this raises the question of how much the robot should explore.

### The value of a policy

[6:46](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=406s) For a policy $\pi$, the *value function* $V^\pi : S \to \mathbb{R}$ gives the
expected total payoff of starting in $s$ and following $\pi$ for ever:

$$
V^\pi(s) = \mathbb{E}\left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \;\middle|\; s_0 = s,\ \pi \right].
$$

[8:40](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=520s) Ng's example is deliberately poor. From some states it drives straight for the
$-1$. The left half of Figure 17.1 uses a concrete policy of that kind, East from every
state. On the top row it reaches the $+1$ and is worth about 0.6 to 0.76. Along the middle
and bottom rows it runs into the $-1$, or stalls against the east wall paying $-0.02$ a
step, and is worth about $-0.3$ to $-1.1$. The $\pm 1$ cells are *absorbing*: once the robot
reaches one, it collects that reward and the world ends.

[10:57](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=657s) The equation that governs $V^\pi$ comes from splitting the payoff after its first
term. Factor one $\gamma$ out of everything after $R(s_0)$:

$$
R(s_0) + \gamma \bigl( R(s_1) + \gamma R(s_2) + \cdots \bigr).
$$

The bracket is the payoff of a robot that starts in $s_1$ and follows $\pi$. Its expectation
is $V^\pi(s_1)$, and $s_1$ is drawn from $P_{s\pi(s)}$, because in state $s$ the policy takes
action $\pi(s)$. Writing $s'$ for the next state gives *Bellman's equation* for $V^\pi$:

$$
V^\pi(s) = R(s) + \gamma \sum_{s' \in S} P_{s\pi(s)}(s')\, V^\pi(s').
$$ (17.1)

The first term is the *immediate reward*, which the robot gets just for waking up in $s$.
The second is the discounted expected future reward.

[19:42](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=1182s) For a fixed $\pi$, Equation 17.1 is *linear* in the unknowns $V^\pi(s)$. For
$s = (3, 1)$, with $\pi$ going North, it reads

$$
V^\pi(3,1) = R(3,1) + \gamma\bigl[0.8\,V^\pi(3,2) + 0.1\,V^\pi(2,1) + 0.1\,V^\pi(4,1)\bigr],
$$

and one such equation for each of the 11 states gives 11 linear equations in 11 unknowns.
A linear solver handles that instantly. The generator solves the system for the East policy
and checks the result against 5,000 sweeps of repeated substitution into Equation 17.1. The
two agree to $10^{-9}$.

![Figure 17.1. Left: the policy "always go East" and its value $V^\pi$, solved from the 11 Bellman equations. Right: the optimal policy and the optimal value function $V^*$ for $R = -0.02$, $\gamma = 0.99$. Cell $(3, 1)$, whose action the lecture works out by hand, is highlighted.](https://raimbekovm.github.io/cs229-2018-notes/figures/17-value-policy-iteration/diagram_values.svg)

### The optimal value function and the optimal policy

[24:47](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=1487s) The *optimal value function* is the best value any policy achieves from each
state:

$$
V^*(s) = \max_\pi V^\pi(s).
$$

Ng flags a terminology trap. Both $V^\pi$ and $V^*$ are called "value functions", but one
belongs to a particular policy, good or bad, and the other to the best of them. The maximum
is over a combinatorially large set, $4^9$ policies even for this small grid, so the
definition does not suggest how to compute $V^*$. Its Bellman equation does.

[27:12](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=1632s) Run the same argument as for Equation 17.1, but let the robot choose its first
action well. It collects $R(s)$ for waking up in $s$. Whatever action $a$ it takes, it then
lands in $s' \sim P_{sa}$ and can collect at most $V^*(s')$ from there. The best first action
is the one with the largest expected continuation, so

$$
V^*(s) = R(s) + \max_{a \in A}\; \gamma \sum_{s' \in S} P_{sa}(s')\, V^*(s').
$$ (17.2)

[29:44](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=1784s) If $V^*$ were known, the optimal action would be the one that attains that
maximum. Ng asks the room what to do in state $s$, and the answer is already in
Equation 17.2:

$$
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s')\, V^*(s').
$$ (17.3)

$\gamma$ is dropped because a positive constant does not move an argmax. [32:50](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=1970s) Ng
states, without proof, that this policy is optimal from every state at once:
$V^*(s) = V^{\pi^*}(s) \ge V^\pi(s)$ for every $\pi$ and every $s$. One policy is best
everywhere, so no state has to trade off against another. The generator checks both halves.
The value of the greedy policy, solved from its own linear system, equals $V^*$ to
$10^{-9}$, and $V^*$ is at least $V^\pi$, state by state, for 3,000 random policies. The plan
is therefore to find $V^*$ and then read off $\pi^*$ with Equation 17.3.

### Value iteration

[35:22](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=2122s) Value iteration treats Equation 17.2 as an update rule:

1. Initialise $V(s) := 0$ for every $s$.
2. Repeat until convergence: for every $s$,
   $V(s) := R(s) + \max_a \gamma \sum_{s'} P_{sa}(s')\, V(s')$.

[38:07](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=2287s) There are two ways to carry out step 2, and Ng compares them to gradient descent.
A *synchronous* update computes the right-hand side for all 11 states from the old values,
then overwrites all 11 at once. In vector form this is $V := B(V)$, where $B$ is called the
*Bellman backup operator*. An *asynchronous* update sweeps through the states one at a time
and overwrites each value as soon as it is computed, so later states in the sweep already use
the new values. Both converge to $V^*$.

Ng's default is the synchronous form, because it is a matrix operation and vectorises. He
says "asynchronous" at this point in the recording, but the reason he gives, vectorisation,
and the procedure he describes, compute all eleven and then update, are the synchronous one.
The asynchronous form has the opposite advantage: it needs fewer sweeps. On the gridworld
with $\gamma = 0.99$, the max-norm error falls below $10^{-6}$ after 21 asynchronous sweeps
against 29 synchronous ones (Figure 17.2).

[40:32](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=2432s) A student asks how to code the absorbing states. One way is to give them
transition probabilities of zero to every state, so that their value is just $R$. Another is
to add a twelfth, zero-reward "sink" state that they lead to and that loops on itself. Ng
calls the second cleaner and the first more common. Both give the same values.

[43:07](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=2587s) Why does it converge? The problem set shows that $B$ is a *contraction* in the
max norm: $\lVert B(V) - B(V')\rVert_\infty \le \gamma \lVert V - V'\rVert_\infty$. Since
$V^* = B(V^*)$, each synchronous sweep shrinks the error by at least a factor $\gamma$, so
$\lVert V_k - V^*\rVert_\infty \le \gamma^k \lVert V_0 - V^*\rVert_\infty$. Ng turns this
into a rule of thumb: at $\gamma = 0.99$, value iteration needs a few hundred iterations, and
at $\gamma = 0.9$ a few dozen.

The bound is right but, for this MDP, very loose. Taken literally, it needs 1,375 sweeps per
unit of initial error to reach $10^{-6}$ at $\gamma = 0.99$, and 132 at $\gamma = 0.9$. The
gridworld gets there in 29 sweeps at $\gamma = 0.99$ and 21 at $\gamma = 0.9$. The generator
asserts that every synchronous sweep is within the bound, so nothing here contradicts the
theorem. The reason is the absorbing states. After $k$ sweeps the remaining error is roughly
the discounted value of what happens beyond step $k$. Under $\pi^*$, starting from $(1, 1)$, the
robot is still wandering after 10 steps with probability 0.04, and after 20 steps with probability
essentially zero, so there is little left to discount. The factor $\gamma$ only governs the
rate when the process can run on for a long time without ending. The more useful fact for
practice is that the *policy* converges before the values. With $\gamma = 0.99$, the greedy
policy Equation 17.3 of the synchronous estimate is already exactly $\pi^*$ after 11 sweeps,
while the values are still off by as much as 0.08.

![Figure 17.2. Value iteration on the gridworld: base-10 logarithm of the max-norm error against the number of sweeps, synchronous and asynchronous, at $\gamma = 0.99$ and $\gamma = 0.9$. Dotted lines are the guarantee $\gamma^k \lVert V_0 - V^*\rVert_\infty$, which every synchronous curve stays below, by a wide margin here because the robot is absorbed within a few steps.](https://raimbekovm.github.io/cs229-2018-notes/figures/17-value-policy-iteration/diagram_vi_convergence.svg)

### Reading the policy off $V^*$

[45:15](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=2715s) Ng writes the converged $V^*$ on the board and uses it to justify the West arrow at
$(3, 1)$ that lecture 16 asserted. Going West reaches $(2, 1)$ with probability 0.8, where
$V^* \approx 0.75$. It slips North to $(3, 2)$, worth 0.69, with probability 0.1, and South
into the wall, staying at $(3, 1)$, worth 0.71, with probability 0.1. The expected value is
0.740. Going North gives $0.8 \times 0.69 + 0.1 \times 0.75 + 0.1 \times 0.49 = 0.676$, so
West wins.

The generator's $V^*$ (Figure 17.1, right) matches the board values to two decimals: 0.746,
0.687, 0.709 and 0.491. Computed from the unrounded values, the two expectations are 0.736 and
0.674. Ng adds that a full check compares all four actions, and the full check is
instructive. South, at 0.691, beats North: bumping into the wall and staying put is the
second-best action, because North risks the slip into $-1$. East is last at 0.532.

### Policy iteration

[50:18](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3018s) Policy iteration keeps a policy, not a value function, at the centre:

1. Initialise $\pi$ randomly.
2. Repeat until $\pi$ stops changing:
   a. $V := V^\pi$, by solving the linear system Equation 17.1;
   b. for every $s$, $\pi(s) := \arg\max_a \sum_{s'} P_{sa}(s')\, V(s')$.

Step (b) pretends that the current $V$ is optimal and acts greedily on it. Value iteration
waits until $V$ has converged to extract a policy. Policy iteration extracts a new one on
every round. [54:44](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3284s) The trade-off, as Ng puts it, is the linear solve in step (a). For
11 states, or a few hundred, it is trivial, and policy iteration is fast. For 10,000 states
or a million, solving the system on every round is expensive, and he would use value
iteration. He also notes a more academic difference. Value iteration approaches $V^*$ and
never exactly reaches it, as gradient descent approaches the minimum of a least-squares cost.
Policy iteration stops after a finite number of rounds, at exactly $\pi^*$ and exactly
$V^*$, the way the normal equations jump straight to the answer. It must stop, because each
round improves the policy, by the policy improvement theorem, and there are only finitely
many policies to improve through.

The generator checks this on the gridworld. Started from 200 random policies, policy
iteration needs between 2 and 8 evaluations, 5 on average, and always ends at the $\pi^*$ of
Figure 17.1. From "always East" it needs 3. The values never decrease from one round to the
next, in any state. In practice, Ng says, he sees value iteration used more, because real
problems are large. For small ones policy iteration may converge a little faster.

**In the official notes.** Both algorithms, the synchronous and asynchronous variants, and the statement that policy
iteration terminates are in cs229-notes12, section 2. The contraction argument for value
iteration is a homework problem in the course.
### Learning the model

[58:41](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3521s) So far $P_{sa}$ has been given. For a real robot, a helicopter or a chess
opponent, it usually is not, and has to be estimated from experience. Run some policy for a
while, count what happens, and use the maximum-likelihood estimate

$$
\hat P_{sa}(s') = \frac{\#\{\text{times action } a \text{ was taken in } s \text{ and led to } s'\}}{\#\{\text{times action } a \text{ was taken in } s\}}.
$$ (17.4)

When the denominator is zero, because $a$ has never been tried in $s$, the common heuristic
is a uniform guess, $\hat P_{sa}(s') = 1/\lvert S\rvert$.

[1:01:41](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3701s) Laplace smoothing would also avoid the $0/0$: add 1 to every numerator and
$\lvert S\rvert$ to every denominator. Ng says it is allowed but uncommon, because, unlike
naive Bayes, MDP solvers are not badly hurt by zero estimates. The generator suggests a
stronger statement: on the gridworld, smoothing makes things worse. It runs a uniformly
random policy for a given number of moves, builds $\hat P$ with and without smoothing, solves
for the policy, and scores it by its true loss, the largest $V^*(s) - V^{\hat\pi}(s)$ over
states, averaged over 40 runs. The plain counts lose 0.078 after 1,000 moves and 0.010 after
10,000. With smoothing the losses are 0.263 and 0.023. The reason is visible in the formula.
Smoothing gives every action a small probability of moving the robot to *every* state,
including straight to the $+1$ or $-1$ from across the grid. With 11 states and only a few
visits per state-action pair, that invented mass is large, and it is exactly the mass the
optimal policy is most sensitive to. Exact recovery of $\pi^*$ is rare either way, 14 runs in
40 even after 10,000 moves without smoothing, because of the 0.0005 margin at $(3, 2)$ noted
in lecture 16. Measuring the loss, not whether the policy is exactly $\pi^*$, is the fair
test.

[1:03:01](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3781s) Put together, the practical loop is:

1. Take actions according to some policy $\pi$ to gain experience in the MDP.
2. Update the estimates of $P_{sa}$ from the counts, and of $R$ if it is unknown.
3. Solve Bellman's equations for the estimated MDP by value iteration, warm-started from the
   previous $V$ if convenient, to get a new $V$.
4. Update $\pi$ to be greedy with respect to $V$, and repeat.

[1:06:03](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=3963s) The reward is usually part of the problem specification, but not always. For a
trading application the reward is a day's return, and for a car it might be a penalty for
every pothole. Both are random functions of the state, so the same loop estimates the
expected reward of each state by averaging what it observes.

### Exploration versus exploitation

[1:07:33](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=4053s) The loop above is greedy: it always acts on its current best estimate. Ng
shows the trouble with a different MDP. A robot starts with a $+1$ nearby and a $+10$ farther
away. If its first runs happen to reach the $+1$, the estimated model says that going there
works, and the greedy policy keeps going there. Every trip ends at the $+1$, so the robot's
experience stays in that half of the state space, and the $+10$ is never found. Acting on
what is known is *exploitation*. Trying actions that look worse to learn something new is
*exploration*. The tension between them is not academic. Ng's example is online advertising.
Showing only the ads a user is predicted to click maximises short-term revenue. Showing
occasional others is how the platform discovers interests nobody knew the user had.

[1:13:28](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=4408s) The standard fix is *ε-greedy* exploration. On each step, with probability
$1-\varepsilon$ take the action $\pi(s)$, and with probability $\varepsilon$, for instance
0.1, take a random one. Ng points out that the name is backwards, since $\varepsilon$ is the
probability of acting *randomly*, not greedily, but it is the name everyone uses. With
$\varepsilon > 0$ every action keeps being tried in every state the robot can reach, so the
estimated model converges to the true one and the planned policy to $\pi^*$, though finding a
remote reward by chance can take a long time.

Figure 17.3 makes Ng's example concrete in one dimension, in a toy of our own: an
eight-cell corridor with the $+10$ five steps to the left of the start and the $+1$ two steps
to the right. Moves slip 20% of the time, each step costs 0.02, $\gamma = 0.95$, and the robot
learns both $P_{sa}$ and $R$ as above. Going left is worth about 7.65 and going right 0.86, so
the right answer is not close. Over 200 independent learners, the fraction whose policy heads
for the $+10$ after 150 episodes is 0.62 for the greedy learner, 0.68 with $\varepsilon = 0.1$
and 0.78 with $\varepsilon = 0.2$. Two details are worth noticing. The greedy learner is not
completely stuck: its curve keeps creeping up, presumably because slipping moves sometimes
carry it left without being asked. And none of the curves has reached 1, which is Ng's caveat about how long a remote reward can take to find.

![Figure 17.3. A toy version of the lecture's exploration example, not from the lecture. Left: the corridor, with the start between a distant $+10$ and a nearby $+1$. Right: over 200 learners that estimate the model from experience and re-plan after each episode, the fraction whose current policy heads for the $+10$, for ε = 0 (greedy), 0.1 and 0.2.](https://raimbekovm.github.io/cs229-2018-notes/figures/17-value-policy-iteration/diagram_exploration.svg)

[1:16:17](https://www.youtube.com/watch?v=d5gaWTo6kDM&t=4577s) Asked whether $\varepsilon$ should stay fixed, Ng says there are many
heuristics, and a reasonable one is to start large and shrink it. Another is *Boltzmann
exploration*. Instead of a uniformly random action with probability $\varepsilon$, choose
action $a$ with probability proportional to $\exp\bigl(Q(s, a)/\tau\bigr)$, where
$Q(s, a) = \sum_{s'} P_{sa}(s') V(s')$ is the estimated value of taking $a$ and $\tau$ is a
temperature. Actions that look nearly as good as the best are tried often, and actions that
look terrible almost never. A student asks about rewarding the robot for visiting states it
has never seen. That is *intrinsic motivation*, a research line Ng suggests looking up. And
asked how often to re-plan, he says as often as the computation allows. It does no harm.
With a physical robot or helicopter, where each flight is a session, re-planning between
flights is the natural rhythm.

**In the official notes.** Estimating $P_{sa}$ and $R$ from experience, with the $1/\lvert S\rvert$ rule for untried
actions and the warm-started value iteration loop, is in cs229-notes12, section 3. The notes
do not treat exploration. Lecture 18 extends the framework to continuous states, covered in
section 4 of the same notes.
