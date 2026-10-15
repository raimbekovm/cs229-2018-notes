# 14. EM Algorithm and Factor Analysis

> How the general EM algorithm turns into concrete updates for the mixture of Gaussians, EM as coordinate ascent on the lower bound, why a Gaussian cannot be fitted when there are fewer examples than dimensions, the factor analysis model for high-dimensional data from few examples, marginals and conditionals of Gaussians, and EM for factor analysis.

Web version: https://raimbekovm.github.io/cs229-2018-notes/lectures/14-factor-analysis.html

Lecture video: [Lecture 14: EM Algorithm and Factor Analysis](https://www.youtube.com/watch?v=tw6cmL5STuY)

*Based on the lecture by Andrew Ng — [CS229](https://cs229.stanford.edu/), Stanford
University, Autumn 2018.*

Factor analysis is a density model for data with many more dimensions than examples. For
example, a hundred temperature sensors give a hundred-dimensional reading, but you may have
only thirty days of data. An ordinary Gaussian cannot be fitted to such data, because its
maximum-likelihood covariance is singular. The model assumes that each observation is a
low-dimensional hidden cause $z$, mapped linearly into the high-dimensional space, plus
independent noise in every coordinate. Because $z$ is hidden, the model is fitted with EM,
and doing so means working out the conditional distributions of a joint Gaussian.

The lecture first closes off the previous one. It shows how the abstract E- and M-steps of
Section 13.7 become code for the mixture of Gaussians, and it recasts EM as
coordinate ascent. It then spends the rest of the hour on factor analysis.

### From the general algorithm to the mixture updates

[3:00](https://www.youtube.com/watch?v=tw6cmL5STuY&t=180s) The general M-step says to maximise the lower bound over $\theta$ with the
$Q_i$ held fixed. For the mixture of Gaussians, $Q_i(z^{(i)} = j)$ is the responsibility
$w^{(i)}_j = p(z^{(i)} = j \mid x^{(i)}; \phi, \mu, \Sigma)$ that the E-step computes and
stores. The joint density factorises as
$p(x^{(i)} \mid z^{(i)} = j)\,p(z^{(i)} = j) = \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)\,\phi_j$.
So the quantity the M-step maximises is a double sum that can be written out in full:

$$
J(\phi, \mu, \Sigma) = \sum_{i=1}^m \sum_{j=1}^k w^{(i)}_j
\log \frac{\dfrac{1}{(2\pi)^{n/2} |\Sigma_j|^{1/2}}
\exp\!\Bigl(-\tfrac12 (x^{(i)} - \mu_j)^\top \Sigma_j^{-1} (x^{(i)} - \mu_j)\Bigr)\,\phi_j}{w^{(i)}_j} .
$$ (14.1)

Ng's point is that this objective is what gives you the updates. The rule "move each mean
to the weighted average of the points" was an intuition in the last lecture, and here it
is derived. Differentiating Equation 14.1 with respect to $\mu_j$ and setting the
result to zero gives

$$
\mu_j = \frac{\sum_i w^{(i)}_j x^{(i)}}{\sum_i w^{(i)}_j},
$$

and the same procedure gives $\Sigma_j$.

For $\phi$, "differentiate and set to zero" does not quite work, and the board skips over
why. The only terms in $\phi$ are $\sum_i \sum_j w^{(i)}_j \log \phi_j$, whose derivative
with respect to $\phi_j$ is $\sum_i w^{(i)}_j / \phi_j$. That derivative is positive for
every value of $\phi_j$, so it never vanishes. Taken alone, the objective wants every
$\phi_j$ to be infinite. What stops it is the constraint $\sum_j \phi_j = 1$, so the step
needs a Lagrange multiplier. With the multiplier, the solution is
$\phi_j = \frac1m \sum_i w^{(i)}_j$. The generator confirms this on a three-component
one-dimensional mixture. At the closed-form M-step, the unconstrained derivative equals
exactly $m = 200$ for all three components (the multiplier), and none of 300 random
perturbations of $(\phi, \mu, \sigma^2)$ that stay on the simplex raises $J$.

[14:11](https://www.youtube.com/watch?v=tw6cmL5STuY&t=851s) Ng then gives a second view of EM that the factor-analysis derivation leans on.
Treat the lower bound as a function of *both* arguments,

$$
J(\theta, Q) = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
\;\le\; \ell(\theta) \quad \text{for every } \theta \text{ and } Q .
$$

The E-step maximises $J$ over $Q$ with $\theta$ fixed, and the maximum is the choice that
makes $J = \ell$. The M-step maximises $J$ over $\theta$ with $Q$ fixed. EM is therefore
*coordinate ascent* on $J$: it maximises over one block of variables, then the other, and
alternates. This view also tells you how to check that an implementation is working. $J$
cannot decrease in either step, and since it equals $\ell(\theta)$ after every E-step, the
log-likelihood you print at each iteration must not decrease either. If it ever goes down,
there is a bug. On the mixture above, the generator asserts that $J = \ell$ after each
E-step to $10^{-8}$, and that $\ell$ never falls over 40 iterations: from $-544.2$ it
reaches $-388.9$ after three iterations and $-358.1$ at the end.

When $z$ is continuous, as it will be in factor analysis, every step of the derivation in
Section 13.7 goes through with the sum over $z^{(i)}$ replaced by an integral.
Jensen's inequality does not care which kind of expectation it is applied to.

### Fewer examples than dimensions

[17:46](https://www.youtube.com/watch?v=tw6cmL5STuY&t=1066s) A mixture of Gaussians suits data like Ng's two-dimensional picture with a
hundred points: $m \gg n$, and each cluster has plenty of points to estimate its
covariance. Factor analysis is for the opposite regime, where $m \approx n$ or
$m \ll n$. Ng's running example is a Stanford PhD student's project with temperature
sensors placed around campus buildings: a hundred sensors and thirty days of readings, so
$n = 100$ and $m = 30$. The goal was anomaly detection by density estimation, as in
Section 13.3. A window left open on a winter day produces a reading with low $p(x)$,
and in practice the point was saving energy rather than detecting fires. To do this you
need a joint density over all hundred sensors, because neighbouring sensors are strongly
correlated, and you have thirty examples to fit it with.

[22:01](https://www.youtube.com/watch?v=tw6cmL5STuY&t=1321s) The obvious model is a single Gaussian with its maximum-likelihood estimates
$\mu = \frac1m \sum_i x^{(i)}$ and $\Sigma = \frac1m \sum_i (x^{(i)} - \mu)(x^{(i)} - \mu)^\top$.
If $m \le n$, this $\Sigma$ is singular. The reason is that it is a sum of $m$ rank-one
matrices whose vectors $x^{(i)} - \mu$ sum to zero, so its rank is at most $m - 1$. The
generator checks the exact statement on random data for $n = 2, 5, 10$ and every $m$ from
2 to $n + 2$: the rank is $\min(m - 1, n)$, so the estimate is invertible exactly when
$m \ge n + 1$. A singular $\Sigma$ breaks the density in two ways. The normaliser contains
$|\Sigma|^{1/2} = 0$ and the exponent contains $\Sigma^{-1}$, which does not exist, so
evaluating $p(x)$ gives $0/0$ at best. Ng's picture with $m = n = 2$ shows the second
problem (Figure 14.1, first panel). The fitted Gaussian has its contours squashed to
zero width along the segment joining the two points. All of its probability mass lies on
that line, so any test point slightly off the line gets density zero. That is a numerical
failure, and it is also a bad model of the data.

![Figure 14.1. Maximum-likelihood Gaussians on two-dimensional data, with the one- and two-standard-deviation contours. With two points (left) the full covariance is singular and the Gaussian collapses onto the line through them. With twelve strongly correlated points (sample correlation 0.93) the full covariance follows the tilt of the data, the diagonal covariance gives axis-aligned ellipses that miss it, and the spherical covariance gives circles.](https://raimbekovm.github.io/cs229-2018-notes/figures/14-factor-analysis/diagram_covariance.svg)

[26:53](https://www.youtube.com/watch?v=tw6cmL5STuY&t=1613s) Ng adds that one of the earliest applications of factor analysis had the same
shape: psychological testing, with a hundred questionnaire items and thirty subjects, where
the question is how attributes like verbal and mathematical ability are correlated.

### Restricting the covariance

[28:15](https://www.youtube.com/watch?v=tw6cmL5STuY&t=1695s) One way out is to fit fewer parameters. If $\Sigma$ is constrained to be
diagonal, the maximum-likelihood estimate is what you would guess: each feature gets its own
sample variance, $\Sigma_{jj} = \frac1m \sum_i (x^{(i)}_j - \mu_j)^2$. That makes $n$
parameters in place of about $n^2/2$, and they can be fitted from any $m \ge 2$. The
contours are axis-aligned ellipses. The price is that the model assumes the features are
uncorrelated, and for a hundred sensors in one room that is plainly false
(Figure 14.1, third panel).

[31:36](https://www.youtube.com/watch?v=tw6cmL5STuY&t=1896s) A stronger restriction is $\Sigma = \sigma^2 I$, with one parameter and
circular contours. Its estimate is the average squared deviation over every coordinate of
every example, which is the mean of the diagonal model's $n$ variances. A student asks why
anyone would bother with a model that is worse still. Ng's answer is that it is a building
block: factor analysis uses a diagonal covariance for its noise, not for the data.

[34:38](https://www.youtube.com/watch?v=tw6cmL5STuY&t=2078s) Another student asks about the standard fix, a Wishart prior on $\Sigma$. In
practice this means adding a constant $\lambda$ to the diagonal of the maximum-likelihood
estimate. Ng's verdict is that it removes the singularity "in a technical way" but is not
the best model for many data sets. The generator puts that verdict to a test at the end of
the chapter (Section 14.6). What Ng wants instead is a model that captures the main
correlations, such as the room warming and cooling as a whole or one side heating up under
the lamps, without paying for a full covariance matrix.

He follows this with an aside about small data that is worth keeping. The large technology
companies, he says, all work on the same few big-data problems (translation, speech, face
recognition), while many real applications in healthcare and manufacturing have a hundred
or a thousand examples. The best algorithm for a thousand examples is often not the best
for a million, and the field does not yet know how to make one algorithm work along the
whole range. He mentions using a modified factor analysis on a manufacturing problem that
same week.

### The factor analysis model

[40:32](https://www.youtube.com/watch?v=tw6cmL5STuY&t=2432s) Like the mixture of Gaussians, the model is a joint density
$p(x, z) = p(x \mid z)\,p(z)$ with $z$ hidden. This time $z$ is continuous and of lower
dimension than $x$:

$$
\begin{aligned}
z &\sim \mathcal{N}(0, I), \qquad z \in \mathbb{R}^d,\ d < n, \\
x &= \mu + \Lambda z + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Psi),
\end{aligned}
$$ (14.2)

with $z$ and $\varepsilon$ independent. The parameters are the mean $\mu \in \mathbb{R}^n$,
the *factor loading* matrix $\Lambda \in \mathbb{R}^{n \times d}$, and a **diagonal**
$\Psi \in \mathbb{R}^{n \times n}$. Equivalently, $x \mid z \sim \mathcal{N}(\mu + \Lambda z, \Psi)$.

[43:30](https://www.youtube.com/watch?v=tw6cmL5STuY&t=2610s) Ng's reading of the model uses the room. Suppose three strong forces drive the
temperatures in the room: the temperature outside, the lights on the left side, and the
lights on the right. Then $d = 3$, and $z$ holds those three numbers. Each sensor reads a
linear combination of the three, weighted by where it sits, so a sensor near the door
leans on the outdoor term. On top of that each sensor adds noise of its own, such as a gust
of wind or a bad thermistor. The assumption that $\Psi$ is diagonal says that this noise is
independent across sensors once the shared factors are accounted for. All correlation
between sensors has to come through $\Lambda z$.

[46:21](https://www.youtube.com/watch?v=tw6cmL5STuY&t=2781s) The lecture's example makes this concrete with $d = 1$, $n = 2$, $m = 7$,
$\Lambda = (2, 1)^\top$, $\mu = 0$ and $\Psi = \operatorname{diag}(1, 2)$
(Figure 14.2). Seven draws of $z$ fall on the real line. Multiplying by $\Lambda$
places them on the line $x_2 = x_1 / 2$ in the plane. Around each of these points, the noise
$\varepsilon$ has axis-aligned elliptical contours, taller than wide in the ratio
$\sqrt2 : 1$. Drawing one sample from each gives the observed points. Only those crosses
are data. The $z$ values, and the points on the line, are latent.

![Figure 14.2. The factor-analysis model as a generative process, with the lecture's parameters. Seven draws of $z$ (left) map through $\Lambda z + \mu$ onto a line in the plane (centre). Independent axis-aligned noise with covariance $\Psi$ moves each one to the observed $x$, the crosses (right). The thin lines join each latent point to its observation.](https://raimbekovm.github.io/cs229-2018-notes/figures/14-factor-analysis/diagram_fa_generative.svg)

[51:05](https://www.youtube.com/watch?v=tw6cmL5STuY&t=3065s) Ng repeats the construction with $d = 2$ and $n = 3$, carrying the five
sampled $z$ points on a sheet of cardboard and holding it up into three-dimensional space.
The data then lie close to a two-dimensional "pancake" in $\mathbb{R}^3$, with some fuzz off
it. Factor analysis takes data in a hundred dimensions and models it as lying near a
three-dimensional subspace, plus independent noise in each coordinate. [54:57](https://www.youtube.com/watch?v=tw6cmL5STuY&t=3297s) Asked
what happens when the data do not lie near a subspace, he concedes that the model is then
not the best one. He adds that in a hundred dimensions you rarely know, because you cannot
look.

### Marginals and conditionals of Gaussians

[56:43](https://www.youtube.com/watch?v=tw6cmL5STuY&t=3403s) Fitting the model needs two facts about multivariate Gaussians. Partition a
Gaussian vector into blocks $x_1 \in \mathbb{R}^r$ and $x_2 \in \mathbb{R}^s$:

$$
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \sim
\mathcal{N}\!\left( \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix},
\begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix} \right).
$$

The first fact is that the marginal of $x_1$ is $\mathcal{N}(\mu_1, \Sigma_{11})$. You can
prove it by integrating $x_2$ out of the joint density, but the result is the obvious one:
the mean and covariance of $x_1$ are the corresponding blocks. The second fact is that the
conditional is also Gaussian,

$$
x_1 \mid x_2 \sim \mathcal{N}\bigl(\mu_{1|2}, \Sigma_{1|2}\bigr), \qquad
\mu_{1|2} = \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2), \qquad
\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} .
$$ (14.3)

This one is less obvious, and the lecture states it without proof. The conditional mean is
a linear regression of $x_1$ on $x_2$. The conditional covariance does not depend on the
value of $x_2$ that was observed.

[1:02:32](https://www.youtube.com/watch?v=tw6cmL5STuY&t=3752s) Applying this to factor analysis starts with the observation that $(z, x)$ is
jointly Gaussian, since $x$ is a linear function of the Gaussians $z$ and $\varepsilon$. Its
mean is $(0, \mu)$. Ng derives the lower-right block of the covariance and leaves the others
to the notes. Expanding

$$
\mathbb{E}\bigl[(x - \mu)(x - \mu)^\top\bigr]
= \mathbb{E}\bigl[(\Lambda z + \varepsilon)(\Lambda z + \varepsilon)^\top\bigr]
= \Lambda\,\mathbb{E}[z z^\top]\,\Lambda^\top + \mathbb{E}[\varepsilon \varepsilon^\top]
= \Lambda \Lambda^\top + \Psi .
$$

Here the cross terms vanish because $z$ and $\varepsilon$ are independent with mean zero.
The same expansion gives $\mathbb{E}[z (x - \mu)^\top] = \Lambda^\top$, so

$$
\begin{bmatrix} z \\ x \end{bmatrix} \sim
\mathcal{N}\!\left( \begin{bmatrix} 0 \\ \mu \end{bmatrix},
\begin{bmatrix} I & \Lambda^\top \\ \Lambda & \Lambda\Lambda^\top + \Psi \end{bmatrix} \right).
$$ (14.4)

For the lecture's example, Equation 14.4 gives the covariance of $(z, x_1, x_2)$ as rows
$(1, 2, 1)$, $(2, 5, 2)$ and $(1, 2, 3)$. A Monte Carlo estimate from 200,000 samples in
the generator agrees to within 0.015 in every entry.

By the marginal rule, $x \sim \mathcal{N}(\mu, \Lambda\Lambda^\top + \Psi)$. This explains
why the model escapes the singularity: $\Psi$ is diagonal with positive entries, so
$\Lambda\Lambda^\top + \Psi$ is positive definite however small $m$ is. The covariance
has $nd + n$ free parameters: a rank-$d$ part that carries the correlations, plus a
diagonal part. [1:10:09](https://www.youtube.com/watch?v=tw6cmL5STuY&t=4209s) The log-likelihood
$\sum_i \log \mathcal{N}(x^{(i)}; \mu, \Lambda\Lambda^\top + \Psi)$ is perfectly well
defined, but setting its gradient to zero has no closed-form solution in $\Lambda$ and
$\Psi$. That is why the lecture turns to EM.

### EM for factor analysis

[1:11:03](https://www.youtube.com/watch?v=tw6cmL5STuY&t=4263s) In the mixture of Gaussians, $Q_i$ was a list of $k$ numbers. Here $z^{(i)}$
is continuous, so the question is how to represent $Q_i$ in a computer. The answer is that
$Q_i(z^{(i)}) = p(z^{(i)} \mid x^{(i)})$ is Gaussian, by Equation 14.3 applied to
Equation 14.4, so it is stored as a mean vector and a covariance matrix:

$$
\mu_{z^{(i)} \mid x^{(i)}} = \Lambda^\top (\Lambda\Lambda^\top + \Psi)^{-1} (x^{(i)} - \mu),
\qquad
\Sigma_{z^{(i)} \mid x^{(i)}} = I - \Lambda^\top (\Lambda\Lambda^\top + \Psi)^{-1} \Lambda .
$$ (14.5)

The generator checks Equation 14.5 against Bayes' rule applied directly, multiplying the
prior on $z$ by the likelihood of $x$ on a grid of 12,000 points. For three different
observations in the lecture's example, the two agree to $10^{-6}$ in mean and variance. As
the formula says, the posterior variance is the same for every observation:
$1/(1 + 4 + \tfrac12) = 0.1818$. In code, the $n \times n$ inverse is avoided with the
matrix-inversion lemma,
$\Sigma_{z \mid x} = (I + \Lambda^\top \Psi^{-1} \Lambda)^{-1}$, which is $d \times d$.

[1:13:39](https://www.youtube.com/watch?v=tw6cmL5STuY&t=4419s) The M-step derivation is long, and Ng gives only its two tricks. The first is
to recognise expectations instead of expanding integrals. The objective contains terms
like $\int Q_i(z^{(i)})\, z^{(i)}\, dz^{(i)}$. Substituting the Gaussian formula for $Q_i$
and integrating is the wrong approach. The integral is $\mathbb{E}_{z^{(i)} \sim Q_i}[z^{(i)}]$,
which is $\mu_{z^{(i)} \mid x^{(i)}}$, already computed in the E-step. [1:18:13](https://www.youtube.com/watch?v=tw6cmL5STuY&t=4693s) The
second is to expand a Gaussian density when a logarithm sits in front of it. Then the log
cancels the exponential and leaves a quadratic in the parameters, whose expectation under
$Q_i$ is again a quadratic that can be maximised in closed form. The resulting update for
$\Lambda$ is

$$
\Lambda = \left( \sum_{i=1}^m (x^{(i)} - \mu)\, \mu_{z^{(i)} \mid x^{(i)}}^\top \right)
\left( \sum_{i=1}^m \Bigl[ \mu_{z^{(i)} \mid x^{(i)}} \mu_{z^{(i)} \mid x^{(i)}}^\top + \Sigma_{z^{(i)} \mid x^{(i)}} \Bigr] \right)^{-1},
$$ (14.6)

with $\mu$ the sample mean. $\Psi$ is the diagonal of a similar expression, given in the
official notes. Equation 14.6 has the form of the normal equations for regressing $x$ on
$z$, except that the unknown $z^{(i)}$ are replaced by their posterior moments. The bracket
in the second factor is $\mathbb{E}[z^{(i)} z^{(i)\top}]$ under $Q_i$. That is the mean's
outer product *plus* the posterior covariance, because $\mathbb{E}[zz^\top]$ is not
$\mathbb{E}[z]\,\mathbb{E}[z]^\top$.

The official notes warn that dropping $\Sigma_{z \mid x}$ here is a common mistake, which
treats the posterior mean as if it were the true $z$. They do not say what the mistake
costs, so the generator runs both versions (Figure 14.3). The data are 30 examples in
$n = 100$ dimensions, drawn from a three-factor model; the sample covariance has rank 29,
as Section 14.2 says it must. Correct EM increases $\ell$ on every one of 200
iterations, from $-5744.2$ at the start to $-3254.7$ after 10 iterations and $-3203.0$ at
200. Over the last 50 iterations it is still gaining 1.3 nats. The shortcut version is
deceptive. It climbs *faster* at first and reaches $-3221.0$ at iteration 4, ahead of the
correct algorithm. Then the likelihood falls on every remaining iteration, 196 of 200 in
all. Correct EM overtakes it at iteration 38, and the shortcut finishes at $-3235.3$. The
shortcut is not coordinate ascent on any bound, so the monitoring rule of
Section 14.1 is exactly what catches it: an implementation whose log-likelihood ever
decreases has a bug, even if its first few iterations look better than the correct one.

![Figure 14.3. EM for factor analysis with $n = 100$, $m = 30$, $d = 3$. Left: training log-likelihood per iteration for the correct M-step and for the shortcut that uses $\mathbb{E}[z]\mathbb{E}[z]^\top$ in place of $\mathbb{E}[zz^\top]$. The first few iterations climb from far below and are clipped. The shortcut leads early, then declines on every iteration. Right: average log-likelihood of 300 held-out examples under factor analysis, a full Gaussian with $\lambda I$ added to its covariance (with $\lambda$ chosen on the test set itself, which favours it), and a diagonal Gaussian.](https://raimbekovm.github.io/cs229-2018-notes/figures/14-factor-analysis/diagram_fa_em.svg)

The same run tests Ng's remark about the Wishart fix from Section 14.3. On 300
held-out examples, factor analysis scores an average log-likelihood of $-129.0$ per
example. The full Gaussian with $\lambda I$ added scores $-138.5$, even though $\lambda$
was chosen from a grid using the test set itself, which is an unfair advantage. The
diagonal Gaussian scores $-198.8$. The comparison does favour factor analysis, because the
data were generated from a factor model. What it shows is narrower than "factor analysis
is better" in general. When the correlations really are low-rank, adding $\lambda I$ makes
the covariance invertible without capturing that structure, and the diagonal model throws
the correlations away entirely.

**In the official notes.** Restrictions on $\Sigma$, the marginal and conditional formulas with a derivation of the
conditional, the joint distribution of $(z, x)$, and the full M-step for $\mu$, $\Lambda$
and $\Psi$ are in cs229-notes9, sections 1 to 4. The mixture-of-Gaussians M-step with the
Lagrange multiplier for $\phi$, and the coordinate-ascent view of EM, are in cs229-notes8,
sections 2 and 3. The EM algorithm for factor analysis was given by
[Rubin and Thayer (1982)](https://doi.org/10.1007/BF02293851).
