# 2. Linear Regression and Gradient Descent

> Linear regression fitted two ways: by gradient descent on the least-squares cost — batch and stochastic, with Ng's rules for the learning rate — and in one step by the normal equations, derived with matrix derivatives and the trace.

Web version: https://raimbekovm.github.io/cs229-2018-notes/lectures/02-linear-regression.html

Lecture video: [Lecture 2: Linear Regression and Gradient Descent](https://www.youtube.com/watch?v=4b4MUYve_U8)

*Based on the lecture by Andrew Ng — [CS229](https://cs229.stanford.edu/), Stanford
University, Autumn 2018.*

The second lecture builds the first complete learning algorithm of the course and then
solves the same problem a second way. Linear regression predicts a number as a
weighted sum of features; the weights are chosen to make the squared error on the
training set as small as possible; and that minimum can be found either by walking
downhill on the error surface, one gradient step at a time, or by solving a linear
system for it directly. Gradient descent is the half that matters later: the normal
equations work for linear regression and its weighted variant and nothing else in the course, while the walk
downhill reappears in logistic regression, generalised linear models and neural
networks. Ng uses this lecture to set up the vocabulary all of them share —
hypothesis, parameters, cost function, learning rate — and it is worth getting that
vocabulary straight here, where the mathematics is still simple enough to check by
hand.

### The housing problem and the hypothesis

[1:55](https://www.youtube.com/watch?v=4b4MUYve_U8&t=115s) The running example is the one that closed lecture 1: predict the asking
price of a house from its size. Ng's data are real, a few dozen listings from
Portland, Oregon collected off Craigslist — a 2,104 ft² house listed at \$400,000, a
1,416 ft² one at \$232,000, and so on — and the task is to fit a straight line
through them.

Before writing any formula he describes the shape every supervised algorithm in the
course will have. A *training set* goes into a *learning algorithm*, and what comes
out is a function, by convention called the *hypothesis* $h$, which takes the size of a
house it has never seen and returns an estimated price. The learning algorithm's job
is to produce $h$; $h$'s job is to make predictions. [4:43](https://www.youtube.com/watch?v=4b4MUYve_U8&t=283s) The first design
decision is therefore not how to fit, but how to *represent* $h$, and for linear
regression the answer is

$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2,
$$ (2.1)

with $x_1$ the size and $x_2$ the number of bedrooms. Ng concedes the point a
mathematician in the room would raise: with the $\theta_0$ term this is an affine
function, not a linear one, and machine learning calls it linear anyway.

The intercept is awkward to carry around, so it is absorbed into the sum by inventing
a feature that is always one, $x_0 = 1$. Then $x$ and $\theta$ are both vectors in
$\mathbb{R}^{n+1}$, and

$$
h_\theta(x) = \sum_{j=0}^{n} \theta_j x_j = \theta^\top x .
$$ (2.2)

The rest of the notation is fixed here for the whole quarter. The $\theta_j$ are the
*parameters*, and choosing them is all that learning means for this model. $m$ is the
number of training examples (rows of the table), $n$ the number of features (two
here, so $x$ and $\theta$ are three-dimensional), $x$ the input features and $y$ the
output or *target*. The $i$-th training example is written $(x^{(i)}, y^{(i)})$, and
the superscript in parentheses is an index into the table, not a power: $x^{(1)}_1 =
2104$ is the size of the first house. Ng is open that some of this terminology is an
accident of history — he does not think "hypothesis" is a good name, but someone used
it in a paper decades ago and the field is stuck with it.

### The least-squares cost

[14:41](https://www.youtube.com/watch?v=4b4MUYve_U8&t=881s) "Choose $\theta$ so that $h(x)$ is close to $y$ on the training examples"
needs a number that measures closeness, and Ng builds it one piece at a time on the
board: the squared difference between prediction and truth for one house, summed over
all $m$ houses, with a factor of one half in front:

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 .
$$ (2.3)

$J$ is the *cost function*, and linear regression — ordinary least squares — is the
choice of $\theta$ that minimises it. The one half changes nothing about where the
minimum is; it is there so that the 2 brought down by differentiating a square cancels
it, as it will in Section 2.4.

[16:39](https://www.youtube.com/watch?v=4b4MUYve_U8&t=999s) The question Ng says he is asked most often is why the error is squared
rather than taken in absolute value or raised to the fourth power. His answer here is
a deferral with a hint: squared error corresponds to assuming the noise in $y$ is
Gaussian, and linear regression turns out to be one member of a family, the
generalised linear models, in which that choice is derived rather than assumed. The
next lecture makes the Gaussian argument precise.

### Gradient descent

[18:27](https://www.youtube.com/watch?v=4b4MUYve_U8&t=1107s) The algorithm for minimising $J$ starts from some initial $\theta$ — the
zero vector is a reasonable default, a random start is also fine — and keeps changing
$\theta$ in a way that lowers $J$.

[19:40](https://www.youtube.com/watch?v=4b4MUYve_U8&t=1180s) Ng's picture for it is a landscape. Put $\theta_0$ and $\theta_1$ on the two
horizontal axes and $J$ as the height (with three parameters the surface lives in four
dimensions, which is why the picture uses two). Stand at the starting point, turn
through 360 degrees, and ask: if I took one tiny step, which direction would take me
downhill fastest? Take that step, look around again from the new spot, and repeat.

[22:03](https://www.youtube.com/watch?v=4b4MUYve_U8&t=1323s) The picture also exposes the method's weakness. On a landscape with more
than one valley, where you end up depends on where you start: begin a few steps to one
side and the same procedure walks you into a different local minimum
(Figure 2.1). For linear regression this turns out not to matter, for a reason
Section 2.5 gives, but it matters a great deal for most of what comes later in the
course.

![Figure 2.1. Gradient descent on a one-parameter cost with two valleys. Started just left of the hump, it runs into the deeper valley; started just right, it stops in the shallower one and has no way of knowing a better minimum exists. *Computed by `figures/02-linear-regression/make_diagrams.py`.*](https://raimbekovm.github.io/cs229-2018-notes/figures/02-linear-regression/diagram_local_minima.svg)

[23:54](https://www.youtube.com/watch?v=4b4MUYve_U8&t=1434s) Formally, one step of gradient descent updates every parameter at once:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta), \qquad j = 0, 1, \dots, n .
$$ (2.4)

The partial derivative gives the direction of steepest ascent, so subtracting a
multiple of it goes downhill. The multiple $\alpha$ is the *learning rate*, which sets
the size of the step. The `:=` is assignment, and Ng is deliberate about it: $a := a
+ 1$ increments $a$, whereas $a = b$ asserts a fact, and he promises never to write $a
= a + 1$. Throughout the update, the training set is fixed and $J$ is a fixed
function of $\theta$; only $\theta$ moves.

[26:04](https://www.youtube.com/watch?v=4b4MUYve_U8&t=1564s) A student asked at once how to choose $\alpha$. "There's a theory and
there's a practice," and the practice, to laughter, is $0.01$. The serious version
followed: if the features are scaled to roughly $[-1, 1]$, start at $0.01$ and try
larger and smaller values to see which drives $J$ down fastest. Section 2.5 says why the
scaling clause is not optional.

### The LMS update

[27:27](https://www.youtube.com/watch?v=4b4MUYve_U8&t=1647s) To run Equation 2.4 the partial derivative has to be computed, and Ng does it
for a training set of a single example $(x, y)$ first, so that the sum does not get in
the way. The chain rule brings the 2 down from the square, where it cancels the
half, and leaves the derivative of what was inside:

$$
\frac{\partial}{\partial \theta_j} \frac{1}{2}\left(h_\theta(x) - y\right)^2
= \left(h_\theta(x) - y\right) \frac{\partial}{\partial \theta_j}
\left( \theta_0 x_0 + \theta_1 x_1 + \dots + \theta_n x_n - y \right)
= \left(h_\theta(x) - y\right) x_j .
$$ (2.5)

Of all the terms in the expanded sum only $\theta_j x_j$ depends on $\theta_j$, so
every other term differentiates to zero and this one to $x_j$.

[31:43](https://www.youtube.com/watch?v=4b4MUYve_U8&t=1903s) With $m$ examples, the derivative of a sum is the sum of the derivatives,
and the full algorithm is

$$
\text{repeat until convergence:} \quad
\theta_j := \theta_j - \alpha \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\quad \text{for every } j .
$$ (2.6)

This is the least-mean-squares or Widrow–Hoff rule, and it reads well once written
out. Each example pushes $\theta_j$ in proportion to its own error: a house whose price
is already predicted almost exactly barely moves the parameters, one that is badly
mispredicted moves them a lot, and the push on $\theta_j$ is weighted by how large the
feature $x_j$ was for that house. The generator behind this chapter checks
Equation 2.6's sum against finite differences of $J$ on the housing data, and it
agrees.

**In the official notes.** The same derivation, and the remark that the update is proportional to the error,
are in cs229-notes1, section 1.
### A bowl, and the learning rate

[33:52](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2032s) The local-minimum worry of Figure 2.1 disappears for this cost.
$J$ is a sum of squares of functions linear in $\theta$, so it is a quadratic in
$\theta$, and a quadratic of this kind is a single bowl: its only local minimum is the
global one. Looked at from above, its contours are nested ellipses, and Ng adds a fact
he half-remembers from school — "I think it's true" — that each step of steepest
descent crosses the contour it starts on at right angles. It is true: the gradient is
normal to the level set through any point, and the generator confirms it on the
housing cost to within a cosine of 0.003.

What the board picture does not show is how *thin* the bowl can be.
Figure 2.2 runs gradient descent on synthetic housing data shaped like Ng's, with
size in thousands of square feet, starting from $\theta = 0$. The first step drops
almost straight across the contours into the floor of a long valley; every step after
that inches along the floor towards the minimum, and it takes more than four hundred
of them to bring the intercept within \$500 of its final value. The ratio of the
bowl's steepest to its shallowest curvature — the eigenvalues of $X^\top X$, in the
notation of Section 2.8 — is about a hundred here, and gradient descent's
pace is set by the shallow direction while its step size is capped by the steep one.

![Figure 2.2. Batch and stochastic gradient descent on the contours of the least-squares cost, both from $\theta = 0$, on 47 synthetic houses shaped after the Portland data. Batch descent (left) reaches the valley floor in one step and then crawls along it; the stochastic version (right) takes the same route noisily and keeps moving near the minimum, marked by the dot. *Computed by `figures/02-linear-regression/make_diagrams.py`.*](https://raimbekovm.github.io/cs229-2018-notes/figures/02-linear-regression/diagram_contours.svg)

That cap is the precise version of Ng's [36:08](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2168s) advice on the learning rate. Too
small and the algorithm needs very many iterations; too large and each step
overshoots the minimum, lands higher on the far side, and diverges
(Figure 2.3). For a quadratic the boundary is exact: with $\lambda_{\max}$ the
largest eigenvalue of $X^\top X$, gradient descent converges for $\alpha <
2/\lambda_{\max}$ and diverges above it, which the generator checks on both sides of
the line. His practical rules follow from watching $J$ rather than computing
eigenvalues. If $J$ goes *up* over iterations, the learning rate is almost certainly
too large. And rather than search finely, try values on a roughly doubling scale —
0.01, 0.02, 0.04, 0.08 — and keep whichever drives $J$ down fastest.

![Figure 2.3. Gradient descent on $J(\theta) = \theta^2$, where the update multiplies $\theta$ by $1 - 2\alpha$. A small learning rate creeps down one side; a moderate one reaches the bottom in a few steps; above $\alpha = 1$ every step overshoots to the other side, higher than before. *Computed by `figures/02-linear-regression/make_diagrams.py`.*](https://raimbekovm.github.io/cs229-2018-notes/figures/02-linear-regression/diagram_learning_rate.svg)

The same bound explains the feature-scaling clause in "set it to 0.01". On the
synthetic data with size measured in raw square feet, $\lambda_{\max}$ is so large
that the stable range ends at $\alpha \approx 4 \times 10^{-9}$, and $0.01$ diverges
immediately; with size rescaled to about $[-1, 1]$ the limit is about $0.04$, and
$0.01$ is comfortably safe. One further caveat follows from writing $J$ as a sum.
$\lambda_{\max}$ grows with $m$, so the safe learning rate shrinks as the training set
grows, and $0.01$ that works on 47 houses will not on 47,000. Many implementations
average the squared error instead of summing it, which removes that dependence; the
minimum is unchanged either way.

[37:14](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2234s) Ng's second view of the same run is in the data rather than in parameter
space. With $\theta = 0$ the hypothesis is the horizontal line $h(x) = 0$, predicting
every house free; each step of gradient descent replaces it with a line that fits a
little better, until it settles on the least-squares fit (Figure 2.4).

![Figure 2.4. The hypothesis after successive steps of batch gradient descent from $\theta = 0$, drawn over the synthetic housing data; the solid line is the least-squares fit the steps converge to. *Computed by `figures/02-linear-regression/make_diagrams.py`.*](https://raimbekovm.github.io/cs229-2018-notes/figures/02-linear-regression/diagram_gd_lines.svg)

[39:30](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2370s) A student asked why the update *subtracts* $\alpha$ times the gradient
rather than adding it. Ng's suggestion was to work one case by hand, and it settles
the matter: take $J(\theta) = \theta^2$ at $\theta = 1$. The derivative there is $2$
(Ng said 1 at the board; the sign is what matters), positive, and to reach the
minimum at zero $\theta$ has to decrease — so the step must go against the gradient.
Adding it would climb.

### Stochastic gradient descent

[41:05](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2465s) Equation 2.6 is called *batch* gradient descent because every single update
looks at the whole training set as one batch. That is its weakness. If $m$ is the
size of the US census rather than of a Craigslist scrape — Ng mentions training on
hundreds of millions of examples — then one step requires a full pass over the data,
possibly terabytes read from disk, before $\theta$ moves at all, and convergence can
take hundreds or thousands of such steps.

[44:13](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2653s) *Stochastic* gradient descent updates after every example instead:

$$
\text{loop: for } i = 1, \dots, m: \quad
\theta_j := \theta_j - \alpha \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\quad \text{for every } j .
$$ (2.7)

[46:02](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2762s) In Ng's words it looks at one house and adjusts the parameters to predict
that house better, then the next house, then the next. No single house's gradient
points straight at the minimum, so the path is noisy, but on average it heads
downhill, and it starts making progress after the first example rather than after the
millionth. The price is that it never quite converges: because each update chases a
different house, $\theta$ keeps wandering around the minimum, as on the right of
Figure 2.2. For a large dataset that trade is almost always worth making.

The questions from the room fill in how it is used in practice. Asked [47:58](https://www.youtube.com/watch?v=4b4MUYve_U8&t=2878s)
whether one could start stochastic and switch to batch near the end, Ng said yes, and
that it does no harm — but that people rarely bother, because batch steps on a large
dataset are too slow to be worth it, and because the parameters stochastic gradient
descent ends near are usually perfectly good even if they are not exactly the global
minimum. The middle ground, which he names and leaves to CS230, is *mini-batch*
gradient descent, which updates on a hundred or so examples at a time. What is common
instead is [50:03](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3003s) to decrease the learning rate slowly over time: the
oscillations shrink with the step size, so $\theta$ ends up bouncing around a smaller
region closer to the minimum. Asked when to stop, his answer was to plot $J$ over time
and stop when it stops going down — easy here, where there are no local optima to
confuse the picture, and much harder for the neural networks later in the course.

[51:35](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3095s) The last question was whether stochastic gradient descent's learning rate
should be $1/m$ times batch's. No, Ng said, it is usually much bigger, since otherwise
it would be as slow as batch. The arithmetic behind that answer is worth having.
Because $J$ is a sum, a full pass of Equation 2.7 at rate $\alpha$ adds up, to first order
in $\alpha$, to one batch step at the same $\alpha$; the generator confirms the gap
between them shrinking as $\alpha$ does. Dividing by $m$ would therefore make a whole
pass of stochastic updates worth only $1/m$ of a batch step. And a single-example
update tolerates a far larger rate in the first place: its stability limit is $2 /
\lVert x^{(i)} \rVert^2$ rather than $2 / \lambda_{\max}$ of the whole of $X^\top X$,
which on the synthetic housing data is a factor of about 25 more room.

[51:51](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3111s) Ng's own rule of thumb closes the topic. With a dataset of hundreds or
thousands of examples, where a pass is cheap, he would almost always use batch
gradient descent, because it is one less thing to fiddle with — no oscillating
parameters to worry about. When the data are too large for that, almost everyone uses
stochastic gradient descent or something like it.

### Matrix derivatives and the trace

[53:25](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3205s) For linear regression, and for exactly linear regression — none of the
models of the following lectures beyond its weighted variant in lecture 3 — there is a way to skip the iterations altogether and
jump to the minimum in one step. Deriving it means setting the gradient of $J$ to zero
and solving, and Ng's aim is to do that "in roughly four lines of linear algebra
rather than pages and pages", which takes a little notation first. In his experience,
he says, the right notation is what makes such problems tractable.

[56:02](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3362s) For a function $J : \mathbb{R}^{n+1} \to \mathbb{R}$, the derivative with
respect to the vector $\theta$ is the vector of partial derivatives,

$$
\nabla_\theta J = \begin{bmatrix} \partial J / \partial \theta_0 \\ \vdots \\ \partial J / \partial \theta_n \end{bmatrix} ,
$$ (2.8)

and more generally, for a function $f$ that maps an $m \times n$ matrix $A$ to a real
number, $\nabla_A f(A)$ is the $m \times n$ matrix whose $(i, j)$ entry is $\partial f /
\partial A_{ij}$. Ng's example [58:32](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3512s) is a made-up function of a $2 \times 2$
matrix, $f(A) = A_{11} + A_{12}^2$, so that $f$ applied to the matrix with rows
$(5, 6)$ and $(7, 8)$ is $5 + 36 = 41$, and

$$
\nabla_A f(A) = \begin{bmatrix} 1 & 2A_{12} \\ 0 & 0 \end{bmatrix} .
$$ (2.9)

A student asked whether the vector definition was a special case of the matrix one;
it is — a vector is an $(n+1) \times 1$ matrix, and the two agree.

[1:03:35](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3815s) The second tool is the *trace* of a square matrix, the sum of its
diagonal entries, $\operatorname{tr} A = \sum_i A_{ii}$. Transposing a matrix leaves
its diagonal where it was, so $\operatorname{tr} A = \operatorname{tr} A^\top$. The
trace of a product is unchanged by moving the last factor to the front,
$\operatorname{tr} AB = \operatorname{tr} BA$ and $\operatorname{tr} ABC =
\operatorname{tr} CAB$, and its derivative has a clean form:

$$
\nabla_A \operatorname{tr} AB = B^\top .
$$ (2.10)

[1:06:23](https://www.youtube.com/watch?v=4b4MUYve_U8&t=3983s) Ng then gives one identity as the matrix analogue of $\frac{d}{da}
a^2 c = 2ac$, a rule to be used without re-deriving it every time, the way one uses
the derivative of $x^2$. The analogy is a good way to remember the shape of the
result, but the generator caught that the shorthand needs a condition the board left
out:

$$
\nabla_A \operatorname{tr}\!\left(A A^\top C\right) = \left(C + C^\top\right) A ,
$$ (2.11)

which equals $2CA$ only when $C$ is symmetric. For a random non-symmetric $C$, the
numerical gradient matches Equation 2.11 and misses $2CA$ by a visible margin;
for the symmetric part of the same $C$ the two coincide. In the derivation that
follows $A$ is $\theta$ and the matrix playing the role of $C$ is $X^\top X$, which is
symmetric, so nothing goes wrong, but
the general rule is the one with $C + C^\top$. The official notes state the
fully general version, $\nabla_A \operatorname{tr} ABA^\top C = CAB + C^\top A
B^\top$, which reduces to Equation 2.11 when $B$ is the identity.

### The normal equations

[1:08:39](https://www.youtube.com/watch?v=4b4MUYve_U8&t=4119s) To write $J$ with matrices, stack the training inputs as the rows of the
*design matrix* $X$, an $m \times (n+1)$ matrix whose $i$-th row is
$(x^{(i)})^\top$, and stack the targets into a vector $\vec{y} \in \mathbb{R}^m$. Then
$X\theta$ is the vector of every prediction at once, since its $i$-th entry is
$(x^{(i)})^\top \theta = h_\theta(x^{(i)})$, and $X\theta - \vec{y}$ is the vector of
every error. A vector dotted with itself is the sum of its squared entries, $z^\top z =
\sum_i z_i^2$, so

$$
J(\theta) = \frac{1}{2} \left( X\theta - \vec{y} \right)^\top \left( X\theta - \vec{y} \right) .
$$ (2.12)

[1:13:08](https://www.youtube.com/watch?v=4b4MUYve_U8&t=4388s) Now take the gradient and set it to zero. Expanding the product the way
one expands $(ax - b)^2$ gives four terms, $\theta^\top X^\top X \theta - \theta^\top
X^\top \vec{y} - \vec{y}^\top X \theta + \vec{y}^\top \vec{y}$. The last does not
depend on $\theta$; the middle two are the same scalar, each the transpose of the
other; and the trace rules of Section 2.7 differentiate the first to
$2X^\top X\theta$ and the middle pair to $2X^\top\vec{y}$. Halving,

$$
\nabla_\theta J(\theta) = X^\top X \theta - X^\top \vec{y} .
$$ (2.13)

This is Equation 2.6's sum written in matrix form, which is a useful check on both.
[1:16:19](https://www.youtube.com/watch?v=4b4MUYve_U8&t=4579s) Setting it to zero gives the *normal equations*,

$$
X^\top X \theta = X^\top \vec{y} ,
\qquad
\theta = \left( X^\top X \right)^{-1} X^\top \vec{y} .
$$ (2.14)

A few matrix products land on the global minimum of the bowl in one step, the same
point that gradient descent spends hundreds of steps approaching in Figure 2.2; the
generator runs both and confirms they agree to six decimal places. On the real
Portland data the official notes report the fit $\theta_0 = 71.27$, $\theta_1 =
0.1345$: about \$71,000 plus \$134.50 per square foot.

[1:17:01](https://www.youtube.com/watch?v=4b4MUYve_U8&t=4621s) The common question is what to do if $X^\top X$ is not invertible. That
usually means the features are linearly dependent — redundant — and the matrix
pseudo-inverse will still return a sensible answer. But Ng's preferred fix is
diagnostic: a singular $X^\top X$ most often means the same feature is in the data
twice (size in square feet and size in square metres, say), and the right move is to
find the duplicate and drop it. With a feature duplicated, every way of splitting its
weight between the two copies gives exactly the same predictions and the same $J$,
which is why no single $\theta$ can be picked out by the equations; the generator
checks this on a copy of the housing data with the size column repeated.

His last answer, to a final question on how to choose the learning rate, came back to
where Section 2.3 left it: it is empirical, and most people try several
values and keep the one that works.

**In the official notes.** The matrix-derivative identities and the full derivation of the normal equations,
including the trace manipulation Ng skips, are in cs229-notes1, sections 2.1 and 2.2.
Section 3 justifies the least-squares cost as maximum likelihood under Gaussian noise,
which is where the next lecture begins.
