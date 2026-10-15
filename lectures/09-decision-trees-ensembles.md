# 9. Decision Trees and Ensemble Methods

> Decision trees as greedy recursive partitioning of the input space, why splits are chosen by a strictly concave loss such as cross-entropy rather than misclassification, regression trees and categorical features, how trees are regularised and why they are fast, their weakness on additive structure — and how ensembles fix them: the variance of an average of correlated models, bagging, random forests, and boosting with AdaBoost.

Web version: https://raimbekovm.github.io/cs229-2018-notes/lectures/09-decision-trees-ensembles.html

Lecture video: [Lecture 9: Decision Trees and Ensemble Methods](https://www.youtube.com/watch?v=wr9gUr-eWdA)

*Based on the lecture by Raphael Townshend, head TA — [CS229](https://cs229.stanford.edu/),
Stanford University, Autumn 2018.*

A decision tree splits the input space into rectangles by asking one yes-or-no
question about one feature at a time, and predicts a constant in each rectangle. It is
the course's first genuinely non-linear model, it is easy to explain to anyone, and on
its own it is not very accurate: a fully grown tree has low bias and very high variance.
The second half of the lecture is about the cure, which is to average many trees.
Averaging cuts variance by an amount the correlation between the trees controls, and
that single formula explains bagging, the extra randomness of random forests, and why
boosting, which works on bias instead, uses deliberately weak trees.

The lecture is given by Raphael Townshend, one of the course's head TAs, while Ng is
travelling. It is almost entirely board work, built on one running example.

### A model that cuts the space into boxes

[1:24](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=84s) Townshend, who is Canadian and likes to ski, poses a classifier: given a
month and a latitude, can you ski there? Plot month (1 for January to 12 for December)
against latitude ($+90°$ at the North Pole, $-90°$ at the South). The positives sit in
two separate blocks, the northern winter at high northern latitudes and the southern
winter, mid-year, at high southern ones, with nothing near the equator. No line
separates that, and while a kernel might lift it into a space where a hyperplane does,
the structure is plainly a set of boxes, and it would be simpler to find the boxes
directly.

[3:38](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=218s) Finding the best set of regions is intractable in general. A decision tree
finds a good one *greedily, top-down and recursively*. It starts with the whole space
and asks the single question that best separates the classes — "is the latitude above
30°?" — which cuts the plane in two. It then treats each half as a fresh problem and
asks another question there, "is the month before March?", and so on: a game of twenty
questions with the data. Figure 9.1 shows the result on synthetic ski data. Seven
leaves isolate both skiing seasons.

![Figure 9.1. Synthetic ski data: one can ski in the northern winter above 30° and in the southern winter below −30°. A tree grown greedily until its leaves are pure cuts the plane into axis-aligned rectangles; the shaded ones are those it labels "ski".](https://raimbekovm.github.io/cs229-2018-notes/figures/09-decision-trees-ensembles/diagram_ski.svg)

[6:08](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=368s) Formally, a split of a parent region $R_p$ is a feature $j$ and a threshold
$t$, which divide it into two children:

$$
R_1 = \{x \in R_p : x_j < t\}, \qquad R_2 = \{x \in R_p : x_j \ge t\}.
$$

Each leaf predicts the most common class among the training examples that land in it.
What remains is to say which $(j, t)$ is "best".

### Choosing a split

[8:32](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=512s) Define a loss $L(R)$ for a region and choose the split that reduces it the
most, from the loss of the parent to the combined loss of the children. Since the
parent's loss is fixed, this is the same as minimising the children's loss. With
$\hat p_c$ the proportion of the region's examples in class $c$, the obvious loss is the
misclassification rate, $L_{\text{mis}}(R) = 1 - \max_c \hat p_c$: predict the majority
class and count everything else as an error.

[12:29](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=749s) Townshend's example shows why that loss is the wrong one. A region holds
900 positives and 100 negatives. One split sends 700 positives and 100 negatives one
way and 200 positives the other; another sends 400 and 100 one way and 500 positives
the other. The second split looks better — it isolates a pure region of 500 — but both
children of both splits still predict "positive", so each split makes exactly the 100
mistakes the parent made. Misclassification registers no gain for either, and a tree
built on it would have no reason to prefer the split that obviously helps.

[15:21](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=921s) The fix is a loss that rewards purity even when it does not change any
prediction. The standard choice is the **cross-entropy**,

$$
L_{\text{cross}}(R) = -\sum_c \hat p_c \log_2 \hat p_c ,
$$ (9.1)

the number of bits needed, on average, to tell someone the class of an example when
they already know the proportions: zero for a pure region, one bit for a 50/50 binary
region. [51:56](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3116s) Asked whether the logarithm must be base 2, Townshend is "pretty
sure" it does not matter. It does not: changing the base multiplies every loss and every
gain by the same constant, so the ranking of splits cannot change. The generator
confirms the ratio is exactly $\ln 2$ for both splits.

One detail on the board needs care. The children's loss must be weighted by their
sizes,

$$
\frac{|R_1|\,L(R_1) + |R_2|\,L(R_2)}{|R_1| + |R_2|},
$$ (9.2)

as cs229-notes-dt writes it. The board writes $L(R_1) + L(R_2)$, which is correct for
misclassification measured in *counts*, as in the example, but not for losses defined
on *proportions* such as Equation 9.1. Unweighted, the sum can be made larger
than the parent's loss by peeling off a single positive and a single negative — a
region of two with a full bit of entropy — even though such a split barely changes
anything. With the weighting, cross-entropy gives the lecture's two splits gains of
$0.034$ and $0.108$ bits, preferring the second as intuition does.

[21:44](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=1304s) The geometry explains why this works, and Figure 9.2 draws it. Plot a
binary loss against $\hat p$, the share of positives. A parent's $\hat p$ is the
size-weighted average of its children's, so the parent sits on the curve at that
average, while the children's weighted loss sits on the *chord* between the children's
points. For a strictly concave curve such as cross-entropy, the chord lies strictly
below the curve whenever the children differ, so every split that changes the class
proportions gains something. Misclassification is a tent, linear on each side of
$\frac12$. When both children fall on the same side as the parent, the chord lies on the
curve and the gain is exactly zero, which is what happened in the example. Over random
splits of random regions the generator finds this case — positive cross-entropy gain,
zero misclassification gain — in about half of all splits.

![Figure 9.2. Three impurities against the share of positives. The 900/100 parent sits at 0.9; the 400/100 and 500/0 children sit at 0.8 and 1.0, and their average loss is the midpoint of the dashed chord. The gain is the vertical gap between the parent's point (teal) and the midpoint (ring): zero for misclassification, positive for cross-entropy and Gini.](https://raimbekovm.github.io/cs229-2018-notes/figures/09-decision-trees-ensembles/diagram_impurity.svg)

[29:36](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=1776s) The **Gini** impurity $\sum_c \hat p_c (1 - \hat p_c)$ has the same shape
and is the other common choice. It is tempting to treat the two as interchangeable,
and usually they are; but the generator, offering both the same ten random candidate
splits of a 60/40 region 2,000 times, finds that they choose different splits in
$5.8\,\%$ of cases. The family resemblance is the concavity, and that, Townshend says,
is what the losses that work in practice have in common.

### Regression trees and categorical features

[31:03](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=1863s) Nothing above depends on the labels being classes. To predict snowfall
instead of "can ski", let each leaf $R_m$ predict the mean of its targets,
$\hat y_m = \frac{1}{|R_m|}\sum_{i \in R_m} y^{(i)}$, and measure the region by squared
error, $L_{\text{sq}}(R_m) = \frac{1}{|R_m|}\sum_{i \in R_m} (y^{(i)} - \hat y_m)^2$.
These are **regression trees**, and splitting proceeds exactly as before.
[34:34](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2074s) Asked how the best split is actually found, Townshend says it is found by
brute force, efficiently: sort the examples along each feature, and every distinct gap
between consecutive values is a candidate threshold that can be scored with running
sums in one pass.

[35:06](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2106s) Trees also take **categorical** features naturally. If location is one of
*north*, *equator* or *south*, a split is a question such as "is the location in the
northern hemisphere?" — in general, "is the category in subset $A$?". [36:27](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2187s) The
catch is the number of subsets. The lecture counts $2^q$ for $q$ categories; the number
of distinct binary splits is $2^{q-1} - 1$, since a subset and its complement define the
same split and the empty set defines none. Either way it is exponential.
[37:01](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2221s) For binary classification there is a well-known escape, which the lecture
mentions without proof: sort the categories by their proportion of positives, and the
best split is always one of the $q - 1$ splits of that sorted list into a prefix and a
suffix. The generator checks this against brute force over all subsets for 500 random
six-category problems under cross-entropy, and the two always agree.

**In the official notes.** cs229-notes-dt, sections 2 to 4.1, give the recursive partitioning algorithm, the
size-weighted split criterion and the treatment of categorical variables.
### Regularising a tree

[37:39](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2259s) Grown until every leaf is pure, a tree can put each training example in a
region of its own. That overfits, and trees are in this sense high-variance models.
[38:24](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2304s) The usual controls are heuristics: stop splitting a node with fewer than a
minimum number of examples, cap the depth, or cap the number of nodes.

[39:07](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2347s) A fourth heuristic, stopping when the best split decreases the loss by less
than some threshold, is one Townshend calls "very tempting" and advises against.
Interactions between features can make the first question worthless on its own and the
second one decisive. In the ski data the latitude question alone helps little; the
latitude question followed by the month question nails it. The extreme case is XOR:
label a point positive when exactly one of its two coordinates exceeds a threshold. The
generator builds a balanced XOR set of 64 points, on which *every* single split leaves
the cross-entropy exactly where it was, so a tree that splits only on positive gain
stops at the root, with 32 mistakes. Taking the zero-gain split anyway, the two children
are each solved by one more split, with no mistakes at depth 2.

[40:07](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2407s) The better approach is to grow the tree fully and then **prune** it
backwards: remove a split if doing so does not hurt the misclassification error on a
held-out validation set — the dev set of lecture 8, used to choose the size of the tree.

### Runtime

[42:54](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2574s) Let $n$ be the number of examples, $f$ the number of features and $d$ the
depth of the tree. Prediction walks one path from root to leaf, so it costs $O(d)$, and
for a reasonably balanced tree $d$ is about $\log_2 n$. Training is the surprise.
[44:46](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2686s) Each example lands in exactly one node per level, so it takes part in at
most $d$ split searches, and in each it is scanned once per feature. The total is
$O(nfd)$. Since $nf$ is the size of the data matrix, training a tree costs about as
much as reading the data $d$ times. (For continuous features the scan needs the examples
sorted along each feature; the sort is done once up front, or the bound picks up a
logarithmic factor.)

### What trees cannot do

[47:58](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2878s) The main weakness is **additive structure**. Label points positive above the
line $x_2 = x_1$. Logistic regression represents that boundary exactly with one
weight per feature. A tree can only approximate it with a staircase of axis-aligned
steps, and every step costs leaves. On 400 training points, a depth-1 tree reaches
$75.8\,\%$ test accuracy, depth 3 reaches $89.0\,\%$, and a tree grown until its leaves
are pure stops at 18 leaves and $96.9\,\%$, the staircase being as fine as the training
data allow (Figure 9.3).

![Figure 9.3. Points above the dashed diagonal are positive. Trees of depth 1 and 3, and one grown until every leaf is pure, each trained on 400 points, shade the region they call positive. The boundary is always a staircase, and following the diagonal more closely takes more leaves and more data.](https://raimbekovm.github.io/cs229-2018-notes/figures/09-decision-trees-ensembles/diagram_staircase.svg)

[49:36](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=2976s) The scorecard, then. Trees are easy to explain — twenty questions — and
short ones can be read directly. They handle categorical features and they are fast.
Against that, they have high variance, they are bad at additive structure, and, mostly
for those two reasons, their predictive accuracy is low. [51:09](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3069s) "I know what you
guys are thinking," Townshend says: why spend half a lecture on a model that, on its
own, is not very good? Because trees are the ideal raw material for ensembles, which
(as of 2018) underpin many of the methods winning Kaggle competitions.

### Why averaging helps

[53:11](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3191s) Let $X_1, \dots, X_n$ be independent random variables with variance
$\sigma^2$. Their average has variance $\sigma^2 / n$: each independent source divides
the variance further. [55:00](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3300s) Models trained on related data are not independent,
though. If the $X_i$ are identically distributed with pairwise correlation $\rho$, then

$$
\operatorname{Var}\Bigl(\frac1n \sum_{i=1}^n X_i\Bigr) = \rho\,\sigma^2 + \frac{1-\rho}{n}\,\sigma^2 .
$$ (9.3)

The second term vanishes as $n$ grows; the first does not. With $\rho = 1$ averaging
does nothing at all, and in general the variance can never fall below $\rho\sigma^2$
(Figure 9.4). [57:01](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3421s) So the game has two moves: average many models to
shrink the second term, and make them as decorrelated as possible to shrink the first.
The generator checks Equation 9.3 by simulation for $\rho = 0, 0.3, 0.6$.

![Figure 9.4. The variance of an average of n equally correlated models, relative to one model's. Independent models (ρ = 0) average away to nothing; correlated ones level off at ρ, the dotted lines. Rings are simulated averages.](https://raimbekovm.github.io/cs229-2018-notes/figures/09-decision-trees-ensembles/diagram_correlation.svg)

[57:48](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3468s) The most decorrelated ensembles come from different algorithms — averaging a
neural network, a random forest and an SVM is a common Kaggle tactic that works well —
or from different training sets, which is rarely an option, since collecting one was
hard enough. Bagging approximates the second; boosting does something else.

### Bagging

[1:00:20](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3620s) **Bagging** stands for *bootstrap aggregation*. The bootstrap is a
statistician's device for measuring the uncertainty of an estimate. The training set
$S$ was sampled from a population $P$; ideally we would draw fresh training sets from
$P$, but we cannot. So pretend $S$ *is* the population, and draw new training sets
$Z$ from it: $n$ examples sampled from $S$ *with replacement*, which is what sampling
from a population means. [1:03:25](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3805s) Bagging takes $M$ bootstrap samples
$Z_1, \dots, Z_M$, trains a model $G_m$ on each, and averages:

$$
G(x) = \frac1M \sum_{m=1}^M G_m(x).
$$

[1:05:06](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=3906s) In the terms of Equation 9.3, the bootstrap lowers $\rho$
(the models see different data) and $M$ controls the second term. A consequence
Townshend stresses is that more bootstrap models cannot make the ensemble overfit more:
they only shrink the second term, so the usual practice is to keep adding models until
the error stops falling. Figure 9.5 shows exactly that.

[1:07:33](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4053s) Asked how far bootstrapping can push $\rho$ down, he has no formula, only
the reason there is a floor: every bootstrap sample is drawn from the same $S$, and each
contains about two-thirds of it. The exact fraction of distinct examples is
$1 - (1 - 1/n)^n \to 1 - 1/e \approx 0.632$, which the generator confirms by sampling.
[1:08:29](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4109s) That missing third is also the price: each model trains on fewer distinct
examples, so bagging slightly raises bias. In Figure 9.5 a single tree on a bootstrap
sample has test error $1.52$ against $1.08$ for one tree on the full data, but by three
or four trees the average has overtaken the full-data tree, and a hundred bagged trees
reach $0.75$.

[1:11:23](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4283s) This is why trees and bagging belong together. A deep tree has low bias
and high variance; bagging trades a little bias for a large cut in variance; and so a
model whose error is mostly variance gains the most. Bagging a high-bias model would
gain little.

![Figure 9.5. Test error of an average of M regression trees grown on bootstrap samples, for plain bagging and for a random forest that considers three of eight features at each split. Both fall steeply and then flatten; neither rises as trees are added. One full tree on all the data is far worse; the dotted line is the noise floor.](https://raimbekovm.github.io/cs229-2018-notes/figures/09-decision-trees-ensembles/diagram_forest.svg)

The method is due to [Breiman (1996)](https://doi.org/10.1007/BF00058655).

### Random forests

[1:12:29](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4349s) A **random forest** is bagged trees with one more source of randomness: at
each split, the tree may only consider a random subset of the features. [1:13:56](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4436s)
Townshend gives two reasons this helps, which are the same reason. It lowers $\rho$.
And concretely, if one feature is a very strong predictor, every bagged tree will split
on it first whatever its bootstrap sample, so the trees are alike from the root down;
forbidding it at some splits forces them apart.

The decorrelation is real and easy to measure. On the problem of Figure 9.5 — eight
noisy measurements, five of one latent variable and three of another — the correlation
between two bagged trees' predictions across training sets is about $0.18$; letting
each split see only three features drops it to $0.07$, with each tree's own variance
roughly unchanged, so $\rho\sigma^2$ falls from about $0.14$ to $0.05$. The test error
of 100 trees falls from $0.738$ with all features to about $0.70$ with two to four.

But the argument above is only about variance, and the lecture does not mention the
other side. Forbidding a feature can also make a tree worse. On a second problem where
one feature carries most of the signal and the others are weak, restricting each split
to two of six features roughly doubles each tree's variance — many splits must make do
with weak features — and the forest's test error rises from $0.44$ to $0.51$; with one
feature per split it reaches $0.74$. Random feature selection pays when the features it
forces the trees to use are nearly as good as the ones it hides, as with redundant or
correlated measurements. The number of features per split is a hyperparameter to tune
like any other, which is how [Breiman (2001)](https://doi.org/10.1023/A:1010933404324)
treated it.

### Boosting

[1:15:16](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4516s) Bagging fights variance. **Boosting** works in the opposite direction: it
reduces *bias*, by building an ensemble additively, each new model trained to fix what
the previous ones got wrong. It starts from deliberately weak models — typically
**decision stumps**, trees of depth one that ask a single question. A stump has high
bias and low variance, which is the right raw material for a method that lowers bias.
(The lecture's sentence at [1:16:44](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4604s) runs these two the wrong way round, saying that
restricting to depth one decreases bias before saying it increases it; the second is
right.)

[1:16:59](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4619s) The first stump draws its line and misclassifies some examples. Boosting
increases the weight of those examples and trains the next stump on the reweighted set,
which pushes it to get the hard ones right; and so on, each example's weight reflecting
how often the ensemble so far has got it wrong. Figure 9.6 runs it on the
diagonal boundary that defeats a single tree. [1:18:17](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4697s) Each stump $G_m$ also gets a
vote $\alpha_m$, larger for more accurate stumps, and the final classifier is the
weighted vote $G(x) = \operatorname{sign}\bigl(\sum_m \alpha_m G_m(x)\bigr)$. The
algorithm is **AdaBoost**, due to [Freund and Schapire
(1997)](https://doi.org/10.1006/jcss.1997.1504). In the form the official notes give,
with labels $y \in \{-1, +1\}$ and weights starting at $w_i = 1/n$, each round:

1. fits a stump $G_m$ to the weighted data;
2. computes its weighted error $\operatorname{err}_m = \sum_i w_i\, 1\{y_i \ne G_m(x_i)\} / \sum_i w_i$;
3. sets its vote to the log-odds $\alpha_m = \log\bigl((1 - \operatorname{err}_m)/\operatorname{err}_m\bigr)$;
4. multiplies the weight of every misclassified example by $e^{\alpha_m}$.

![Figure 9.6. AdaBoost with decision stumps on 60 points labelled by the diagonal. Left: the first stump, with its nine mistakes ringed. Middle: point areas show the weights before round 2, the mistakes now heavy, and the second stump that responds to them. Right: the region the weighted vote of 40 stumps calls positive, an additive staircase that classifies every training point correctly.](https://raimbekovm.github.io/cs229-2018-notes/figures/09-decision-trees-ensembles/diagram_adaboost.svg)

The weights look arbitrary, and the notes justify them: AdaBoost is *forward stagewise
additive modelling* under the exponential loss $e^{-y f(x)}$, adding one stump at a time
and choosing each stump's coefficient to minimise that loss with the earlier ones fixed.
That derivation gives a coefficient of $\frac12 \log\bigl((1 - \operatorname{err}_m)/
\operatorname{err}_m\bigr)$ — half the vote above. The generator runs both versions side
by side for 40 rounds: they choose the same stump every round, their coefficients differ
by a factor of exactly 2, and since doubling every vote does not change the sign of the
sum, they make the same prediction everywhere on a fine grid.

On the diagonal data the first stump has weighted error $0.15$ and the second $0.18$.
Training error is $15\,\%$ after two rounds, $5\,\%$ after five, $1.7\,\%$ after ten,
and zero from round twenty, each stump being only a little better than chance — by round
40 the newest stump's weighted error is $0.35$. Freund and Schapire's bound guarantees
this kind of progress: training error is at most
$\prod_m 2\sqrt{\operatorname{err}_m(1 - \operatorname{err}_m)}$, which shrinks
geometrically as long as every stump beats chance by some margin. The generator asserts
the bound at every round; it is $0.068$ at round 20 and $0.023$ at round 40.

The additive structure that trees lack (Section 9.6) is exactly what boosting adds,
since the ensemble is a sum. That cuts both ways, and the notes say so where the lecture
does not: because each new model depends on the earlier ones, adding rounds raises the
risk of overfitting, unlike adding trees to a bagged ensemble. The number of rounds is
chosen on held-out data. [1:20:08](https://www.youtube.com/watch?v=wr9gUr-eWdA&t=4808s) The same stagewise idea with other losses gives
gradient boosting, and with engineering on top, XGBoost.

**In the official notes.** cs229-notes-ensemble, section 1, covers the bootstrap, aggregation and their combination
with trees; section 2 gives AdaBoost, states that it is forward stagewise additive modelling
under the exponential loss (leaving the derivation to *The Elements of Statistical
Learning*, section 10.4), and sketches gradient boosting.
