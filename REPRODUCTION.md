# What this implementation reproduces

This repository implements the **two-view semi-supervised BBCSport experiment** in Section 6.2.1 of Xu et al. (2016). It is an independent reimplementation, not an author-provided package. It does not reproduce the paper's MIL, Reuters, or outlier-detection experiments, or every comparison baseline.

## Data and protocol

- Original creator-provided BBCSport term-frequency matrix: **737 documents × 4,613 words**, with five categories.
- The original data already includes stemming, stop-word removal and low-frequency filtering. We use those features, rather than fitting a new text vectorizer.
- For each class, a seeded random permutation sends ceil(n/2) documents to training. This yields **370 training / 367 test articles**, matching Table 2. This rounding convention is explicit because classes have odd sizes.
- Exactly **two training labels per class** are supplied. The other **360** training labels are replaced with a sentinel before the learner is called.
- Independently shuffle the vocabulary, split it into **2,306 / 2,307** word features, and L1-normalize each document within each view. The same document IDs occur in both views.
- One binary classifier per category (one-versus-rest). Final category scores average the two views; the largest averaged score gives a category prediction.
- The default experiment declares **seeds 0–9 before training**. Each method uses the same splits and feature views. The demo recording is always the **first** seed, never the best-scoring seed.
- Primary metric: mean of the five one-versus-rest average precision scores, multiplied by 100. Sample standard deviation across splits uses `ddof=1`. Accuracy is a separate secondary measure.

The feature vocabulary was supplied by the dataset creator and predates our train/test splitting, as in the paper. No test labels are used for normalization, fitting, stopping, or parameter selection.

## Candidate labelings: Algorithms 1–3 and Equation 3

Each view starts from a linear SVM fitted to the ten supplied labels. For each binary category, its initial positive fraction is estimated using supplied labels only (2/10 = 0.2).

For each other-view score vector, we generate 11 candidates with fractions `0.2 + [-0.10, -0.08, ..., +0.10]`, as in Section 4.2. To construct a candidate, retain supplied labels and rank unknown examples by the teacher's score. Assign the top `floor(fraction × unknown_count + 0.5)` as positive; all remaining unknowns are negative. Ties follow stable document order. The paper does not prescribe the integer rounding convention.

Candidate sets accumulate across outer rounds. Exact duplicate labelings **within the same bias group** are merged. This preserves the optimization: duplicate kernels in an L1 history group can be represented by their combined weight. Duplicates across different bias groups are retained, since those groups have distinct regularization.

Outer fitting stops when no new candidates are added, both per-view objective changes are relatively below `1e-6`, or the configured maximum of ten rounds is reached. Binary categories may stop at different times; the replay holds their last learned state on later global rounds.

## What the kernel solver actually optimizes

For view features X, the input kernel is **K = XXᵀ + 11ᵀ**, including the regularized bias term in Equation 5. Each candidate yₘ supplies the input-output kernel **Qₘ = K ⊙ (yₘyₘᵀ)**. These are the paper's label-dependent kernels; this is not majority voting or averaging independently trained pseudo-label classifiers.

For fixed candidate coefficients d, Equation 9 is solved as:

```text
minimize   0.5 αᵀ [Σₘ dₘ Qₘ + I/C] α
subject to α ≥ 0,  1ᵀα = 1.
```

This is a strictly convex QP. The implementation uses a warm-started active-set solver, with OSQP as a fallback. An inactive coefficient can enter only when its reduced gradient violates the KKT condition; a negative free-set target is approached only until a bound is hit. Tests compare this solver with both OSQP and an independent SLSQP optimizer.

For **CoL(1-layer)**, coefficients satisfy `d ≥ 0` and `sum(d) ≤ 1`.

For **CoL(2-layer)** in the two-view setting, the other-view dimension has one node. The coefficient constraint is:

```text
sqrt(sum_over_bias (sum_over_round d[bias, round])²) ≤ 1.
```

Thus history uses an L1 norm and bias groups an L2 norm. This is the two-view reduction of the paper's grouped formulation; with just two views the third view layer adds no additional node, as explained near Table 1.

The solver alternates QP fitting with the analytical specialization of Equations 10–17 for fixed primal SVM norms. Multiplicative weight updates can approach sparse solutions slowly. When necessary, SLSQP polishes **the same convex kernel-weight objective and the same constraint**, using exact QP values and envelope gradients. The gradient for a candidate is `-0.5 αᵀQₘα`. This is a numerical implementation choice, not a replacement learning objective. The polishing budget is at most `max(100, 2 × inner_iterations)`.

Convergence is checked with a **global saddle gap**, not just a small change in objective. For one layer the kernel support function is `maxₘ αᵀQₘα`; for grouped weights it is the L2 norm of the maximum energy in each bias group. Half the difference between that support and the current weighted energy gives the gap. Requested relative tolerance: **1e-5**, scaled by the rho-SVM value. Unconverged fits are explicitly counted in results rather than silently marked as exact.

Prediction uses:

```text
βᵢ = αᵢ Σₘ dₘ yₘᵢ
ρ  = αᵀ [Σₘ dₘQₘ + I/C] α
f(x) = Σᵢ βᵢ [k(xᵢ,x) + 1] / ρ.
```

The rho normalization is stated explicitly in the authors' earlier [2012 co-labeling paper, Section IV-B](https://wenli-vision.github.io/papers/Li_ICDM2012.pdf). It is important when averaging views and comparing one-versus-rest category scores. A unit test verifies recovery of the standard single-label rho-SVM and its KKT prediction scale.

## Declared choices that prevent an exact-number claim

- The journal paper does not fully specify C selection, random seeds, or numerical tolerances. We fix **C=10** before evaluation for all methods, rather than choose it by looking at test scores.
- Initialization and the known-label baseline use linear C-SVMs. The co-labeling learner uses the squared-hinge rho-SVM objective in the paper.
- The co-training baseline exchanges the five highest and five lowest ranked still-unassigned examples per binary class/view per round. This is an explicit classical confident-sample selection policy, consistent with the earlier paper's five-positive/five-negative example, rather than a claim to recover the authors' undocumented baseline configuration.
- The initial candidate source, deterministic tie/rounding conventions, duplicate merging and stopping details are declared here.
- A finite numerical solver and its convergence diagnostics are part of every saved report.

The independent experiment may perform differently from Table 3. Scores in the demo's **Our implementation** section always come from this code. The paper's 80.65 / 90.31 / 94.81 / 95.44 scores are labeled as a separate published comparison.

## Validation

`python -m pytest -q` checks known-answer preservation and exact class proportions, stable tie handling, grouped/flat analytical updates against a separate constrained optimizer, QP optimality against two numerical references, global MKL gap convergence, rho-SVM prediction recovery, and the original data/split/view dimensions.

Browser checks cover the real replay controls, source-article retrieval, answer reveal, candidate table, numerical result labels, projector/phone layouts and the absence of external network dependencies. The committed recording consists of genuine model outputs and derived word clues; full original article text remains in ignored local data.

## References

1. X. Xu, W. Li, D. Xu and I. W. Tsang. *Co-Labeling for Multi-View Weakly Labeled Learning*. IEEE TPAMI 38(6):1113–1125, 2016. [DOI](https://doi.org/10.1109/TPAMI.2015.2476813).
2. W. Li, L. Duan, I. W.-H. Tsang and D. Xu. *Co-Labeling: A New Multi-view Learning Approach for Ambiguous Problems*. ICDM, 2012. [Author PDF](https://wenli-vision.github.io/papers/Li_ICDM2012.pdf).
3. D. Greene and P. Cunningham. *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*. ICML, 2006. [Dataset](https://derekgreene.com/bbc/).
