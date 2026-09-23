"""Dependence, and whether it helps the task.

Four distinct quantities live here and are never merged, because three of them
have been confused with one another in earlier reports:

* **sum of 45 pairwise mutual informations** -- a pairwise statistic. It is not
  total correlation, and a parity distribution has zero pairwise MI with
  positive total correlation.
* **actual total correlation**, estimated without a huge joint histogram from
  the model's own conditional vectors at its own generated prefixes:
  ``TC = sum_i [H(mean_b q_bi) - mean_b H(q_bi)]``. In population this is
  ``sum_i I(X_i; X_<i) = sum_i H(X_i) - H(X_1..X_10)``. In finite samples it is
  a nonnegative Jensen estimate, generally downward biased, identically zero at
  ``n = 1``, and its first editable position contributes exactly zero under a
  fixed conditioning prefix.
* **the parent-context drift decomposition** ``T_i = E_P KL(P_i || Q_i)`` and
  ``B_i = KL(E_P P_i || E_P Q_i)``. The ``Q`` average here uses *parent*
  prefixes, so it is not ``Q``'s generated marginal, and ``T - B`` measures
  context dependence of the distributional change -- not biological epistasis.
* **a discriminative interaction comparator**: a regularized three-class
  logistic model with and without pair terms. Its ``P(high)`` is a ranking
  score. It is not a normalized generative Potts model, its scores are never
  exponentiated into a density, and they are never fed to a yield calculation.

The useful result is a *task-relevant* dependency contribution that survives the
proximity challenge. A larger MI sum, a correlation across selected arms, or a
large Jensen gap does not establish that.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .her2_data import CANONICAL, CORE_LENGTH
from .her2_nf_contract import NF_SCHEMA
from .her2_runtime import require

ALPHABET = len(CANONICAL)
PAIRS = tuple(itertools.combinations(range(CORE_LENGTH), 2))
MAIN_EFFECTS = CORE_LENGTH * (ALPHABET - 1)
PAIR_FEATURES = len(PAIRS) * (ALPHABET - 1) ** 2


# ---------------------------------------------------------------------------
# generated marginals and total correlation
# ---------------------------------------------------------------------------

def generated_log_marginals(log_conditionals):
    """``(length, alphabet)`` log marginals ``log mhat_Qi`` from Q's own context vectors.

    ``logsumexp`` over the bank axis minus ``log n``, in float64. Counting drawn
    tokens and adding a pseudocount would answer a different question and would
    need an undocumented floor; using the full log-softmax vectors does not.

    A column every row assigns exactly zero probability has ``largest = -inf``,
    and ``-inf + log(0)`` is ``nan`` on the shifted path -- a bank supported only
    on residue zero produced 190 such columns. Those columns are ``-inf``, which
    is the correct log marginal, and they are returned as ``-inf``.
    """
    values = np.asarray(log_conditionals, dtype=np.float64)
    require(values.ndim == 3, "Expected an (N, length, alphabet) block of LOG conditional vectors")
    rows = values.shape[0]
    require(rows > 0, "No rows to average")
    largest = values.max(axis=0)
    empty = ~np.isfinite(largest)
    shifted_from = np.where(empty[None, :, :], 0.0, largest[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.exp(values - shifted_from).sum(axis=0)
        out = np.where(empty, 0.0, largest) + np.log(total) - math.log(rows)
    return np.where(empty, -np.inf, out)


def _entropy_from_log(log_probabilities):
    probabilities = np.exp(np.asarray(log_probabilities, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0,
                         probabilities * np.asarray(log_probabilities, dtype=np.float64), 0.0)
    return -terms.sum(axis=-1)


def total_correlation(log_conditionals):
    """``TC_hat`` plus its per-position terms and the marginals it used."""
    values = np.asarray(log_conditionals, dtype=np.float64)
    log_marginals = generated_log_marginals(values)
    marginal_entropy = _entropy_from_log(log_marginals)
    conditional_entropy = _entropy_from_log(values).mean(axis=0)
    terms = marginal_entropy - conditional_entropy
    return {"schema_version": NF_SCHEMA, "record_kind": "total_correlation",
            "rows": int(values.shape[0]),
            "total_correlation": float(terms.sum()),
            "per_position": [float(value) for value in terms],
            "marginal_entropy": [float(value) for value in marginal_entropy],
            "mean_conditional_entropy": [float(value) for value in conditional_entropy],
            "log_marginals": log_marginals,
            "estimand": "sum_i I_Q(X_i; X_<i) = sum_i H_Q(X_i) - H_Q(X_1..X_10)",
            "properties": ("nonnegative finite-sample Jensen estimate, generally DOWNWARD biased, "
                           "identically zero at n = 1. The first editable position's term is zero "
                           "for a fixed conditioning prefix. Nonnegativity does not certify "
                           "accuracy and a bootstrap does not remove bias."),
            "not": "the sum of pairwise mutual informations"}


def bootstrap_total_correlation(log_conditionals, *, seed, draws=200, batch_size=16):
    """Resample whole trajectories, recomputing all generated marginals each time."""
    values = np.asarray(log_conditionals, dtype=np.float64)
    require(values.ndim == 3 and values.shape[0] > 0, "No trajectories to bootstrap")
    n, length, alphabet = values.shape
    flat = values.reshape(n, -1)
    maximum = flat.max(axis=0)
    shifted = np.exp(flat - np.where(np.isfinite(maximum), maximum, 0.0))
    entropies = _entropy_from_log(values)
    rng = np.random.default_rng(int(seed))
    estimates = []
    for first in range(0, int(draws), int(batch_size)):
        count = min(int(batch_size), int(draws) - first)
        weights = np.stack([np.bincount(rng.integers(n, size=n), minlength=n)
                            for _ in range(count)]).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = maximum + np.log(weights @ shifted) - math.log(n)
        logs[:, ~np.isfinite(maximum)] = -np.inf
        marginal_h = _entropy_from_log(logs.reshape(count, length, alphabet))
        conditional_h = weights @ entropies / n
        estimates.extend((marginal_h - conditional_h).sum(axis=1).tolist())
    return {"rows": n, "draws": int(draws), "seed": int(seed), "values": estimates,
            "percentile_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "unit": "one entire generated trajectory and all its conditional vectors",
            "meaning": "Monte Carlo sensitivity of the finite-bank Jensen estimate; "
                       "not bias correction or trained-seed uncertainty"}


def pairwise_mutual_information(index, *, laplace=0.0):
    """All 45 pairwise MI values in nats, their sum, and the position marginals.

    Plug-in estimates from the realized draws. Their sum is a pairwise statistic
    and is reported under that name; it is not total correlation.
    """
    values = np.asarray(index).astype(np.int64, copy=False)
    require(values.ndim == 2 and values.shape[1] >= 2, "Expected an (N, length) core block")
    rows, length = values.shape
    require(rows > 1, "Pairwise MI needs more than one draw")
    alphabet = max(int(ALPHABET), int(values.max()) + 1)
    pairs = tuple(itertools.combinations(range(length), 2))
    counts = np.zeros((length, alphabet), dtype=np.float64)
    for position in range(length):
        counts[position] = np.bincount(values[:, position], minlength=alphabet)
    marginals = (counts + laplace) / (counts.sum(axis=1, keepdims=True) + laplace * alphabet)
    per_pair = {}
    total = 0.0
    for first, second in pairs:
        joint = np.zeros((alphabet, alphabet), dtype=np.float64)
        np.add.at(joint, (values[:, first], values[:, second]), 1.0)
        joint = (joint + laplace) / (rows + laplace * alphabet * alphabet)
        outer = np.outer(marginals[first], marginals[second])
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(joint > 0, joint * (np.log(joint) - np.log(outer)), 0.0)
        value = float(terms.sum())
        per_pair[f"{first}-{second}"] = value
        total += value
    return {"schema_version": NF_SCHEMA, "record_kind": "pairwise_mutual_information",
            "rows": int(rows), "pairs": len(pairs), "per_pair": per_pair,
            "sum": float(total), "marginals": marginals,
            "estimator": "plug-in from realized draws",
            "not": ("total correlation. Sample-size matched comparisons only; bank size is part "
                    "of the estimate and is reported with it.")}


def permutation_floor(index, *, seed, draws=20):
    """Independence floor: permute each column independently, keeping the marginals.

    A diagnostic floor for the MI sum, not a general bias correction. It removes
    dependence while preserving the empirical marginals, so the residual is the
    part of the estimate that finite samples produce even under independence.
    """
    values = np.asarray(index)
    generator = np.random.default_rng(int(seed))
    sums = []
    for _ in range(int(draws)):
        shuffled = np.empty_like(values)
        for position in range(values.shape[1]):
            shuffled[:, position] = values[generator.permutation(values.shape[0]), position]
        sums.append(pairwise_mutual_information(shuffled)["sum"])
    array = np.asarray(sums, dtype=np.float64)
    return {"draws": int(draws), "seed": int(seed), "rows": int(values.shape[0]),
            "mean": float(array.mean()), "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "quantiles": {"0.05": float(np.quantile(array, 0.05)),
                          "0.5": float(np.quantile(array, 0.5)),
                          "0.95": float(np.quantile(array, 0.95))},
            "status": ("a column-permutation independence floor. It preserves empirical marginals "
                       "and removes dependence; it is a diagnostic, not a bias correction, and "
                       "numerical spacing of RNG seeds is not an independence test.")}


# ---------------------------------------------------------------------------
# does dependence improve discrimination?
# ---------------------------------------------------------------------------

def score_decomposition(index, sequence_log_probability, log_marginals):
    """``s = ell/10``, ``g = sum_i log mhat_i(x_i)/10``, ``c = s - g`` per row.

    The identity ``s_Q - s_P = (g_Q - g_P) + (c_Q - c_P)`` is exact for the
    estimated marginals, which is why the three vectors are produced together.
    """
    values = np.asarray(index).astype(np.int64, copy=False)
    marginals = np.asarray(log_marginals, dtype=np.float64)
    require(marginals.ndim == 2, "Expected (length, alphabet) log marginals")
    require(values.ndim == 2 and values.shape[1] == marginals.shape[0],
            "the core block and the log marginals disagree about the sequence length")
    length = float(values.shape[1])
    s = np.asarray(sequence_log_probability, dtype=np.float64) / length
    positions = np.arange(values.shape[1])[None, :]
    g = marginals[positions, values].sum(axis=1) / length
    return {"full_score": s, "marginal_score": g, "residual_score": s - g,
            "definition": ("s = ell_Q/10; g = sum_i log mhat_Qi(x_i)/10; c = s - g. Length is "
                           "fixed, so dividing by ten is an order-preserving rescale.")}


def hybrid_scores(parent, policy):
    """The two declared diagnostic hybrids, on the same rows.

    ``s_P + (g_Q - g_P)`` and ``s_P + (c_Q - c_P)``. Ranking diagnostics only:
    they are not normalized generators, not internal causal interventions, and
    not valid inputs to a yield calculation.
    """
    return {"marginal_hybrid": parent["full_score"] + (policy["marginal_score"]
                                                       - parent["marginal_score"]),
            "residual_hybrid": parent["full_score"] + (policy["residual_score"]
                                                       - parent["residual_score"]),
            "status": ("ranking diagnostics. Never exponentiated into a density and never used "
                       "for expected yield.")}


def decomposition_report(index, parent_scores, policy_scores, parent_log_marginals,
                         policy_log_marginals, labels, classes):
    """Full/marginal/residual ranking, the hybrids, and within-class mean changes.

    Class-mean *scores* are decomposed additively, which is exact. AP differences
    are not added: AP is not linear in the score, and the schema deliberately has
    no field for an additive AP decomposition.
    """
    parent = score_decomposition(index, parent_scores, parent_log_marginals)
    policy = score_decomposition(index, policy_scores, policy_log_marginals)
    hybrids = hybrid_scores(parent, policy)
    positive = np.asarray(labels).astype(bool)
    class_values = np.asarray(classes)
    from .her2_nf_metrics import rank_block
    ranking = {
        "full_parent": rank_block(parent["full_score"], positive, prevalence_note=False),
        "full_policy": rank_block(policy["full_score"], positive, prevalence_note=False),
        "marginal_parent": rank_block(parent["marginal_score"], positive, prevalence_note=False),
        "marginal_policy": rank_block(policy["marginal_score"], positive, prevalence_note=False),
        "residual_parent": rank_block(parent["residual_score"], positive, prevalence_note=False),
        "residual_policy": rank_block(policy["residual_score"], positive, prevalence_note=False),
        "marginal_hybrid": rank_block(hybrids["marginal_hybrid"], positive, prevalence_note=False),
        "residual_hybrid": rank_block(hybrids["residual_hybrid"], positive, prevalence_note=False)}
    within_class = {}
    for name in sorted(set(class_values.tolist())):
        mask = class_values == name
        within_class[str(name)] = {
            "rows": int(mask.sum()),
            "full_change": float((policy["full_score"] - parent["full_score"])[mask].mean()),
            "marginal_change": float((policy["marginal_score"]
                                      - parent["marginal_score"])[mask].mean()),
            "residual_change": float((policy["residual_score"]
                                      - parent["residual_score"])[mask].mean())}
        block = within_class[str(name)]
        block["additivity_residual"] = float(block["full_change"] - block["marginal_change"]
                                             - block["residual_change"])
    return {"schema_version": NF_SCHEMA, "record_kind": "coupling_decomposition",
            "rows": int(np.asarray(index).shape[0]), "ranking": ranking,
            "within_class_mean_changes": within_class,
            "identity": "s_Q - s_P = (g_Q - g_P) + (c_Q - c_P), exactly, for the estimated "
                        "marginals",
            "forbidden": ("AP differences are not additive and no field here decomposes them. "
                          "Mean SCORE changes are decomposed; ranking metrics are reported per "
                          "score, not summed.")}


def bank_size_sensitivity(log_conditionals, *, sizes, seed):
    """Recompute the generated marginals and TC at several bank sizes.

    The strongest residual-score claims rest on ``mhat_Qi``, which is a finite
    average; this shows how much of the claim moves with ``n``.
    """
    values = np.asarray(log_conditionals, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    out = []
    for size in sorted(int(value) for value in sizes):
        require(size <= values.shape[0], f"Requested {size} rows from a bank of {values.shape[0]}")
        rows = generator.choice(values.shape[0], size=size, replace=False)
        block = total_correlation(values[rows])
        out.append({"rows": size, "total_correlation": block["total_correlation"],
                    "per_position": block["per_position"]})
    return {"sizes": [entry["rows"] for entry in out], "points": out, "seed": int(seed),
            "note": "subsamples of ONE bank; independent banks answer the repeat question"}


# ---------------------------------------------------------------------------
# parent-context drift
# ---------------------------------------------------------------------------

def parent_context_drift(teacher_probabilities, teacher_log_probabilities,
                         student_log_probabilities):
    """Per-position ``T_i``, ``B_i`` and ``T_i - B_i`` on a common parent bank.

    ``T_i`` averages the per-context KL; ``B_i`` is the KL between the averaged
    conditionals. Jensen gives ``B <= T``; the numeric check is a sanity check
    and is reported as one, not as a correctness proof.
    """
    probabilities = np.asarray(teacher_probabilities, dtype=np.float64)
    teacher_log = np.asarray(teacher_log_probabilities, dtype=np.float64)
    student_log = np.asarray(student_log_probabilities, dtype=np.float64)
    require(probabilities.shape == teacher_log.shape == student_log.shape
            and probabilities.ndim == 3, "Expected aligned (N, 10, 20) blocks")
    with np.errstate(invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * (teacher_log - student_log), 0.0)
    conditional = terms.sum(axis=2).mean(axis=0)
    parent_bar = probabilities.mean(axis=0)
    student_bar = np.exp(student_log).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        averaged_terms = np.where(parent_bar > 0,
                                  parent_bar * (np.log(parent_bar) - np.log(student_bar)), 0.0)
    averaged = averaged_terms.sum(axis=1)
    gap = conditional - averaged
    return {"schema_version": NF_SCHEMA, "record_kind": "parent_context_drift",
            "rows": int(probabilities.shape[0]),
            "conditional_T": [float(value) for value in conditional],
            "averaged_B": [float(value) for value in averaged],
            "gap_T_minus_B": [float(value) for value in gap],
            "sum_T": float(conditional.sum()), "sum_B": float(averaged.sum()),
            "B_le_T_holds": bool(np.all(gap >= -1e-9)),
            "first_position_gap": float(gap[0]),
            "interpretation": ("T - B measures context dependence of the distributional change "
                               "under PARENT prefixes. It is not biological epistasis, and the "
                               "Q average here is not Q's generated marginal."),
            "sanity_check_note": "B <= T by Jensen; checking it numerically is not a proof"}


class DriftAccumulator:
    """Streaming ``T``, ``B`` and ``T - B`` over a whole parent bank.

    The record is identical to :func:`parent_context_drift`'s, but the bank is
    consumed in chunks and only ``(length, alphabet)`` sufficient statistics are
    retained: two float64 sums for the averaged conditionals and one for the
    per-context KL. A 50k bank therefore costs a chunk at a time instead of two
    resident 80 MB blocks, so the measurement is not shrunk to fit an allocation.
    """

    def __init__(self):
        self.rows = 0
        self._conditional = None
        self._parent_bar = None
        self._student_bar = None

    def add(self, teacher_log_probabilities, student_log_probabilities):
        teacher_log = np.asarray(teacher_log_probabilities, dtype=np.float64)
        student_log = np.asarray(student_log_probabilities, dtype=np.float64)
        require(teacher_log.shape == student_log.shape and teacher_log.ndim == 3,
                "Expected aligned (N, length, alphabet) LOG conditional blocks")
        probabilities = np.exp(teacher_log)
        with np.errstate(invalid="ignore"):
            terms = np.where(probabilities > 0, probabilities * (teacher_log - student_log), 0.0)
        if self._conditional is None:
            shape = teacher_log.shape[1:]
            self._conditional = np.zeros(shape[0], dtype=np.float64)
            self._parent_bar = np.zeros(shape, dtype=np.float64)
            self._student_bar = np.zeros(shape, dtype=np.float64)
        self._conditional += terms.sum(axis=2).sum(axis=0)
        self._parent_bar += probabilities.sum(axis=0)
        self._student_bar += np.exp(student_log).sum(axis=0)
        self.rows += int(teacher_log.shape[0])
        return self.rows

    def finish(self):
        require(self.rows > 0, "the drift accumulator consumed no rows")
        conditional = self._conditional / float(self.rows)
        parent_bar = self._parent_bar / float(self.rows)
        student_bar = self._student_bar / float(self.rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            averaged_terms = np.where(parent_bar > 0,
                                      parent_bar * (np.log(parent_bar) - np.log(student_bar)), 0.0)
        averaged = averaged_terms.sum(axis=1)
        gap = conditional - averaged
        return {"schema_version": NF_SCHEMA, "record_kind": "parent_context_drift",
                "rows": int(self.rows),
                "conditional_T": [float(value) for value in conditional],
                "averaged_B": [float(value) for value in averaged],
                "gap_T_minus_B": [float(value) for value in gap],
                "sum_T": float(conditional.sum()), "sum_B": float(averaged.sum()),
                "B_le_T_holds": bool(np.all(gap >= -1e-9)),
                "first_position_gap": float(gap[0]),
                "estimator": ("streamed sufficient statistics over the FULL declared bank; no row "
                              "slice, and float64 accumulation throughout"),
                "interpretation": ("T - B measures context dependence of the distributional change "
                                   "under PARENT prefixes. It is not biological epistasis, and the "
                                   "Q average here is not Q's generated marginal."),
                "sanity_check_note": "B <= T by Jensen; checking it numerically is not a proof"}


# ---------------------------------------------------------------------------
# the discriminative interaction comparator
# ---------------------------------------------------------------------------

def design_matrix(index, *, pairwise):
    """Sparse reference-coded design: 190 main effects, plus 16,245 pair terms.

    Residue index 0 is the reference level at every position, so each position
    contributes 19 indicators and each pair contributes ``19 x 19``. The matrix
    is CSR because a dense 16k-column Hessian is the thing to avoid here.

    The residue codes arrive as ``int8`` -- that is how a bank is stored -- and
    every column offset here is larger than ``int8`` holds. NumPy 2 raises
    ``OverflowError`` on ``int8_array + 133`` rather than wrapping, and a pair
    product would wrap silently, so the codes are widened to ``int64`` once,
    before any arithmetic touches them.
    """
    from scipy import sparse

    values = np.asarray(index).astype(np.int64, copy=False)
    require(values.ndim == 2 and values.shape[1] == CORE_LENGTH, "Expected an (N, 10) core block")
    require(bool(((values >= 0) & (values < ALPHABET)).all()),
            f"a residue code outside 0..{ALPHABET - 1} reached the design matrix; reference "
            "coding would silently place it in another position's column block")
    rows = values.shape[0]
    row_ids, column_ids = [], []
    for position in range(CORE_LENGTH):
        active = values[:, position] > 0
        row_ids.append(np.flatnonzero(active))
        column_ids.append(position * (ALPHABET - 1) + values[active, position] - 1)
    width = MAIN_EFFECTS
    if pairwise:
        for offset, (first, second) in enumerate(PAIRS):
            active = (values[:, first] > 0) & (values[:, second] > 0)
            row_ids.append(np.flatnonzero(active))
            block = ((values[active, first] - 1) * (ALPHABET - 1) + (values[active, second] - 1))
            column_ids.append(MAIN_EFFECTS + offset * (ALPHABET - 1) ** 2 + block)
        width += PAIR_FEATURES
    row_index = np.concatenate(row_ids).astype(np.int64, copy=False)
    column_index = np.concatenate(column_ids).astype(np.int64, copy=False)
    require(bool((column_index < width).all()) and bool((column_index >= 0).all()),
            "a design column landed outside the declared feature width")
    data = np.ones(row_index.size, dtype=np.float64)
    return sparse.csr_matrix((data, (row_index, column_index)), shape=(rows, width))


def enumerate_features(core, *, pairwise):
    """The active column ids of one core, enumerated independently of the encoder.

    Written from the coding rule rather than from :func:`design_matrix`, so a
    test can compare the two and an off-by-one in either shows up as a set
    difference instead of as a slightly worse classifier.
    """
    values = [int(value) for value in np.asarray(core).reshape(-1)]
    require(len(values) == CORE_LENGTH, f"Expected {CORE_LENGTH} residue codes")
    columns = []
    for position, value in enumerate(values):
        if value > 0:
            columns.append(position * (ALPHABET - 1) + value - 1)
    if pairwise:
        for offset, (first, second) in enumerate(PAIRS):
            if values[first] > 0 and values[second] > 0:
                columns.append(MAIN_EFFECTS + offset * (ALPHABET - 1) ** 2
                               + (values[first] - 1) * (ALPHABET - 1) + (values[second] - 1))
    return sorted(columns)


def fit_interaction_classifier(train_index, train_classes, *, development_index,
                               development_classes, pairwise, regularization_grid=(0.01, 0.1, 1.0),
                               max_iterations=200, seed=0, class_order=("low", "mid", "high")):
    """Regularized three-class logistic fit; regularization tuned on development rows only.

    The solver is chosen for a sparse multinomial problem and the removed
    ``multi_class`` argument is not passed: scikit-learn >= 1.9 is multinomial by
    default for three or more classes. ``P(high)`` is the ranking score.
    """
    from sklearn.linear_model import LogisticRegression

    import sklearn

    order = list(class_order)
    train_y = np.asarray([order.index(value) for value in np.asarray(train_classes)])
    development_y = np.asarray([order.index(value)
                                for value in np.asarray(development_classes)])
    require(len(np.unique(train_y)) == len(order),
            "the interaction comparator needs all three classes in its training rows")
    train_x = design_matrix(train_index, pairwise=pairwise)
    development_x = design_matrix(development_index, pairwise=pairwise)
    attempts = []
    best = None
    for strength in regularization_grid:
        model = LogisticRegression(penalty="l2", C=float(strength), solver="saga",
                                   max_iter=int(max_iterations), tol=1e-4, random_state=int(seed))
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(development_x)
        high = probabilities[:, order.index("high")]
        from .her2_nf_metrics import average_precision
        score = average_precision(high, development_y == order.index("high"))
        attempts.append({"C": float(strength), "development_average_precision": float(score),
                         "converged": bool(np.all(np.asarray(model.n_iter_) < int(max_iterations))),
                         "iterations": [int(value) for value in np.atleast_1d(model.n_iter_)]})
        if best is None or score > best["score"]:
            best = {"C": float(strength), "score": float(score), "model": model}
    return {"schema_version": NF_SCHEMA, "record_kind": "interaction_classifier",
            "pairwise": bool(pairwise), "features": int(train_x.shape[1]),
            "main_effects": MAIN_EFFECTS, "pair_features": PAIR_FEATURES if pairwise else 0,
            "class_order": order, "attempts": attempts,
            "selected_C": best["C"], "development_average_precision": best["score"],
            "model": best["model"], "sklearn_version": str(sklearn.__version__),
            "solver": "saga", "tuned_on": "development/C rows only",
            "status": ("a DISCRIMINATIVE interaction comparator. P(high) is a ranking score; it is "
                       "not a normalized generative Potts model, is never exponentiated into a "
                       "density, and is never an input to a yield calculation.")}


def classifier_scores(fit, index, *, class_order=None):
    """``P(high)`` on new rows, carrying the reminder of what it is not."""
    order = list(class_order or fit["class_order"])
    matrix = design_matrix(index, pairwise=fit["pairwise"])
    probabilities = fit["model"].predict_proba(matrix)
    return {"p_high": probabilities[:, order.index("high")],
            "all_class_probabilities": probabilities, "class_order": order,
            "status": "discriminative ranking score; not a generative probability"}
