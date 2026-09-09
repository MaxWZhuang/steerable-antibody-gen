"""Statistical protocol (v2) for the frozen HCDR3 contrast benchmark.

This module reads the *cached* score artifact and recomputes the endpoints; it
never touches a model, so a full re-analysis costs seconds and cannot perturb
the frozen inputs that `contrast_scoring` produced.

Why it exists. The v1 headline was `mean_group_concordance` pooled over every
scored group, and on the first real run that number was dominated twice over:
579 of 956 groups contributed exactly one pairwise comparison (so their group
score could only be 0 or 1), while a single TIGIT panel supplied 50,086 of the
57,903 comparisons. Averaging groups and pooling pairs therefore answer two
different questions, and neither answers "does this checkpoint rank variants
across design backgrounds".

The endpoints here separate those questions deliberately:

- **Primary** (`primary_endpoint`): equal-target, equal-group concordance. Every
  target contributes equally, and within a target every background contributes
  equally, so no single oversized panel can carry the benchmark.
- **Secondary** (`pair_pooled_by_target`): pair-pooled concordance *within* a
  target. Comparisons are never pooled between groups or between targets --
  variants in different backgrounds were never measured against each other.
- **Diagnostics**: `weight_concentration` reports how concentrated the pair mass
  is, and `drop_largest_group` re-runs the endpoints without each target's
  biggest panel.

Every reported interval is a cluster bootstrap over *groups*, stratified by
target, because groups -- not pairs -- are the independent unit.

Floors (change-control Rule 4). A constant scorer earns exactly 0.500 on both
endpoints: `contrasts.concordance` excludes measured ties and counts score ties
as half, so a scorer with no resolution lands on the floor by construction, not
by convention. `permutation_null` gives the empirical null for a given set of
group sizes, and `power_simulation` says what effect that set could detect.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist, fmean, pstdev
from typing import Any, Callable, Iterable, Sequence

from .contrasts import concordance

#: Two variants with one comparable measurement pair is the smallest rankable
#: panel. Single-pair groups are noisy but carry real information, so they are
#: kept in the primary population; raising this threshold defines a *secondary*
#: population and must be pre-registered rather than chosen after looking.
DEFAULT_MIN_VARIANTS = 2

#: Pairs per group above which the power simulation switches from exact
#: Bernoulli draws to a normal approximation (one real panel has >50k pairs).
_EXACT_BINOMIAL_LIMIT = 200


@dataclass(frozen=True)
class GroupStat:
    """One contrast group's recomputed ranking counts."""

    target: str
    group_id: str
    concordant: int
    discordant: int
    score_ties: int
    comparable_pairs: int
    n_variants: int

    @property
    def concordance(self) -> float:
        """Order agreement with score ties counted half; 0.5 for a constant scorer."""
        if not self.comparable_pairs:
            raise ValueError("group {!r} has no comparable pairs".format(self.group_id))
        return (self.concordant + self.score_ties / 2) / self.comparable_pairs


def _directions(manifest_doc: dict) -> dict[str, str]:
    """Map group id to measurement direction, refusing to guess a missing one."""
    out = {}
    for group in manifest_doc["groups"]:
        direction = group.get("direction")
        if direction not in ("higher", "lower"):
            raise ValueError(
                "group {!r} has no usable measurement direction; refusing to assume "
                "one because the sign decides every concordance".format(group.get("group_id"))
            )
        out[group["group_id"]] = direction
    return out


def group_stats(scores_doc: dict, manifest_doc: dict, *, score_key: str = "native",
                min_variants: int = DEFAULT_MIN_VARIANTS, fold: str | None = None,
                verify: bool = True) -> list[GroupStat]:
    """Recompute per-group ranking counts from the cached per-variant scores.

    With ``score_key="native"`` and ``verify=True`` the recomputation is checked
    against the counts the scoring run froze into the artifact. A disagreement
    means the artifact and this code no longer describe the same experiment, so
    it raises instead of silently reporting the newer arithmetic.
    """
    directions = _directions(manifest_doc)
    stats: list[GroupStat] = []
    for group in scores_doc["groups"]:
        if fold is not None and group.get("fold") != fold:
            continue
        group_id = group["group_id"]
        if group_id not in directions:
            raise ValueError("group {!r} is scored but absent from the manifest".format(group_id))
        variants = group["variants"]
        if len(variants) < min_variants:
            continue
        values = [float(v["measurement"]) for v in variants]
        scores = [float(v["scores"][score_key]) for v in variants]
        counts = concordance(values, scores, direction=directions[group_id])
        if verify and score_key == "native":
            cached = group["native_ranking"]
            for key in ("concordant", "discordant", "score_ties",
                        "measurement_ties", "comparable_pairs"):
                if cached.get(key) != counts[key]:
                    raise ValueError(
                        "recomputed {} for group {!r} disagrees with the frozen artifact "
                        "({} vs {}); the cached scores and this code describe different "
                        "experiments".format(key, group_id, counts[key], cached.get(key))
                    )
        if not counts["comparable_pairs"]:
            continue
        stats.append(GroupStat(target=group["target"], group_id=group_id,
                               concordant=counts["concordant"], discordant=counts["discordant"],
                               score_ties=counts["score_ties"],
                               comparable_pairs=counts["comparable_pairs"],
                               n_variants=len(variants)))
    return stats


# --- endpoints -------------------------------------------------------------

def _equal_target_mean(pairs: Sequence[tuple[str, float]]) -> float:
    """Mean over targets of the mean over that target's groups."""
    if not pairs:
        raise ValueError("no groups survive the population filters")
    by_target: dict[str, list[float]] = defaultdict(list)
    for target, value in pairs:
        by_target[target].append(value)
    return fmean([fmean(values) for _, values in sorted(by_target.items())])


def _as_pairs(stats: Iterable[GroupStat]) -> list[tuple[str, float]]:
    return [(s.target, s.concordance) for s in stats]


def equal_group_by_target(stats: Iterable[GroupStat]) -> dict[str, float]:
    """Per target, the unweighted mean of its groups' concordances."""
    by_target: dict[str, list[float]] = defaultdict(list)
    for stat in stats:
        by_target[stat.target].append(stat.concordance)
    return {target: fmean(values) for target, values in sorted(by_target.items())}


def primary_endpoint(stats: Iterable[GroupStat]) -> float:
    """Equal-target, equal-group concordance. Floor 0.5, perfect 1.0."""
    return _equal_target_mean(_as_pairs(stats))


def primary_endpoint_se(stats: Iterable[GroupStat]) -> float:
    """Analytic standard error of `primary_endpoint` treating groups as the unit."""
    by_target: dict[str, list[float]] = defaultdict(list)
    for stat in stats:
        by_target[stat.target].append(stat.concordance)
    if not by_target:
        raise ValueError("no groups survive the population filters")
    variance = sum(pstdev(v) ** 2 / len(v) for v in by_target.values() if len(v) > 1)
    return math.sqrt(variance) / len(by_target)


def pair_pooled_by_target(stats: Iterable[GroupStat]) -> dict[str, float]:
    """Pair-pooled concordance within each target. Never pooled across targets."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for stat in stats:
        totals[stat.target][0] += stat.concordant + stat.score_ties / 2
        totals[stat.target][1] += stat.comparable_pairs
    return {target: won / total for target, (won, total) in sorted(totals.items()) if total}


def weight_concentration(stats: Iterable[GroupStat]) -> dict[str, dict[str, Any]]:
    """How much of each target's pair mass sits in its largest panels."""
    by_target: dict[str, list[GroupStat]] = defaultdict(list)
    for stat in stats:
        by_target[stat.target].append(stat)
    report = {}
    for target, group in sorted(by_target.items()):
        ordered = sorted(group, key=lambda s: (-s.comparable_pairs, s.group_id))
        total = sum(s.comparable_pairs for s in ordered)
        report[target] = {
            "groups": len(ordered),
            "comparable_pairs": total,
            "largest_group_id": ordered[0].group_id,
            "largest_group_pair_share": ordered[0].comparable_pairs / total,
            "top5_pair_share": sum(s.comparable_pairs for s in ordered[:5]) / total,
            "herfindahl": sum((s.comparable_pairs / total) ** 2 for s in ordered),
            "median_pairs_per_group": sorted(s.comparable_pairs for s in ordered)[len(ordered) // 2],
        }
    return report


def drop_largest_group(stats: Sequence[GroupStat]) -> list[GroupStat]:
    """Every group except each target's largest panel, in the original order."""
    largest = {}
    for stat in stats:
        current = largest.get(stat.target)
        if current is None or stat.comparable_pairs > current.comparable_pairs:
            largest[stat.target] = stat
    dropped = {(s.target, s.group_id) for s in largest.values()}
    return [s for s in stats if (s.target, s.group_id) not in dropped]


# --- inference -------------------------------------------------------------

def _quantile(ordered: Sequence[float], q: float) -> float:
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _bootstrap_pairs(pairs: Sequence[tuple[str, float]], *, n_resamples: int, seed: int,
                     alpha: float) -> dict[str, Any]:
    """Cluster bootstrap over groups, resampling within each target stratum."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(seed)
    by_target: dict[str, list[float]] = defaultdict(list)
    for target, value in pairs:
        by_target[target].append(value)
    strata = [(target, values) for target, values in sorted(by_target.items())]
    draws = []
    for _ in range(n_resamples):
        draws.append(fmean([
            fmean([values[rng.randrange(len(values))] for _ in range(len(values))])
            for _, values in strata
        ]))
    draws.sort()
    return {"estimate": _equal_target_mean(pairs),
            "low": _quantile(draws, alpha / 2), "high": _quantile(draws, 1 - alpha / 2),
            "n_groups": len(pairs), "n_resamples": n_resamples, "seed": seed}


def cluster_bootstrap(stats: Sequence[GroupStat], *, n_resamples: int = 20000, seed: int = 42,
                      alpha: float = 0.05) -> dict[str, Any]:
    """Target-stratified cluster-bootstrap interval for `primary_endpoint`."""
    return _bootstrap_pairs(_as_pairs(stats), n_resamples=n_resamples, seed=seed, alpha=alpha)


def permutation_null(scores_doc: dict, manifest_doc: dict, *, n_permutations: int = 2000,
                     seed: int = 42, score_key: str = "native",
                     min_variants: int = DEFAULT_MIN_VARIANTS, fold: str | None = None,
                     alpha: float = 0.05) -> dict[str, Any]:
    """Empirical null from shuffling scores among variants inside each group.

    Permuting *within* a group preserves that group's size, its measured ties and
    its score multiset, so the null keeps the dependence structure that a
    per-pair coin-flip model would throw away.
    """
    directions = _directions(manifest_doc)
    panels = []
    for group in scores_doc["groups"]:
        if fold is not None and group.get("fold") != fold:
            continue
        variants = group["variants"]
        if len(variants) < min_variants:
            continue
        values = [float(v["measurement"]) for v in variants]
        scores = [float(v["scores"][score_key]) for v in variants]
        if concordance(values, scores, direction=directions[group["group_id"]])["comparable_pairs"]:
            panels.append((group["target"], values, scores, directions[group["group_id"]]))
    if not panels:
        raise ValueError("no groups survive the population filters")

    observed = _equal_target_mean([
        (target, concordance(values, scores, direction=direction)["concordance"])
        for target, values, scores, direction in panels
    ])
    rng = random.Random(seed)
    draws = []
    for _ in range(n_permutations):
        shuffled = []
        for target, values, scores, direction in panels:
            permuted = list(scores)
            rng.shuffle(permuted)
            shuffled.append((target, concordance(values, permuted, direction=direction)["concordance"]))
        draws.append(_equal_target_mean(shuffled))
    draws.sort()
    extreme = sum(1 for d in draws if abs(d - 0.5) >= abs(observed - 0.5))
    return {"observed": observed, "null_mean": fmean(draws),
            "low": _quantile(draws, alpha / 2), "high": _quantile(draws, 1 - alpha / 2),
            "p_value": (1 + extreme) / (1 + len(draws)),
            "n_permutations": n_permutations, "seed": seed}


def power_simulation(stats: Sequence[GroupStat], *, effect: float = 0.03,
                     n_simulations: int = 2000, seed: int = 42,
                     alpha: float = 0.05) -> dict[str, Any]:
    """Probability this group population detects a `effect`-sized shift off the floor.

    Each group's pairs are simulated as independent draws at ``0.5 + effect``.
    Real pairs inside a panel are *not* independent -- they come from ranking the
    same handful of variants -- so the simulated variance is too small and the
    returned power is an upper bound. A design that fails to clear the bar here
    cannot clear it in reality.
    """
    if not 0.0 <= 0.5 + effect <= 1.0:
        raise ValueError("effect must keep the success probability inside [0, 1]")
    rng = random.Random(seed)
    critical = NormalDist().inv_cdf(1 - alpha / 2)
    probability = 0.5 + effect
    sizes = [(s.target, s.comparable_pairs) for s in stats]
    if not sizes:
        raise ValueError("no groups survive the population filters")
    rejections = 0
    for _ in range(n_simulations):
        simulated = []
        for target, pairs in sizes:
            if pairs <= _EXACT_BINOMIAL_LIMIT:
                wins = sum(1 for _ in range(pairs) if rng.random() < probability)
            else:  # normal approximation keeps 50k-pair panels tractable
                mean = pairs * probability
                spread = math.sqrt(pairs * probability * (1 - probability))
                wins = min(pairs, max(0, int(round(rng.gauss(mean, spread)))))
            simulated.append((target, wins / pairs))
        estimate = _equal_target_mean(simulated)
        by_target: dict[str, list[float]] = defaultdict(list)
        for target, value in simulated:
            by_target[target].append(value)
        variance = sum(pstdev(v) ** 2 / len(v) for v in by_target.values() if len(v) > 1)
        error = math.sqrt(variance) / len(by_target)
        if error > 0 and abs(estimate - 0.5) > critical * error:
            rejections += 1
    return {"power": rejections / n_simulations, "effect": effect, "alpha": alpha,
            "n_groups": len(sizes), "n_simulations": n_simulations, "seed": seed,
            "caveat": "upper bound: pairs within a group are simulated independently, "
                      "so the true variance is larger and the true power lower"}


def paired_contrast(stats_a: Sequence[GroupStat], stats_b: Sequence[GroupStat], *,
                    n_resamples: int = 20000, seed: int = 42,
                    alpha: float = 0.05) -> dict[str, Any]:
    """Per-group difference `a - b` on the groups both populations share."""
    lookup = {(s.target, s.group_id): s for s in stats_b}
    differences = [(a.target, a.concordance - lookup[(a.target, a.group_id)].concordance)
                   for a in stats_a if (a.target, a.group_id) in lookup]
    if not differences:
        raise ValueError("the two populations share no groups")
    return _bootstrap_pairs(differences, n_resamples=n_resamples, seed=seed, alpha=alpha)


def analyze(scores_doc: dict, manifest_doc: dict, *, min_variants: int = DEFAULT_MIN_VARIANTS,
            fold: str | None = None, n_resamples: int = 20000, n_permutations: int = 2000,
            n_simulations: int = 2000, effect: float = 0.03, seed: int = 42,
            alpha: float = 0.05) -> dict[str, Any]:
    """Run the whole v2 protocol over a cached scoring run."""
    native = group_stats(scores_doc, manifest_doc, score_key="native",
                         min_variants=min_variants, fold=fold)
    substituted = group_stats(scores_doc, manifest_doc, score_key="substituted_unmeasured",
                              min_variants=min_variants, fold=fold, verify=False)
    without_largest = drop_largest_group(native)
    return {
        "protocol": {"primary": "equal-target, equal-group concordance",
                     "secondary": "pair-pooled concordance within target",
                     "floor": 0.5, "min_variants": min_variants, "fold": fold,
                     "seed": seed, "alpha": alpha},
        "population": {"groups": len(native),
                       "comparable_pairs": sum(s.comparable_pairs for s in native),
                       "targets": sorted({s.target for s in native})},
        "primary": cluster_bootstrap(native, n_resamples=n_resamples, seed=seed, alpha=alpha),
        "primary_se": primary_endpoint_se(native),
        "by_target": {
            "equal_group": equal_group_by_target(native),
            "pair_pooled": pair_pooled_by_target(native),
            "bootstrap": {
                target: cluster_bootstrap([s for s in native if s.target == target],
                                          n_resamples=n_resamples, seed=seed, alpha=alpha)
                for target in sorted({s.target for s in native})
            },
        },
        "weight_concentration": weight_concentration(native),
        "leave_largest_group_out": cluster_bootstrap(
            without_largest, n_resamples=n_resamples, seed=seed, alpha=alpha),
        "permutation_null": permutation_null(scores_doc, manifest_doc,
                                             n_permutations=n_permutations, seed=seed,
                                             min_variants=min_variants, fold=fold, alpha=alpha),
        "power": power_simulation(native, effect=effect, n_simulations=n_simulations,
                                  seed=seed, alpha=alpha),
        "native_minus_substituted": paired_contrast(native, substituted,
                                                    n_resamples=n_resamples, seed=seed,
                                                    alpha=alpha),
        "claims": {
            "antigen_correctness_established": False,
            "interpretation": "native-variant ranking on a frozen scoring run; the "
                              "substituted-antigen arm has no measured labels and is an "
                              "input ablation, not a specificity measurement",
        },
    }
