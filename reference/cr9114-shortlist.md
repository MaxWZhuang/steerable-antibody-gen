# Quality-constrained shortlist comparison

2026-09-17. **The locked selector improved shortlist diversity, but failed the
combined affinity/diversity screen at K=32.** No model was trained or promoted.
The training-only calibration chose admission from the top 4*K SFT scores.

| Selection | K | Mean measured H1 affinity | Mean pair Hamming | Minimum pair Hamming | Above training quality threshold |
|---|---:|---:|---:|---:|---:|
| Ordinary SFT ranking | 16 | 9.572041 | 3.6750 | 1 | 16 |
| Diversity-aware | 16 | 9.564898 | 5.2333 | 3 | 16 |
| Ordinary SFT ranking | 32 | 9.559854 | 4.2964 | 1 | 31 |
| Diversity-aware | 32 | 9.537314 | 5.6996 | 3 | 29 |

The affinity changes were -0.007143 and -0.022540. The latter exceeds the
predeclared 0.02 engineering tolerance. Both budgets improved mean Hamming by
more than 0.25 and eliminated pairs at distance <=1. Among quality-qualified
selected candidates, the diverse sets also had minimum distance 3; their mean
Hamming distances were 5.2333 and 5.5862. More sequence variety did not guarantee
preserved affinity. The tolerance is not a biological noninferiority margin.

## How this was done

The code and [protocol](../specs/cr9114_shortlist.md) were committed as `353ded5`
before scoring. Four disjoint training cohorts, each with 2,048 identities,
calibrated admission multipliers {1,2,4,8}. Each option had to retain mean affinity
within 0.02 for every cohort and both budgets; the acceptable option with greatest
mean Hamming gain was chosen. Ties favored the smaller window. These are model
training records, so calibration success alone is not independent validation.

The choice was saved before scoring 2,048 new development identities. Selection
accepted only genotype and model score; assay labels were joined after the selected
identities were saved. Greedy selection began with the highest score, then chose
the greatest minimum distance to the existing set, breaking ties by score and
identity. This is a likelihood-admission heuristic, not an affinity constraint.

An [independent audit](../scripts/audit_cr9114_shortlist.py) reimplemented selection,
recomputed calibration and key metrics, verified five saved artifact hashes,
and checked identity-defined cohorts and earlier-evaluation exclusions. It passed.
The selector tests passed (13 tests). [Compact evidence](evidence/cr9114-shortlist-2026-09-17.json)
includes all selected identities, calibration cells, input hashes and results.

## Limits and next comparison

This is one development cohort in one antibody lineage with three development
blocks. The ordinary top-16 all belong to block 7; diversity-aware selection
spreads them across blocks 7 and 11. That does not establish generalization to
another lineage or antigen. Reserved test labels remain untouched, and 3,046
eligible development identities have not yet been evaluated.

A post-hoc split-locus decomposition checks whether mixing blocks alone explains
the gain. Excluding the four split-defining sites, mean pair Hamming still rises
from 3.6750 to 4.5833 at K=16 and from 3.9456 to 4.8125 at K=32. Block mixing
explains part, but not all, of the observed increase. This diagnostic changes no
selection rule or screening decision.

The four-arm, three-seed regularization comparison was specified before seeing
these results. It keeps the affinity objective fixed and compares no regularizer,
explicit SFT-reference KL, KL+entropy, and KL+trainable embedding cosine. The same
locked selector will be evaluated alongside ordinary ranking; its failure is
retained, and the development cohort will be explicitly marked reused.
