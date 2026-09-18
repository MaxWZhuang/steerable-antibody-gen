"""Render completed HER2 experiment artifacts; performs no fitting or selection.

Run from the repository root. Matplotlib is a reporting dependency, installed
separately from the training environment requirements.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/her2_posttrain_20260918'
REFERENCE = ROOT / 'reference'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_runs(document):
    return {name: {key: value for key, value in entry.items() if key != 'history'}
            for name, entry in document['runs'].items()}


def render():
    result_path = RUN / 'evaluation_math/results.json'
    result = load(result_path)
    if result['status'] != 'completed':
        raise ValueError('Only a completed frozen evaluation may be reported')
    freeze_path = ROOT / result['selection_freeze_path']
    if sha(freeze_path) != result['selection_freeze_sha256']:
        raise ValueError('Final selection changed after evaluation')
    freeze = load(freeze_path)
    validation_path = ROOT / freeze['validation_records']['path']
    if sha(validation_path) != freeze['validation_records']['sha256']:
        raise ValueError('Validation evidence changed after selection')
    numerical_path = ROOT / freeze['numerical_evaluation']['path']
    if sha(numerical_path) != freeze['numerical_evaluation']['sha256']:
        raise ValueError('Numerical evaluation manifest changed after selection')
    if result.get('numerical_evaluation') != freeze['numerical_evaluation']:
        raise ValueError('Evaluation and selection numerical provenance differ')
    validation = load(validation_path)
    continuation = load(RUN / 'continuation/continuation_results.json')
    training = load(RUN / 'training_results.json')
    config = load(ROOT / 'configs/experiments/her2_posttrain.json')
    selected_names = {choice['selected']
        for choices in result['selection_within_budget'].values()
        for choice in choices.values() if choice.get('selected')}

    rows = []
    for name, entry in result['selected'].items():
        if name not in validation:
            continue
        val = validation[name]
        test = result['test_metrics'][name]
        generation = result['generation'][name]
        assay = result['assay']['endpoints'][name]
        parent_score = result['test_metrics'].get(name + '_minus_parent', {})
        parent_assay = result['assay']['endpoints'].get(name + '_minus_parent', {})
        seen = entry.get('distinct_exposures') or {}
        novelty = generation['exact_train_novelty']['fraction_not_in_training_cores']
        heldout = generation['labelled_hits']['heldout']
        rows.append(dict(name=name, method=entry.get('method'), seed=entry.get('seed'),
            role=entry['role'], target_minutes=entry.get('budget_seconds', 0) / 60,
            actual_minutes=(entry.get('actual_gpu_seconds') or 0) / 60,
            updates=entry.get('updates'), distinct_high=seen.get('distinct_chosen_rows'),
            distinct_low=seen.get('distinct_rejected_rows'),
            val_ap=val['val_metrics']['average_precision'], test_ap=test['average_precision'],
            val_high_nll=val['val_positive_nll_per_residue'],
            test_auc=test['auroc'], p1000=test['precision_at_k']['1000'],
            parent_relative_ap=parent_score.get('average_precision'),
            spr_spearman=assay['quantitative']['spearman'],
            spr_binding_auc=assay['binary']['auroc'],
            spr_quantitative_n=assay['quantitative']['n'],
            spr_parent_relative_spearman=parent_assay.get('quantitative', {}).get('spearman'),
            entropy=generation['entropy']['entropy_nats'],
            site_entropy=generation['sum_site_entropy_nats'],
            mean_hamming=generation['mean_pairwise_hamming'],
            unique=generation['unique_fraction'], max_frequency=generation['max_single_core_frequency'],
            not_in_train=novelty, heldout_matches=heldout['draws_matching'],
            heldout_high=heldout['conditional_high']['rate'],
            kl_parent=generation.get('kl_to_sft_parent', {}).get('kl_nats', 0.0),
            eligible=val['diversity']['eligible'], selected=name in selected_names))
    frame = pd.DataFrame(rows)
    figures = REFERENCE / 'figures'
    evidence = REFERENCE / 'evidence'
    figures.mkdir(exist_ok=True)
    evidence.mkdir(exist_ok=True)
    frame.to_csv(evidence / 'her2-posttrain-scaling-2026-09-18.csv', index=False)

    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [('test_ap', 'Test average precision'),
              ('spr_spearman', 'Independent SPR Spearman'),
              ('entropy', 'Generated sequence entropy (nats)'),
              ('unique', 'Unique fraction of 10,000 draws'),
              ('kl_parent', 'KL to own SFT parent (nats)'),
              ('not_in_train', 'Fraction not exactly in training')]
    colors = {'continued_sft': '#cf782b', 'dpo': '#266a9c'}
    labels = {'continued_sft': 'Continued SFT', 'dpo': 'DPO'}
    for ax, (metric, title) in zip(axes.flat, panels):
        for method, color in colors.items():
            group = frame[frame.method == method].sort_values('target_minutes')
            for seed in config['continuation']['seeds']:
                raw = group[group.seed == seed]
                parent = frame[frame.name == f'policy_sft_seed{seed}']
                trajectory = pd.concat([parent, raw]).sort_values('target_minutes')
                ax.plot(trajectory.actual_minutes, trajectory[metric], color=color,
                        alpha=0.25, linewidth=1)
            summary = group.groupby('target_minutes').agg(
                x=('actual_minutes', 'mean'), mid=(metric, 'mean'),
                low=(metric, 'min'), high=(metric, 'max'))
            ax.plot(summary.x, summary.mid, 'o-', color=color, label=labels[method])
            ax.fill_between(summary.x, summary.low, summary.high, color=color, alpha=0.15)
            failed = group[~group.eligible]
            ax.scatter(failed.actual_minutes, failed[metric], marker='x', color=color, s=30)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Additional measured GPU minutes')
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].axhline(0, color='grey', linewidth=0.6)
    axes[1, 0].axhline(0.9, color='grey', linewidth=0.8, linestyle='--')
    axes[1, 2].axhline(0.5, color='grey', linewidth=0.8, linestyle='--')
    fig.suptitle('HER2 post-training: compute, measured-population ranking and generation',
                 fontsize=14)
    fig.savefig(figures / 'her2-posttrain-scaling.svg')
    fig.savefig(RUN / 'evaluation_math/her2-posttrain-scaling.png', dpi=170)
    plt.close(fig)

    snapshot = dict(evaluation=result, validation=validation, selection_freeze=freeze,
        numerical_evaluation=load(numerical_path),
        training=training, continuation_without_update_history=compact_runs(continuation),
        provenance={'evaluation_sha256': sha(result_path),
                    'validation_sha256': sha(validation_path),
                    'config_sha256': sha(ROOT / 'configs/experiments/her2_posttrain.json'),
                    'render_script_sha256': sha(Path(__file__))},
        reporting_note='No fitting or checkpoint selection in this renderer; all raw budgets retained')
    (evidence / 'her2-posttrain-2026-09-18.json').write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')

    raw = frame[frame.role == 'raw_budget']
    aggregate = raw.groupby(['method', 'target_minutes']).agg(
        actual_minutes=('actual_minutes', 'mean'), test_ap=('test_ap', 'mean'),
        ap_min=('test_ap', 'min'), ap_max=('test_ap', 'max'),
        spr_spearman=('spr_spearman', 'mean'), spr_binding_auc=('spr_binding_auc', 'mean'),
        entropy=('entropy', 'mean'),
        unique=('unique', 'mean'), eligible_seeds=('eligible', 'sum')).reset_index()
    diagnostics = raw[['method', 'seed', 'target_minutes', 'val_high_nll', 'test_ap', 'parent_relative_ap',
                       'spr_spearman', 'spr_parent_relative_spearman', 'kl_parent',
                       'mean_hamming', 'unique', 'max_frequency', 'not_in_train', 'eligible']]
    decisions = []
    for run_name, choices in result['selection_within_budget'].items():
        for budget, choice in choices.items():
            decisions.append({'run': run_name, 'budget_minutes': float(budget) / 60,
                'selected': choice.get('selected') or 'none eligible',
                'validation_ap': choice.get('value')})
    paired = []
    for seed in config['continuation']['seeds']:
        for budget in sorted(set(config['continuation']['budgets_gpu_seconds']['dpo'])
                             & set(config['continuation']['budgets_gpu_seconds']['continued_sft'])):
            name = f'dpo_seed{seed}_budget{budget}_minus_continued_sft_seed{seed}_budget{budget}'
            difference = result['paired_differences']['differences'][name]['average_precision']
            paired.append({'seed': seed, 'budget_minutes': budget / 60,
                'DPO_minus_SFT_AP': difference['observed'],
                'CI_low': difference['ci_low'], 'CI_high': difference['ci_high']})
    baseline_rows = []
    for name, metric in result['test_metrics'].items():
        if '_minus_' in name or 'budget' in name:
            continue
        assay = result['assay']['endpoints'][name]
        baseline_rows.append({'scorer': name, 'test_ap': metric['average_precision'],
            'test_auroc': metric['auroc'], 'p1000': metric['precision_at_k']['1000'],
            'spr_spearman': assay['quantitative']['spearman'],
            'spr_binding_auc': assay['binary']['auroc']})
    initial = []
    for name, entry in training['policies'].items():
        for epoch, values in entry['epochs'].items():
            if 'val_positive_nll_per_residue' in values:
                initial.append({'run': name, 'epoch': int(epoch),
                    'val_high_nll': values['val_positive_nll_per_residue'],
                    'val_ap_diagnostic': values['val_ranking_diagnostic']['average_precision']})
    lines = ['# HER2 p-IgGen SFT and DPO benchmark', '',
        'Completed campaign, 2026-09-18. See the [fixed protocol](../specs/her2_hcdr3_benchmark.md).', '',
        'p-IgGen is the generator. The CNN and additive model are auxiliary ranking comparators '
        'on measured populations; neither supplies rewards, preferences, or checkpoint selection.', '',
        '![Scaling curves](figures/her2-posttrain-scaling.svg)', '',
        'Lines show seed means, bands show the range across three seeds, and faint lines show '
        'individual trajectories. Crosses mark raw checkpoints that fail diversity eligibility. '
        'These bands are not confidence intervals. The initial SFT cost is separate from the '
        'additional GPU budgets. DPO reference scoring is included in its budget.', '',
        '## Compute comparison', '', aggregate.to_markdown(index=False, floatfmt='.4f'), '',
        '## Ranking baselines and initial policies', '',
        pd.DataFrame(baseline_rows).to_markdown(index=False, floatfmt='.4f'), '',
        'SPR Spearman uses only finite positive KD measurements; the binary SPR AUROC '
        'uses measured binding versus nonbinding, including unquantified binders as positive. '
        'No numerical KD is assigned to nonbinding or unquantified records.', '',
        '## Every continuation checkpoint', '',
        diagnostics.to_markdown(index=False, floatfmt='.4f'), '',
        'Parent-relative scores are policy log density minus the same SFT parent that DPO '
        'used as its reference. They are a declared diagnostic. Checkpoint selection used raw '
        'validation density ranking and the fixed diversity criteria.', '',
        '## Matched-budget paired differences', '',
        pd.DataFrame(paired).to_markdown(index=False, floatfmt='.5f'), '',
        'Intervals are 95% paired percentile intervals from 1,000 row resamples of the test '
        'population. They describe row-resampling uncertainty, separately from variation '
        'between training seeds. They do not account for dependence among nearby sequences '
        'or establish generalization to a different target or scaffold.', '',
        '## Frozen checkpoint decisions', '',
        pd.DataFrame(decisions).to_markdown(index=False, floatfmt='.4f'), '',
        'Selections were fixed using validation AP and the declared diversity gates before '
        'model-based final test and SPR evaluation. An empty budget stays empty; no parent '
        'or failed checkpoint is silently promoted.', '',
        '## Initial SFT validation', '',
        pd.DataFrame(initial).to_markdown(index=False, floatfmt='.4f'), '',
        'Initial checkpoints were selected by high-bin validation NLL. Their validation AP '
        'was diagnostic only. No best seed was selected.', '',
        '## Initial-policy sampling', '',
        frame[frame.role == 'initial'][['name', 'entropy', 'unique', 'max_frequency',
            'not_in_train', 'mean_hamming']].to_markdown(index=False, floatfmt='.4f'), '',
        '## Numerical evaluation amendment', '',
        'After fitting and before final selection, one automatic-SDPA sampler/scorer '
        'comparison failed its FP32 tolerance. An FP64 diagnosis found full teacher '
        'forcing and autoregression agreeing to 8.53e-14 nats on the worst rows. All '
        '31 policies were then revalidated uniformly using native math SDPA, retaining '
        'FP32, the original tolerances, weights, seeds and selection rules. Final '
        'scoring uses the same backend. The original partial results are preserved '
        'separately. The [numerical audit](evidence/her2-numerical-audit-2026-09-18.json) '
        'and the [dated protocol amendment](../specs/her2_hcdr3_benchmark.md#9-numerical-evaluation-amendment--2026-09-18) '
        'document the change. The final freeze binds the added evaluation driver '
        'and backend through a separate manifest.', '',
        '## Interpretation limits', '',
        '- Library labels are high/mid/low binding bins, not numerical KD. The fixed-scaffold '
        'task edits ten HCDR3 positions and has no antigen encoder.',
        '- The random split is dominated by close training neighbors. Proximity-stratified '
        'results and paired bootstrap intervals are retained in the evidence.',
        '- Independent SPR evaluates other authors\' designs, after removing exact library '
        'overlaps. It does not measure the binding of our newly generated sequences. '
        'Within-method results reduce, but do not remove, author selection bias.',
        '- Generated catalogue hits are lookups of existing measurements; unmeasured outputs '
        'receive no inferred experimental affinity. Repeated draws are not new assay replicates.',
        '- Diversity thresholds are distributional checks, not proof of functional diversity '
        'or absence of overfitting. Three seeds do not resolve every small effect.',
        '- Equal GPU time is not equal update count or label information: DPO also reads '
        'low-bin examples, while continued SFT trains on eligible high-bin examples.',
        '- p-IgGen\'s HER2 pretraining exposure remains unresolved. The scratch control tests '
        'random initialization at the same training schedule, not a tuned scratch optimum.',
        '- Initial source auditing read aggregate counts and two examples from every split, '
        'including test. Model-based test and SPR evaluation followed the final selection freeze.',
        '- The workbook identifies SPR; the upstream README abstract describes BLI. The '
        'metadata discrepancy is retained. Absci Corporation (2023) data were used solely for '
        'a support-compatibility audit, not these training or evaluation results.', '',
        '## Reproducibility', '',
        'The [evidence snapshot](evidence/her2-posttrain-2026-09-18.json) retains every scorer, '
        'distance stratum, paired interval, seed result, selection decision, exposure count '
        'and source/checkpoint digest. [Scaling data](evidence/her2-posttrain-scaling-2026-09-18.csv) '
        'are available separately. Weights, draw files and full logs remain under '
        '`outputs/her2_posttrain_20260918/`.', '']
    (REFERENCE / 'her2-posttrain.md').write_text('\n'.join(lines), encoding='utf-8')
    print(aggregate.to_string(index=False))
    print('Report rendered without fitting or selection.')


if __name__ == '__main__':
    render()
