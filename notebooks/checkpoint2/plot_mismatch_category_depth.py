"""Four mismatch-category curves by PA-CCS depth quarter and model size.

Run: python notebooks/checkpoint2/plot_mismatch_category_depth.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

from analyze_mismatch_cases_baseline import CASE_TYPES, plt
from analyze_pa_ccs import DEPTH_GROUPS, MODEL_GROUPS
from artifacts import MODELS, PROJECT

TITLES = ['Both aligned', 'Behavior aligned · latent misaligned',
          'Behavior misaligned · latent aligned', 'Both misaligned']
DEPTH_LABELS = ['Early\n0–25%', 'Middle-early\n25–50%',
                'Middle-late\n50–75%', 'Late\n75–100%']


def aggregate_depth(layers):
    """Average category fractions across layers, then equally across models."""
    frame = layers.copy()
    keys = ['dataset', 'model', 'layer']
    if frame.empty or not frame.split.eq('test').all():
        raise ValueError('Expected nonempty test-only category shares')
    if frame.duplicated([*keys, 'mismatch_category']).any():
        raise ValueError('Duplicate layer/category')
    if not frame.mismatch_category.isin(CASE_TYPES).all():
        raise ValueError('Unknown mismatch category')
    grouped = frame.groupby(keys)
    if not grouped.size().eq(4).all():
        raise ValueError('Every layer requires all four categories, including zeros')
    if not frame.fraction.between(0, 1).all() or not np.allclose(grouped.fraction.sum(), 1):
        raise ValueError('Category fractions must sum to one at each layer')
    frame['model_group'] = frame.model.map(MODEL_GROUPS)
    if frame.model_group.isna().any():
        raise ValueError('Unknown model group')
    last = frame.model.map({name: spec.n_layers - 1 for name, spec in MODELS.items()})
    depth = frame.layer / last
    if not depth.between(0, 1).all():
        raise ValueError('Layer outside model depth')
    frame['depth_group'] = pd.cut(depth, [0, .25, .5, .75, 1],
                                  labels=DEPTH_GROUPS, include_lowest=True)
    per_model = (frame.groupby(['dataset', 'model_group', 'model', 'depth_group',
                                'mismatch_category'], observed=True, as_index=False)
                 .agg(fraction=('fraction', 'mean'), n_layers=('layer', 'size'),
                      first_layer=('layer', 'min'), last_layer=('layer', 'max')))
    means = (per_model.groupby(['dataset', 'model_group', 'depth_group', 'mismatch_category'],
                                observed=True, as_index=False)
             .agg(fraction=('fraction', 'mean'), n_models=('model', 'nunique'),
                  n_layers=('n_layers', 'sum')))
    return per_model, means


def plot_curves(means):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=False)
    for ax, case, title in zip(axes.flat, CASE_TYPES, TITLES):
        for dataset, color in [('mixed', '#2874a6'), ('not', '#d97706')]:
            for size, style, marker in [('big', '-', 'o'), ('small', '--', 's')]:
                group = means[(means.dataset == dataset) & (means.model_group == size)
                              & (means.mismatch_category == case)]
                values = group.set_index('depth_group').reindex(DEPTH_GROUPS)
                ax.plot(range(4), values.fraction, color=color, linestyle=style,
                        marker=marker, markersize=6, linewidth=2, label=f'{size} / {dataset}')
        upper = {'both_aligned': 1.,
                 'aligned_output_misaligned_latent': .4,
                 'misaligned_output_aligned_latent': .25,
                 'both_misaligned': .2}[case]
        ax.set(title=title, ylim=(0, upper), ylabel='Share of test objects')
        ax.yaxis.set_major_locator(MultipleLocator(.2 if upper == 1 else .05))
        ax.set_xticks(range(4), DEPTH_LABELS)
        ax.tick_params(labelbottom=True)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .92),
               ncol=4, frameon=False)
    fig.suptitle('Mismatch categories by relative depth\n'
                 'Big: Gemma 9B / 9B-it; small: Gemma 2B / 2B-it + DeBERTa', fontsize=16, y=.99)
    fig.text(.5, .025, 'Categories computed at each layer on the test split using saved behavioral calibration.\n'
             'Mean across layers within each model and depth quarter, then equal-weight mean across models.',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.13, left=.075, right=.98, hspace=.32, wspace=.16)
    return fig


def accuracy_from_categories(means):
    """Recover true-label accuracies from the same equally weighted shares."""
    keys = ['dataset', 'model_group', 'depth_group']
    if 'model' in means:
        keys.insert(2, 'model')
    wide = means.pivot(index=keys, columns='mismatch_category', values='fraction')
    result = pd.DataFrame({
        'latent_accuracy': wide['both_aligned'] + wide['misaligned_output_aligned_latent'],
        'behavioral_calibrated_accuracy': wide['both_aligned'] + wide['aligned_output_misaligned_latent'],
    }).reset_index()
    result.columns.name = None
    return result


def plot_accuracy(accuracy):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, title in zip(axes,
            ['latent_accuracy', 'behavioral_calibrated_accuracy'],
            ['Latent accuracy', 'Behavioral calibrated accuracy']):
        for dataset, color in [('mixed', '#2874a6'), ('not', '#d97706')]:
            for size, style, marker in [('big', '-', 'o'), ('small', '--', 's')]:
                values = (accuracy[(accuracy.dataset == dataset) & (accuracy.model_group == size)]
                          .set_index('depth_group').reindex(DEPTH_GROUPS))
                ax.plot(range(4), values[metric], color=color, linestyle=style,
                        marker=marker, markersize=6, linewidth=2, label=f'{size} / {dataset}')
        lower = max(0, np.floor((accuracy[metric].min() - .025) * 20) / 20)
        upper = min(1, np.ceil((accuracy[metric].max() + .025) * 20) / 20)
        ax.set(title=title, ylim=(lower, upper), ylabel='Test accuracy')
        ax.set_xticks(range(4), DEPTH_LABELS)
        ax.yaxis.set_major_locator(MultipleLocator(.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .88),
               ncol=4, frameon=False)
    fig.suptitle('Accuracy by relative depth\n'
                 'Big: Gemma 9B / 9B-it; small: Gemma 2B / 2B-it + DeBERTa', fontsize=15, y=.99)
    fig.text(.5, .025, 'Layer accuracies averaged within each model and quarter, then equally across models.\n'
             'Latent uses saved train orientation; calibrated behavioral predictions do not depend on layer.',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=.75, bottom=.23, left=.075, right=.98, wspace=.22)
    return fig


def run(root):
    root = Path(root) / 'mismatch_calibrated_test'
    per_model, means = aggregate_depth(pd.read_csv(root / 'mismatch_summary.csv'))
    output = root / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(output / 'mismatch_category_model_depth_means.csv', index=False)
    means.to_csv(output / 'mismatch_category_group_depth_means.csv', index=False)
    fig = plot_curves(means)
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'mismatch_category_depth_curves.{extension}', dpi=200)
    plt.close(fig)
    accuracy = accuracy_from_categories(means)
    accuracy.to_csv(output / 'accuracy_group_depth_means.csv', index=False)
    accuracy_from_categories(per_model).to_csv(output / 'accuracy_model_depth_means.csv', index=False)
    fig = plot_accuracy(accuracy)
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'accuracy_depth_curves.{extension}', dpi=200)
    plt.close(fig)
    print(f'Saved category and accuracy curves with depth means to {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=PROJECT / 'results/checkpoint2')
    run(parser.parse_args().results_root)
