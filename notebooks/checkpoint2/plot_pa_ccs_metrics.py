"""Plot and aggregate already-computed PA-CCS layer metrics.

Run: python notebooks/checkpoint2/plot_pa_ccs_metrics.py
"""
import argparse
import os
from pathlib import Path
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/checkpoint2-pa-ccs-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artifacts import DATASETS, MODELS, PROJECT


METRIC_COLUMNS = {
    'accuracy': 'ESA',
    'polar_consistency_↓': 'Polar Consistency',
    'contradiction_idx_↓': 'Contradiction Index',
}
MODEL_MEAN_COLUMNS = [
    'dataset', 'model', 'esa_mean', 'polar_consistency_mean',
    'contradiction_idx_mean', 'n_layers',
]
DATASET_MEAN_COLUMNS = [
    'dataset', 'esa_mean', 'polar_consistency_mean',
    'contradiction_idx_mean', 'n_models',
]


def load_layer_metrics(path, dataset, model):
    """Read, validate, and numerically sort one layer-metrics table."""
    frame = pd.read_csv(path)
    required = {'layer', *METRIC_COLUMNS}
    if missing := required - set(frame.columns):
        raise ValueError(
            f'{dataset}/{model}: missing columns {sorted(missing)} in {path}')

    numeric_layers = pd.to_numeric(frame['layer'], errors='coerce')
    if numeric_layers.isna().any() or not np.isfinite(numeric_layers).all():
        raise ValueError(f'{dataset}/{model}: layer must be numeric in {path}')
    frame = frame.copy()
    frame['layer'] = numeric_layers

    for column in METRIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
        if frame[column].isna().all():
            raise ValueError(
                f'{dataset}/{model}: {column} contains only NaN in {path}')

    return frame.sort_values('layer', kind='stable').reset_index(drop=True)


def _plot_model(frame, destination, dataset, model):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for column, label in METRIC_COLUMNS.items():
        ax.plot(frame['layer'], frame[column], marker='o', markersize=4,
                linewidth=1.5, label=label)
    ax.axhline(0, color='#666666', linestyle='--', linewidth=1)
    ax.set(xlabel='Layer', ylabel='Metric value',
           title=f'PA-CCS — {model} — {dataset}')
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _model_mean(frame, dataset, model):
    means = frame[list(METRIC_COLUMNS)].mean()
    return {
        'dataset': dataset,
        'model': model,
        'esa_mean': means['accuracy'],
        'polar_consistency_mean': means['polar_consistency_↓'],
        'contradiction_idx_mean': means['contradiction_idx_↓'],
        'n_layers': len(frame),
    }


def _dataset_means(model_means):
    rows = []
    for dataset, group in model_means.groupby('dataset', sort=False):
        rows.append({
            'dataset': dataset,
            'esa_mean': group['esa_mean'].mean(),
            'polar_consistency_mean': group['polar_consistency_mean'].mean(),
            'contradiction_idx_mean': group['contradiction_idx_mean'].mean(),
            'n_models': len(group),
        })
    return pd.DataFrame(rows, columns=DATASET_MEAN_COLUMNS)


def _plot_summary(row, destination):
    labels = ['ESA', 'Polar Consistency', 'Contradiction Index']
    values = [row.esa_mean, row.polar_consistency_mean,
              row.contradiction_idx_mean]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, values, color=['#2874a6', '#d68910', '#b03a2e'])
    ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=3)
    ax.set(ylabel='Mean metric value',
           title=f'Mean PA-CCS metrics — {row.dataset}')
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _console_summary(counts, output):
    lines = ['PA-CCS plots generated.', '']
    for dataset, count in counts.items():
        lines.extend([
            f'{dataset}:',
            f'  {count} models',
            f'  {count} individual plots',
            '',
        ])
    lines.extend(['Saved:', f'  {output}/'])
    return '\n'.join(lines)


def run(results_root, *, datasets=None, models=None):
    """Generate plots and equal-model-weight summaries from layer CSV files."""
    results_root = Path(results_root)
    datasets = list(DATASETS if datasets is None else datasets)
    models = list(MODELS if models is None else models)
    output = results_root / 'pa_ccs_analysis'
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {}
    for dataset in datasets:
        counts[dataset] = 0
        for model in models:
            source = (results_root / dataset / 'analysis'
                      / 'checkpoint1_compatible' / model / 'layer_metrics.csv')
            if not source.is_file():
                warnings.warn(
                    f'{dataset}/{model}: layer_metrics.csv not found; skipping',
                    stacklevel=2)
                continue
            frame = load_layer_metrics(source, dataset, model)
            _plot_model(frame, output / 'by_model' / dataset / f'{model}.png',
                        dataset, model)
            rows.append(_model_mean(frame, dataset, model))
            counts[dataset] += 1

    model_means = pd.DataFrame(rows, columns=MODEL_MEAN_COLUMNS)
    model_means.to_csv(output / 'pa_ccs_model_means.csv', index=False)
    dataset_means = _dataset_means(model_means)
    dataset_means.to_csv(output / 'pa_ccs_dataset_means.csv', index=False)
    for row in dataset_means.itertuples(index=False):
        _plot_summary(row, output / 'summary' / f'{row.dataset}_mean_metrics.png')

    summary = _console_summary(counts, output)
    print(summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path,
                        default=PROJECT / 'results/checkpoint2')
    args = parser.parse_args()
    run(args.results_root)
