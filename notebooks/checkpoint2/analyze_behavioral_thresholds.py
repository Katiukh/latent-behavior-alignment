"""Fit one behavioral P(Yes) decision threshold per dataset/model on train.

This is a standalone experiment. It does not change the fixed 0.5 boundary used
by the mismatch analysis.

Run: python notebooks/checkpoint2/analyze_behavioral_thresholds.py
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/checkpoint2-behavioral-threshold-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

REQUIRED_COLUMNS = [
    'sample_idx', 'split', 'true_label', 'behavioral_probability_yes', 'layer']
METRIC_COLUMNS = [
    'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
    'tn', 'fp', 'fn', 'tp']


@dataclass
class AnalysisResult:
    threshold: dict
    metrics: pd.DataFrame
    curve: pd.DataFrame


def _unique_objects(scores, dataset, model):
    scores = scores.copy()
    if ('behavioral_probability_yes' not in scores
            and model == 'deberta-hate-tuned'
            and 'behavioral_probability_hate' in scores):
        # DeBERTa is a direct harmful/not-harmful classifier. Its P(hate) is
        # the same positive-class probability represented by P(Yes) for Gemma.
        scores['behavioral_probability_yes'] = scores['behavioral_probability_hate']
    if missing := set(REQUIRED_COLUMNS) - set(scores.columns):
        raise ValueError(f'{dataset}/{model}: missing columns {sorted(missing)}')
    if scores.empty or scores[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f'{dataset}/{model}: empty table or missing required values')
    if not scores.true_label.isin([0, 1]).all():
        raise ValueError(f'{dataset}/{model}: true_label must be 0 or 1')
    probabilities = scores.behavioral_probability_yes.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f'{dataset}/{model}: behavioral_probability_yes must be in [0,1]')
    if scores.duplicated(['sample_idx', 'layer']).any():
        raise ValueError(f'{dataset}/{model}: duplicate sample_idx within a layer')
    invariant = ['split', 'true_label', 'behavioral_probability_yes']
    if (scores.groupby('sample_idx')[invariant].nunique() > 1).any().any():
        raise ValueError(f'{dataset}/{model}: object metadata or probability changes across layers')

    # The behavioral probability is repeated once per latent layer. Thresholds
    # and metrics operate on objects, never on those repeated layer rows.
    objects = scores.drop_duplicates(subset=['sample_idx']).copy()
    unexpected = set(objects.split) - {'train', 'test'}
    if unexpected or not {'train', 'test'}.issubset(set(objects.split)):
        raise ValueError(f'{dataset}/{model}: split must contain train and test only')
    return objects


def _metric_row(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(labels, predicted),
        'balanced_accuracy': balanced_accuracy_score(labels, predicted),
        'precision': precision_score(labels, predicted, zero_division=0),
        'recall': recall_score(labels, predicted, zero_division=0),
        'f1': f1_score(labels, predicted, zero_division=0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def _threshold_curve(train):
    candidates = np.unique(np.r_[0.0, .5, train.behavioral_probability_yes.to_numpy(float), 1.0])
    rows = []
    labels = train.true_label.to_numpy(int)
    probabilities = train.behavioral_probability_yes.to_numpy(float)
    for threshold in candidates:
        metrics = _metric_row(labels, probabilities, threshold)
        rows.append({'behavioral_threshold': float(threshold), **metrics})
    return pd.DataFrame(rows)


def _select_threshold(curve):
    best = curve[curve.accuracy == curve.accuracy.max()].copy()
    best['distance_from_0.5'] = (best.behavioral_threshold - .5).abs()
    distances = best['distance_from_0.5']
    closest = best[np.isclose(distances, distances.min(), rtol=0, atol=1e-12)]
    return float(closest.behavioral_threshold.min())


def analyze_scores(scores, dataset, model):
    objects = _unique_objects(scores, dataset, model)
    train = objects[objects.split == 'train']
    test = objects[objects.split == 'test']
    if train.true_label.nunique() != 2 or test.true_label.nunique() != 2:
        raise ValueError(f'{dataset}/{model}: train and test must each contain both classes')

    curve = _threshold_curve(train)
    threshold = _select_threshold(curve)
    metric_rows = []
    for threshold_type, value in [('fixed_0.5', .5), ('train_optimized', threshold)]:
        for split, frame in [('train', train), ('test', test)]:
            metric_rows.append({
                'dataset': dataset,
                'model': model,
                'threshold_type': threshold_type,
                'threshold': value,
                'split': split,
                **_metric_row(frame.true_label.to_numpy(int),
                              frame.behavioral_probability_yes.to_numpy(float), value),
            })
    metrics = pd.DataFrame(metric_rows, columns=[
        'dataset', 'model', 'threshold_type', 'threshold', 'split', *METRIC_COLUMNS])
    selected_train = metrics[
        (metrics.threshold_type == 'train_optimized') & (metrics.split == 'train')].iloc[0]
    threshold_row = {
        'dataset': dataset,
        'model': model,
        'behavioral_threshold': threshold,
        'train_accuracy': selected_train.accuracy,
        'train_balanced_accuracy': selected_train.balanced_accuracy,
        'n_train': len(train),
        'n_test': len(test),
    }
    return AnalysisResult(threshold_row, metrics, curve)


def _plot_curve(curve, threshold, destination, dataset, model):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.plot(curve.behavioral_threshold, curve.accuracy, color='#2874a6',
            label='train accuracy')
    ax.axvline(.5, color='#777777', linestyle='--', label='fixed 0.5')
    ax.axvline(threshold, color='#b03a2e', linestyle='--',
               label=f'train optimized {threshold:.3f}')
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='behavioral_threshold',
           ylabel='accuracy', title=f'{dataset} / {model}')
    ax.legend(frameon=False)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _summary(metrics, thresholds):
    lines = []
    for row in thresholds.itertuples(index=False):
        pair = metrics[(metrics.dataset == row.dataset) & (metrics.model == row.model)]
        values = pair.set_index(['split', 'threshold_type'])
        fixed_train = values.loc[('train', 'fixed_0.5')]
        fitted_train = values.loc[('train', 'train_optimized')]
        fixed_test = values.loc[('test', 'fixed_0.5')]
        fitted_test = values.loc[('test', 'train_optimized')]
        lines.extend([
            f'{row.dataset} / {row.model}',
            f'best train threshold: {row.behavioral_threshold:.3f}',
            f'train accuracy:          {fixed_train.accuracy:.3f} -> {fitted_train.accuracy:.3f}',
            f'test accuracy:           {fixed_test.accuracy:.3f} -> {fitted_test.accuracy:.3f}',
            f'train balanced accuracy: {fixed_train.balanced_accuracy:.3f} -> {fitted_train.balanced_accuracy:.3f}',
            f'test balanced accuracy:  {fixed_test.balanced_accuracy:.3f} -> {fitted_test.balanced_accuracy:.3f}',
            '',
        ])
    return '\n'.join(lines).rstrip()


def run(results_root):
    results_root = Path(results_root)
    plans = []
    for dataset_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        for path in sorted((dataset_dir / 'analysis').rglob('scores.csv')):
            plans.append((dataset, path.parent.name,
                          analyze_scores(pd.read_csv(path), dataset, path.parent.name)))
    if not plans:
        raise ValueError(f'No scores.csv files found under {results_root}')
    pairs = [(dataset, model) for dataset, model, _ in plans]
    if len(pairs) != len(set(pairs)):
        raise ValueError('Ambiguous scores.csv files for a dataset/model pair')

    output = results_root / 'behavioral_threshold_analysis'
    output.mkdir(parents=True, exist_ok=True)
    thresholds = pd.DataFrame([result.threshold for _, _, result in plans])
    thresholds.to_csv(output / 'behavioral_thresholds.csv', index=False)
    metrics = pd.concat([result.metrics for _, _, result in plans], ignore_index=True)
    metrics.to_csv(output / 'behavioral_threshold_metrics.csv', index=False)
    for dataset, model, result in plans:
        _plot_curve(result.curve, result.threshold['behavioral_threshold'],
                    output / 'plots' / dataset / f'{model}.png', dataset, model)
    summary = _summary(metrics, thresholds)
    print(summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'results/checkpoint2')
    args = parser.parse_args()
    run(args.results_root)
