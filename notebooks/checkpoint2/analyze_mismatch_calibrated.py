"""Test-only mismatch using existing train-fitted behavioral thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_mismatch_cases_baseline import CASE_TYPES, COLORS, KEYS, plt
from artifacts import DATASETS, MODELS, PROJECT


def analyze_scores(source, dataset, model, threshold, train_idx, test_idx):
    """Classify positive-class probabilities, then compare classes with truth.

    Calibration uses P(harmful) >= fitted threshold; saved oriented CCS uses
    P(harmful) > 0.5. A safe object's alignment therefore reverses the inequality.
    """
    frame = source.copy()
    probability = ('behavioral_probability_hate' if 'behavioral_probability_hate' in frame
                   else 'behavioral_probability_yes')
    required = ['sample_idx', 'statement', 'true_label', 'split', 'layer', 'latent_score', probability]
    if missing := set(required) - set(frame):
        raise ValueError(f'{dataset}/{model}: missing columns {sorted(missing)}')
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Empty or incomplete scores')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid saved behavioral threshold')
    if not frame.true_label.isin([0, 1]).all():
        raise ValueError('Invalid true_label')
    for col in ['sample_idx', 'layer']:
        if not (np.isfinite(frame[col]) & (frame[col] >= 0) & (frame[col] % 1 == 0)).all():
            raise ValueError(f'Invalid {col}')
    for col in [probability, 'latent_score']:
        if not np.isfinite(frame[col]).all() or not frame[col].between(0, 1).all():
            raise ValueError(f'Invalid probability: {col}')
    if frame.duplicated(['sample_idx', 'layer']).any():
        raise ValueError('Duplicate sample/layer')
    if (frame.groupby('sample_idx')[['statement', 'true_label', 'split', probability]].nunique() > 1).any().any():
        raise ValueError('Object metadata changes across layers')
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    if (not len(train_idx) or not len(test_idx)
            or not np.array_equal(np.sort(np.r_[train_idx, test_idx]),
                                  np.arange(len(train_idx) + len(test_idx)))):
        raise ValueError('Invalid saved split')
    for _, group in frame.groupby('layer'):
        for split, idx in [('train', train_idx), ('test', test_idx)]:
            if not np.array_equal(np.sort(group.loc[group.split.eq(split), 'sample_idx']), np.sort(idx)):
                raise ValueError(f'{dataset}/{model}: scores do not match saved {split} split')
    if not frame.split.isin(['train', 'test']).all():
        raise ValueError('Unknown split')
    frame = frame.loc[frame.split.eq('test')].copy()
    frame['dataset'], frame['model'] = dataset, model
    # Preserve the historical binary behavioral_score under an explicit name.
    if 'behavioral_score' in frame:
        frame = frame.rename(columns={'behavioral_score': 'baseline_behavioral_prediction'})
    frame['behavioral_score'] = frame[probability]
    frame['behavioral_threshold'] = float(threshold)
    frame['latent_threshold'] = .5
    frame['behavioral_prediction'] = (frame.behavioral_score >= threshold).astype(int)
    frame['latent_prediction'] = (frame.latent_score > .5).astype(int)
    b = frame.behavioral_prediction.eq(frame.true_label)
    l = frame.latent_prediction.eq(frame.true_label)
    frame['behavioral_class'] = np.where(b, 'aligned', 'misaligned')
    frame['latent_class'] = np.where(l, 'aligned', 'misaligned')
    frame['mismatch_category'] = np.select([b & l, b & ~l, ~b & l], CASE_TYPES[:3], default=CASE_TYPES[3])
    return frame


def summarize(objects, keys=KEYS):
    index = pd.MultiIndex.from_tuples([
        (*key, case) for key in objects.groupby(keys, sort=True).groups for case in CASE_TYPES
    ], names=[*keys, 'mismatch_category'])
    counts = objects.groupby([*keys, 'mismatch_category']).size().reindex(index, fill_value=0)
    frame = counts.rename('n').reset_index()
    frame['denominator_n'] = frame.groupby(keys).n.transform('sum')
    frame['fraction'] = frame.n / frame.denominator_n
    frame['counting_unit'] = 'test sample' if 'layer' in keys else 'test sample × layer'
    thresholds = objects.groupby(keys)[['behavioral_threshold', 'latent_threshold']].first().reset_index()
    return frame.merge(thresholds, on=keys, validate='many_to_one').assign(split='test')


def plot_layer(objects, destination, dataset, model, layer):
    """Same axis orientation as baseline; raw scores show the fitted boundary."""
    threshold = objects.behavioral_threshold.iloc[0]
    # Separate labels make the meaning of the four categories unambiguous.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    for label, ax in zip([0, 1], axes):
        subset = objects[objects.true_label.eq(label)]
        for case in CASE_TYPES:
            group = subset[subset.mismatch_category.eq(case)]
            ax.scatter(group.behavioral_score, group.latent_score, s=16, alpha=.6,
                       edgecolors='none', color=COLORS[case], label=f'{case} (n={len(group)})')
        ax.axvline(threshold, color='#444444', linestyle='--', label=f'behavioral threshold = {threshold:.8g}')
        ax.axhline(.5, color='#444444', linestyle=':', label='latent threshold = 0.5')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Behavioral score: P(harmful)',
               ylabel='Latent score: oriented P(harmful)',
               title=f'True label: {"harmful" if label else "safe"}; n={len(subset)}')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), fontsize=7, frameon=False)
    fig.suptitle(f'{dataset} / {model} / layer {int(layer):02d}\n'
                 f'Calibrated behavioral threshold (train fitted); test split only')
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run(root):
    root = Path(root)
    threshold_path = root / 'behavioral_threshold_analysis/behavioral_thresholds.csv'
    thresholds = pd.read_csv(threshold_path)
    required = {'dataset', 'model', 'behavioral_threshold', 'n_train', 'n_test'}
    if not required.issubset(thresholds) or thresholds.duplicated(['dataset', 'model']).any():
        raise ValueError('Missing or duplicate saved thresholds')
    expected = {(d, m) for d in DATASETS for m in MODELS}
    if set(zip(thresholds.dataset, thresholds.model)) != expected:
        raise ValueError('Exactly one existing threshold per dataset/model is required')
    plans, sources = [], []
    for row in thresholds.itertuples(index=False):
        path = root / row.dataset / 'analysis/checkpoint1_compatible' / row.model / 'scores.csv'
        split_path = root / row.dataset / 'artifacts' / row.model / 'split.npz'
        with np.load(split_path, allow_pickle=False) as split:
            train, test = split['train_idx'], split['test_idx']
        if len(train) != row.n_train or len(test) != row.n_test:
            raise ValueError('Saved calibration sample counts differ from split.npz')
        objects = analyze_scores(pd.read_csv(path), row.dataset, row.model,
                                 row.behavioral_threshold, train, test)
        plans.append((row.dataset, row.model, objects))
        sources.append({'dataset': row.dataset, 'model': row.model,
                        'scores': str(path), 'scores_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest()})
    output = root / 'mismatch_calibrated_test'
    output.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(output / 'behavioral_thresholds.csv', index=False)
    summaries, model_summaries = [], []
    for dataset, model, objects in plans:
        folder = output / dataset / model
        (folder / 'plots').mkdir(parents=True, exist_ok=True)
        objects.to_csv(folder / 'mismatch_cases.csv', index=False)
        summary = summarize(objects)
        summary.to_csv(folder / 'mismatch_summary.csv', index=False)
        summaries.append(summary)
        model_summary = summarize(objects, keys=['dataset', 'model'])
        model_summary['n_test_objects'] = objects.sample_idx.nunique()
        model_summary['n_layers'] = objects.layer.nunique()
        model_summaries.append(model_summary)
        for layer, group in objects.groupby('layer', sort=True):
            plot_layer(group, folder / 'plots' / f'layer_{int(layer):02d}.png', dataset, model, layer)
        print(f'{dataset}/{model}: {objects.sample_idx.nunique()} test objects × {objects.layer.nunique()} layers', flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(output / 'mismatch_summary.csv', index=False)
    pd.concat(model_summaries, ignore_index=True).to_csv(output / 'mismatch_model_summary.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps({
        'sources': sources, 'threshold_source': str(threshold_path),
        'threshold_source_sha256': hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
        'behavioral_rule': 'P(harmful) >= saved train threshold',
        'latent_rule': 'saved oriented latent_score > 0.5',
        'aligned': 'predicted class equals true_label',
        'split': 'test only, verified against split.npz for every layer',
        'model_summary_denominator': 'test objects × layers; no layer selection',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=PROJECT / 'results/checkpoint2')
    run(parser.parse_args().results_root)
