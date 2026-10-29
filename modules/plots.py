"""Figures for model notebooks and four independent dataset/size slices."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .experiments import CASES, DEPTHS, depth_summary

COLORS = ['#229954', '#d97706', '#2874a6', '#b03a2e']
LABELS = ['Both aligned', 'Behavior aligned / latent misaligned',
          'Behavior misaligned / latent aligned', 'Both misaligned']


def _title(result):
    manifest = result['manifest']
    sig = manifest['signature']
    suffix = ' | SMOKE/PARTIAL' if not (manifest['full_hyperparameters'] and manifest['complete_layers']) else ''
    return f"{sig['dataset']} | {sig['model']}{suffix}"


def _finish(fig, folder, name, *, rect=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(folder / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    if plt.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)


def _curve(ax, frame, metric, label, color=None, x='layer'):
    stats = frame.groupby(x)[metric].agg(['mean', 'std']).sort_index()
    xx = stats.index.to_numpy()
    mean, std = stats['mean'].to_numpy(), stats['std'].fillna(0).to_numpy()
    line, = ax.plot(xx, mean, label=label, color=color, marker='.', markersize=3)
    ax.fill_between(xx, mean-std, mean+std, alpha=.12, color=line.get_color())
    ax.grid(alpha=.2)


def accuracy(results):
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 3.6*len(results)), squeeze=False, sharey=True)
    for row, (protocol, result) in enumerate(results.items()):
        metrics = result['metrics'].query("calibration == 'calibrated'")
        for col, split in enumerate(('train', 'test')):
            ax = axes[row, col]
            frame = metrics.loc[metrics.split.eq(split)]
            for metric, label, color in [('latent_accuracy', 'Latent: train orientation', '#2874a6'),
                                          ('behavioral_accuracy', 'Behavior: train threshold', '#d97706'),
                                          ('behavioral_accuracy_05', 'Behavior: threshold 0.5', '#229954')]:
                _curve(ax, frame, metric, label, color)
            ax.set(title=f'{protocol} | {split}', xlabel='Layer', ylabel='Accuracy', ylim=(0, 1))
            ax.legend(fontsize=8)
    first = next(iter(results.values()))
    sig = first['manifest']['signature']
    fig.suptitle(f"{_title(first)} | mean ± fold std (pairwise)", y=1.02)
    _finish(fig, first['folder'].parent / 'plots', 'accuracy')


def mismatch(results, *, baseline=False):
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3.6*len(results)), squeeze=False, sharey=True)
    calibration = 'baseline_alignment' if baseline else 'calibrated'
    for row, (protocol, result) in enumerate(results.items()):
        frame = result['metrics'].query('calibration == @calibration')
        for col, split in enumerate(('train', 'test')):
            ax = axes[row, col]
            selected = frame.loc[frame.split.eq(split)]
            for case, label, color in zip(CASES, LABELS, COLORS):
                _curve(ax, selected, f'{case}_fraction', label, color)
            ax.set(title=f'{protocol} | {split}', xlabel='Layer', ylabel='Fraction of evaluated objects', ylim=(0, 1))
            ax.legend(fontsize=7)
    first = next(iter(results.values()))
    sig = first['manifest']['signature']
    fig.suptitle(f"{_title(first)} | {calibration}", y=1.02)
    _finish(fig, first['folder'].parent / 'plots', f'mismatch_{calibration}')


def pa_comparison(results):
    for metric in ('esa', 'pc', 'ci'):
        fig, axes = plt.subplots(len(results), 2, figsize=(12, 3.6*len(results)), squeeze=False, sharey=True)
        for row, (protocol, result) in enumerate(results.items()):
            frame = result['pa_metrics']
            for col, split in enumerate(('train', 'test')):
                ax = axes[row, col]
                for variant, color in [('reference', '#d97706'), ('consistent', '#2874a6')]:
                    selected = frame.loc[frame.split.eq(split) & frame.pa_variant.eq(variant)]
                    _curve(ax, selected, metric, variant, color)
                ax.set(title=f'{protocol} | {split}', xlabel='Layer', ylabel=metric.upper())
                ax.set_ylim(*{'esa': (.5, 1), 'pc': (-1, 1), 'ci': (0, 2)}[metric])
                ax.legend()
        first = next(iter(results.values()))
        sig = first['manifest']['signature']
        fig.suptitle(f"{_title(first)} | PA {metric.upper()} on paired rows", y=1.02)
        _finish(fig, first['folder'].parent / 'plots', f'pa_{metric}')
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 3.6*len(results)), squeeze=False)
    for row, (protocol, result) in enumerate(results.items()):
        for col, variant in enumerate(('reference', 'consistent')):
            ax = axes[row, col]
            frame = result['pa_metrics'].query("split == 'test' and pa_variant == @variant")
            frame = frame.groupby('layer')[['pc', 'ci', 'esa']].mean()
            points = ax.scatter(frame.pc, frame.ci, c=frame.esa, cmap='viridis', vmin=.5, vmax=1)
            fig.colorbar(points, ax=ax, label='ESA')
            ax.set(title=f'{protocol} | {variant} | test fold means', xlabel='PC', ylabel='CI',
                   xlim=(-1, 1), ylim=(0, 2))
            ax.grid(alpha=.2)
    fig.suptitle(f'{_title(first)} | one point per layer', y=1.02)
    _finish(fig, first['folder'].parent / 'plots', 'pa_pc_ci')


def scatter(result, *, layer=None):
    frame = result['predictions'].query("split == 'test'")
    layer = int(frame.layer.max()) if layer is None else layer
    frame = frame.loc[frame.layer.eq(layer)]
    if frame.empty:
        raise ValueError(f'No saved test predictions for layer {layer}')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (label, title) in zip(axes, [(0, 'Safe'), (1, 'Harmful')]):
        selected = frame.loc[frame.label.eq(label)]
        for case, case_label, color in zip(CASES, LABELS, COLORS):
            group = selected.loc[selected.category_calibrated.eq(case)]
            ax.scatter(group.behavioral_score, group.latent_score, s=10, alpha=.55,
                       color=color, label=f'{case_label}: {len(group)}')
        for value in selected.behavioral_threshold.unique():
            ax.axvline(value, color='grey', linestyle='--', alpha=.4)
        ax.axhline(.5, color='grey', linestyle='--')
        ax.set(title=title, xlabel='P(harmful), behavior', ylabel='P(harmful), oriented latent',
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    sig = result['manifest']['signature']
    fig.suptitle(f"{_title(result)} | {sig['protocol']} | test | layer {layer}\n"
                 'Vertical lines: train thresholds per fold', y=1.06)
    _finish(fig, result['folder'] / 'plots', f'mismatch_layer_{layer:02d}')


def aggregation(long, grouped, output, *, metrics=('latent_accuracy', 'behavioral_accuracy', 'esa', 'pc', 'ci'),
                splits=('test',), calibrations=('calibrated', 'reference', 'consistent'), smoke=False):
    """Four categorical depth quarters; group curves only, matching old figures."""
    if 'std' not in grouped or 'n_folds' not in grouped:
        grouped, _ = depth_summary(long, require_complete=bool(grouped.complete.all()))
    grouped = grouped.loc[grouped.metric.isin(metrics) & grouped.split.isin(splits)
                          & grouped.calibration.isin(calibrations)]
    folder = Path(output) / 'aggregation/plots'
    families = [
        ('accuracy', [('latent_accuracy', 'Latent accuracy'),
                      ('behavioral_accuracy', 'Behavioral calibrated accuracy')]),
        ('mismatch', [(f'{case}_fraction', title) for case, title in zip(CASES, LABELS)]),
        ('pa', [('esa', 'ESA'), ('ci', 'CI'), ('pc', 'PC')])]
    for (protocol, split, calibration), selected in grouped.groupby(['protocol', 'split', 'calibration']):
        for family, candidates in families:
            panels = [(metric, title) for metric, title in candidates if metric in set(selected.metric)]
            if not panels:
                continue
            if family == 'pa':
                fig, axes = plt.subplots(2, len(panels), figsize=(5*len(panels), 8), squeeze=False)
                for row, size in enumerate(('big', 'small')):
                    for col, (metric, title) in enumerate(panels):
                        _depth_curves(axes[row, col], selected, metric, size=size)
                        axes[row, col].set(title=title, ylabel=f'{size.capitalize()} models — mean {title}')
            else:
                ncols = 2 if len(panels) > 1 else 1
                nrows = int(np.ceil(len(panels)/ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.7*nrows), squeeze=False)
                for ax, (metric, title) in zip(axes.flat, panels):
                    _depth_curves(ax, selected, metric)
                    ax.set(title=title, ylabel='Accuracy' if family == 'accuracy' else 'Share of objects')
                    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                    if family == 'mismatch':
                        ax.set_ylim(0, 1)
                for ax in list(axes.flat)[len(panels):]:
                    ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91),
                       ncol=4, frameon=False, fontsize=10)
            title = {'accuracy': 'Accuracy', 'mismatch': 'Mismatch categories',
                     'pa': f'{calibration.upper()} PA-CCS'}[family]
            suffix = ' | SMOKE' if smoke else ''
            fig.suptitle(f'{title} by relative depth — {split.upper()} | {protocol}{suffix}\n'
                         'Big: Gemma 9B / 9B-IT; small: Gemma 2B / 2B-IT + DeBERTa', fontsize=13, y=.99)
            fig.text(.5, .025, 'Blue = mixed; orange = not. Solid = big; dashed = small. '
                     'Shading = std of group means across folds (not a confidence interval).',
                     ha='center', fontsize=9)
            _finish(fig, folder, f'{protocol}_{split}_{calibration}_{family}_depth', rect=(0, .07, 1, .86))


def _depth_curves(ax, frame, metric, *, size=None):
    labels = ['Early\n0–25%', 'Middle-early\n25–50%', 'Middle-late\n50–75%', 'Late\n75–100%']
    for dataset, color in [('mixed', '#2874a6'), ('not', '#d97706')]:
        for group, style in [('big', '-'), ('small', '--')]:
            if size is not None and size != group:
                continue
            curve = frame.loc[frame.dataset.eq(dataset) & frame['size'].eq(group) & frame.metric.eq(metric)]
            curve = curve.set_index('depth').reindex(DEPTHS)
            mean, std = curve['mean'].to_numpy(float), curve['std'].fillna(0).to_numpy(float)
            if not np.isfinite(mean).any():
                continue
            ax.plot(range(4), mean, color=color, linestyle=style, marker='o', markersize=5,
                    linewidth=2, label=f'{group} / {dataset}' if size is None else dataset)
            ax.fill_between(range(4), mean-std, mean+std, color=color, alpha=.15, linewidth=0)
            if curve.complete.isna().any() or not curve.complete.fillna(False).all():
                ax.text(.02, .02, 'INCOMPLETE COVERAGE', transform=ax.transAxes, fontsize=8, color='#b03a2e')
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_xlim(-.12, 3.12)
    ax.grid(alpha=.22)


def hidden_geometry(payloads):
    """Two dataset rows, reference/shared projections and real CCS histograms."""
    palette = ['#2171b5', '#238b45', '#d98b28', '#cb4665']
    group_labels = ['safe, Yes', 'safe, No', 'harmful, Yes', 'harmful, No']
    for protocol, datasets in payloads.items():
        order = [dataset for dataset in ('mixed', 'not') if dataset in datasets]
        fig, axes = plt.subplots(len(order), 4, figsize=(20, 4.3*len(order)), squeeze=False)
        for row, dataset in enumerate(order):
            payload = datasets[dataset]
            for col, variant in enumerate(('reference', 'consistent')):
                ax = axes[row, col]
                for points, color, label in zip(payload['xy'][variant], palette, group_labels):
                    ax.scatter(points[:, 0], points[:, 1], color=color, s=9, alpha=.5, label=label)
                    ax.scatter(points[:, 0].mean(), points[:, 1].mean(), marker='x', color=color,
                               s=75, linewidths=2)
                ax.axhline(0, color='#8293a0', linewidth=.7)
                ax.axvline(0, color='#8293a0', linewidth=.7)
                ax.set(title='Reference: four group means' if variant == 'reference' else 'CCS-consistent: train Yes/No means',
                       xlabel='Projection on fixed unit CCS direction',
                       ylabel=f'{dataset} | layer {payload["layer"]}\nOrthogonal PCA component')
            cloud = np.concatenate([g for variant in payload['xy'].values() for g in variant])
            xpad, ypad = max(np.ptp(cloud[:, 0])*.08, .001), max(np.ptp(cloud[:, 1])*.08, .001)
            for ax in axes[row, :2]:
                ax.set(xlim=(cloud[:, 0].min()-xpad, cloud[:, 0].max()+xpad),
                       ylim=(cloud[:, 1].min()-ypad, cloud[:, 1].max()+ypad))
            for col, variant in enumerate(('reference', 'consistent'), 2):
                ax = axes[row, col]
                for name, color in [('safe', '#238b45'), ('harmful', '#cb4665')]:
                    score = payload['scores'][variant][name]
                    ax.hist(score, bins=np.linspace(0, 1, 21), weights=np.ones(len(score))/len(score),
                            histtype='step', linewidth=1.8, color=color, label=name)
                metric = payload['metrics'][variant]
                ax.axvline(.5, linestyle=':', color='#8293a0', linewidth=1)
                ax.set(title=f'{variant.capitalize()}: ESA={metric["esa"]:.3f}\nPC={metric["pc"]:.3f}, CI={metric["ci"]:.3f}',
                       xlabel='Train-oriented CCS harmfulness score', ylabel='Within-class fraction', xlim=(0, 1))
                ax.legend(fontsize=8)
                ax.grid(alpha=.16)
        first = datasets[order[0]]
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
        outside = ', '.join(f'{d}: {datasets[d]["counterpart_outside_split"]}' for d in order)
        scope = f'pairwise test, fold {first["fold"]}' if protocol == 'pairwise' else 'rowwise historical PA pairs'
        fig.suptitle(f'{first["model"]}: real last-layer hidden states and CCS scores | {scope}\n'
                     'Common axes for both centering variants within each dataset. × = group mean.', fontsize=14, y=.99)
        fig.text(.5, .012, f'Counterparts outside test — {outside}. Fixed probe; projection-axis sign is arbitrary. '
                 'PCA is descriptive only; folds are not pooled.', ha='center', fontsize=9)
        _finish(fig, first['figure_folder'], f'hidden_geometry_fold_{first["fold"]}', rect=(0, .08, 1, .92))
