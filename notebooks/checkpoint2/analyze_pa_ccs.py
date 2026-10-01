"""CCS-consistent PA-CCS from saved probes and hidden states; never trains."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import DATASETS, MODELS, PROJECT, load_cache, load_dataset, reference_module
from pa_ccs_reference import pa_metrics
from probes import load_probes, prepare_layer, probe_signature, restore_probe
from plot_pa_ccs_metrics_reference import (
    _dataset_means, _model_mean, _plot_model, _plot_summary, load_layer_metrics,
)


def centered_layer(cache, saved, layer, index):
    """Use the CCS L2 routine and the two saved prediction offsets, exactly once."""
    pos, neg = prepare_layer(cache, layer, 'checkpoint1')
    pos -= saved['prediction_offset_pos'][index]
    neg -= saved['prediction_offset_neg'][index]
    return pos, neg


class SavedCenterProbabilities:
    """Only replace reference PA preprocessing; inherit all metric formulas."""

    def get_contrastive_probas(self, *groups):
        import torch
        with torch.no_grad():
            return tuple(self.best_probe(torch.as_tensor(
                group, dtype=torch.float32, device=self.device)).cpu().numpy()
                for group in groups)


def consistent_probe(ccs_class, saved, index):
    cls = type('CCSConsistentPA', (SavedCenterProbabilities, ccs_class), {})
    return restore_probe(cls, saved, index, device='cpu')


def esa_regime(esa):
    """Use the ESA ranges from polarity-probing/teaser/ccs_teaser.png, panel E.

    ESA is sign-invariant and therefore >= 0.5. The teaser's probability-level
    polarity cases have no published aggregate-layer rule; do not invent one.
    """
    values = np.asarray(esa)
    if not np.isfinite(values).all() or not ((values >= .5) & (values <= 1)).all():
        raise ValueError('Expected sign-invariant ESA in [0.5, 1]')
    return np.where(values >= .75, 'strong ESA (>= 0.75)', '0.5 <= ESA < 0.75')


def plot_scatter(frame, destination, dataset):
    from plot_pa_ccs_metrics_reference import plt
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    for category, color in [('strong ESA (>= 0.75)', '#22b500'),
                            ('0.5 <= ESA < 0.75', '#b8db00')]:
        group = frame[frame.esa_regime.eq(category)]
        ax.scatter(group.pc, group.ci, color=color, edgecolors='#444444',
                   linewidths=.35, alpha=.8, s=36, label=f'{category} (n={len(group)})')
    ax.set(xlabel='Polar Consistency (PC)', ylabel='Contradiction Index (CI)',
           title=f'CCS-consistent PA-CCS — {dataset}\nOne point per model/layer; n={len(frame)}')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


DEPTH_GROUPS = ['Early', 'Middle-early', 'Middle-late', 'Late']
MODEL_GROUPS = {
    'gemma-2-9b': 'big', 'gemma-2-9b-it': 'big',
    'gemma-2-2b': 'small', 'gemma-2-2b-it': 'small',
    'deberta-hate-tuned': 'small',
}


def depth_group_means(layers):
    """Average four relative-depth quarters within models, then across models."""
    frame = layers.copy()
    frame['model_group'] = frame.model.map(MODEL_GROUPS)
    if frame.model_group.isna().any() or frame.duplicated(['dataset', 'model', 'layer']).any():
        raise ValueError('Unknown model group or duplicate model/layer')
    max_layer = frame.model.map({name: spec.n_layers - 1 for name, spec in MODELS.items()})
    frame['relative_depth'] = frame.layer / max_layer
    if not frame.relative_depth.between(0, 1).all():
        raise ValueError('Layer outside the saved model depth')
    frame['depth_group'] = pd.cut(frame.relative_depth, [0, .25, .5, .75, 1],
                                  labels=DEPTH_GROUPS, include_lowest=True)
    per_model = (frame.groupby(['model_group', 'dataset', 'model', 'depth_group'],
                               observed=True, as_index=False)
                 .agg(esa=('esa', 'mean'), ci=('ci', 'mean'), pc=('pc', 'mean'),
                      n_layers=('layer', 'size'), first_layer=('layer', 'min'),
                      last_layer=('layer', 'max')))
    means = (per_model.groupby(['model_group', 'dataset', 'depth_group'],
                               observed=True, as_index=False)
             .agg(esa=('esa', 'mean'), ci=('ci', 'mean'), pc=('pc', 'mean'),
                  n_models=('model', 'nunique'), n_layers=('n_layers', 'sum')))
    return per_model, means


def plot_group_depth_means(layers, output, *, title='CCS-consistent PA-CCS'):
    """One 2×3 figure: big/small rows, ESA/CI/PC columns, dataset colors."""
    from plot_pa_ccs_metrics_reference import plt
    per_model, means = depth_group_means(layers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(output / 'pa_ccs_model_depth_means.csv', index=False)
    means.to_csv(output / 'pa_ccs_group_depth_means.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True,
                             sharey='col', layout='constrained')
    colors = {'mixed': '#2874a6', 'not': '#d97706'}
    x = np.arange(len(DEPTH_GROUPS))
    for row, group in enumerate(['big', 'small']):
        for col, metric in enumerate(['esa', 'ci', 'pc']):
            ax = axes[row, col]
            for dataset, color in colors.items():
                values = (means[(means.model_group == group) & (means.dataset == dataset)]
                          .set_index('depth_group').reindex(DEPTH_GROUPS))
                ax.plot(x, values[metric], color=color, marker='o',
                        markersize=6, linewidth=2, label=dataset)
            ax.set_title(metric.upper(), fontsize=14)
            ax.set_xticks(x, ['Early\n0–25%', 'Middle-early\n25–50%',
                             'Middle-late\n50–75%', 'Late\n75–100%'])
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_ylabel(f'{group.capitalize()} models — mean {metric.upper()}')
            ax.grid(alpha=.25)
            if metric == 'esa':
                ax.set_ylim(.5, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper right', frameon=False)
    fig.suptitle(f'{title} by relative depth\n'
                 'Big: Gemma 9B / 9B-it; small: Gemma 2B / 2B-it + DeBERTa', fontsize=16)
    fig.supxlabel('Depth = layer / last layer (embedding = 0). '
                  'Mean within each model and depth group, then equal-weight mean across models.', fontsize=10)
    destination = output / 'summary' / 'mixed_not_group_depth_means.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return means


def analyze_model(root, dataset, model):
    spec = MODELS[model]
    data = load_dataset(dataset=dataset)
    path = root / dataset / 'artifacts' / model
    cache = load_cache(path, spec, data)
    with np.load(path / 'ccs_probes.npz', allow_pickle=False) as f:
        stored = json.loads(f['metadata_json'].item())
    config = stored['config']
    if (config['preprocessing'] != 'checkpoint1' or not config['linear']
            or config['var_normalize'] or config['predict_normalize']):
        raise ValueError(f'{dataset}/{model}: expected existing checkpoint1 linear probes')
    train, test = cache['train_idx'], cache['test_idx']
    if stored['train_idx'] != train.tolist() or stored['test_idx'] != test.tolist():
        raise ValueError(f'{dataset}/{model}: probe split differs from split.npz')
    saved = load_probes(path / 'ccs_probes.npz',
                        probe_signature(cache, spec, train, test, config))
    if saved is None:
        raise ValueError(f'{dataset}/{model}: compatible saved probes required; no training allowed')
    source = root / dataset / 'analysis/checkpoint1_compatible' / model / 'layer_metrics.csv'
    frame = load_layer_metrics(source, dataset, model)
    if frame.layer.tolist() != sorted(saved['layer'].tolist()):
        raise ValueError(f'{dataset}/{model}: source metric layers differ from saved probes')
    labels = 1 - data.raw.is_harmfull_opposition.to_numpy()
    cls = reference_module('ccs').CCS
    for index, layer in enumerate(saved['layer']):
        pos, neg = centered_layer(cache, saved, int(layer), index)
        ccs = consistent_probe(cls, saved, index)
        esa = ccs.get_acc(neg[test], pos[test], labels[test])
        mask = frame.layer.eq(layer)
        if not np.isclose(esa, frame.loc[mask, 'accuracy'].item(), atol=1e-12, rtol=0):
            raise ValueError(f'{dataset}/{model}/{layer}: reconstructed ESA differs from saved CCS')
        for key, value in pa_metrics(ccs, pos, neg, test).items():
            frame.loc[mask, key] = value
    frame.insert(0, 'model', model)
    frame.insert(0, 'dataset', dataset)
    frame['esa'] = frame.accuracy
    frame['pc'] = frame['polar_consistency_↓']
    frame['ci'] = frame['contradiction_idx_↓']
    frame['esa_regime'] = esa_regime(frame.esa)
    frame['preprocessing'] = 'CCS-consistent: L2 + saved train Yes/No offsets'
    provenance = {
        'dataset': dataset, 'model': model,
        'probe': str(path / 'ccs_probes.npz'),
        'probe_sha256': hashlib.sha256((path / 'ccs_probes.npz').read_bytes()).hexdigest(),
        'hidden_sha256': stored['hidden_sha256'],
        'source_metrics_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'offsets': ['prediction_offset_pos', 'prediction_offset_neg'],
        'n_train': len(train), 'n_test': len(test),
    }
    return frame, provenance


def run(root):
    import torch
    torch.set_num_threads(2)
    root = Path(root)
    output = root / 'pa_ccs_analysis'
    frames, sources, means = [], [], []
    for dataset in DATASETS:
        for model in MODELS:
            frame, provenance = analyze_model(root, dataset, model)
            folder = output / 'by_model' / dataset
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_csv(folder / f'{model}.csv', index=False)
            _plot_model(frame, folder / f'{model}.png', dataset, model)
            frames.append(frame)
            sources.append(provenance)
            means.append(_model_mean(frame, dataset, model))
            print(f'{dataset}/{model}: {len(frame)} layers; saved centers; no training', flush=True)
    layers = pd.concat(frames, ignore_index=True)
    layers.to_csv(output / 'pa_ccs_layer_metrics.csv', index=False)
    plot_group_depth_means(layers, output)
    model_means = pd.DataFrame(means)
    model_means.to_csv(output / 'pa_ccs_model_means.csv', index=False)
    dataset_means = _dataset_means(model_means)
    dataset_means.to_csv(output / 'pa_ccs_dataset_means.csv', index=False)
    for row in dataset_means.itertuples(index=False):
        _plot_summary(row, output / 'summary' / f'{row.dataset}_mean_metrics.png')
        plot_scatter(layers[layers.dataset.eq(row.dataset)],
                     output / 'summary' / f'{row.dataset}_pc_ci.png', row.dataset)
    (output / 'manifest.json').write_text(json.dumps({
        'sources': sources,
        'esa_regime_source': 'https://github.com/SadSabrina/polarity-probing/blob/main/teaser/ccs_teaser.png (panel E)',
        'metrics': 'Unchanged reference ESA, get_agreement and get_contradiction_idx',
        'pairing': 'Safe second-half test rows with first-half counterparts; counterparts may be train',
        'preprocessing': 'prepare_layer checkpoint1, then saved prediction_offset_pos/neg; no group centering',
    }, indent=2) + '\n')
    return layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=PROJECT / 'results/checkpoint2')
    run(parser.parse_args().results_root)
