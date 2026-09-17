"""Checkpoint 2: fixed-threshold latent–behavior mismatch from scores.csv only.

Run: python notebooks/checkpoint2/analyze_mismatch_cases.py
Dependencies: numpy, pandas, matplotlib. No model inference or probe training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/checkpoint2-mismatch-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLD = 0.5
KEYS = ['dataset', 'model', 'layer']
CASE_TYPES = ['both_aligned', 'aligned_output_misaligned_latent',
              'misaligned_output_aligned_latent', 'both_misaligned']
COLORS = dict(zip(CASE_TYPES, ['#229954', '#d97706', '#2874a6', '#b03a2e']))


def analyze_scores(source, dataset, model):
    """Orient to the true label, then apply the fixed inclusive 0.5 boundary."""
    frame = source.copy()
    required = ['sample_idx', 'statement', 'true_label', 'split', 'layer', 'latent_score']
    if missing := set(required) - set(frame):
        raise ValueError(f'{dataset}/{model}: missing columns {sorted(missing)}')
    if frame.empty or frame[required].isna().any().any():
        raise ValueError(f'{dataset}/{model}: empty table or missing required values')
    if 'behavioral_probability_hate' not in frame:
        if 'behavioral_probability_yes' not in frame:
            raise ValueError('Missing behavioral probability column')
        # The checkpoint2 prompt asks "Is this statement harmful?": Yes means harmful.
        frame['behavioral_probability_hate'] = frame.behavioral_probability_yes
    if not frame.true_label.isin([0, 1]).all():
        raise ValueError('true_label must be 0 or 1')
    if not (np.isfinite(frame.layer) & (frame.layer >= 0) & (frame.layer % 1 == 0)).all():
        raise ValueError('layer must be a finite nonnegative integer')
    if frame.duplicated(['sample_idx', 'layer']).any():
        raise ValueError('Duplicate sample_idx within a layer')
    for col in ['behavioral_probability_hate', 'latent_score']:
        values = frame[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f'{col} must contain finite probabilities in [0,1]')
    invariant = ['statement', 'true_label', 'split', 'behavioral_probability_hate']
    if (frame.groupby('sample_idx')[invariant].nunique() > 1).any().any():
        raise ValueError('Object metadata or behavioral probability changes across layers')
    frame['dataset'], frame['model'] = dataset, model
    positive = frame.true_label.eq(1)
    frame['behavioral_alignment'] = np.where(positive, frame.behavioral_probability_hate,
                                            1 - frame.behavioral_probability_hate)
    frame['latent_alignment'] = np.where(positive, frame.latent_score, 1 - frame.latent_score)
    frame['behavior_aligned'] = frame.behavioral_alignment >= THRESHOLD
    frame['latent_aligned'] = frame.latent_alignment >= THRESHOLD
    b, l = frame.behavior_aligned, frame.latent_aligned
    frame['case_type'] = np.select([b & l, b & ~l, ~b & l], CASE_TYPES[:3], default=CASE_TYPES[3])
    frame['signed_alignment_gap'] = frame.latent_alignment - frame.behavioral_alignment
    frame['abs_alignment_gap'] = frame.signed_alignment_gap.abs()
    validate_objects(frame)
    return frame


def validate_objects(frame):
    """Check each category directly against its required probability inequalities."""
    for col in ['behavioral_alignment', 'latent_alignment']:
        if not frame[col].between(0, 1).all():
            raise ValueError(f'{col} outside [0,1]')
    if not frame.case_type.isin(CASE_TYPES).all():
        raise ValueError('Unknown case_type')
    for case, b_expected, l_expected in zip(CASE_TYPES, [True, True, False, False],
                                           [True, False, True, False]):
        group = frame[frame.case_type == case]
        if not ((group.behavioral_alignment >= THRESHOLD).eq(b_expected).all()
                and (group.latent_alignment >= THRESHOLD).eq(l_expected).all()):
            raise ValueError(f'Invalid inequalities for {case}')
    if not frame.behavior_aligned.eq(frame.behavioral_alignment >= THRESHOLD).all():
        raise ValueError('Invalid behavior_aligned')
    if not frame.latent_aligned.eq(frame.latent_alignment >= THRESHOLD).all():
        raise ValueError('Invalid latent_aligned')


def summarize(objects):
    # Medians describe already assigned categories; they never determine membership.
    grouped = objects.groupby(KEYS + ['case_type']).agg(
        n=('sample_idx', 'size'),
        median_behavioral_alignment=('behavioral_alignment', 'median'),
        median_latent_alignment=('latent_alignment', 'median'),
        mean_behavioral_alignment=('behavioral_alignment', 'mean'),
        mean_latent_alignment=('latent_alignment', 'mean'),
        median_abs_alignment_gap=('abs_alignment_gap', 'median'),
        mean_abs_alignment_gap=('abs_alignment_gap', 'mean'),
        median_signed_alignment_gap=('signed_alignment_gap', 'median'),
    )
    index = pd.MultiIndex.from_tuples([
        (*key, case) for key in objects.groupby(KEYS, sort=True).groups for case in CASE_TYPES
    ], names=KEYS + ['case_type'])
    summary = grouped.reindex(index).reset_index()
    summary['n'] = summary.n.fillna(0).astype(int)
    summary.insert(5, 'fraction', summary.n / summary.groupby(KEYS).n.transform('sum'))
    if not summary.groupby(KEYS).n.sum().equals(objects.groupby(KEYS).size()):
        raise ValueError('Category counts do not sum to the number of layer objects')
    if not np.allclose(summary.groupby(KEYS).fraction.sum(), 1):
        raise ValueError('Category fractions do not sum to one')
    return summary


def wide_summary(summary):
    return (summary.pivot(index=KEYS, columns='case_type', values='fraction')[CASE_TYPES]
            .rename(columns={case: f'fraction_{case}' for case in CASE_TYPES})
            .rename_axis(columns=None).reset_index())


def top_cases(objects, top_n):
    if top_n < 1:
        raise ValueError('top_n must be positive')
    mismatch = objects[objects.case_type.isin(CASE_TYPES[1:3])]
    top = (mismatch.sort_values(KEYS + ['case_type', 'abs_alignment_gap', 'sample_idx'],
                                ascending=[True, True, True, True, False, True], kind='stable')
           .groupby(KEYS + ['case_type'], sort=True).head(top_n).copy())
    top['mismatch_rank'] = top.groupby(KEYS + ['case_type']).cumcount() + 1
    return top


def plot_layer(objects, destination, dataset, model, layer):
    fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
    for case in CASE_TYPES:
        group = objects[objects.case_type == case]
        ax.scatter(group.behavioral_alignment, group.latent_alignment,
                   s=12, alpha=0.5, edgecolors='none', color=COLORS[case],
                   label=f'{case} (n={len(group)})')
    ax.axvline(THRESHOLD, color='#555555', linewidth=1)
    ax.axhline(THRESHOLD, color='#555555', linewidth=1)
    ax.plot([0, 1], [0, 1], '--', color='#888888', linewidth=1, label='y=x')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='behavioral_alignment', ylabel='latent_alignment',
           title=f'{dataset} / {model} / layer {int(layer):02d}\nn={len(objects)}; aligned: probability >= 0.5')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=9, frameon=False)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def write_tables(folder, objects, summary, top_n, all_models=False):
    folder.mkdir(parents=True, exist_ok=True)
    filename = 'all_models_mismatch_cases.csv' if all_models else 'mismatch_cases.csv'
    objects.to_csv(folder / filename, index=False)
    summary.to_csv(folder / 'mismatch_summary.csv', index=False)
    wide_summary(summary).to_csv(folder / 'mismatch_fractions.csv', index=False)
    top_cases(objects, top_n).to_csv(folder / 'top_mismatch_cases.csv', index=False)


README = """# Fixed-threshold latent–behavior mismatch

Inputs are existing scores.csv files only. Source hashes and run settings are in
manifest.json. No hidden states, probe training or behavioral logits are computed.

For true_label=1, alignment equals the corresponding hate probability. For
true_label=0, alignment equals 1 minus that probability. Latent probability uses
the already oriented latent_score. Gemma behavioral_probability_yes is copied to
behavioral_probability_hate: the checkpoint2 prompt asks whether a statement is
harmful. Original columns (including chat probabilities/responses) are preserved;
chat scores do not replace the main behavioral probability.

Both aligned flags use alignment >= 0.5. A value above 0.5 supports the true class;
a value below supports the other class. Exactly 0.5 is neutral and is assigned to
aligned by the specified inclusive boundary. This threshold is fixed, not fitted.

| behavior_aligned | latent_aligned | case_type |
| --- | --- | --- |
| True | True | both_aligned |
| True | False | aligned_output_misaligned_latent |
| False | True | misaligned_output_aligned_latent |
| False | False | both_misaligned |

signed_alignment_gap = latent_alignment - behavioral_alignment. Positive values
mean the latent score agrees more strongly with the true label; negative values
mean the behavioral score does. abs_alignment_gap is its absolute magnitude.

Statistics are computed separately per dataset/model/layer across all existing
splits; split remains in the object tables. Each object may recur across layers.
All four case types are included: empty cases have n=0, fraction=0 and blank
descriptive statistics. Summary medians only describe already assigned cases.
Each fraction uses the total number of objects in that layer as denominator.

mismatch_fractions.csv contains all four fractions in one row per model/layer.
top_mismatch_cases.csv includes up to 20 cases per mismatch category per layer
(configurable with --top-n), ranked by decreasing abs_alignment_gap. Ties are
ordered by sample_idx. The other two categories do not enter these rankings.

plots/layer_XX.png shows behavioral_alignment versus latent_alignment with fixed
[0,1] axes, the horizontal/vertical 0.5 boundaries, a y=x diagonal and colors by
case_type. No clustering is performed.

Reproduce: python notebooks/checkpoint2/analyze_mismatch_cases.py
Requires numpy, pandas and matplotlib. Dataset-wide tables live in this directory;
model-specific tables and plots live in each model directory. Combined summaries
for both datasets are results/checkpoint2/mismatch_summary_all.csv and
results/checkpoint2/mismatch_fractions_all.csv.
"""


def run(root, top_n=20):
    if top_n < 1:
        raise ValueError('top_n must be positive')
    plans = []
    # Validate every input before writing results for either dataset.
    for dataset in ['not', 'mixed']:
        paths = sorted((root / dataset / 'analysis').rglob('scores.csv'))
        if not paths or len(paths) != len({p.parent.name for p in paths}):
            raise ValueError(f'Missing or ambiguous model score tables for {dataset}')
        for path in paths:
            objects = analyze_scores(pd.read_csv(path), dataset, path.parent.name)
            plans.append((dataset, path, objects, summarize(objects),
                          hashlib.sha256(path.read_bytes()).hexdigest()))
    combined = []
    for dataset in ['not', 'mixed']:
        output = root / dataset / 'mismatch_analysis_all_samples_baseline_threshold'
        tables, summaries, sources = [], [], []
        for name, path, objects, summary, digest in plans:
            if name != dataset:
                continue
            model = path.parent.name
            folder = output / model
            write_tables(folder, objects, summary, top_n)
            (folder / 'plots').mkdir(exist_ok=True)
            for layer, group in objects.groupby('layer', sort=True):
                plot_layer(group, folder / 'plots' / f'layer_{int(layer):02d}.png', dataset, model, layer)
            tables.append(objects)
            summaries.append(summary)
            sources.append({'path': str(path), 'sha256': digest, 'rows': len(objects),
                            'layers': objects.layer.nunique()})
            print(f'{dataset}/{model}: {len(objects)} rows, {objects.layer.nunique()} layers', flush=True)
        summary = pd.concat(summaries, ignore_index=True)
        write_tables(output, pd.concat(tables, ignore_index=True), summary, top_n, all_models=True)
        (output / 'README.md').write_text(README)
        (output / 'manifest.json').write_text(json.dumps({
            'sources': sources, 'threshold': THRESHOLD, 'aligned_rule': 'alignment >= 0.5',
            'grouping': KEYS, 'splits': 'all existing splits pooled within layer',
            'top_n_per_mismatch_case_per_layer': top_n,
            'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                         'matplotlib': matplotlib.__version__},
        }, indent=2) + '\n')
        combined.append(summary)
    summary = pd.concat(combined, ignore_index=True)
    summary.to_csv(root / 'mismatch_summary_all.csv', index=False)
    wide_summary(summary).to_csv(root / 'mismatch_fractions_all.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'results' / 'checkpoint2')
    parser.add_argument('--top-n', type=int, default=20)
    args = parser.parse_args()
    run(args.results_root, args.top_n)
