"""Per-model CCS experiments with a single frozen train transform.

Only input-cache contracts and original CCS formulas are imported from the old
pipeline. No old analysis, plotting, output paths or saved probes are reused.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
INPUT = PROJECT / 'data'
OUTPUT = PROJECT / 'results/full'
from . import artifacts as _artifacts
MODELS = _artifacts.MODELS
CASES = ('both_aligned', 'aligned_output_misaligned_latent',
         'misaligned_output_aligned_latent', 'both_misaligned')
SIZES = {name: 'big' if '9b' in name else 'small' for name in MODELS}
DEPTHS = ('Early', 'Middle-early', 'Middle-late', 'Late')
DEFAULTS = dict(epochs=1500, restarts=10, lr=.015, weight_decay=.01, seed=0,
                split_seed=71, device='cpu', threads=2)
TABLES = ('metrics', 'predictions', 'pa_metrics', 'examples', 'audit', 'thresholds')


@lru_cache(maxsize=1)
def reference_ccs():
    return _artifacts.reference_module('ccs').CCS


def l2(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Representations must be a finite matrix')
    return normalize(values, norm='l2', axis=1)


@dataclass(frozen=True)
class State:
    yes_center: np.ndarray
    no_center: np.ndarray
    weight: np.ndarray
    bias: float
    flipped: bool
    loss: float

    def __post_init__(self):
        for key in ('yes_center', 'no_center', 'weight'):
            value = np.asarray(getattr(self, key), dtype=np.float32).copy()
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite {key}')
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        if not np.isfinite([self.bias, self.loss]).all():
            raise ValueError('Nonfinite fitted parameters')

    def digest(self):
        h = hashlib.sha256()
        for x in (self.yes_center, self.no_center, self.weight):
            h.update(x.tobytes())
        h.update(repr((self.bias, self.flipped, self.loss)).encode())
        return h.hexdigest()


def probability(centered, state):
    return expit(centered @ state.weight + np.float32(state.bias))


def predict(state, yes_l2, no_l2):
    """L2 inputs -> subtract saved means once -> raw, un-oriented CCS score."""
    return .5 * (probability(no_l2 - state.no_center, state)
                 + 1 - probability(yes_l2 - state.yes_center, state))


def fit(yes_train, no_train, labels_train, *, epochs=1500, restarts=10, seed=0,
        lr=.015, weight_decay=.01, device='cpu', ccs_class=None):
    """Accepts train rows only. Input vectors must already be L2 normalized."""
    import torch
    cls = ccs_class or reference_ccs()

    class TrainCenteredCCS(cls):
        def normalize(self, values):
            # Constructor must not subtract another mean, even residual rounding.
            return values.copy()

    if epochs < 1 or restarts < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError('Invalid training hyperparameters')
    if len(yes_train) != len(no_train) or len(yes_train) != len(labels_train):
        raise ValueError('Train array lengths differ')
    if set(np.unique(labels_train)) != {0, 1}:
        raise ValueError('Train must contain both labels')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    yes_center = yes_train.mean(0, dtype=np.float64).astype(np.float32)
    no_center = no_train.mean(0, dtype=np.float64).astype(np.float32)
    yes, no = yes_train - yes_center, no_train - no_center
    np.testing.assert_allclose(yes.mean(0, dtype=np.float64), 0, atol=2e-7)
    np.testing.assert_allclose(no.mean(0, dtype=np.float64), 0, atol=2e-7)
    probe = TrainCenteredCCS(no, yes, y_train=labels_train, nepochs=epochs,
                            ntries=restarts, lr=lr, weight_decay=weight_decay,
                            device=device, batch_size=-1, linear=True,
                            var_normalize=False, lambda_classification=0.,
                            predict_normalize=False)
    np.testing.assert_array_equal(probe.x0, no)
    np.testing.assert_array_equal(probe.x1, yes)
    loss = probe.repeated_train()
    weight, bias = probe.get_weights()
    provisional = State(yes_center, no_center, np.asarray(weight).reshape(-1), bias, False, loss)
    raw = predict(provisional, yes_train, no_train)
    # Confirm saved parameters reproduce actual reference prediction.
    _, actual = probe.predict(no, yes)
    np.testing.assert_allclose(raw, actual.cpu().numpy().ravel(), atol=2e-6)
    flipped = np.mean(((1-raw) > .5) == labels_train) > np.mean((raw > .5) == labels_train)
    return State(yes_center, no_center, provisional.weight, bias, bool(flipped), loss)


def threshold(scores_train, labels_train):
    """Same candidate grid/accuracy/tie-breaks as checkpoint2; train only."""
    scores_train, labels_train = np.asarray(scores_train), np.asarray(labels_train)
    if not len(scores_train) or not np.isfinite(scores_train).all():
        raise ValueError('Invalid train behavioral scores')
    candidates = np.unique(np.r_[0., .5, scores_train.astype(float), 1.])
    accuracy = np.array([np.mean((scores_train >= t) == labels_train) for t in candidates])
    best = candidates[accuracy == accuracy.max()]
    distance = np.abs(best - .5)
    return float(best[np.isclose(distance, distance.min(), rtol=0, atol=1e-12)].min())


def splits(protocol, pair_ids, row_train, row_test, seed=71):
    if protocol == 'rowwise':
        result = [(0, np.asarray(row_train), np.asarray(row_test))]
    elif protocol == 'pairwise':
        unique = np.unique(pair_ids)
        result = [(fold, np.flatnonzero(np.isin(pair_ids, unique[a])),
                   np.flatnonzero(np.isin(pair_ids, unique[b])))
                  for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=seed).split(unique))]
    else:
        raise ValueError(f'Unknown protocol {protocol}')
    for _, train, test in result:
        if set(train) & set(test) or set(np.r_[train, test]) != set(range(len(pair_ids))):
            raise ValueError('Invalid train/test partition')
        if protocol == 'pairwise' and set(pair_ids[train]) & set(pair_ids[test]):
            raise ValueError('Pair leakage')
    if protocol == 'pairwise':
        np.testing.assert_array_equal(np.sort(np.concatenate([test for _, _, test in result])),
                                      np.arange(len(pair_ids)))
    return result


def samples_from(data):
    raw = data.raw
    label = 1 - raw.is_harmfull_opposition.to_numpy(dtype=int)
    identifiers = [c for c in ('pair_id', 'polar_pair_id', 'opposition_pair_id') if c in raw]
    if identifiers:
        if raw[identifiers].isna().any().any():
            raise ValueError('Missing semantic pair identifiers')
        pairs = raw[identifiers[0]].astype(str).to_numpy()
        for name in identifiers[1:]:
            relation = pd.DataFrame({'a': pairs, 'b': raw[name].astype(str)})
            if not relation.groupby('a').b.nunique().eq(1).all() or not relation.groupby('b').a.nunique().eq(1).all():
                raise ValueError('Conflicting pair identifiers')
        for frame in (data.yes, data.no):
            for name in identifiers:
                if name in frame and not np.array_equal(raw[name].astype(str), frame[name].astype(str)):
                    raise ValueError('Pair identifiers differ between answer tables')
    else:
        n = len(raw)
        if n % 2 or not (np.all(label[:n//2] == 1) and np.all(label[n//2:] == 0)):
            raise ValueError('Reference half-offset pairing is invalid')
        pairs = np.tile(np.arange(n//2).astype(str), 2)
    frame = pd.DataFrame(dict(sample_idx=np.arange(len(raw)), pair_id=pairs,
                              label=label, statement=raw.statement.to_numpy()))
    if not frame.groupby('pair_id').size().eq(2).all() or not frame.groupby('pair_id').label.sum().eq(1).all():
        raise ValueError('Each semantic pair must have one harmful and one safe member')
    return frame


def category(behavior_aligned, latent_aligned):
    return np.select([behavior_aligned & latent_aligned, behavior_aligned & ~latent_aligned,
                      ~behavior_aligned & latent_aligned], CASES[:3], default=CASES[3])


def predictions(samples, behavioral, raw, selected_threshold, flipped):
    frame = samples.copy().reset_index(drop=True)
    frame['behavioral_score'], frame['raw_latent_score'] = behavioral, raw
    frame['latent_score'] = 1 - raw if flipped else raw
    frame['latent_pred'] = (frame.latent_score > .5).astype(int)
    frame['behavioral_pred'] = (frame.behavioral_score >= selected_threshold).astype(int)
    frame['behavioral_threshold'] = selected_threshold
    frame['latent_aligned'] = frame.latent_pred.eq(frame.label)
    frame['behavioral_aligned'] = frame.behavioral_pred.eq(frame.label)
    # Historical all-samples baseline uses label-relative >= .5, including ties.
    frame['baseline_latent_aligned'] = np.where(frame.label.eq(1), frame.latent_score, 1-frame.latent_score) >= .5
    frame['baseline_behavioral_aligned'] = np.where(frame.label.eq(1), behavioral, 1-behavioral) >= .5
    frame['category_baseline'] = category(frame.baseline_behavioral_aligned, frame.baseline_latent_aligned)
    frame['category_calibrated'] = category(frame.behavioral_aligned, frame.latent_aligned)
    return frame


def summarize(frame, calibration):
    baseline = calibration == 'baseline_alignment'
    b = frame.baseline_behavioral_aligned if baseline else frame.behavioral_aligned
    l = frame.baseline_latent_aligned if baseline else frame.latent_aligned
    counts = (frame.category_baseline if baseline else frame.category_calibrated).value_counts()
    row = dict(calibration=calibration, n_samples=len(frame), latent_accuracy=l.mean(),
               behavioral_accuracy=b.mean(), behavioral_accuracy_05=((frame.behavioral_score >= .5) == frame.label).mean(),
               esa_ccs=max(np.mean((frame.raw_latent_score > .5) == frame.label),
                           1-np.mean((frame.raw_latent_score > .5) == frame.label)))
    for case in CASES:
        row[f'{case}_count'] = int(counts.get(case, 0))
        row[f'{case}_fraction'] = row[f'{case}_count'] / len(frame)
    if sum(row[f'{case}_count'] for case in CASES) != len(frame):
        raise ValueError('Mismatch counts do not sum to evaluation sample count')
    return row


class PAProbabilities:
    """Original PA formulas select this probability transform, nothing else."""
    def __init__(self, state, variant):
        if variant not in ('reference', 'consistent'):
            raise ValueError('Unknown PA variant')
        self.state, self.variant = state, variant

    def get_contrastive_probas(self, *groups):
        result = []
        for i, values in enumerate(groups):
            center = values.mean(0, keepdims=True) if self.variant == 'reference' else (
                self.state.no_center if i % 2 == 0 else self.state.yes_center)
            result.append(probability(values-center, self.state).reshape(-1, 1))
        return tuple(result)


def pa_metrics(state, yes, no, samples, idx):
    # Historical rowwise selection: safe members from evaluated split and
    # their harmful partners; the latter may be in the other split.
    safe = samples.loc[idx].query('label == 0')
    harmful = samples.query('label == 1').set_index('pair_id')
    A = safe.sample_idx.to_numpy()
    notA = harmful.loc[safe.pair_id].sample_idx.to_numpy()
    rows = []
    for variant in ('reference', 'consistent'):
        row = dict(pa_variant=variant, n_pairs=len(A), counterpart_outside_split=int((~np.isin(notA, idx)).sum()))
        if not len(A):
            row.update(esa=np.nan, pc=np.nan, ci=np.nan)
        else:
            adapter = PAProbabilities(state, variant)
            args = (no[A], yes[A], no[notA], yes[notA])
            p0, p1, pn0, pn1 = adapter.get_contrastive_probas(*args)
            scores = np.r_[.5*(p0 + 1-p1).ravel(), .5*(pn0 + 1-pn1).ravel()]
            labels = np.r_[np.zeros(len(A), dtype=int), np.ones(len(A), dtype=int)]
            accuracy = np.mean((scores > .5) == labels)
            cls = reference_ccs()
            row.update(esa=max(accuracy, 1-accuracy),
                       pc=float(cls.get_agreement(adapter, *args).mean()),
                       ci=float(cls.get_contradiction_idx(adapter, *args).mean()))
        rows.append(row)
    return rows


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_result(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError(f'Incomplete experiment: {folder}')
    for filename, fingerprint in manifest['outputs'].items():
        if sha(folder / filename) != fingerprint:
            raise ValueError(f'Changed experiment output: {folder / filename}')
    result = {name: pd.read_csv(folder / f'{name}.csv', dtype={'pair_id': str},
                               keep_default_na=False,
                               na_values={key: [''] for key in ('esa', 'pc', 'ci')}) for name in TABLES}
    return {**result, 'folder': folder, 'manifest': manifest}


def run_dataset(model, dataset, *, output=OUTPUT, input_root=INPUT, layers=None,
                protocols=('rowwise', 'pairwise'), **settings):
    """Run/reuse one model × dataset. Incompatible results are never overwritten."""
    import torch
    if model not in MODELS or dataset not in ('not', 'mixed'):
        raise ValueError('Unknown model/dataset')
    config = {**DEFAULTS, **settings}
    if set(config) != set(DEFAULTS):
        raise ValueError(f'Unknown settings: {set(config)-set(DEFAULTS)}')
    if not config['device'].startswith(('cpu', 'cuda')) or config['threads'] < 1:
        raise ValueError('Invalid device/threads')
    torch.set_num_threads(config['threads'])
    spec = MODELS[model]
    layers = list(range(spec.n_layers)) if layers is None else sorted(set(layers))
    if not layers or any(not isinstance(x, int) or x < 0 or x >= spec.n_layers for x in layers):
        raise ValueError('Invalid layers')
    if not protocols or len(set(protocols)) != len(protocols) or set(protocols)-{'rowwise', 'pairwise'}:
        raise ValueError('Invalid protocols')
    folder = Path(input_root) / dataset / 'artifacts' / model
    output = Path(output).resolve()
    if output.is_relative_to(INPUT.resolve()) or output == PROJECT.resolve():
        raise ValueError('Use a separate output root; historical results are read-only')
    data = _artifacts.load_dataset(dataset=dataset)
    provenance = {name: sha(folder / name) for name in _artifacts.FILES}
    source = {str(path): sha(path) for path in
              (Path(__file__), Path(_artifacts.__file__), HERE / 'ccs.py')}
    signature = dict(model=model, dataset=dataset, config=config, layers=layers,
                     dataset_sha256=data.fingerprint, inputs=provenance, source=source,
                     preprocessing='L2 -> separate train arithmetic Yes/No means -> subtract once',
                     versions=dict(numpy=np.__version__, pandas=pd.__version__, torch=str(torch.__version__)))
    results, pending = {}, []
    for protocol in protocols:
        destination = output / dataset / model / protocol
        expected = {**signature, 'protocol': protocol}
        if (destination / 'manifest.json').exists():
            result = load_result(destination)
            if result['manifest']['signature'] != expected:
                raise ValueError(f'Incompatible cached experiment: {destination}. Choose a fresh output root.')
            results[protocol] = result
            print(f'Reused {dataset}/{model}/{protocol}', flush=True)
        else:
            if destination.exists() and any(destination.iterdir()):
                raise ValueError(f'Incomplete output at {destination}; choose a fresh output root.')
            pending.append((protocol, destination, expected))
    if pending:
        cache = _artifacts.load_cache(folder, spec, data)
        samples = samples_from(data)
        logits = cache['behavioral']
        zero, one = ('logit_class_0', 'logit_class_1') if spec.kind == 'encoder' else ('logit_no', 'logit_yes')
        # Match existing float32 two-class softmax exactly.
        values = np.stack([logits[zero], logits[one]], axis=1).astype(np.float32)
        values = np.exp(values-values.max(axis=1, keepdims=True))
        behavior = values[:, 1] / values.sum(axis=1)
        for protocol, destination, expected in pending:
            with threadpool_limits(limits=config['threads']):
                results[protocol] = _run(cache, samples, behavior, layers, protocol, destination, expected)
    return results


def _run(cache, samples, behavior, layers, protocol, destination, signature):
    config = signature['config']
    dataset, model = signature['dataset'], signature['model']
    folds = splits(protocol, samples.pair_id.to_numpy(), cache['train_idx'], cache['test_idx'], config['split_seed'])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'probes').mkdir(exist_ok=True)
    labels = samples.label.to_numpy()
    thresholds = {fold: threshold(behavior[train], labels[train]) for fold, train, _ in folds}
    threshold_rows = []
    for fold, train, test in folds:
        for split, idx in (('train', train), ('test', test)):
            for calibration, value in (('baseline', .5), ('calibrated', thresholds[fold])):
                pred = behavior[idx] >= value
                y = labels[idx]
                threshold_rows.append(dict(dataset=dataset, model=model, protocol=protocol,
                    fold=fold, split=split, calibration=calibration, threshold=value,
                    n_samples=len(idx), accuracy=np.mean(pred == y),
                    tp=int(np.sum(pred & (y == 1))), tn=int(np.sum(~pred & (y == 0))),
                    fp=int(np.sum(pred & (y == 0))), fn=int(np.sum(~pred & (y == 1)))))
    metrics, pa, audit, examples = [], [], [], []
    predictions_path = destination / 'predictions.csv'
    first = True
    for layer in layers:
        yes, no = l2(cache['X_pos'][:, layer]), l2(cache['X_neg'][:, layer])
        for fold, train, test in folds:
            seed = config['seed'] + fold*MODELS[model].n_layers + layer
            state = fit(yes[train], no[train], labels[train], seed=seed,
                        **{k: config[k] for k in ('epochs', 'restarts', 'lr', 'weight_decay', 'device')})
            before = state.digest()
            probe_path = destination / 'probes' / f'layer_{layer:02d}_fold_{fold}.npz'
            np.savez(probe_path, yes_center=state.yes_center, no_center=state.no_center,
                     weight=state.weight, bias=state.bias, flipped=state.flipped, loss=state.loss,
                     train_idx=train, test_idx=test, behavioral_threshold=thresholds[fold], seed=seed)
            both = []
            for split, idx in (('train', train), ('test', test)):
                keys = dict(dataset=dataset, model=model, protocol=protocol, layer=layer, fold=fold, split=split)
                frame = predictions(samples.iloc[idx], behavior[idx], predict(state, yes[idx], no[idx]),
                                    thresholds[fold], state.flipped).assign(**keys)
                frame.to_csv(predictions_path, mode='w' if first else 'a', header=first, index=False)
                first = False
                both.append(frame)
                for calibration in ('baseline_alignment', 'calibrated'):
                    metrics.append({**keys, **summarize(frame, calibration)})
                for row in pa_metrics(state, yes, no, samples, idx):
                    if protocol == 'pairwise' and row['counterpart_outside_split']:
                        raise ValueError('PA pair leakage')
                    pa.append({**keys, **row})
                if split == 'test':
                    candidates = frame.loc[frame.category_calibrated.isin(CASES[1:3])].copy()
                    candidates['alignment_gap'] = (np.where(candidates.label.eq(1), candidates.behavioral_score, 1-candidates.behavioral_score)
                                                   - np.where(candidates.label.eq(1), candidates.latent_score, 1-candidates.latent_score))
                    candidates['abs_gap'] = candidates.alignment_gap.abs()
                    examples.append(candidates.sort_values('abs_gap', ascending=False).groupby('category_calibrated', sort=False).head(5))
            if protocol == 'rowwise':
                metrics.append({**keys, 'split': 'all', **summarize(pd.concat(both), 'baseline_alignment')})
            if before != state.digest():
                raise ValueError('Evaluation mutated fitted state')
            audit.append(dict(dataset=dataset, model=model, protocol=protocol, layer=layer, fold=fold,
                              fit_split='train', n_train=len(train), n_test=len(test), seed=seed,
                              loss=state.loss, behavioral_threshold=thresholds[fold], flipped=state.flipped,
                              state_sha256=before, evaluation_state_unchanged=True,
                              yes_center_residual=float(np.max(np.abs((yes[train]-state.yes_center).mean(0, dtype=np.float64)))),
                              no_center_residual=float(np.max(np.abs((no[train]-state.no_center).mean(0, dtype=np.float64))))))
            print(f'{dataset}/{model}/{protocol}: layer {layer}, fold {fold}', flush=True)
    tables = dict(metrics=pd.DataFrame(metrics), pa_metrics=pd.DataFrame(pa),
                  audit=pd.DataFrame(audit), thresholds=pd.DataFrame(threshold_rows),
                  examples=pd.concat(examples, ignore_index=True))
    for name, frame in tables.items():
        frame.to_csv(destination / f'{name}.csv', index=False)
    if protocol == 'pairwise':
        oof = pd.read_csv(predictions_path, dtype={'pair_id': str}, keep_default_na=False).query("split == 'test'")
        if oof.duplicated(['sample_idx', 'layer']).any() or len(oof) != len(samples)*len(layers):
            raise ValueError('Invalid out-of-fold coverage')
        for _, group in oof.groupby('layer'):
            if set(group.sample_idx) != set(samples.sample_idx):
                raise ValueError('Missing or unexpected out-of-fold sample IDs')
        if not oof.groupby(['layer', 'pair_id']).fold.nunique().eq(1).all():
            raise ValueError('Pairwise OOF leakage')
        if not oof.groupby('sample_idx').fold.nunique().eq(1).all():
            raise ValueError('OOF fold changes across layers')
    outputs = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest = dict(complete=True, signature=signature, outputs=outputs,
                    full_hyperparameters=config['epochs'] == 1500 and config['restarts'] == 10,
                    complete_layers=layers == list(range(MODELS[model].n_layers)))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return load_result(destination)


def metric_rows(result):
    """One long schema; PA variant and calibration are never collapsed."""
    keys = ['dataset', 'model', 'protocol', 'layer', 'fold', 'split']
    metrics = result['metrics']
    values = ['latent_accuracy', 'behavioral_accuracy', 'behavioral_accuracy_05', 'esa_ccs',
              *[f'{c}_fraction' for c in CASES]]
    main = metrics.melt(id_vars=[*keys, 'calibration'], value_vars=values,
                        var_name='metric', value_name='value')
    pa = result['pa_metrics'].rename(columns={'pa_variant': 'calibration'})
    pa = pa.melt(id_vars=[*keys, 'calibration'], value_vars=['esa', 'pc', 'ci'],
                 var_name='metric', value_name='value')
    return pd.concat([main, pa], ignore_index=True)


def aggregate(rows, *, require_complete=True):
    """Fold -> model/depth quarter -> dataset/size. No pooled overall mean."""
    rows = rows.copy()
    keys = ['dataset', 'model', 'protocol', 'split', 'calibration', 'metric']
    if rows.empty or rows.duplicated([*keys, 'layer', 'fold']).any():
        raise ValueError('Empty or duplicated aggregation input')
    if not set(rows.dataset) <= {'not', 'mixed'} or not set(rows.model) <= set(MODELS):
        raise ValueError('Unknown aggregation dataset/model')
    if not set(rows.protocol) <= {'rowwise', 'pairwise'}:
        raise ValueError('Unknown aggregation protocol')
    coverage = {}
    for key, group in rows.groupby(keys, observed=True):
        model, protocol = key[1], key[2]
        expected_layers = set(range(MODELS[model].n_layers))
        expected_folds = set(range(5)) if protocol == 'pairwise' else {0}
        good = (set(group.layer) == expected_layers and np.isfinite(group.value).all()
                and all(set(g.fold) == expected_folds for _, g in group.groupby('layer')))
        if require_complete and not good:
            raise ValueError(f'Incomplete layer/fold coverage: {key}')
        coverage[key] = good
    rows['size'] = rows.model.map(SIZES)
    last = rows.model.map({name: spec.n_layers-1 for name, spec in MODELS.items()})
    rows['relative_depth'] = rows.layer / last
    if not rows.relative_depth.between(0, 1).all():
        raise ValueError('Invalid relative layer depth')
    rows['depth'] = pd.cut(rows.relative_depth, [0, .25, .5, .75, 1], labels=DEPTHS, include_lowest=True)
    # Equal fold weights first, equal layer weights within model/quarter next.
    layer = rows.groupby([*keys, 'size', 'depth', 'layer'], observed=True, as_index=False).agg(
        value=('value', 'mean'), fold_std=('value', 'std'), n_folds=('fold', 'nunique'))
    per_model = layer.groupby([*keys, 'size', 'depth'], observed=True, as_index=False).agg(
        value=('value', 'mean'), n_layers=('layer', 'nunique'))
    group_keys = ['dataset', 'size', 'protocol', 'split', 'calibration', 'metric', 'depth']
    result = []
    for key, group in per_model.groupby(group_keys, observed=True):
        dataset, size, protocol, split, calibration, metric, depth = key
        expected = {m for m in MODELS if SIZES[m] == size}
        complete = (set(group.model) == expected and all(
            coverage[(dataset, m, protocol, split, calibration, metric)] for m in group.model))
        if require_complete and not complete:
            raise ValueError(f'Incomplete model coverage: {key}')
        result.append({**dict(zip(group_keys, key)), 'mean': group.value.mean(),
                       'model_std': group.value.std(), 'n_models': group.model.nunique(),
                       'expected_models': len(expected), 'complete': complete})
    output = pd.DataFrame(result)
    if require_complete:
        for key, group in output.groupby(['protocol', 'split', 'calibration', 'metric'], observed=True):
            if set(zip(group.dataset, group['size'])) != {('not', 'small'), ('not', 'big'), ('mixed', 'small'), ('mixed', 'big')}:
                raise ValueError(f'Incomplete dataset/size coverage: {key}')
    return output


def depth_summary(rows, *, require_complete=True):
    """Layer mean/model/quarter -> equal model mean/fold -> fold mean and std."""
    summary = aggregate(rows, require_complete=require_complete)
    frame = rows.copy()
    frame['size'] = frame.model.map(SIZES)
    relative = frame.layer / frame.model.map({m: s.n_layers-1 for m, s in MODELS.items()})
    frame['depth'] = pd.cut(relative, [0, .25, .5, .75, 1], labels=DEPTHS, include_lowest=True)
    keys = ['dataset', 'size', 'protocol', 'split', 'calibration', 'metric', 'depth']
    per_model = frame.groupby([*keys, 'model', 'fold'], observed=True, as_index=False).agg(value=('value', 'mean'))
    folds = per_model.groupby([*keys, 'fold'], observed=True, as_index=False).agg(
        value=('value', 'mean'), n_models=('model', 'nunique'))
    stats = folds.groupby(keys, observed=True, as_index=False).agg(
        fold_mean=('value', 'mean'), std=('value', 'std'), n_folds=('fold', 'nunique'))
    summary = summary.merge(stats, on=keys, validate='one_to_one')
    np.testing.assert_allclose(summary['mean'], summary.fold_mean, atol=1e-12)
    return summary.drop(columns='fold_mean'), folds


def geometry_projection(state, yes_l2, no_l2, samples, test_idx):
    """Descriptive projections on one frozen CCS direction and a common PCA axis.

    PCA is fitted for visualization only; no probe, center, orientation or
    classification threshold is fitted or changed here.
    """
    from sklearn.decomposition import PCA
    before = state.digest()
    safe = samples.iloc[test_idx].query('label == 0')
    harmful = samples.query('label == 1').set_index('pair_id')
    A = safe.sample_idx.to_numpy()
    notA = harmful.loc[safe.pair_id].sample_idx.to_numpy()
    if not len(A):
        raise ValueError('Geometry needs safe/harmful semantic pairs')
    original = [yes_l2[A], no_l2[A], yes_l2[notA], no_l2[notA]]
    groups = dict(reference=[g-g.mean(0, keepdims=True) for g in original],
                  consistent=[g-(state.yes_center if i % 2 == 0 else state.no_center)
                              for i, g in enumerate(original)])
    direction = state.weight.astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise ValueError('Geometry needs a nonzero CCS direction')
    direction /= norm
    cloud = np.concatenate(groups['consistent']).astype(np.float64)
    residual = cloud - (cloud @ direction)[:, None]*direction
    with threadpool_limits(limits=2):
        orthogonal = PCA(n_components=1, svd_solver='randomized', random_state=71).fit(residual).components_[0]
    orthogonal -= (orthogonal @ direction)*direction
    if np.linalg.norm(orthogonal) <= 1e-10:
        orthogonal = np.eye(len(direction))[np.argmin(np.abs(direction))]
        orthogonal -= (orthogonal @ direction)*direction
    orthogonal /= np.linalg.norm(orthogonal)
    xy = {variant: [np.column_stack((g @ direction, g @ orthogonal)) for g in values]
          for variant, values in groups.items()}
    args = (no_l2[A], yes_l2[A], no_l2[notA], yes_l2[notA])
    scores = {}
    for variant in ('reference', 'consistent'):
        p0, p1, pn0, pn1 = PAProbabilities(state, variant).get_contrastive_probas(*args)
        safe_score, harmful_score = (.5*(p0+1-p1)).ravel(), (.5*(pn0+1-pn1)).ravel()
        if state.flipped:
            safe_score, harmful_score = 1-safe_score, 1-harmful_score
        scores[variant] = dict(safe=safe_score, harmful=harmful_score)
    if before != state.digest():
        raise ValueError('Geometry changed frozen state')
    return dict(groups=groups, xy=xy, scores=scores, ccs_axis=direction,
                orthogonal_axis=orthogonal, safe_idx=A, harmful_idx=notA,
                n_pairs=len(A), counterpart_outside_split=int((~np.isin(notA, test_idx)).sum()),
                metrics={row['pa_variant']: row for row in pa_metrics(state, yes_l2, no_l2, samples, test_idx)})


def load_geometry(datasets, *, layer=None, fold=0, input_root=INPUT):
    """Load real hidden states once/dataset and project rowwise and one CV fold.

    Each protocol has its own frozen weight/basis. Different fold directions
    are never pooled into a single cloud.
    """
    result = {}
    for dataset, protocols in datasets.items():
        first = next(iter(protocols.values()))
        model = first['manifest']['signature']['model']
        data = _artifacts.load_dataset(dataset=dataset)
        samples = samples_from(data)
        raw = Path(input_root) / dataset / 'artifacts' / model / 'hidden_states.npz'
        if any(r['manifest']['signature']['dataset_sha256'] != data.fingerprint for r in protocols.values()):
            raise ValueError('Geometry dataset fingerprint differs from experiment')
        raw_sha = sha(raw)
        if any(r['manifest']['signature']['inputs']['hidden_states.npz'] != raw_sha for r in protocols.values()):
            raise ValueError('Geometry hidden states differ from experiment')
        available = set.intersection(*(set(r['manifest']['signature']['layers']) for r in protocols.values()))
        chosen = max(available) if layer is None else layer
        if chosen not in available:
            raise ValueError('Requested geometry layer has no saved probe')
        with np.load(raw, allow_pickle=False) as f:
            yes = l2(f['X_pos'][:, chosen, :].copy())
            no = l2(f['X_neg'][:, chosen, :].copy())
        for protocol, experiment in protocols.items():
            chosen_fold = fold if protocol == 'pairwise' else 0
            probe = experiment['folder'] / 'probes' / f'layer_{chosen:02d}_fold_{chosen_fold}.npz'
            with np.load(probe, allow_pickle=False) as f:
                train, test = f['train_idx'], f['test_idx']
                state = State(f['yes_center'], f['no_center'], f['weight'], float(f['bias']),
                              bool(f['flipped']), float(f['loss']))
            np.testing.assert_array_equal(state.yes_center, yes[train].mean(0, dtype=np.float64).astype(np.float32))
            np.testing.assert_array_equal(state.no_center, no[train].mean(0, dtype=np.float64).astype(np.float32))
            with threadpool_limits(limits=2):
                payload = geometry_projection(state, yes, no, samples, test)
            if protocol == 'pairwise' and payload['counterpart_outside_split']:
                raise ValueError('Pairwise geometry contains a partner outside test')
            payload.update(dataset=dataset, model=model, protocol=protocol, layer=chosen, fold=chosen_fold,
                           folder=experiment['folder'],
                           figure_folder=experiment['folder'].parents[2] / 'geometry' / model / protocol)
            destination = experiment['folder'] / 'geometry' / f'layer_{chosen:02d}_fold_{chosen_fold}'
            destination.mkdir(parents=True, exist_ok=True)
            coordinates, score_rows = [], []
            for variant in ('reference', 'consistent'):
                for i, (name, answer) in enumerate((('safe', 'Yes'), ('safe', 'No'), ('harmful', 'Yes'), ('harmful', 'No'))):
                    idx = payload['safe_idx'] if name == 'safe' else payload['harmful_idx']
                    coordinates.append(pd.DataFrame(dict(variant=variant, semantic_class=name,
                        answer=answer, sample_idx=idx, x=payload['xy'][variant][i][:, 0],
                        y=payload['xy'][variant][i][:, 1])))
                for name, scores in payload['scores'][variant].items():
                    idx = payload['safe_idx'] if name == 'safe' else payload['harmful_idx']
                    score_rows.append(pd.DataFrame(dict(variant=variant, semantic_class=name,
                                                        sample_idx=idx, oriented_ccs_score=scores)))
            pd.concat(coordinates).to_csv(destination / 'coordinates.csv', index=False)
            pd.concat(score_rows).to_csv(destination / 'scores.csv', index=False)
            pd.DataFrame(payload['metrics'].values()).to_csv(destination / 'metrics.csv', index=False)
            np.savez(destination / 'basis.npz', ccs_axis=payload['ccs_axis'], orthogonal_axis=payload['orthogonal_axis'])
            provenance = dict(dataset=dataset, model=model, protocol=protocol, layer=chosen, fold=chosen_fold,
                split='test PA pairs', n_pairs=payload['n_pairs'],
                counterpart_outside_split=payload['counterpart_outside_split'],
                hidden_states_sha256=raw_sha, probe_sha256=sha(probe),
                projection='unit fixed CCS axis; shared orthogonal PCA axis within dataset/protocol',
                pca_scope='descriptive evaluation cloud only; no classifier fitting',
                centering='consistent: frozen train Yes/No means; reference: four evaluation-group means',
                source_sha256=sha(Path(__file__)))
            (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
            result.setdefault(protocol, {})[dataset] = payload
    return result


def collect(output=OUTPUT, *, require_complete=True):
    output = Path(output)
    rows, inventory, signatures = [], [], []
    for dataset in ('not', 'mixed'):
        for model in MODELS:
            for protocol in ('rowwise', 'pairwise'):
                folder = output / dataset / model / protocol
                if not (folder / 'manifest.json').exists():
                    if require_complete:
                        raise ValueError(f'Missing experiment coverage: {folder}')
                    inventory.append(dict(dataset=dataset, model=model, protocol=protocol,
                                          available=False, full_hyperparameters=False,
                                          complete_layers=False, config_json=''))
                    continue
                result = load_result(folder)
                signature = result['manifest']['signature']
                if (signature['dataset'], signature['model'], signature['protocol']) != (dataset, model, protocol):
                    raise ValueError(f'Misplaced experiment: {folder}')
                rows.append(metric_rows(result))
                signatures.append(signature)
                inventory.append(dict(dataset=dataset, model=model, protocol=protocol,
                                      available=True,
                                      full_hyperparameters=result['manifest']['full_hyperparameters'],
                                      complete_layers=result['manifest']['complete_layers'],
                                      config_json=json.dumps(signature['config'], sort_keys=True)))
    if not rows:
        raise ValueError('No completed experiments found; run the model notebooks first')
    configs = {r['config_json'] for r in inventory if r['available']}
    if len(configs) != 1:
        raise ValueError('Cannot aggregate experiments with different training configurations')
    if len({json.dumps(s['source'], sort_keys=True) for s in signatures}) != 1:
        raise ValueError('Cannot aggregate experiments from different source versions')
    for dataset in ('not', 'mixed'):
        if len({s['dataset_sha256'] for s in signatures if s['dataset'] == dataset}) > 1:
            raise ValueError(f'Cannot aggregate different dataset versions: {dataset}')
    long = pd.concat(rows, ignore_index=True)
    grouped, folds = depth_summary(long, require_complete=require_complete)
    inventory = pd.DataFrame(inventory)
    for (dataset, size), frame in grouped.groupby(['dataset', 'size']):
        destination = output / 'aggregation' / dataset / size
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / 'depth_metrics.csv', index=False)
    (output / 'aggregation').mkdir(exist_ok=True)
    long.to_csv(output / 'aggregation/model_layer_metrics.csv', index=False)
    folds.to_csv(output / 'aggregation/group_depth_fold_metrics.csv', index=False)
    inventory.to_csv(output / 'aggregation/coverage.csv', index=False)
    return long, grouped, inventory
