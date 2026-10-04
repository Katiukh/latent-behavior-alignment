"""Historical PA-CCS: original per-group mean-centering, preserved unchanged."""
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import load_cache, reference_module
from probes import probe_signature, prepare_layer, load_probes, restore_probe


def pa_metrics(ccs, pos, neg, test_idx):
    """Original train_ccs_on_hidden_states pair selection and method calls.

    pos/neg are per-vector L2 normalized, BEFORE train-median subtraction.
    Reference get_contrastive_probas separately mean-centers each subset.
    Counterparts may belong to train; this intentionally preserves the source.
    """
    n = len(pos)
    if n % 2:
        raise ValueError('Original PA-CCS needs two equal dataset halves')
    A = test_idx[test_idx >= n/2]
    notA = (A-n/2).astype(int)
    if not len(A):
        raise ValueError('No second-half test rows for original PA-CCS')
    args = (neg[A],pos[A],neg[notA],pos[notA])
    agreement = ccs.get_agreement(*args)
    contradiction = ccs.get_contradiction_idx(*args)
    return {'polar_consistency_↓':float(np.mean(agreement)),
            'abs_agreement_score':float(np.median(np.abs(agreement))),
            'contradiction_idx_↓':float(np.mean(contradiction)),
            'pa_pair_count':len(A)}



def pa_probabilities(path, spec, data, layer, device='cpu'):
    """Return transient original PA-CCS probabilities using an existing probe.

    Uses the split and preprocessing stored with the probe. Never trains.
    """
    import json
    from artifacts import CacheError
    cache=load_cache(path,spec,data)
    probe_path=Path(path)/'ccs_probes.npz'
    try:
        with np.load(probe_path,allow_pickle=False) as f:
            stored=json.loads(str(f['metadata_json'].item()))
        config=stored['config']
        train,test=np.asarray(stored['train_idx']),np.asarray(stored['test_idx'])
        signature=probe_signature(cache,spec,train,test,config)
        saved=load_probes(probe_path,signature)
        if saved is None:
            raise CacheError('Probe no longer matches hidden states/reference; run analyze first')
        index=config['layers'].index(layer)
    except (OSError,ValueError,KeyError) as e:
        raise CacheError(f'Cannot restore PA-CCS probe: {e}') from e
    pos,neg=prepare_layer(cache,layer,config['preprocessing'])
    ccs=restore_probe(reference_module('ccs').CCS,saved,index,device)
    A=test[test>=len(data)/2]
    notA=(A-len(data)/2).astype(int)
    values=ccs.get_contrastive_probas(neg[A],pos[A],neg[notA],pos[notA])
    return pd.DataFrame(dict(sample_idx=A,opposition_sample_idx=notA,
        pA0=values[0].ravel(),pA1=values[1].ravel(),
        p_notA0=values[2].ravel(),p_notA1=values[3].ravel()))


def esa_from_probabilities(probabilities, labels):
    """CCS accuracy on A and notA together, with one global sign choice."""
    pA0, pA1, pn0, pn1 = probabilities
    scores = np.concatenate((.5 * (pA0 + (1 - pA1)),
                             .5 * (pn0 + (1 - pn1)))).ravel()
    accuracy = np.mean((scores > .5) == labels)
    return float(max(accuracy, 1 - accuracy))


def reference_esa_frame(root, dataset, model):
    """Replace exported ESA with accuracy from the reference PA probabilities.

    Uses exactly the PC/CI pairs and per-group means, with saved probes only.
    Original CCS source tables are never modified.
    """
    import json
    from artifacts import MODELS, load_dataset, CacheError
    from plot_pa_ccs_metrics_reference import load_layer_metrics

    spec = MODELS[model]
    data = load_dataset(dataset=dataset)
    path = Path(root) / dataset / 'artifacts' / model
    cache = load_cache(path, spec, data)
    with np.load(path / 'ccs_probes.npz', allow_pickle=False) as f:
        stored = json.loads(str(f['metadata_json'].item()))
    config = stored['config']
    train, test = np.asarray(stored['train_idx']), np.asarray(stored['test_idx'])
    saved = load_probes(path / 'ccs_probes.npz',
                        probe_signature(cache, spec, train, test, config))
    if saved is None:
        raise CacheError('Compatible saved probes required for reference ESA; no training')
    source = (Path(root) / dataset / 'analysis/checkpoint1_compatible'
              / model / 'layer_metrics.csv')
    frame = load_layer_metrics(source, dataset, model)
    if frame.layer.tolist() != sorted(saved['layer'].tolist()):
        raise CacheError('Source metric layers differ from saved probes')
    A = test[test >= len(data) / 2]
    notA = (A - len(data) / 2).astype(int)
    if len(data) % 2 or not len(A):
        raise CacheError('Reference ESA requires paired dataset halves and second-half test rows')
    labels = 1 - data.raw.is_harmfull_opposition.to_numpy()
    pair_labels = np.concatenate((labels[A], labels[notA]))
    cls = reference_module('ccs').CCS
    frame['ccs_accuracy'] = frame.accuracy
    for index, layer in enumerate(saved['layer']):
        pos, neg = prepare_layer(cache, int(layer), config['preprocessing'])
        ccs = restore_probe(cls, saved, index, device='cpu')
        probabilities = ccs.get_contrastive_probas(neg[A], pos[A], neg[notA], pos[notA])
        frame.loc[frame.layer.eq(layer), 'accuracy'] = esa_from_probabilities(probabilities, pair_labels)
    return frame.assign(dataset=dataset, model=model, esa=frame.accuracy,
                        pc=frame['polar_consistency_↓'], ci=frame['contradiction_idx_↓'],
                        esa_sample_count=2 * len(A),
                        esa_preprocessing='Reference: L2 + separate mean of each A/notA Yes/No group')
