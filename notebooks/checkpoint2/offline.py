"""CCS, original PA-CCS metrics and behavioral scores from saved artifacts only."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from artifacts import load_cache, reference_module
from probes import (probe_signature, prepare_layer, offsets, load_probes,
                    save_probes, restore_probe)


def behavioral_scores(payload, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    b = payload['behavioral']
    df = pd.DataFrame({'sample_idx':b['sample_idx']})
    classifier = payload['metadata']['behavioral_scoring_type']=='classifier_softmax'
    modes = [('', 'logit_class_0','logit_class_1','hate')] if classifier else [('', 'logit_no','logit_yes','yes')]
    if payload['metadata']['chat_behavioral']:
        modes.append(('chat_','chat_logit_no','chat_logit_yes','yes'))
    for prefix,zero,one,label in modes:
        logits = np.stack([b[zero],b[one]],axis=1).astype(np.float32)
        logits -= logits.max(axis=1,keepdims=True)
        exp = np.exp(logits)
        probability = exp[:,1]/exp.sum(axis=1)
        score = (probability>threshold).astype(np.int8)
        df[f'{prefix}behavioral_probability_{label}'] = probability
        df[f'{prefix}behavioral_score'] = score
        df[f'{prefix}behavioral_response'] = np.where(score, 'hate' if classifier else 'Yes', 'non-hate' if classifier else 'No')
    return df


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


def analyze(path, spec, data, *, threshold=0.5, device='cpu', nepochs=1500,ntries=10,
            seed=0,layers=None,preprocessing='checkpoint1',split=None,
            lr=0.015,weight_decay=0.01,force_retrain=False):
    """No path to model loading. Every call starts from validated raw cache.

    seed=0 reproduces the seed set by importing reference ccs.py in Checkpoint 1;
    resetting it here makes repeated offline runs independent of import order.
    """
    import torch
    cache = load_cache(path,spec,data)
    ccs_class = reference_module('ccs').CCS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train,test = (cache['train_idx'],cache['test_idx']) if split is None else tuple(np.asarray(x) for x in split)
    if preprocessing not in ('checkpoint1','mean','median','l2','raw'):
        raise ValueError('Unknown preprocessing')
    if nepochs<1 or ntries<1 or lr<=0 or weight_decay<0:
        raise ValueError('Invalid CCS training parameters')
    labels = 1.0-data.raw.is_harmfull_opposition.to_numpy(dtype=np.float32)
    n = len(data)
    if any(idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or not len(idx) for idx in (train,test)):
        raise ValueError('Offline split needs nonempty integer index vectors')
    if not np.array_equal(np.sort(np.concatenate([train,test])),np.arange(n)):
        raise ValueError('Offline split must be a disjoint partition of all sample indices')
    if n%2 or not (np.all(labels[:n//2]==1) and np.all(labels[n//2:]==0)):
        raise ValueError('PA-CCS expects harmful first half, safe opposition second half')
    frames,metrics,probe_rows = [],[],[]
    layers = list(range(spec.n_layers)) if layers is None else list(layers)
    if not layers or len(set(layers))!=len(layers) or any(i<0 or i>=spec.n_layers for i in layers):
        raise ValueError('layers must be unique valid layer indices')
    config=dict(layers=layers,preprocessing=preprocessing,nepochs=nepochs,ntries=ntries,
                seed=seed,lr=lr,weight_decay=weight_decay,linear=True,
                var_normalize=False,lambda_classification=0.0,batch_size=-1,
                predict_normalize=False,torch_version=torch.__version__)
    signature=probe_signature(cache,spec,train,test,config)
    probe_path=Path(path)/'ccs_probes.npz'
    saved=None if force_retrain else load_probes(probe_path,signature)
    if saved is not None:
        print('Valid CCS probes: training skipped')
    behavior = behavioral_scores(cache,threshold)
    for layer_index,layer in enumerate(layers):
        print(f'Offline CCS: {spec.name}, layer {layer}/{spec.n_layers-1}')
        # Normalize just this layer; raw cache is never mutated.
        pos,neg = prepare_layer(cache,layer,preprocessing)
        pt,pe = pos[train].copy(),pos[test].copy()
        nt,ne = neg[train].copy(),neg[test].copy()
        if saved is None:
            pm,nm = offsets(pos,neg,train,preprocessing)
        else:
            pm,nm = saved['prediction_offset_pos'][layer_index],saved['prediction_offset_neg'][layer_index]
        pt-=pm; pe-=pm; nt-=nm; ne-=nm
        if saved is None:
            ccs = ccs_class(nt,pt,y_train=labels[train],var_normalize=False,
                        lambda_classification=0.0,predict_normalize=False,
                        device=device,nepochs=nepochs,ntries=ntries,lr=lr,weight_decay=weight_decay)
            loss = ccs.repeated_train()
        else:
            ccs=restore_probe(ccs_class,saved,layer_index,device)
            loss=float(saved['ccs_loss'][layer_index])
        _,train_conf = ccs.predict(nt,pt)
        _,test_conf = ccs.predict(ne,pe)
        raw_train = train_conf.cpu().numpy().reshape(-1)
        raw_test = test_conf.cpu().numpy().reshape(-1)
        normal = accuracy_score(labels[train],raw_train>0.5)
        flipped = accuracy_score(labels[train],(1-raw_train)>0.5)
        flip = flipped>normal
        orientation_acc = max(normal,flipped)
        if saved is not None:
            # Orientation is the saved training decision, never refit on test.
            flip=bool(saved['orientation_flipped'][layer_index])
            normal=float(saved['train_accuracy_raw'][layer_index])
            flipped=float(saved['train_accuracy_flipped'][layer_index])
            orientation_acc=max(normal,flipped)
        else:
            weight,bias=ccs.get_weights()
            probe_rows.append(dict(layer=layer,weight=np.asarray(weight).reshape(-1),bias=bias,
                train_mean_pos=pt.mean(axis=0),train_mean_neg=nt.mean(axis=0),
                prediction_offset_pos=pm,prediction_offset_neg=nm,
                ccs_loss=float(loss),orientation_flipped=bool(flip),
                train_accuracy_raw=float(normal),train_accuracy_flipped=float(flipped)))
        score_test = 1-raw_test if flip else raw_test
        for split,idx,raw in [('train',train,raw_train),('test',test,raw_test)]:
            frames.append(pd.DataFrame(dict(sample_idx=idx,statement=data.raw.statement.iloc[idx].to_numpy(),
                true_label=labels[idx].astype(np.int8),split=split,layer=layer,raw_latent_score=raw,
                latent_score=1-raw if flip else raw,orientation_flipped=flip,
                train_orientation_accuracy=orientation_acc)))
        row = dict(layer=layer,ccs_loss=float(loss),orientation_flipped=bool(flip),
                   train_accuracy_raw=float(normal),train_accuracy_flipped=float(flipped),
                   train_orientation_accuracy=float(orientation_acc),
                   accuracy=float(ccs.get_acc(ne,pe,labels[test])),
                   silhouette=float(ccs.get_silhouette(ne,pe)),
                   latent_vs_true=float(np.mean((score_test>0.5)==labels[test])),
                   behavioral_vs_true=float(np.mean(behavior.behavioral_score.to_numpy()[test]==labels[test])),
                   latent_vs_behavioral=float(np.mean((score_test>0.5)==behavior.behavioral_score.to_numpy()[test])))
        row.update(pa_metrics(ccs,pos,neg,test))
        metrics.append(row)
        del ccs
    if saved is None:
        save_probes(probe_path,probe_rows,signature)
    scores = pd.concat(frames,ignore_index=True).merge(behavior,on='sample_idx',how='left',validate='many_to_one')
    return scores,pd.DataFrame(metrics)


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
