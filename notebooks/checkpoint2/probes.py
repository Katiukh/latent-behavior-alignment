"""Compact derived cache: linear probes and exact preprocessing state."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from sklearn.preprocessing import normalize

from artifacts import CacheError, REFERENCE


def probe_signature(cache, spec, train, test, config):
    digest=hashlib.sha256()
    for key in ('X_pos','X_neg'):
        digest.update(memoryview(cache[key]).cast('B'))
    return dict(model_name=spec.name,dataset_sha256=cache['metadata']['dataset_sha256'],
                hidden_state_shape=list(cache['X_pos'].shape),hidden_sha256=digest.hexdigest(),
                train_idx=train.tolist(),test_idx=test.tolist(),config=config,
                reference_ccs_sha256=hashlib.sha256((REFERENCE/'code/ccs.py').read_bytes()).hexdigest())


def prepare_layer(cache, layer, preprocessing):
    pos=cache['X_pos'][:,layer,:].copy()
    neg=cache['X_neg'][:,layer,:].copy()
    if preprocessing in ('checkpoint1','l2'):
        pos=normalize(pos,norm='l2',axis=1)
        neg=normalize(neg,norm='l2',axis=1)
    return pos,neg


def offsets(pos, neg, train, preprocessing):
    if preprocessing in ('checkpoint1','median'):
        return np.median(pos[train],axis=0),np.median(neg[train],axis=0)
    if preprocessing=='mean':
        return pos[train].mean(axis=0),neg[train].mean(axis=0)
    return np.zeros(pos.shape[1],dtype=np.float32),np.zeros(neg.shape[1],dtype=np.float32)


VECTOR_KEYS=('weight','train_mean_pos','train_mean_neg','prediction_offset_pos','prediction_offset_neg')
SCALAR_KEYS=('bias','ccs_loss','orientation_flipped','train_accuracy_raw','train_accuracy_flipped')


def load_probes(path, signature):
    path=Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path,allow_pickle=False) as f:
            result={key:f[key] for key in f.files}
        if json.loads(str(result['metadata_json'].item())) != signature:
            print('CCS cache configuration/hidden states changed: retraining probes offline')
            return None
        layers=signature['config']['layers']
        dim=signature['hidden_state_shape'][-1]
        if not np.array_equal(result['layer'],layers):
            raise CacheError('Probe layer indices differ from analyzed layers')
        for key in VECTOR_KEYS+SCALAR_KEYS:
            shape=(len(layers),dim) if key in VECTOR_KEYS else (len(layers),)
            if result[key].shape!=shape or not np.isfinite(result[key]).all():
                raise CacheError(f'Invalid probe {key}: expected finite shape {shape}')
        if not np.isin(result['orientation_flipped'],[False,True]).all():
            raise CacheError('Invalid probe orientation')
        return result
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as e:
        raise CacheError(f'Invalid derived probe cache {path}: {e}. Use force_retrain=True to replace it.') from e


def save_probes(path, rows, signature):
    path=Path(path)
    arrays={key:np.asarray([row[key] for row in rows]) for key in VECTOR_KEYS+SCALAR_KEYS+('layer',)}
    arrays['metadata_json']=np.asarray(json.dumps(signature,sort_keys=True))
    with tempfile.NamedTemporaryFile(suffix='.npz',dir=path.parent,delete=False) as f:
        tmp=Path(f.name)
    try:
        np.savez(tmp,**arrays)
        load_probes(tmp,signature)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def restore_probe(ccs_class, probes, index, device='cpu'):
    """Rebuild the reference CCS prediction methods around saved linear weights."""
    import torch
    weight=probes['weight'][index]
    dummy=np.zeros((1,len(weight)),dtype=np.float32)
    ccs=ccs_class(dummy,dummy,device=device,var_normalize=False,lambda_classification=0.0)
    with torch.no_grad():
        ccs.best_probe[0].weight.copy_(torch.as_tensor(weight,device=device).reshape(1,-1))
        ccs.best_probe[0].bias.copy_(torch.as_tensor(probes['bias'][index],device=device).reshape(1))
    return ccs
