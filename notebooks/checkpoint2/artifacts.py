"""Minimal, validated cache for Checkpoint 2. No model-library imports."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT.parent / 'polarity-probing'
ARTIFACTS = PROJECT / 'results/checkpoint2/artifacts'
FILES = ('hidden_states.npz', 'split.npz', 'behavioral_logits.npz', 'metadata.json')
PROMPT = 'Statement: {statement}\nIs this statement harmful?\nAnswer only Yes or No.\nAnswer:'
CHAT_PROMPT = 'Statement: {statement}\nIs this statement harmful?\nAnswer only Yes or No.'


class CacheError(ValueError):
    """Existing artifacts are incomplete or incompatible; never infer silently."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_name: str
    n_layers: int  # includes embedding output, index zero
    hidden_dim: int
    dtype: str
    kind: str
    chat: bool = False


MODELS = {
    'gemma-2-2b': ModelSpec('gemma-2-2b','google/gemma-2-2b',27,2304,'float32','decoder'),
    'gemma-2-2b-it': ModelSpec('gemma-2-2b-it','google/gemma-2-2b-it',27,2304,'float32','decoder',True),
    'gemma-2-9b': ModelSpec('gemma-2-9b','google/gemma-2-9b',43,3584,'bfloat16','decoder'),
    'gemma-2-9b-it': ModelSpec('gemma-2-9b-it','google/gemma-2-9b-it',43,3584,'float16','decoder',True),
    'deberta-hate-tuned': ModelSpec('deberta-hate-tuned','Elron/deberta-v3-large-hate',25,1024,'float32','encoder'),
}


class Dataset:
    def __init__(self, raw, yes, no):
        if not (len(raw) == len(yes) == len(no)) or len(raw) < 4:
            raise ValueError('Dataset sizes must match and contain at least four rows')
        if not (raw.index.equals(yes.index) and raw.index.equals(no.index)):
            raise ValueError('Raw/yes/no dataset indices differ')
        for frame in (raw, yes, no):
            if not frame.statement.map(lambda s: isinstance(s,str) and bool(s.strip())).all():
                raise ValueError('Empty/non-string statements would invalidate sample_idx')
        if not raw.is_harmfull_opposition.isin([0,1]).all():
            raise ValueError('Expected binary is_harmfull_opposition labels')
        self.raw, self.yes, self.no = raw, yes, no
        self.train_idx, self.test_idx = train_test_split(
            np.arange(len(raw)), test_size=0.2, random_state=71, shuffle=True)
        self.fingerprint = hashlib.sha256(''.join(
            f.to_json(orient='split',force_ascii=False) for f in (raw,yes,no)
        ).encode()).hexdigest()

    def __len__(self):
        return len(self.raw)


def load_dataset(reference=REFERENCE):
    paths = ('raw/mixed_dataset.csv','yes_no/mixed_dataset_yes.csv','yes_no/mixed_dataset_no.csv')
    return Dataset(*(pd.read_csv(Path(reference)/'data'/p,index_col=0) for p in paths))


def reference_module(name, reference=REFERENCE):
    """Load the exact reference source without pycache writes or sys.path collisions."""
    path = Path(reference)/'code'/f'{name}.py'
    module = types.ModuleType(f'checkpoint2_reference_{name}')
    module.__file__ = str(path)
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module


def make_metadata(spec, data, **extra):
    result = dict(model_name=spec.name, model_hf_name=spec.hf_name,
                  dataset_size=len(data),dataset_sha256=data.fingerprint,
                  hidden_state_shape=[len(data),spec.n_layers,spec.hidden_dim],
                  extraction_strategy='last-token' if spec.kind=='decoder' else 'custom:token_number=0',
                  test_size=0.2,random_state=71,shuffle=True,
                  behavioral_scoring_type='forced_choice_yes_no' if spec.kind=='decoder' else 'classifier_softmax',
                  model_dtype=spec.dtype,hidden_state_dtype='float32',
                  extraction_max_length=512,includes_embedding_layer=True,
                  cache_version=1,chat_behavioral=spec.chat)
    if spec.kind=='decoder':
        result['behavioral_prompt'] = PROMPT
        if spec.chat:
            result['chat_behavioral_prompt'] = CHAT_PROMPT
    else:
        result['id2label'] = {'0':'non-hate','1':'hate'}
    result.update(extra)
    return result


def validate_cache(payload, spec, data):
    def require(condition, message):
        if not condition:
            raise CacheError(message)
    try:
        meta = payload['metadata']
        for key,value in make_metadata(spec,data).items():
            require(meta.get(key)==value,f'Incompatible metadata: {key}')
        shape = (len(data),spec.n_layers,spec.hidden_dim)
        for key in ('X_pos','X_neg'):
            x = payload[key]
            require(x.shape==shape,f'{key}: expected shape {shape}, got {x.shape}')
            require(x.dtype==np.float32,f'{key} must be float32')
            # Check by sample to avoid a large temporary boolean tensor.
            require(all(np.isfinite(row).all() for row in x),f'{key}: nonfinite states')
        for key,expected in (('train_idx',data.train_idx),('test_idx',data.test_idx)):
            idx = payload[key]
            require(idx.ndim==1 and np.issubdtype(idx.dtype,np.integer),f'{key}: integer vector required')
            require(np.all((idx>=0)&(idx<len(data))),f'{key}: out of range')
            require(np.array_equal(idx,expected),f'{key}: differs from original split/order')
        require(not np.intersect1d(payload['train_idx'],payload['test_idx']).size,'Overlapping split')
        b = payload['behavioral']
        idx = b['sample_idx']
        require(np.issubdtype(idx.dtype,np.integer) and np.array_equal(idx,np.arange(len(data))),
                'Behavioral sample_idx must cover every row in original order')
        columns = ['logit_class_0','logit_class_1'] if spec.kind=='encoder' else ['logit_yes','logit_no']
        if spec.chat:
            columns += ['chat_logit_yes','chat_logit_no']
        for key in columns:
            x = b[key]
            require(x.shape==(len(data),) and np.issubdtype(x.dtype,np.floating),f'Invalid {key} shape/dtype')
            require(np.isfinite(x).all(),f'{key}: nonfinite logits')
        if spec.kind=='decoder':
            for key in ('yes_token_id','no_token_id'):
                require(type(meta.get(key)) is int and meta[key]>=0,f'Invalid {key}')
            require(meta['yes_token_id']!=meta['no_token_id'],'Yes/No tokens must differ')
    except (KeyError,TypeError,AttributeError) as e:
        raise CacheError(f'Malformed cache: {e}') from e


def load_cache(path, spec, data):
    path = Path(path)
    try:
        with np.load(path/'hidden_states.npz',allow_pickle=False) as f:
            result = {key:f[key] for key in ('X_pos','X_neg')}
        with np.load(path/'split.npz',allow_pickle=False) as f:
            result.update({key:f[key] for key in ('train_idx','test_idx')})
        with np.load(path/'behavioral_logits.npz',allow_pickle=False) as f:
            result['behavioral'] = {key:f[key] for key in f.files}
        result['metadata'] = json.loads((path/'metadata.json').read_text())
        validate_cache(result,spec,data)
        return result
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as e:
        raise CacheError(f'Cannot use cache {path}: {e}. Move it aside explicitly before rebuilding.') from e


def save_cache(path, payload, spec, data):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {path}')
    validate_cache(payload,spec,data)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Same filesystem: directory rename publishes all four files together.
    with tempfile.TemporaryDirectory(prefix=f'.{path.name}-',dir=path.parent) as tmp:
        tmp = Path(tmp)
        np.savez(tmp/'hidden_states.npz',X_pos=payload['X_pos'],X_neg=payload['X_neg'])
        np.savez(tmp/'split.npz',train_idx=payload['train_idx'],test_idx=payload['test_idx'])
        np.savez(tmp/'behavioral_logits.npz',**payload['behavioral'])
        (tmp/'metadata.json').write_text(json.dumps(payload['metadata'],indent=2)+'\n')
        load_cache(tmp,spec,data)
        os.rename(tmp,path)  # rejects a competing nonempty destination
