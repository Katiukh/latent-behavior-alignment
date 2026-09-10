"""One model session per cache miss; this module does not import transformers eagerly."""
from contextlib import contextmanager
import gc
import os
from pathlib import Path

import numpy as np

from artifacts import (CHAT_PROMPT, PROMPT, make_metadata, reference_module,
                       load_cache, save_cache)


def load_model(spec):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    if spec.kind == 'encoder':
        import json
        from huggingface_hub import hf_hub_download
        from transformers import DebertaV2Config
        with open(hf_hub_download(spec.hf_name,'config.json')) as f:
            config = json.load(f)
        config.update(id2label={0:'non-hate',1:'hate'},label2id={'non-hate':0,'hate':1})
        config = DebertaV2Config.from_dict(config)
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_name,config=config)
        model = AutoModelForSequenceClassification.from_pretrained(spec.hf_name,config=config)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
        kwargs = dict(torch_dtype=getattr(torch,spec.dtype),device_map='auto',attn_implementation='eager')
        if os.environ.get('CHECKPOINT2_MAX_GPU_MEMORY'):
            # Reserve VRAM for forward activations while preserving model dtype.
            kwargs['max_memory'] = {0:os.environ['CHECKPOINT2_MAX_GPU_MEMORY'],'cpu':'10GiB'}
        if spec.name == 'gemma-2-9b-it':
            kwargs.update(low_cpu_mem_usage=True,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(spec.hf_name,**kwargs)
    model.eval()
    return model,tokenizer


def raw_behavioral_logits(spec, data, model, tokenizer):
    import torch
    from tqdm.auto import tqdm
    result = {'sample_idx':np.arange(len(data),dtype=np.int64)}
    extra = {}
    device = next(model.parameters()).device
    def move(inputs):
        return {k:v.to(device) for k,v in inputs.items()}
    if spec.kind == 'encoder':
        rows = []
        for statement in tqdm(data.raw.statement,desc='Classifier logits'):
            inputs = move(tokenizer(statement,return_tensors='pt',truncation=True,max_length=512))
            with torch.no_grad():
                logits = model(**inputs).logits[0].float().cpu().numpy()
            if logits.shape != (2,):
                raise ValueError(f'Expected two classifier logits, got {logits.shape}')
            rows.append(logits)
        rows = np.asarray(rows,dtype=np.float32)
        result.update(logit_class_0=rows[:,0],logit_class_1=rows[:,1])
        return result,extra

    def token_id(text):
        ids = tokenizer.encode(text,add_special_tokens=False)
        if len(ids)!=1:
            raise ValueError(f'Expected one token for {text!r}, got {ids}')
        return ids[0]
    yes,no = token_id(' Yes'),token_id(' No')
    extra.update(yes_token_id=int(yes),no_token_id=int(no))
    batched = spec.name == 'gemma-2-9b-it'
    batch_size = 2 if batched else 1
    if batched:
        # Set only AFTER hidden extraction, exactly as Checkpoint 1.
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = model.get_input_embeddings().weight.device
    plain,chat = [],[]
    for start in tqdm(range(0,len(data),batch_size),desc='Yes/No logits'):
        statements = data.raw.statement.iloc[start:start+batch_size].tolist()
        prompts = [PROMPT.format(statement=s) for s in statements]
        inputs = tokenizer(prompts,return_tensors='pt',padding=True) if batched else tokenizer(prompts[0],return_tensors='pt')
        with torch.no_grad():
            logits = model(**move(inputs)).logits[:,-1,:][:,[no,yes]].float().cpu().numpy()
        plain.append(logits)
        if spec.chat:
            conversations = [[{'role':'user','content':CHAT_PROMPT.format(statement=s)}] for s in statements]
            kwargs = dict(add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
            if batched:
                kwargs['padding'] = True
            inputs = tokenizer.apply_chat_template(conversations if batched else conversations[0],**kwargs)
            with torch.no_grad():
                logits = model(**move(inputs)).logits[:,-1,:][:,[no,yes]].float().cpu().numpy()
            chat.append(logits)
    plain = np.concatenate(plain)
    result.update(logit_no=plain[:,0],logit_yes=plain[:,1])
    if spec.chat:
        chat = np.concatenate(chat)
        result.update(chat_logit_no=chat[:,0],chat_logit_yes=chat[:,1])
    return result,extra


@contextmanager
def extract_model(spec, data):
    import torch
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model,tokenizer = load_model(spec)
    try:
        expected = (spec.n_layers-1,spec.hidden_dim)
        actual = (model.config.num_hidden_layers,model.config.hidden_size)
        if actual != expected:
            raise ValueError(f'Model config {actual} differs from expected {expected}')
        extract = reference_module('extract')
        kwargs = dict(layer_index=None,strategy='last-token' if spec.kind=='decoder' else 'custom',
                      model_type=spec.kind,use_decoder=False,get_all_hs=True,device=None,
                      token_number=0 if spec.kind=='encoder' else None)
        pos = np.asarray(extract.vectorize_df(data.yes.statement,model,tokenizer,**kwargs),dtype=np.float32)
        neg = np.asarray(extract.vectorize_df(data.no.statement,model,tokenizer,**kwargs),dtype=np.float32)
        logits,extra = raw_behavioral_logits(spec,data,model,tokenizer)
        # The caller persists while this one model instance is still alive.
        yield dict(X_pos=pos,X_neg=neg,train_idx=data.train_idx,test_idx=data.test_idx,
                   behavioral=logits,metadata=make_metadata(spec,data,**extra))
    finally:
        del model,tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def ensure_cache(path, spec, data):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        load_cache(path,spec,data)
        print(f'Valid cache: {path}; inference skipped')
        return path
    with extract_model(spec,data) as payload:
        save_cache(path,payload,spec,data)
    print(f'Saved artifacts and released model: {path}')
    return path
