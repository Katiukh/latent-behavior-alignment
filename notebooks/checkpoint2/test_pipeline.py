"""Contract tests: cache corruption, costly inference reuse, offline parity."""
import copy
from contextlib import nullcontext
import ast
import gc
import random
import subprocess
import sys
import weakref
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import artifacts as a


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'toy'
        self.spec = a.ModelSpec('toy', 'toy/model', 2, 3, 'float32', 'decoder')
        raw = pd.DataFrame({'statement': [f's{i}' for i in range(20)],
                            'is_harmfull_opposition': [0]*10+[1]*10})
        self.data = a.Dataset(raw, raw.assign(statement=raw.statement+' Yes.'),
                              raw.assign(statement=raw.statement+' No.'))
        self.pos = np.random.default_rng(42).normal(size=(20, 2, 3)).astype('float32')
        self.neg = self.pos + 2
        self.logits = {'sample_idx': np.arange(20), 'logit_yes': np.zeros(20),
                       'logit_no': np.zeros(20)}
        self.meta = a.make_metadata(self.spec, self.data, yes_token_id=1, no_token_id=2)
        self.payload = dict(X_pos=self.pos, X_neg=self.neg,
                            train_idx=self.data.train_idx, test_idx=self.data.test_idx,
                            behavioral=self.logits, metadata=self.meta)

    def test_roundtrip_preserves_raw_and_split(self):
        a.save_cache(self.path, self.payload, self.spec, self.data)
        loaded = a.load_cache(self.path, self.spec, self.data)
        np.testing.assert_array_equal(loaded['X_pos'], self.pos)
        np.testing.assert_array_equal(loaded['X_neg'], self.neg)
        np.testing.assert_array_equal(loaded['test_idx'], self.data.test_idx)
        self.assertEqual({p.name for p in self.path.iterdir()},
                         {'hidden_states.npz','split.npz','behavioral_logits.npz','metadata.json'})
        with self.assertRaises(FileExistsError):
            a.save_cache(self.path, self.payload, self.spec, self.data)

    def test_rejects_incompatible_payloads(self):
        for mutate in [
            lambda p: p.update(X_neg=self.neg[:-1]),
            lambda p: p.update(X_pos=self.pos[:, :1]),
            lambda p: p['metadata'].update(model_name='wrong'),
            lambda p: p['metadata'].update(dataset_size=19),
            lambda p: p.update(train_idx=np.array([20])),
            lambda p: p.update(test_idx=self.data.train_idx),
            lambda p: p.update(train_idx=self.data.train_idx[::-1]),
            lambda p: p['behavioral'].update(sample_idx=np.arange(20)[::-1]),
            lambda p: p['behavioral'].update(logit_yes=np.full(20,np.nan)),
        ]:
            with self.subTest(mutate=mutate):
                payload = copy.deepcopy(self.payload)
                mutate(payload)
                with self.assertRaises(a.CacheError):
                    a.validate_cache(payload, self.spec, self.data)

    def test_dataset_content_change_is_rejected(self):
        a.save_cache(self.path, self.payload, self.spec, self.data)
        raw = self.data.raw.copy()
        raw.loc[0,'statement'] = 'changed'
        changed = a.Dataset(raw,self.data.yes,self.data.no)
        with self.assertRaises(a.CacheError):
            a.load_cache(self.path, self.spec, changed)

    def test_partial_cache_does_not_trigger_inference(self):
        from inference import ensure_cache
        self.path.mkdir()
        (self.path/'metadata.json').write_text('{}')
        with patch('inference.extract_model', side_effect=AssertionError('model loaded')):
            with self.assertRaises(a.CacheError):
                ensure_cache(self.path,self.spec,self.data)

    def test_cache_hit_never_calls_model(self):
        from inference import ensure_cache
        a.save_cache(self.path,self.payload,self.spec,self.data)
        with patch('inference.extract_model', side_effect=AssertionError('model loaded')):
            self.assertEqual(ensure_cache(self.path,self.spec,self.data),self.path)

    def test_cache_miss_persists_then_reuses(self):
        from inference import ensure_cache
        with patch('inference.extract_model', side_effect=lambda *args: nullcontext(self.payload)) as extract:
            ensure_cache(self.path,self.spec,self.data)
            ensure_cache(self.path,self.spec,self.data)
        self.assertEqual(extract.call_count,1)
        np.testing.assert_array_equal(a.load_cache(self.path,self.spec,self.data)['X_pos'],self.pos)

    def test_offline_logits_and_threshold(self):
        from offline import behavioral_scores
        self.logits['logit_yes'][:3] = [0, np.log(3), -10000]
        df = behavioral_scores(self.payload, threshold=0.8)
        np.testing.assert_allclose(df.behavioral_probability_yes[:3], [.5,.75,0],atol=1e-7)
        np.testing.assert_array_equal(df.behavioral_score[:3],[0,0,0])
        df = behavioral_scores(self.payload)
        np.testing.assert_array_equal(df.behavioral_score[:3],[0,1,0])

    def test_offline_retraining_preserves_cache(self):
        from offline import analyze
        a.save_cache(self.path,self.payload,self.spec,self.data)
        scores, metrics = analyze(self.path,self.spec,self.data,device='cpu',nepochs=2,ntries=1)
        self.assertEqual(len(scores),40)
        self.assertEqual(len(metrics),2)
        self.assertFalse(scores.behavioral_score.isna().any())
        self.assertTrue(np.isfinite(metrics['contradiction_idx_↓']).all())
        np.testing.assert_array_equal(a.load_cache(self.path,self.spec,self.data)['X_pos'],self.pos)

    def test_checkpoint1_scores_and_original_pa_metrics_match(self):
        # Executes the actual Checkpoint 1 function, not a rewritten expectation.
        import functools
        import json
        import torch
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import normalize
        from offline import analyze
        notebook = a.PROJECT/'notebooks/checkpoint1/ccs_gemma-2-2b_fixed.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        source = next(''.join(c['source']) for c in cells if 'def train_ccs_and_collect_scores' in ''.join(c['source']))
        function = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train_ccs_and_collect_scores')
        ref = a.reference_module('ccs')
        ref.CCS = functools.partial(ref.CCS,nepochs=2,ntries=1)
        namespace = dict(np=np,pd=pd,torch=torch,gc=gc,CCS=ref.CCS,CCS_DEVICE='cpu',
                         accuracy_score=accuracy_score,mixed_data=self.data.raw)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(notebook),'exec'),namespace)
        pos = normalize(self.pos.reshape(-1,3)).reshape(self.pos.shape)
        neg = normalize(self.neg.reshape(-1,3)).reshape(self.neg.shape)
        labels = 1-self.data.raw.is_harmfull_opposition.to_numpy(dtype=np.float32)
        torch.manual_seed(0);np.random.seed(0);random.seed(0)
        expected,_ = namespace['train_ccs_and_collect_scores'](pos,neg,labels,self.data.train_idx,self.data.test_idx,device='cpu')
        torch.manual_seed(0);np.random.seed(0);random.seed(0)
        expected_pa = ref.train_ccs_on_hidden_states(pos,neg,pd.Series(labels),self.data.train_idx,self.data.test_idx,normalizing='median',device='cpu')
        a.save_cache(self.path,self.payload,self.spec,self.data)
        actual,metrics = analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)
        pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False)
        for layer,r in expected_pa.items():
            self.assertAlmostEqual(metrics.iloc[layer]['accuracy'],r['accuracy'])
            self.assertAlmostEqual(metrics.iloc[layer]['silhouette'],r['silhouette'])
            self.assertAlmostEqual(metrics.iloc[layer]['polar_consistency_↓'],np.mean(r['agreement']))
            self.assertAlmostEqual(metrics.iloc[layer]['abs_agreement_score'],np.median(np.abs(r['agreement'])))
            self.assertAlmostEqual(metrics.iloc[layer]['contradiction_idx_↓'],np.mean(r['contradiction idx']))

    def test_deberta_and_chat_scores_from_raw_logits(self):
        from offline import behavioral_scores
        self.payload['metadata']['behavioral_scoring_type']='classifier_softmax'
        self.payload['behavioral']={'sample_idx':np.arange(2),'logit_class_0':np.array([0.,2.]),'logit_class_1':np.array([0.,0.])}
        df=behavioral_scores(self.payload)
        np.testing.assert_allclose(df.behavioral_probability_hate,[.5,.11920292],atol=1e-7)
        self.assertEqual(df.behavioral_response.tolist(),['non-hate','non-hate'])
        self.payload['metadata'].update(behavioral_scoring_type='forced_choice_yes_no',chat_behavioral=True)
        self.payload['behavioral']={'sample_idx':np.arange(2),'logit_no':np.zeros(2),'logit_yes':np.zeros(2),
                                    'chat_logit_no':np.zeros(2),'chat_logit_yes':np.array([0.,2.])}
        df=behavioral_scores(self.payload)
        self.assertEqual(df.chat_behavioral_score.tolist(),[0,1])
        self.assertEqual(df.behavioral_score.tolist(),[0,0])

    def test_one_model_session_collects_raw_outputs_then_releases(self):
        import torch
        from types import SimpleNamespace
        from inference import ensure_cache
        forwarded=[]
        class Inputs(dict):
            def to(self,device):
                return Inputs({k:v.to(device) for k,v in self.items()})
        class Tokenizer:
            def __call__(self,text,**kwargs):
                forwarded.append((text,kwargs))
                return Inputs(input_ids=torch.tensor([[1,2]]),attention_mask=torch.tensor([[1,1]]))
            def encode(self,text,**kwargs):
                return {' Yes':[1],' No':[0]}[text]
        class Output(dict):
            __getattr__=dict.__getitem__
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.ones(1))
                self.config=SimpleNamespace(num_hidden_layers=1,hidden_size=3)
                self.device=torch.device('cpu')
            def forward(self,**kwargs):
                hs=torch.tensor([[[1.,2.,3.],[4.,5.,6.]]])
                return Output(hidden_states=(hs,hs+10),logits=torch.tensor([[[0.,0.],[7.,9.]]]))
        refs=[]
        def load(spec):
            model=Model()
            refs.append(weakref.ref(model))
            return model,Tokenizer()
        real_save=a.save_cache
        def save(*args):
            self.assertIsNotNone(refs[0]())  # save precedes unload
            return real_save(*args)
        with patch('inference.load_model',side_effect=load) as loader,patch('inference.save_cache',side_effect=save):
            ensure_cache(self.path,self.spec,self.data)
            ensure_cache(self.path,self.spec,self.data)
        self.assertEqual(loader.call_count,1)
        self.assertIsNone(refs[0]())
        cached=a.load_cache(self.path,self.spec,self.data)
        np.testing.assert_array_equal(cached['X_pos'][0],[[4,5,6],[14,15,16]])
        np.testing.assert_array_equal(cached['behavioral']['logit_no'],np.full(20,7))
        np.testing.assert_array_equal(cached['behavioral']['logit_yes'],np.full(20,9))
        self.assertEqual(len(forwarded),60)
        self.assertEqual(forwarded[0][0],'s0 Yes.')
        self.assertEqual(forwarded[20][0],'s0 No.')
        self.assertEqual(forwarded[40][0],'Statement: s0\nIs this statement harmful?\nAnswer only Yes or No.\nAnswer:')

    def test_saved_probes_skip_training_and_reproduce_scores(self):
        from offline import analyze
        a.save_cache(self.path,self.payload,self.spec,self.data)
        first,metrics=analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)
        self.assertTrue((self.path/'ccs_probes.npz').is_file())
        ref=a.reference_module('ccs')
        with patch.object(ref.CCS,'repeated_train',side_effect=AssertionError('CCS retrained')):
            with patch('offline.reference_module',return_value=ref):
                second,again=analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)
        pd.testing.assert_frame_equal(first,second)
        pd.testing.assert_frame_equal(metrics,again)

    def test_changed_probe_configuration_retrains_offline(self):
        from offline import analyze
        a.save_cache(self.path,self.payload,self.spec,self.data)
        analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)
        # New preprocessing is allowed offline, while the raw artifact is fixed.
        scores,_=analyze(self.path,self.spec,self.data,nepochs=2,ntries=1,preprocessing='mean')
        self.assertEqual(len(scores),40)
        with np.load(self.path/'ccs_probes.npz',allow_pickle=False) as probes:
            self.assertEqual(probes['weight'].shape,(2,3))
            self.assertEqual(probes['train_mean_pos'].shape,(2,3))
        np.testing.assert_array_equal(a.load_cache(self.path,self.spec,self.data)['X_pos'],self.pos)

    def test_bad_probe_dimensions_are_rejected(self):
        from offline import analyze
        a.save_cache(self.path,self.payload,self.spec,self.data)
        analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)
        with np.load(self.path/'ccs_probes.npz',allow_pickle=False) as f:
            probes={key:f[key] for key in f.files}
        probes['weight']=np.zeros((2,2),dtype=np.float32)
        np.savez(self.path/'ccs_probes.npz',**probes)
        with self.assertRaises(a.CacheError):
            analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)

    def test_fresh_process_needs_neither_transformers_nor_ccs_training(self):
        from offline import analyze
        a.save_cache(self.path,self.payload,self.spec,self.data)
        expected,metrics=analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)
        expected.to_csv(self.path/'expected.csv',index=False)
        metrics.to_csv(self.path/'metrics.csv',index=False)
        script=r'''
import sys
from pathlib import Path
class BlockTransformers:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]=='transformers':
            raise AssertionError('Offline process attempted to import transformers')
sys.meta_path.insert(0,BlockTransformers())
import numpy as np
import pandas as pd
import artifacts as a
import offline
raw=pd.DataFrame({'statement':[f's{i}' for i in range(20)],'is_harmfull_opposition':[0]*10+[1]*10})
data=a.Dataset(raw,raw.assign(statement=raw.statement+' Yes.'),raw.assign(statement=raw.statement+' No.'))
spec=a.ModelSpec('toy','toy/model',2,3,'float32','decoder')
ref=a.reference_module('ccs')
def forbidden(*args,**kwargs):
    raise AssertionError('Offline process attempted to retrain CCS')
ref.CCS.repeated_train=forbidden
offline.reference_module=lambda name:ref
path=Path(sys.argv[1])
scores,metrics=offline.analyze(path,spec,data,nepochs=2,ntries=1)
pd.testing.assert_frame_equal(scores,pd.read_csv(path/'expected.csv'),check_dtype=False)
pd.testing.assert_frame_equal(metrics,pd.read_csv(path/'metrics.csv'),check_dtype=False)
probabilities=offline.pa_probabilities(path,spec,data,0)
pA0,pA1,pN0,pN1=(probabilities[k].to_numpy() for k in ['pA0','pA1','p_notA0','p_notA1'])
np.testing.assert_allclose(np.mean(pA1*pN1+pA0*pN0),metrics.iloc[0]['contradiction_idx_↓'],rtol=1e-6)
assert 'transformers' not in sys.modules
print('Fresh-process reuse verified')
'''
        result=subprocess.run([sys.executable,'-B','-c',script,str(self.path)],
            cwd=Path(__file__).parent,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Fresh-process reuse verified',result.stdout)


if __name__ == '__main__':
    unittest.main()
