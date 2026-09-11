"""Dataset routing must never load or overwrite the other experiment."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import pandas as pd
import numpy as np
import artifacts as a
import test_pipeline


class DatasetTests(unittest.TestCase):
    def test_real_sources_and_independent_split(self):
        for variant, raw_name, n in [('mixed','mixed_dataset',1244),('not','not_hate_dataset',1250)]:
            data = a.load_dataset(dataset=variant)
            self.assertEqual(data.identifier, variant)
            self.assertEqual(len(data), n)
            for role, filename in [('raw',raw_name),('yes',f'{variant}_dataset_yes'),('no',f'{variant}_dataset_no')]:
                path = a.REFERENCE/'data'/('raw' if role=='raw' else 'yes_no')/(filename+'.csv')
                pd.testing.assert_frame_equal(getattr(data,role), pd.read_csv(path,index_col=0))
                self.assertEqual(data.sources[role],str(path))
            self.assertEqual(len(data.test_idx), 249 if variant=='mixed' else 250)
            np.testing.assert_array_equal(data.raw.is_harmfull_opposition, [0]*(n//2)+[1]*(n//2))
        self.assertNotEqual(a.load_dataset(dataset='mixed').fingerprint,a.load_dataset(dataset='not').fingerprint)

    def test_preflight_cli_routes_every_output_without_loading_models(self):
        for variant in ('mixed','not'):
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('run_all.py')),
                '--dataset',variant,'--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(result.stdout)
            self.assertEqual(report['dataset'],variant)
            self.assertEqual(report['output_root'],str(a.PROJECT/'results/checkpoint2'/variant))
            self.assertEqual(set(report['models']),set(a.MODELS))

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            a.load_dataset(dataset='../mixed')

    def test_inference_only_runner_propagates_not_to_each_process(self):
        import run_all
        commands=[]
        def child(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'not'
            with patch.object(run_all, 'output_root', return_value=root), \
                 patch.object(run_all.subprocess, 'run', side_effect=child), \
                 patch.object(sys, 'argv', ['run_all.py','--dataset','not','--stage','inference']):
                run_all.main()
            status=json.loads((root/'logs/run_status.json').read_text())
            self.assertEqual(status['dataset'],'not')
            self.assertEqual(len(commands),5)
            for command in commands:
                self.assertEqual(command[command.index('--dataset')+1],'not')
                self.assertEqual(command[command.index('--stage')+1],'inference')
            self.assertFalse((root/'analysis').exists())

    def test_stage_reads_not_and_writes_not_artifacts(self):
        import run_all
        import torch
        def extract(path,spec,data):
            self.assertEqual(path,a.PROJECT/'results/checkpoint2/not/artifacts/gemma-2-2b')
            self.assertEqual(data.identifier,'not')
            self.assertEqual(len(data),1250)
            self.assertTrue(data.sources['yes'].endswith('/not_dataset_yes.csv'))
            self.assertTrue(data.sources['no'].endswith('/not_dataset_no.csv'))
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.cuda,'get_device_name',return_value='test'), \
             patch('inference.ensure_cache',side_effect=extract):
            run_all.run_stage('gemma-2-2b','inference','not')


class DatasetCacheTests(unittest.TestCase):
    setUp = test_pipeline.CacheTests.setUp

    def test_identifier_mismatch_rejected_even_with_identical_content(self):
        other=copy.copy(self.data)
        other.identifier='not'
        with self.assertRaises(a.CacheError):
            a.validate_cache(self.payload,self.spec,other)

    def test_validation_mode_refuses_missing_probes(self):
        from offline import analyze
        a.save_cache(self.path,self.payload,self.spec,self.data)
        with self.assertRaisesRegex(ValueError,'training is disabled'):
            analyze(self.path,self.spec,self.data,allow_training=False)
        self.assertFalse((self.path/'ccs_probes.npz').exists())

    def test_validation_exports_same_tables_without_retraining(self):
        from offline import analyze
        from validation import export_validation
        a.save_cache(self.path,self.payload,self.spec,self.data)
        scores,metrics=analyze(self.path,self.spec,self.data,nepochs=2,ntries=1)
        root=Path(self.tmp.name)/'experiment'
        (root/'artifacts').mkdir(parents=True)
        self.path.rename(root/'artifacts/toy')
        dest=root/'analysis/checkpoint1_compatible/toy'
        dest.mkdir(parents=True)
        scores.to_csv(dest/'scores.csv',index=False)
        metrics.to_csv(dest/'layer_metrics.csv',index=False)
        result=export_validation(root,self.data,{'toy':self.spec},nepochs=2,ntries=1)
        self.assertEqual(result['validation'],'passed')
        self.assertEqual(result['total_score_rows'],40)
        self.assertTrue(result['models'][0]['probe_reuse_verified'])
        self.assertTrue((dest.parent/'model_summary.csv').is_file())
        self.assertTrue((dest.parent/'all_layer_metrics.csv').is_file())
        scores.loc[0,'latent_score']+=0.1
        scores.to_csv(dest/'scores.csv',index=False)
        with self.assertRaises(AssertionError):
            export_validation(root,self.data,{'toy':self.spec},nepochs=2,ntries=1)
