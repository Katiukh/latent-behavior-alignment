"""Contracts for plotting and aggregating existing PA-CCS layer metrics."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from plot_pa_ccs_metrics_reference import METRIC_COLUMNS, load_layer_metrics, run


class PaCcsMetricsTests(unittest.TestCase):
    @staticmethod
    def write_metrics(root, dataset, model, rows):
        source = (
            root / dataset / 'analysis' / 'checkpoint1_compatible' / model
            / 'layer_metrics.csv'
        )
        source.parent.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(source, index=False)
        return source

    def test_load_validates_numeric_layers_and_sorts_before_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'layer_metrics.csv'
            pd.DataFrame({
                'layer': ['10', '2', '1'],
                'accuracy': [.9, .5, .4],
                'polar_consistency_↓': [.3, .2, .1],
                'contradiction_idx_↓': [.6, .5, .4],
            }).to_csv(source, index=False)

            actual = load_layer_metrics(source, 'mixed', 'toy')

            self.assertEqual(actual.layer.tolist(), [1, 2, 10])
            self.assertEqual(actual.accuracy.tolist(), [.4, .5, .9])

    def test_load_rejects_missing_columns_nonnumeric_layers_and_all_nan_metric(self):
        valid = {
            'layer': [0, 1],
            'accuracy': [.4, .6],
            'polar_consistency_↓': [.1, .2],
            'contradiction_idx_↓': [.3, .4],
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'layer_metrics.csv'
            for mutation, message in [
                ({key: value for key, value in valid.items() if key != 'accuracy'},
                 'missing columns'),
                ({**valid, 'layer': [0, 'last']}, 'numeric'),
                ({**valid, 'polar_consistency_↓': [np.nan, np.nan]},
                 'only NaN'),
            ]:
                pd.DataFrame(mutation).to_csv(source, index=False)
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        load_layer_metrics(source, 'mixed', 'toy')

    def test_run_skips_missing_files_and_uses_equal_model_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_metrics(root, 'mixed', 'shallow', [
                {'layer': 1, 'accuracy': 0.2, 'polar_consistency_↓': 0.1,
                 'contradiction_idx_↓': 0.4},
                {'layer': 0, 'accuracy': 0.4, 'polar_consistency_↓': 0.3,
                 'contradiction_idx_↓': 0.2},
            ])
            self.write_metrics(root, 'mixed', 'deep', [
                {'layer': layer, 'accuracy': 0.9, 'polar_consistency_↓': 0.5,
                 'contradiction_idx_↓': 0.1}
                for layer in range(4)
            ])

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                summary = run(root, datasets=['mixed', 'not'],
                              models=['shallow', 'deep', 'missing'])

            output = root / 'pa_ccs_analysis_reference'
            model_means = pd.read_csv(output / 'pa_ccs_model_means.csv')
            dataset_means = pd.read_csv(output / 'pa_ccs_dataset_means.csv')
            self.assertEqual(model_means.columns.tolist(), [
                'dataset', 'model', 'esa_mean', 'polar_consistency_mean',
                'contradiction_idx_mean', 'n_layers'])
            self.assertEqual(dataset_means.columns.tolist(), [
                'dataset', 'esa_mean', 'polar_consistency_mean',
                'contradiction_idx_mean', 'n_models'])
            self.assertEqual(model_means.model.tolist(), ['shallow', 'deep'])
            self.assertEqual(model_means.n_layers.tolist(), [2, 4])
            mixed = dataset_means.iloc[0]
            self.assertEqual(mixed.dataset, 'mixed')
            self.assertAlmostEqual(mixed.esa_mean, .6)
            self.assertAlmostEqual(mixed.polar_consistency_mean, .35)
            self.assertAlmostEqual(mixed.contradiction_idx_mean, .2)
            self.assertEqual(mixed.n_models, 2)
            self.assertEqual(len(caught), 4)
            self.assertTrue((output / 'by_model/mixed/shallow.png').is_file())
            self.assertTrue((output / 'by_model/mixed/deep.png').is_file())
            self.assertTrue((output / 'summary/mixed_mean_metrics.png').is_file())
            self.assertFalse((output / 'summary/not_mean_metrics.png').exists())
            self.assertIn('mixed:\n  2 models\n  2 individual plots', summary)
            self.assertIn('not:\n  0 models\n  0 individual plots', summary)


if __name__ == '__main__':
    unittest.main()
