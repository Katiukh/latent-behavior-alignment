"""Contracts for the standalone behavioral decision-threshold experiment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_behavioral_thresholds import (
    METRIC_COLUMNS,
    analyze_scores,
    run,
)


class BehavioralThresholdTests(unittest.TestCase):
    @staticmethod
    def layered(rows):
        return pd.DataFrame([
            dict(sample_idx=sample_idx, split=split, true_label=label,
                 behavioral_probability_yes=probability, layer=layer)
            for layer in [0, 1]
            for sample_idx, split, label, probability in rows
        ])

    def test_optimization_uses_train_accuracy_not_balanced_accuracy(self):
        # At 1.0 all ten train objects are predicted negative: accuracy=.9.
        # At .7 the positive is recovered at the cost of two false positives:
        # accuracy=.8 but balanced accuracy=.889, so the two objectives disagree.
        rows = [
            *[(i, 'train', 0, probability) for i, probability in enumerate(
                [.1, .2, .3, .4, .5, .6, .65, .75, .8])],
            (9, 'train', 1, .7),
            (10, 'test', 0, .99),
            (11, 'test', 1, .01),
        ]
        result = analyze_scores(self.layered(rows), 'toy', 'model')

        self.assertEqual(result.threshold['behavioral_threshold'], 1.0)
        self.assertEqual(result.threshold['n_train'], 10)
        self.assertEqual(result.threshold['n_test'], 2)
        optimized = result.metrics[
            (result.metrics.threshold_type == 'train_optimized')
            & (result.metrics.split == 'train')
        ].iloc[0]
        self.assertAlmostEqual(optimized.accuracy, .9)
        self.assertAlmostEqual(optimized.balanced_accuracy, .5)

    def test_accuracy_tie_uses_threshold_closest_to_half_then_lower(self):
        rows = [
            (0, 'train', 0, .1),
            (1, 'train', 1, .3),
            (2, 'train', 0, .5),
            (3, 'train', 1, .7),
            (4, 'train', 0, .9),
            (5, 'test', 0, .2),
            (6, 'test', 1, .8),
        ]
        result = analyze_scores(self.layered(rows), 'toy', 'model')

        self.assertAlmostEqual(result.threshold['behavioral_threshold'], .3)
        self.assertEqual(len(result.curve), 7)  # five unique probabilities plus 0 and 1

    def test_layer_duplicates_are_removed_and_probability_must_be_invariant(self):
        rows = [
            (0, 'train', 0, .2), (1, 'train', 1, .8),
            (2, 'test', 0, .3), (3, 'test', 1, .7),
        ]
        source = self.layered(rows)
        result = analyze_scores(source, 'toy', 'model')
        self.assertEqual(result.threshold['n_train'], 2)
        self.assertEqual(result.threshold['n_test'], 2)

        source.loc[(source.sample_idx == 0) & (source.layer == 1),
                   'behavioral_probability_yes'] = .9
        with self.assertRaisesRegex(ValueError, 'changes across layers'):
            analyze_scores(source, 'toy', 'model')

    def test_deberta_hate_probability_is_the_positive_class_probability(self):
        rows = [
            (0, 'train', 0, .2), (1, 'train', 1, .8),
            (2, 'test', 0, .3), (3, 'test', 1, .7),
        ]
        source = self.layered(rows).rename(
            columns={'behavioral_probability_yes': 'behavioral_probability_hate'})
        result = analyze_scores(source, 'toy', 'deberta-hate-tuned')
        self.assertEqual(result.threshold['behavioral_threshold'], .5)
        self.assertTrue(result.metrics.accuracy.eq(1).all())

    def test_metrics_use_inclusive_boundary_and_standard_confusion_order(self):
        rows = [
            (0, 'train', 0, .5), (1, 'train', 1, .5),
            (2, 'test', 0, .2), (3, 'test', 0, .6),
            (4, 'test', 1, .4), (5, 'test', 1, .8),
        ]
        result = analyze_scores(self.layered(rows), 'toy', 'model')
        fixed_test = result.metrics[
            (result.metrics.threshold_type == 'fixed_0.5')
            & (result.metrics.split == 'test')
        ].iloc[0]
        self.assertEqual(fixed_test[METRIC_COLUMNS].tolist(),
                         [.5, .5, .5, .5, .5, 1, 1, 1, 1])

    def test_run_writes_combined_tables_and_one_plot_per_model(self):
        rows = [
            (0, 'train', 0, .2), (1, 'train', 1, .8),
            (2, 'test', 0, .3), (3, 'test', 1, .7),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'mixed/analysis/checkpoint1_compatible/toy'
            source.mkdir(parents=True)
            self.layered(rows).to_csv(source / 'scores.csv', index=False)

            summary = run(root)
            output = root / 'behavioral_threshold_analysis'
            thresholds = pd.read_csv(output / 'behavioral_thresholds.csv')
            metrics = pd.read_csv(output / 'behavioral_threshold_metrics.csv')
            self.assertEqual(thresholds.columns.tolist(), [
                'dataset', 'model', 'behavioral_threshold',
                'train_accuracy', 'train_balanced_accuracy', 'n_train', 'n_test'])
            self.assertEqual(metrics.columns.tolist(), [
                'dataset', 'model', 'threshold_type', 'threshold', 'split',
                *METRIC_COLUMNS])
            self.assertEqual(len(metrics), 4)
            self.assertTrue((output / 'plots/mixed/toy.png').is_file())
            self.assertIn('mixed / toy', summary)
            self.assertIn('best train threshold: 0.500', summary)


if __name__ == '__main__':
    unittest.main()
