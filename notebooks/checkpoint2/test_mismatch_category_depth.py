"""Category shares are averaged within models before pooling model sizes."""
import unittest

import pandas as pd

from analyze_mismatch_cases_baseline import CASE_TYPES
from plot_mismatch_category_depth import aggregate_depth, accuracy_from_categories


class CategoryDepthTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for model, layers in [('deberta-hate-tuned', [(0, 0.), (6, 0.), (7, .2), (12, .4), (13, .6), (18, .8), (19, 1.), (24, 1.)]),
                              ('gemma-2-2b', [(0, 1.), (7, 1.), (14, 1.), (20, 1.)])]:
            for layer, fraction in layers:
                for case, value in zip(CASE_TYPES, [fraction, 1-fraction, 0., 0.]):
                    rows.append(dict(dataset='mixed', model=model, layer=layer,
                                     mismatch_category=case, fraction=value, split='test'))
        return pd.DataFrame(rows)

    def test_boundaries_and_equal_model_weight(self):
        per_model, groups = aggregate_depth(self.fixture())
        model = per_model[(per_model.model == 'deberta-hate-tuned') &
                          (per_model.mismatch_category == 'both_aligned')]
        self.assertEqual(model.first_layer.tolist(), [0, 7, 13, 19])
        self.assertEqual(model.last_layer.tolist(), [6, 12, 18, 24])
        shares = groups[groups.mismatch_category.eq('both_aligned')]
        for actual, expected in zip(shares.fraction, [.5, .65, .85, 1.]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(shares.n_models.eq(2).all())
        totals = groups.groupby(['dataset', 'model_group', 'depth_group'], observed=True).fraction.sum()
        self.assertTrue(totals.eq(1).all())

    def test_accuracy_counts_the_correct_categories(self):
        frame = pd.DataFrame({
            'dataset': ['mixed'] * 4, 'model_group': ['big'] * 4,
            'depth_group': ['Early'] * 4, 'mismatch_category': CASE_TYPES,
            'fraction': [.4, .3, .2, .1], 'n_models': [2] * 4,
        })
        result = accuracy_from_categories(frame)
        self.assertAlmostEqual(result.latent_accuracy.item(), .6)
        self.assertAlmostEqual(result.behavioral_calibrated_accuracy.item(), .7)

    def test_rejects_missing_category_or_duplicate_layer_category(self):
        frame = self.fixture()
        for invalid in [frame.iloc[1:], pd.concat([frame, frame.iloc[:1]])]:
            with self.assertRaises(ValueError):
                aggregate_depth(invalid)


if __name__ == '__main__':
    unittest.main()
