"""Contracts for fixed-threshold agreement with the true label."""
import unittest

import numpy as np
import pandas as pd

from analyze_mismatch_cases import CASE_TYPES, analyze_scores, summarize, top_cases, wide_summary


class MismatchTests(unittest.TestCase):
    def source(self):
        return pd.DataFrame([
            dict(sample_idx=i, statement=f'object {i}', split='test', true_label=label,
                 layer=layer, behavioral_probability_yes=b, latent_score=l,
                 raw_latent_score=0.123, behavioral_response='preserve')
            for layer in [0, 1]
            for i, (label, b, l) in enumerate([
                (1, 0.9, 0.9 if layer == 0 else 0.5),
                (0, 0.1, 0.9 if layer == 0 else 0.5),
                (1, 0.1, 0.9 if layer == 0 else 0.5),
                (0, 0.9, 0.9 if layer == 0 else 0.5),
            ])
        ])

    def test_orientation_four_categories_and_empty_categories(self):
        source = self.source()
        objects = analyze_scores(source, 'toy', 'model')
        pd.testing.assert_frame_equal(objects[source.columns], source)
        np.testing.assert_allclose(objects.behavioral_alignment, [0.9, 0.9, 0.1, 0.1]*2)
        np.testing.assert_allclose(objects.latent_alignment, [0.9, 0.1, 0.9, 0.1]+[0.5]*4)
        self.assertEqual(objects.loc[objects.layer == 0, 'case_type'].tolist(), CASE_TYPES)
        np.testing.assert_allclose(objects.signed_alignment_gap, [0, -0.8, 0.8, 0, -0.4, -0.4, 0.4, 0.4], atol=1e-15)
        summary = summarize(objects)
        self.assertEqual(summary.groupby('layer').n.sum().tolist(), [4, 4])
        self.assertEqual(summary.groupby('layer').size().tolist(), [4, 4])
        empty = summary[(summary.layer == 1) & summary.case_type.isin([CASE_TYPES[1], CASE_TYPES[3]])]
        self.assertTrue(empty.n.eq(0).all())
        self.assertTrue(empty.median_abs_alignment_gap.isna().all())
        wide = wide_summary(summary)
        np.testing.assert_allclose(wide.fraction_aligned_output_misaligned_latent, [0.25, 0])
        np.testing.assert_allclose(wide.fraction_misaligned_output_aligned_latent, [0.25, 0.5])

    def test_fixed_boundary_and_strong_agreement(self):
        source = self.source()
        source['behavioral_probability_yes'] = [0.5, 0.5, 0.499, 0.501]*2
        source.loc[source.layer == 1, 'latent_score'] = [0.91, 0.01, 0.99, 0.01]
        objects = analyze_scores(source, 'toy', 'model')
        self.assertEqual(objects.behavior_aligned.tolist(), [True, True, False, False]*2)
        self.assertTrue(objects.loc[objects.layer == 1, 'latent_aligned'].all())
        objects = analyze_scores(self.source(), 'toy', 'model')
        self.assertTrue(objects.loc[objects.layer == 1, 'latent_aligned'].all())

    def test_top_cases_are_separate_per_layer_and_category(self):
        objects = analyze_scores(self.source(), 'toy', 'model')
        top = top_cases(objects, 1)
        self.assertEqual(len(top), 3)
        self.assertEqual(top[top.layer == 0].sample_idx.tolist(), [1, 2])
        self.assertEqual(top[top.layer == 1].sample_idx.tolist(), [2])
        self.assertTrue(top.mismatch_rank.eq(1).all())
        self.assertTrue(top_cases(objects[objects.case_type == CASE_TYPES[0]], 20).empty)

    def test_invalid_input_is_rejected(self):
        for value in [np.nan, np.inf, -0.1, 1.1]:
            source = self.source()
            source.loc[0, 'latent_score'] = value
            with self.assertRaises(ValueError):
                analyze_scores(source, 'toy', 'model')
        source = self.source()
        with self.assertRaises(ValueError):
            analyze_scores(pd.concat([source, source.iloc[:1]]), 'toy', 'model')
        source.loc[0, 'true_label'] = 2
        with self.assertRaises(ValueError):
            analyze_scores(source, 'toy', 'model')


if __name__ == '__main__':
    unittest.main()
