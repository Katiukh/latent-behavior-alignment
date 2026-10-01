"""Regression checks for saved-center PA-CCS and calibrated test mismatch."""
import unittest

import numpy as np
import pandas as pd


class CorrectedPaTests(unittest.TestCase):
    def test_esa_boundary_matches_reference_legend(self):
        from analyze_pa_ccs import esa_regime
        self.assertEqual(esa_regime([.5, .749, .75, 1.]).tolist(),
                         ['0.5 <= ESA < 0.75', '0.5 <= ESA < 0.75',
                          'strong ESA (>= 0.75)', 'strong ESA (>= 0.75)'])

    def test_shared_saved_offsets_and_no_group_recentering(self):
        from analyze_pa_ccs import centered_layer, consistent_probe
        from artifacts import reference_module

        # Unit vectors: group-specific centering would erase their difference.
        cache = {'X_pos': np.array([[[1., 0.]], [[0., 1.]],
                                   [[-1., 0.]], [[0., -1.]]], dtype=np.float32),
                 'X_neg': np.array([[[0., 1.]], [[1., 0.]],
                                   [[0., -1.]], [[-1., 0.]]], dtype=np.float32)}
        saved = {'prediction_offset_pos': np.array([[.2, .3]], dtype=np.float32),
                 'prediction_offset_neg': np.array([[-.4, .1]], dtype=np.float32),
                 'weight': np.array([[2., -1.]], dtype=np.float32),
                 'bias': np.array([.25], dtype=np.float32)}
        pos, neg = centered_layer(cache, saved, 0, 0)
        np.testing.assert_allclose(pos, [[.8, -.3], [-.2, .7], [-1.2, -.3], [-.2, -1.3]])
        np.testing.assert_allclose(neg, [[.4, .9], [1.4, -.1], [.4, -1.1], [-.6, -.1]])
        probe = consistent_probe(reference_module('ccs').CCS, saved, 0)
        values = probe.get_contrastive_probas(neg[2:], pos[2:], neg[:2], pos[:2])
        # Independent linear logits; no subtraction of means of the four groups.
        logits = [[2.15, -.85], [-1.85, 1.15], [.15, 3.15], [2.15, -.85]]
        for actual, expected in zip(values, logits):
            np.testing.assert_allclose(actual.ravel(), 1/(1+np.exp(-np.array(expected))), atol=1e-7)
        # Mutating an evaluation object cannot change other objects' centering.
        cache['X_pos'][3] = [9., 2.]
        changed, _ = centered_layer(cache, saved, 0, 0)
        np.testing.assert_array_equal(pos[:3], changed[:3])


class CalibratedMismatchTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({
            'sample_idx': range(6), 'statement': list('abcdef'),
            'true_label': [1, 0, 1, 1, 0, 0],
            'split': ['train', 'train', 'test', 'test', 'test', 'test'],
            'layer': [2]*6, 'latent_score': [.9, .1, .9, .1, .9, .5],
            'behavioral_probability_yes': [.1, .9, .8, .9, .7, .8],
        })

    def test_test_only_and_label_aware_threshold_boundaries(self):
        from analyze_mismatch_calibrated import analyze_scores, summarize
        result = analyze_scores(self.fixture(), 'mixed', 'toy', .8,
                                np.array([0, 1]), np.array([2, 3, 4, 5]))
        self.assertEqual(result.sample_idx.tolist(), [2, 3, 4, 5])
        self.assertEqual(result.behavioral_class.tolist(), ['aligned', 'aligned', 'aligned', 'misaligned'])
        # CCS uses > .5: equality predicts class zero.
        self.assertEqual(result.latent_class.tolist(), ['aligned', 'misaligned', 'misaligned', 'aligned'])
        self.assertEqual(result.mismatch_category.tolist(), [
            'both_aligned', 'aligned_output_misaligned_latent',
            'aligned_output_misaligned_latent', 'misaligned_output_aligned_latent'])
        summary = summarize(result)
        self.assertEqual(summary.n.tolist(), [1, 2, 1, 0])
        self.assertEqual(summary.fraction.tolist(), [.25, .5, .25, 0.])
        self.assertEqual(summary.denominator_n.tolist(), [4, 4, 4, 4])
        self.assertTrue(summary.counting_unit.eq('test sample').all())
        self.assertTrue(summary.behavioral_threshold.eq(.8).all())

    def test_rejects_split_mismatch_and_missing_test_objects(self):
        from analyze_mismatch_calibrated import analyze_scores
        frame = self.fixture()
        frame.loc[2, 'split'] = 'train'
        with self.assertRaises(ValueError):
            analyze_scores(frame, 'mixed', 'toy', .8, np.array([0, 1]), np.array([2, 3, 4, 5]))
        with self.assertRaises(ValueError):
            analyze_scores(self.fixture().iloc[:-1], 'mixed', 'toy', .8,
                           np.array([0, 1]), np.array([2, 3, 4, 5]))


if __name__ == '__main__':
    unittest.main()
