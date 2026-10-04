"""ESA must use the same four centered groups as reference PC/CI."""
import unittest

import numpy as np

from artifacts import reference_module
from probes import restore_probe
from pa_ccs_reference import esa_from_probabilities


class ReferenceEsaTests(unittest.TestCase):
    def test_four_group_centers_and_one_global_orientation(self):
        probe = restore_probe(reference_module('ccs').CCS,
                              {'weight': np.array([[1.]]), 'bias': np.array([0.])}, 0)
        groups = [np.array(x, dtype=np.float32).reshape(-1, 1) for x in
                  ([10, 11, 19], [20, 20, 20], [30, 31, 39], [40, 40, 40])]
        probabilities = probe.get_contrastive_probas(*groups)
        # Each half predicts [0, 0, 1]; labels differ between halves.
        # Flipping each half separately would incorrectly yield 2/3.
        labels = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(esa_from_probabilities(probabilities, labels), .5)
        # A shifted group must have exactly the same reference predictions.
        shifted = probe.get_contrastive_probas(*(g + 100 for g in groups))
        self.assertEqual(esa_from_probabilities(shifted, labels), .5)

    def test_ccs_probability_combination_threshold_and_sign_invariance(self):
        probabilities = tuple(np.array(p).reshape(-1, 1) for p in
                              ([.9, .5], [.1, .5], [.1, .2], [.9, .8]))
        # CCS predictions: [1, 0, 0, 0], equality at .5 predicts zero.
        self.assertEqual(esa_from_probabilities(probabilities, np.array([1, 0, 0, 0])), 1.)
        self.assertEqual(esa_from_probabilities(probabilities, np.array([0, 1, 1, 1])), 1.)
