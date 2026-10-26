"""Regression checks for fair evaluation of the baseline models."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_cnn import standardize_splits
from utils import split_dataset


class TrainingTests(unittest.TestCase):
    def test_held_out_values_do_not_affect_training_normalization(self):
        train = np.arange(48, dtype=np.float32).reshape(4, 2, 3, 2)
        validation = np.full((2, 2, 3, 2), 1000, dtype=np.float32)
        test = -validation
        normalized, val, _ = standardize_splits(train, validation, test)
        changed, _, _ = standardize_splits(train, validation * 10, test * 10)
        np.testing.assert_array_equal(normalized, changed)
        np.testing.assert_allclose(normalized.mean(axis=(0, 2, 3)), 0, atol=1e-6)
        self.assertGreater(float(val.min()), 10)

    def test_models_share_disjoint_reproducible_clip_splits(self):
        clips = np.arange(200)
        labels = np.repeat(["a", "b", "c", "d", "e"], 40)
        encoded = np.repeat(np.arange(5), 40)
        first = split_dataset(clips, labels)
        second = split_dataset(clips, encoded)
        for left, right in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(left, right)
        train, val, test = map(set, first[:3])
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(train | val | test, set(clips))


if __name__ == "__main__":
    unittest.main()
