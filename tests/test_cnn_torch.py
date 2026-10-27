"""Check the PyTorch spatial CNN port against the NumPy reference implementation."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_inference import pose_image, predict_probabilities as numpy_probabilities
from cnn_numpy import SimpleCNN
from spatial_cnn_torch import (SpatialCNN, checkpoint_from_numpy, load_checkpoint, load_numpy_weights,
                               pose_channels, predict_probabilities)
from train_cnn_torch import fit


class SpatialCNNPortTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.rng = np.random.default_rng(42)

    def test_port_reproduces_numpy_logits(self):
        reference = SimpleCNN(5, in_c=2, seed=3)
        port = load_numpy_weights(SpatialCNN(5), reference.state_dict()).eval()
        images = self.rng.normal(size=(6, 2, 21, 3)).astype(np.float32)
        with torch.inference_mode():
            actual = port(torch.from_numpy(images)).numpy()
        np.testing.assert_allclose(actual, reference.forward(images), atol=1e-5)

    def test_port_training_tracks_numpy_sgd_and_early_stopping(self):
        Xtr = self.rng.normal(size=(40, 2, 21, 3)).astype(np.float32)
        Xva = self.rng.normal(size=(12, 2, 21, 3)).astype(np.float32)
        ytr, yva = self.rng.integers(0, 5, 40), self.rng.integers(0, 5, 12)
        reference = SimpleCNN(5, in_c=2, seed=0)
        port = load_numpy_weights(SpatialCNN(5), reference.state_dict())
        reference.fit(Xtr, ytr, Xva, yva, epochs=4, lr=0.05, batch=16, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            _, best_epoch = fit(port, Xtr, ytr, Xva, yva, epochs=4, batch_size=16, learning_rate=0.05)
        self.assertEqual(best_epoch, reference.best_epoch)
        expected = load_numpy_weights(SpatialCNN(5), reference.state_dict()).state_dict()
        for name, value in port.state_dict().items():
            torch.testing.assert_close(value, expected[name], atol=1e-4, rtol=1e-4, msg=name)

    def test_mean_only_ablation_ignores_movement_amplitude(self):
        still = np.tile(self.rng.normal(size=(1, 1, 63)), (1, 30, 1)).astype(np.float32)
        moving = still.copy()
        moving[:, ::2] += 0.5
        moving[:, 1::2] -= 0.5  # same temporal mean, larger temporal std
        both = np.concatenate([still, moving])
        np.testing.assert_array_equal(pose_channels(both, "mean"), pose_image(both)[:, :1])
        for channels, in_channels, should_differ in [("mean", 1, False), ("mean_std", 2, True)]:
            model = SpatialCNN(5, in_channels).eval()
            with torch.inference_mode():
                logits = model(torch.from_numpy(pose_channels(both, channels)))
            self.assertEqual(not torch.allclose(logits[0], logits[1], atol=1e-5), should_differ, channels)

    def test_converted_numpy_checkpoint_preserves_probabilities(self):
        sequences = self.rng.normal(size=(5, 30, 63)).astype(np.float32)
        images = pose_image(sequences)
        reference = SimpleCNN(5, in_c=2, seed=7)
        bundle = {"format_version": 1, "architecture": "cnn", "weights": reference.state_dict(),
                  "classes": list("abcde"), "target_frames": 30, "model": reference,
                  "mean": images.mean(axis=(0, 2, 3), keepdims=True),
                  "std": images.std(axis=(0, 2, 3), keepdims=True) + 1e-6,
                  "data": {"dataset": "synthetic pipeline check"}}
        expected = numpy_probabilities(bundle, sequences)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(checkpoint_from_numpy(bundle), path)
            model, checkpoint = load_checkpoint(path)
            actual = predict_probabilities(model, checkpoint, sequences)
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            np.testing.assert_allclose(actual.sum(1), 1, atol=1e-6)
            with self.assertRaises(ValueError):
                predict_probabilities(model, checkpoint, sequences[:, :10])
            with self.assertRaises(ValueError):
                predict_probabilities(model, checkpoint, sequences * np.nan)


if __name__ == "__main__":
    unittest.main()
