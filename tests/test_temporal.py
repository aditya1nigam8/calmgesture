"""Verify sequence information, learning gradients, and inference persistence."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from temporal_models import create_model, fit_normalizer, load_checkpoint, predict_probabilities


class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_both_models_propagate_gradients_through_sequences(self):
        for architecture in ["1dcnn", "gru"]:
            with self.subTest(architecture=architecture):
                model = create_model(architecture, 5)
                x = torch.randn(4, 30, 63, requires_grad=True)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (4, 5))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_models_do_not_reduce_sequences_to_mean_and_std(self):
        sequence = torch.randn(1, 30, 63)
        reversed_sequence = sequence.flip(1)
        torch.testing.assert_close(sequence.mean(1), reversed_sequence.mean(1))
        for architecture in ["1dcnn", "gru"]:
            model = create_model(architecture, 5).eval()
            with torch.inference_mode():
                self.assertFalse(torch.allclose(model(sequence), model(reversed_sequence), atol=1e-6))

    def test_normalizer_uses_training_only_and_handles_constant_coordinates(self):
        train = np.arange(4 * 30 * 63, dtype=np.float32).reshape(4, 30, 63)
        train[:, :, 0] = 7
        mean, std = fit_normalizer(train)
        held_out = train + 100000
        self.assertGreater(((held_out - mean) / std)[:, :, 1:].min(), 10)
        self.assertTrue(np.isfinite((train - mean) / std).all())
        np.testing.assert_array_equal(((train - mean) / std)[:, :, 0], 0)

    def test_saved_model_preserves_probabilities_and_rejects_wrong_input(self):
        values = np.random.default_rng(42).normal(size=(3, 30, 63)).astype(np.float32)
        mean, std = fit_normalizer(values)
        for architecture in ["1dcnn", "gru"]:
            model = create_model(architecture, 5).eval()
            checkpoint = {"format_version": 1, "architecture": architecture,
                          "state_dict": model.state_dict(), "classes": list("abcde"),
                          "target_frames": 30, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
            expected = predict_probabilities(model, checkpoint, values)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save(checkpoint, path)
                restored, info = load_checkpoint(path)
                actual = predict_probabilities(restored, info, values)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_allclose(actual.sum(1), 1, atol=1e-6)
                with self.assertRaises(ValueError):
                    predict_probabilities(restored, info, values[:, :10])
                with self.assertRaises(ValueError):
                    predict_probabilities(restored, info, values * np.nan)


if __name__ == "__main__":
    unittest.main()
