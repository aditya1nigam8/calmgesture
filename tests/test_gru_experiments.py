"""Behavioral checks for GRU variants and the experiment selection boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gru_experiments import aggregate, run, select_configuration
from temporal_models import create_model, load_checkpoint, predict_probabilities


class GRUExperimentTests(unittest.TestCase):
    def test_pooling_changes_representation_and_preserves_gradients(self):
        last = create_model("gru", 5).eval()
        mean = create_model("gru", 5, {"pooling": "mean"}).eval()
        mean.load_state_dict(last.state_dict())
        sequence = torch.randn(2, 30, 63, requires_grad=True)
        self.assertFalse(torch.allclose(last(sequence), mean(sequence)))
        mean(sequence).sum().backward()
        self.assertGreater(sequence.grad[:, 0].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(sequence.grad).all())

    def test_legacy_and_nondefault_checkpoints_round_trip(self):
        for config in (None, {"pooling": "mean", "hidden_size": 32, "dropout": 0.1}):
            model = create_model("gru", 5, config).eval()
            checkpoint = {"format_version": 1, "architecture": "gru", "classes": list("abcde"),
                          "state_dict": model.state_dict(), "target_frames": 30,
                          "mean": torch.zeros(1, 1, 63), "std": torch.ones(1, 1, 63)}
            if config is not None:
                checkpoint["model_config"] = config
            values = np.random.default_rng(1).normal(size=(2, 30, 63)).astype(np.float32)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save(checkpoint, path)
                restored, metadata = load_checkpoint(path)
                np.testing.assert_array_equal(predict_probabilities(model, checkpoint, values),
                                              predict_probabilities(restored, metadata, values))

    def test_selection_uses_mean_validation_not_test_or_best_seed(self):
        rows = [{"name": "stable", "validation": aggregate([
            {"accuracy": .7, "macro_f1": .7}, {"accuracy": .7, "macro_f1": .7}]), "test": 0},
                {"name": "unstable", "validation": aggregate([
            {"accuracy": .9, "macro_f1": .9}, {"accuracy": .1, "macro_f1": .1}]), "test": 1}]
        self.assertEqual(select_configuration(rows), "stable")
        self.assertIsNone(aggregate([{"accuracy": .5, "macro_f1": .5}])["accuracy"]["sample_std"])

    def test_small_end_to_end_run_freezes_selection_before_test(self):
        rng = np.random.default_rng(5)
        parts = [rng.normal(size=(n, 30, 63)).astype(np.float32) for n in (8, 4, 4)]
        labels = [np.array(["a", "b"] * (len(x) // 2)) for x in parts]
        splits = (*parts, *labels, ["a", "b"])
        provenance = {"dataset": "synthetic pipeline check"}
        configs = [{"name": "mean", "architecture": "gru", "model_config": {"pooling": "mean", "hidden_size": 8}}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experiment"
            from gru_experiments import predict_probabilities as predict

            def guarded_predict(model, checkpoint, values):
                if values is parts[2]:
                    self.assertTrue((output / "selection.json").exists())
                return predict(model, checkpoint, values)

            with patch("gru_experiments.dataset_provenance", return_value=provenance), \
                 patch("train_temporal.dataset_provenance", return_value=provenance), \
                 patch("gru_experiments.load_training_splits", return_value=splits), \
                 patch("train_temporal.load_training_splits", return_value=splits), \
                 patch("train_temporal.predict_probabilities", side_effect=guarded_predict), \
                 patch("gru_experiments.predict_probabilities", side_effect=guarded_predict):
                selection, result = run("unused", output, seeds=[42, 43], epochs=2,
                                        patience=1, configurations=configs, allow_synthetic=True)
            self.assertEqual(selection["selected_configuration"], "mean")
            self.assertEqual(len(result["per_seed"]), 2)
            metrics = json.loads((output / "mean/seed_42/metrics.json").read_text())
            self.assertIsNone(metrics["test"])
            self.assertFalse((output / "mean/seed_42/predictions.npz").exists())
            self.assertTrue((output / "mean/seed_42/selected_test_predictions.npz").exists())


if __name__ == "__main__":
    unittest.main()
