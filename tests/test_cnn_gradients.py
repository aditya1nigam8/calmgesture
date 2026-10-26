"""Finite-difference checks against scalar losses, including batch averaging."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cnn_numpy import Conv2D, Dense, softmax_ce_loss


class GradientTests(unittest.TestCase):
    def test_dense_gradient_matches_mean_cross_entropy(self):
        layer = Dense(3, 2)
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
        x = np.array([[0.2, -0.4, 0.1], [0.6, 0.3, -0.2], [-0.2, 0.8, 0.5]])
        y = np.array([0, 1, 1])
        eps = 1e-5
        old = layer.W[1, 0]
        layer.W[1, 0] = old + eps
        plus = softmax_ce_loss(layer.forward(x), y)[0]
        layer.W[1, 0] = old - eps
        minus = softmax_ce_loss(layer.forward(x), y)[0]
        layer.W[1, 0] = old
        _, gradient, _ = softmax_ce_loss(layer.forward(x), y)
        layer.backward(gradient, lr=1)
        self.assertAlmostEqual(old - layer.W[1, 0], (plus - minus) / (2 * eps), places=6)

    def test_convolution_gradient_matches_mean_squared_loss(self):
        layer = Conv2D(1, 2)
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
        x = np.random.default_rng(42).normal(size=(3, 1, 3, 3))
        eps = 1e-5
        old = layer.W[0, 0, 1, 1]
        layer.W[0, 0, 1, 1] = old + eps
        plus = (layer.forward(x) ** 2).mean()
        layer.W[0, 0, 1, 1] = old - eps
        minus = (layer.forward(x) ** 2).mean()
        layer.W[0, 0, 1, 1] = old
        values = layer.forward(x)
        layer.backward(2 * values / values.size, lr=1)
        self.assertAlmostEqual(old - layer.W[0, 0, 1, 1], (plus - minus) / (2 * eps), places=6)


if __name__ == "__main__":
    unittest.main()
