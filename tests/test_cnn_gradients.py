"""
test_cnn_gradients.py — finite-difference checks for the NumPy CNN (Tanvi Gupta).

Compares the analytic gradients from SimpleCNN.backward with central
differences of the batch-mean cross-entropy loss, for every parameter
tensor. A wrong 1/N factor (the earlier double batch-mean scaling bug)
shows up as a ratio of exactly N and fails these tests.

Run:  python -m pytest tests   (or)   python tests/test_cnn_gradients.py
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cnn_numpy import SimpleCNN, softmax_ce_loss  # noqa: E402


def _make_model_and_batch(in_c=2, n=4, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_c, 21, 3))
    y = rng.integers(0, 5, n)
    model = SimpleCNN(5, in_c=in_c, seed=0)
    # float64 so finite differences are accurate
    model.set_params({k: v.astype(np.float64) for k, v in model.get_params().items()})
    return model, X, y


def _check_all_params(model, X, y, n_samples=6, eps=1e-6, tol=1e-4):
    _, dlogits, _ = softmax_ce_loss(model.forward(X), y)
    model.backward(dlogits)
    rng = np.random.default_rng(0)
    for name, layer in model.layers.items():
        for p in ("W", "b"):
            param, grad = getattr(layer, p), getattr(layer, "d" + p)
            assert grad.shape == param.shape, f"{name}.{p} grad shape"
            # check the largest-gradient entries (non-dead units) plus random ones
            flat_idx = list(np.argsort(-np.abs(grad).ravel())[:n_samples // 2])
            flat_idx += list(rng.integers(0, param.size, n_samples // 2))
            for fi in flat_idx:
                idx = np.unravel_index(fi, param.shape)
                orig = param[idx]
                param[idx] = orig + eps
                lp = softmax_ce_loss(model.forward(X), y)[0]
                param[idx] = orig - eps
                lm = softmax_ce_loss(model.forward(X), y)[0]
                param[idx] = orig
                num = (lp - lm) / (2 * eps)
                err = abs(num - grad[idx]) / max(1e-8, abs(num) + abs(grad[idx]))
                assert err < tol or abs(num - grad[idx]) < 1e-9, \
                    f"{name}.{p}{idx}: numeric={num:.6e} analytic={grad[idx]:.6e}"


def test_gradients_two_channels():
    _check_all_params(*_make_model_and_batch(in_c=2, n=4))


def test_gradients_one_channel():
    _check_all_params(*_make_model_and_batch(in_c=1, n=3))


def test_gradient_is_batch_mean_not_divided_twice():
    """Duplicating a batch must leave the (mean) gradient unchanged."""
    model, X, y = _make_model_and_batch(n=4)
    _, d, _ = softmax_ce_loss(model.forward(X), y)
    model.backward(d)
    g1 = model.fc2.dW.copy()
    X2, y2 = np.concatenate([X, X]), np.concatenate([y, y])
    _, d, _ = softmax_ce_loss(model.forward(X2), y2)
    model.backward(d)
    np.testing.assert_allclose(model.fc2.dW, g1, rtol=1e-10, atol=1e-12)


def test_save_load_roundtrip():
    model, X, _ = _make_model_and_batch()
    mu, sd = np.ones((1, 2, 1, 1)), np.full((1, 2, 1, 1), 2.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cnn.npz")
        model.save(path, mu=mu, sd=sd, classes=np.array(["a", "b", "c", "d", "e"]))
        loaded, extra = SimpleCNN.load(path)
    np.testing.assert_allclose(loaded.forward(X), model.forward(X))
    np.testing.assert_array_equal(extra["mu"], mu)
    assert list(extra["classes"]) == ["a", "b", "c", "d", "e"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
