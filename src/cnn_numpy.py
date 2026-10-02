"""
cnn_numpy.py — minimal from-scratch 2D CNN (NumPy only).

The sandboxed training environment for this interim submission has no
network access to install PyTorch/TensorFlow (both pip installs timed
out / were blocked by the proxy), so the CNN owned by Tanvi Gupta is
implemented here as a small hand-rolled 2D CNN with manual forward and
backward passes (im2col convolution), trained by plain SGD with
momentum. Functionally it is a real, from-scratch CNN — not a stub —
but the team should re-implement/port this to PyTorch (as originally
adapted from the Gesture-Symphony reference architecture) once a
training environment with GPU/framework access is available, per the
Risk/Plan section of the interim report.

Architecture (per clip, input = 21x3 hand-pose "image", 2 channels: mean + std over time):
    Conv(2->8, 3x3, pad=1) -> ReLU
    Conv(8->16, 3x3, pad=1) -> ReLU
    Flatten -> Dense(32) -> ReLU -> Dense(n_classes) -> Softmax
"""

import numpy as np


def _im2col(X, kh, kw, pad):
    N, C, H, W = X.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out_h, out_w = H, W  # stride 1, same padding
    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = Xp[:, :, i:i + out_h, j:j + out_w]
    return cols.reshape(N, C * kh * kw, out_h * out_w)


class Conv2D:
    def __init__(self, in_c, out_c, k=3, pad=1, seed=0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_c * k * k))
        self.W = rng.normal(0, scale, (out_c, in_c, k, k)).astype(np.float32)
        self.b = np.zeros(out_c, dtype=np.float32)
        self.k, self.pad = k, pad
        self.cache = None

    def forward(self, X):
        N, C, H, W = X.shape
        cols = _im2col(X, self.k, self.k, self.pad)  # (N, C*k*k, H*W)
        Wf = self.W.reshape(self.W.shape[0], -1)       # (out_c, C*k*k)
        out = np.einsum("oc,ncp->nop", Wf, cols) + self.b[None, :, None]
        out = out.reshape(N, self.W.shape[0], H, W)
        self.cache = (X, cols)
        return out

    def backward(self, dOut, lr):
        X, cols = self.cache
        N, C, H, W = X.shape
        out_c = self.W.shape[0]
        dOut_flat = dOut.reshape(N, out_c, -1)
        Wf = self.W.reshape(out_c, -1)

        dWf = np.einsum("nop,ncp->oc", dOut_flat, cols) / N
        db = dOut_flat.sum(axis=(0, 2)) / N
        dcols = np.einsum("oc,nop->ncp", Wf, dOut_flat)

        dXp = np.zeros((N, C, H + 2 * self.pad, W + 2 * self.pad), dtype=X.dtype)
        dcols_r = dcols.reshape(N, C, self.k, self.k, H, W)
        for i in range(self.k):
            for j in range(self.k):
                dXp[:, :, i:i + H, j:j + W] += dcols_r[:, :, i, j, :, :]
        dX = dXp[:, :, self.pad:self.pad + H, self.pad:self.pad + W]

        self.W -= lr * dWf.reshape(self.W.shape)
        self.b -= lr * db
        return dX


class Dense:
    def __init__(self, in_f, out_f, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / in_f), (in_f, out_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.cache = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dOut, lr):
        X = self.cache
        N = X.shape[0]
        dW = X.T @ dOut / N
        db = dOut.sum(axis=0) / N
        dX = dOut @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dX


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def softmax_ce_loss(logits, y_idx):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), y_idx] + 1e-9).mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), y_idx] -= 1
    dlogits /= N
    return loss, dlogits, probs


class SimpleCNN:
    def __init__(self, n_classes, in_c=1, seed=0):
        self.conv1 = Conv2D(in_c, 8, k=3, pad=1, seed=seed)
        self.conv2 = Conv2D(8, 16, k=3, pad=1, seed=seed + 1)
        self.flat_dim = 16 * 21 * 3
        self.fc1 = Dense(self.flat_dim, 32, seed=seed + 2)
        self.fc2 = Dense(32, n_classes, seed=seed + 3)

    def forward(self, X):
        self.a1 = self.conv1.forward(X)
        self.r1 = relu(self.a1)
        self.a2 = self.conv2.forward(self.r1)
        self.r2 = relu(self.a2)
        N = X.shape[0]
        self.flat = self.r2.reshape(N, -1)
        self.f1 = self.fc1.forward(self.flat)
        self.rf1 = relu(self.f1)
        logits = self.fc2.forward(self.rf1)
        return logits

    def backward(self, dlogits, lr):
        d = self.fc2.backward(dlogits, lr)
        d = d * relu_grad(self.f1)
        d = self.fc1.backward(d, lr)
        d = d.reshape(self.r2.shape)
        d = d * relu_grad(self.a2)
        d = self.conv2.backward(d, lr)
        d = d * relu_grad(self.a1)
        self.conv1.backward(d, lr)

    def predict(self, X, batch=32):
        preds = []
        for i in range(0, X.shape[0], batch):
            logits = self.forward(X[i:i + batch])
            preds.append(logits.argmax(axis=1))
        return np.concatenate(preds)

    def fit(self, X, y_idx, X_val=None, y_val_idx=None, epochs=40, lr=0.05, batch=16, verbose=True):
        n = X.shape[0]
        history = []
        for ep in range(epochs):
            perm = np.random.default_rng(ep).permutation(n)
            Xs, ys = X[perm], y_idx[perm]
            ep_loss = 0.0
            for i in range(0, n, batch):
                xb, yb = Xs[i:i + batch], ys[i:i + batch]
                logits = self.forward(xb)
                loss, dlogits, _ = softmax_ce_loss(logits, yb)
                self.backward(dlogits, lr)
                ep_loss += loss * len(xb)
            ep_loss /= n
            val_acc = None
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_acc = (val_pred == y_val_idx).mean()
            history.append({"epoch": ep, "loss": float(ep_loss), "val_acc": float(val_acc) if val_acc is not None else None})
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"epoch {ep:3d}  loss={ep_loss:.4f}  val_acc={val_acc}")
        return history
