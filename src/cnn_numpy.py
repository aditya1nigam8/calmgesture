"""
cnn_numpy.py — minimal from-scratch 2D CNN (NumPy only).

The sandboxed training environment for this interim submission has no
network access to install PyTorch/TensorFlow (both pip installs timed
out / were blocked by the proxy), so the CNN owned by Tanvi Gupta is
implemented here as a small hand-rolled 2D CNN with manual forward and
backward passes (im2col convolution), trained by plain mini-batch SGD.
Functionally it is a real, from-scratch CNN — not a stub — but the team
should re-implement/port this to PyTorch (as originally adapted from the
Gesture-Symphony reference architecture) once a training environment
with GPU/framework access is available, per the Risk/Plan section of the
interim report. That port has NOT been done; results come from this file.

Architecture (per clip, input = 21x3 hand-pose "image", C channels,
by default 2: mean + std over time):
    Conv(C->8, 3x3, pad=1) -> ReLU
    Conv(8->16, 3x3, pad=1) -> ReLU
    Flatten -> Dense(32) -> ReLU -> Dense(n_classes) -> Softmax

Gradient scaling: softmax_ce_loss returns dlogits already divided by the
batch size N (the loss is a batch mean). Layer backward passes therefore
must NOT divide by N again. An earlier version did, which shrank every
update by a factor of N (16x at batch 16). tests/test_cnn_gradients.py
checks this against finite differences.
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
        self.dW = self.db = None

    def forward(self, X):
        N, C, H, W = X.shape
        cols = _im2col(X, self.k, self.k, self.pad)  # (N, C*k*k, H*W)
        Wf = self.W.reshape(self.W.shape[0], -1)       # (out_c, C*k*k)
        out = np.einsum("oc,ncp->nop", Wf, cols) + self.b[None, :, None]
        out = out.reshape(N, self.W.shape[0], H, W)
        self.cache = (X, cols)
        return out

    def backward(self, dOut):
        """Store dL/dW, dL/db and return dL/dX. dOut already carries the 1/N batch-mean factor."""
        X, cols = self.cache
        N, C, H, W = X.shape
        out_c = self.W.shape[0]
        dOut_flat = dOut.reshape(N, out_c, -1)
        Wf = self.W.reshape(out_c, -1)

        self.dW = np.einsum("nop,ncp->oc", dOut_flat, cols).reshape(self.W.shape)
        self.db = dOut_flat.sum(axis=(0, 2))
        dcols = np.einsum("oc,nop->ncp", Wf, dOut_flat)

        dXp = np.zeros((N, C, H + 2 * self.pad, W + 2 * self.pad), dtype=X.dtype)
        dcols_r = dcols.reshape(N, C, self.k, self.k, H, W)
        for i in range(self.k):
            for j in range(self.k):
                dXp[:, :, i:i + H, j:j + W] += dcols_r[:, :, i, j, :, :]
        return dXp[:, :, self.pad:self.pad + H, self.pad:self.pad + W]

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Dense:
    def __init__(self, in_f, out_f, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / in_f), (in_f, out_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.cache = None
        self.dW = self.db = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dOut):
        """Store dL/dW, dL/db and return dL/dX. dOut already carries the 1/N batch-mean factor."""
        X = self.cache
        self.dW = X.T @ dOut
        self.db = dOut.sum(axis=0)
        return dOut @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


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
        self.n_classes, self.in_c, self.seed = n_classes, in_c, seed
        self.conv1 = Conv2D(in_c, 8, k=3, pad=1, seed=seed)
        self.conv2 = Conv2D(8, 16, k=3, pad=1, seed=seed + 1)
        self.flat_dim = 16 * 21 * 3
        self.fc1 = Dense(self.flat_dim, 32, seed=seed + 2)
        self.fc2 = Dense(32, n_classes, seed=seed + 3)
        self.layers = {"conv1": self.conv1, "conv2": self.conv2, "fc1": self.fc1, "fc2": self.fc2}

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

    def backward(self, dlogits):
        """Compute and store gradients for every layer (no parameter update)."""
        d = self.fc2.backward(dlogits)
        d = d * relu_grad(self.f1)
        d = self.fc1.backward(d)
        d = d.reshape(self.r2.shape)
        d = d * relu_grad(self.a2)
        d = self.conv2.backward(d)
        d = d * relu_grad(self.a1)
        self.conv1.backward(d)

    def step(self, lr):
        for layer in self.layers.values():
            layer.step(lr)

    def get_params(self):
        return {f"{name}.{p}": getattr(layer, p).copy()
                for name, layer in self.layers.items() for p in ("W", "b")}

    def set_params(self, params):
        for key, value in params.items():
            name, p = key.split(".")
            setattr(self.layers[name], p, value.copy())

    def predict_proba(self, X, batch=32):
        probs = []
        for i in range(0, X.shape[0], batch):
            logits = self.forward(X[i:i + batch])
            logits = logits - logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            probs.append(e / e.sum(axis=1, keepdims=True))
        return np.concatenate(probs)

    def predict(self, X, batch=32):
        return self.predict_proba(X, batch).argmax(axis=1)

    def evaluate_loss(self, X, y_idx, batch=32):
        probs = self.predict_proba(X, batch)
        return float(-np.log(probs[np.arange(len(y_idx)), y_idx] + 1e-9).mean())

    def fit(self, X, y_idx, X_val=None, y_val_idx=None, epochs=40, lr=0.05, batch=16,
            patience=None, verbose=True):
        """Mini-batch SGD. If patience is set (and validation data given), stop once
        validation loss has not improved for `patience` epochs and restore the
        weights from the minimum-validation-loss epoch (stored as self.best_epoch)."""
        n = X.shape[0]
        history = []
        best_loss, best_params, since_best = np.inf, None, 0
        self.best_epoch = None
        for ep in range(epochs):
            perm = np.random.default_rng(ep).permutation(n)
            Xs, ys = X[perm], y_idx[perm]
            ep_loss = 0.0
            for i in range(0, n, batch):
                xb, yb = Xs[i:i + batch], ys[i:i + batch]
                logits = self.forward(xb)
                loss, dlogits, _ = softmax_ce_loss(logits, yb)
                self.backward(dlogits)
                self.step(lr)
                ep_loss += loss * len(xb)
            ep_loss /= n
            val_acc = val_loss = None
            if X_val is not None:
                val_loss = self.evaluate_loss(X_val, y_val_idx)
                val_acc = float((self.predict(X_val) == y_val_idx).mean())
                if val_loss < best_loss:
                    best_loss, best_params, since_best = val_loss, self.get_params(), 0
                    self.best_epoch = ep
                else:
                    since_best += 1
            history.append({"epoch": ep, "loss": float(ep_loss), "val_loss": val_loss, "val_acc": val_acc})
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"epoch {ep:3d}  loss={ep_loss:.4f}  val_loss={val_loss}  val_acc={val_acc}")
            if patience is not None and best_params is not None and since_best >= patience:
                if verbose:
                    print(f"early stop at epoch {ep}; restoring epoch {self.best_epoch}")
                break
        if patience is not None and best_params is not None:
            self.set_params(best_params)
        return history

    def save(self, path, **extra):
        """Save weights plus any preprocessing state (e.g. mu, sd, classes) to one .npz."""
        meta = {"n_classes": self.n_classes, "in_c": self.in_c, "seed": self.seed}
        np.savez(path, **self.get_params(), **{f"meta.{k}": v for k, v in meta.items()},
                 **{f"extra.{k}": np.asarray(v) for k, v in extra.items()})

    @classmethod
    def load(cls, path):
        """Return (model, extra_dict) from a file written by save()."""
        z = np.load(path, allow_pickle=False)
        model = cls(int(z["meta.n_classes"]), in_c=int(z["meta.in_c"]), seed=int(z["meta.seed"]))
        model.set_params({k: z[k] for k in z.files if not k.startswith(("meta.", "extra."))})
        extra = {k[len("extra."):]: z[k] for k in z.files if k.startswith("extra.")}
        return model, extra
