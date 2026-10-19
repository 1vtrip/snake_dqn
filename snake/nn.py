"""A tiny neural network library written from scratch with NumPy.

MLP with ReLU hidden layers and a linear output, trained with backpropagation and Adam.
No PyTorch/TensorFlow: every gradient is computed by hand, see backward().
"""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation: keeps activations at a sane scale with ReLU.
        self.W = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(np.float32)
                  for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n_out, dtype=np.float32) for n_out in sizes[1:]]

    # ---------------------------------------------------------------- forward
    def forward(self, x: np.ndarray, keep_cache: bool = False) -> np.ndarray:
        """x: (batch, in) -> (batch, out). keep_cache stores activations for backward()."""
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            last = i == len(self.W) - 1
            acts.append(z if last else np.maximum(z, 0))  # ReLU on hidden layers
        if keep_cache:
            self._acts = acts
        return acts[-1]

    __call__ = forward

    # --------------------------------------------------------------- backward
    def backward(self, grad_out: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """grad_out = dLoss/dOutput, shape (batch, out). Returns gradients for W and b.
        Must be called right after forward(x, keep_cache=True)."""
        acts = self._acts
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        g = grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g          # dL/dW = input^T · dL/dz
            gb[i] = g.sum(axis=0)          # dL/db = sum over batch
            if i > 0:
                g = g @ self.W[i].T        # propagate to previous layer's output
                g = g * (acts[i] > 0)      # ReLU derivative
        return gW, gb

    # ------------------------------------------------------------ utilities
    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    def copy_from(self, other: "MLP") -> None:
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def save(self, path: str) -> None:
        np.savez(path, *self.W, *self.b, n_layers=len(self.W))

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path)
        n = int(data["n_layers"])
        arrays = [data[f"arr_{i}"] for i in range(2 * n)]
        net = cls.__new__(cls)
        net.W, net.b = arrays[:n], arrays[n:]
        return net


class Adam:
    """Adam optimiser: per-parameter adaptive learning rate using running
    averages of the gradient (m) and of its square (v)."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)  # bias correction for the first steps
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # in-place update
