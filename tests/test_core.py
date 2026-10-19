"""Sanity tests. Run: python -m tests.test_core"""

import numpy as np

from snake.env import SnakeEnv
from snake.nn import MLP, Adam


def test_gradient_check():
    """Compare hand-written backprop with numerical gradients (finite differences).
    This is the standard way to prove backward() is correct."""
    rng = np.random.default_rng(1)
    net = MLP([5, 7, 3], seed=1)
    net.W = [w.astype(np.float64) for w in net.W]
    net.b = [b.astype(np.float64) for b in net.b]
    x = rng.normal(size=(4, 5))
    y = rng.normal(size=(4, 3))
    loss = lambda: 0.5 * ((net(x) - y) ** 2).sum()

    out = net.forward(x, keep_cache=True)
    gW, gb = net.backward(out - y)

    eps, worst = 1e-6, 0.0
    for p, g in zip(net.W + net.b, gW + gb):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps; lp = loss()
            p[i] = old - eps; lm = loss()
            p[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[i]) / max(1e-8, abs(num) + abs(g[i])))
    assert worst < 1e-5, f"relative gradient error too large: {worst}"
    print(f"gradient check OK (max relative error {worst:.2e})")


def test_mlp_learns_xor():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
    y = np.array([[0], [1], [1], [0]], np.float32)
    net = MLP([2, 16, 1], seed=0)
    opt = Adam(net.params(), lr=0.02)
    for _ in range(2000):
        out = net.forward(x, keep_cache=True)
        gW, gb = net.backward((out - y) / len(x))
        opt.step(net.params(), gW + gb)
    pred = net(x).round().ravel()
    assert (pred == y.ravel()).all(), pred
    print("MLP learns XOR OK")


def test_env_rules():
    env = SnakeEnv(6, 6, seed=0)
    s = env.reset()
    assert s.shape == (11,)
    done, steps = False, 0
    while not done:  # going straight must hit the right wall
        s, r, done, info = env.step(0)
        steps += 1
    assert r == -10.0 and steps <= 6
    print("env rules OK")


if __name__ == "__main__":
    test_gradient_check()
    test_mlp_learns_xor()
    test_env_rules()
