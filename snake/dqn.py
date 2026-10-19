"""Deep Q-Network (DQN) agent, implemented from scratch.

Q(s, a) = expected discounted future reward of taking action a in state s.
The network learns it from the Bellman equation:
    target = r + gamma * max_a' Q_target(s', a')      (just r if the episode ended)

Key DQN ingredients implemented here:
  * experience replay  - learn from random past transitions, not only the latest one
                         (breaks correlation between consecutive samples)
  * target network     - a slowly-updated copy used for targets (stabilises training)
  * epsilon-greedy     - explore randomly at first, exploit more as training goes on
  * Double DQN (opt.)  - online net picks a', target net evaluates it (reduces overestimation)
  * Huber loss         - like MSE near zero, linear for big errors (robust to outliers)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nn import MLP, Adam


class ReplayBuffer:
    """Fixed-size circular buffer of transitions (s, a, r, s', done)."""

    def __init__(self, capacity: int, state_size: int, seed: int = 0):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.s = np.zeros((capacity, state_size), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, state_size), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.9            # how much future rewards matter
    batch_size: int = 64
    buffer_size: int = 50_000
    learn_start: int = 1_000      # collect this many steps before learning
    target_sync: int = 500        # copy online -> target net every N learn steps
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 30_000
    double_dqn: bool = True
    huber_delta: float = 1.0
    seed: int = 0


class DQNAgent:
    def __init__(self, state_size: int, n_actions: int, cfg: DQNConfig = DQNConfig()):
        self.cfg, self.n_actions = cfg, n_actions
        sizes = [state_size, *cfg.hidden, n_actions]
        self.q = MLP(sizes, seed=cfg.seed)
        self.q_target = MLP(sizes, seed=cfg.seed)
        self.q_target.copy_from(self.q)
        self.opt = Adam(self.q.params(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, state_size, seed=cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.total_steps = 0
        self.learn_steps = 0

    @property
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.total_steps / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)  # linear decay

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q(state[None])[0]))

    def remember(self, s, a, r, s2, done):
        self.buffer.add(s, a, r, s2, done)
        self.total_steps += 1

    def learn(self) -> float | None:
        c = self.cfg
        if self.buffer.size < max(c.learn_start, c.batch_size):
            return None
        s, a, r, s2, done = self.buffer.sample(c.batch_size)

        # --- Bellman targets (no gradient flows through these)
        q_next_target = self.q_target(s2)
        if c.double_dqn:
            best_next = np.argmax(self.q(s2), axis=1)
            q_next = q_next_target[np.arange(len(a)), best_next]
        else:
            q_next = q_next_target.max(axis=1)
        target = r + c.gamma * q_next * (1.0 - done)

        # --- forward pass on the online network, only the taken action matters
        q_all = self.q.forward(s, keep_cache=True)
        q_taken = q_all[np.arange(len(a)), a]
        err = q_taken - target

        # --- Huber loss and its gradient w.r.t. q_taken
        abs_err = np.abs(err)
        quadratic = abs_err <= c.huber_delta
        loss = np.where(quadratic, 0.5 * err ** 2, c.huber_delta * (abs_err - 0.5 * c.huber_delta)).mean()
        d_q_taken = np.where(quadratic, err, c.huber_delta * np.sign(err)) / len(a)

        grad_out = np.zeros_like(q_all)
        grad_out[np.arange(len(a)), a] = d_q_taken  # other actions get zero gradient

        gW, gb = self.q.backward(grad_out)
        grads = gW + gb
        # Gradient clipping by global norm keeps rare huge updates from breaking training.
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        if norm > 10.0:
            grads = [g * (10.0 / norm) for g in grads]
        self.opt.step(self.q.params(), grads)

        self.learn_steps += 1
        if self.learn_steps % c.target_sync == 0:
            self.q_target.copy_from(self.q)
        return float(loss)
