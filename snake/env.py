"""Snake environment with a Gymnasium-like API: reset() -> state, step(action) -> (state, reward, done, info).

Actions are RELATIVE to the current heading, which makes the problem easier to learn:
    0 = go straight, 1 = turn right, 2 = turn left

State is a vector of 11 binary features (a compact, hand-crafted observation):
    [danger straight, danger right, danger left,
     moving left, moving right, moving up, moving down,
     food left, food right, food up, food down]
"""

from __future__ import annotations

import numpy as np

# Directions in clockwise order: right, down, left, up  (x grows right, y grows down)
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_CLOSER = 0.1   # small shaping reward: moved towards the food
REWARD_FARTHER = -0.15  # moved away from the food


class SnakeEnv:
    n_actions = 3
    state_size = 11

    def __init__(self, width: int = 12, height: int = 12, seed: int | None = None):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> np.ndarray:
        cx, cy = self.w // 2, self.h // 2
        self.dir = 0  # moving right
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]  # head first
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._place_food()
        return self._state()

    def step(self, action: int):
        if action == 1:
            self.dir = (self.dir + 1) % 4
        elif action == 2:
            self.dir = (self.dir - 1) % 4

        dx, dy = DIRS[self.dir]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)
        self.steps += 1
        self.steps_since_food += 1

        # Starvation limit stops the agent from looping forever.
        starving = self.steps_since_food > 100 * len(self.snake)
        if self._collides(new_head) or starving:
            return self._state(), REWARD_DEATH, True, {"score": self.score}

        old_dist = self._dist(head)
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = REWARD_FOOD
            if len(self.snake) == self.w * self.h:  # board filled: perfect game
                return self._state(), reward, True, {"score": self.score}
            self._place_food()
        else:
            self.snake.pop()
            reward = REWARD_CLOSER if self._dist(new_head) < old_dist else REWARD_FARTHER

        return self._state(), reward, False, {"score": self.score}

    # -------------------------------------------------------------- helpers
    def _place_food(self):
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in occupied]
        self.food = free[self.rng.integers(len(free))]

    def _dist(self, p) -> int:
        return abs(p[0] - self.food[0]) + abs(p[1] - self.food[1])

    def _collides(self, p) -> bool:
        x, y = p
        if not (0 <= x < self.w and 0 <= y < self.h):
            return True
        # The tail moves away this step, so it is not an obstacle.
        return p in self.snake[:-1]

    def _danger(self, direction: int) -> bool:
        dx, dy = DIRS[direction]
        hx, hy = self.snake[0]
        return self._collides((hx + dx, hy + dy))

    def _state(self) -> np.ndarray:
        hx, hy = self.snake[0]
        fx, fy = self.food
        d = self.dir
        return np.array([
            self._danger(d),            # straight
            self._danger((d + 1) % 4),  # right
            self._danger((d - 1) % 4),  # left
            d == 2, d == 0, d == 3, d == 1,  # moving left, right, up, down
            fx < hx, fx > hx, fy < hy, fy > hy,  # food left, right, up, down
        ], dtype=np.float32)
