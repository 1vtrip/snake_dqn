"""Evaluate a trained agent against baselines, plot learning curves, record a GIF.

    python evaluate.py                       # uses runs/dqn/best_model.npz
    python evaluate.py --games 200 --gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from snake.env import DIRS, SnakeEnv
from snake.nn import MLP

ROOT = Path(__file__).parent


# ------------------------------------------------------------------ policies
def random_policy(env, s, rng):
    return int(rng.integers(3))


def greedy_heuristic(env, s, rng):
    """Baseline without learning: avoid immediate danger, otherwise move towards food."""
    safe = [a for a, danger in zip((0, 1, 2), s[:3]) if not danger] or [0]
    hx, hy = env.snake[0]

    def dist_after(a):
        d = env.dir if a == 0 else (env.dir + (1 if a == 1 else -1)) % 4
        dx, dy = DIRS[d]
        return abs(hx + dx - env.food[0]) + abs(hy + dy - env.food[1])

    return min(safe, key=dist_after)


def net_policy(net):
    return lambda env, s, rng: int(np.argmax(net(s[None])[0]))


def play(policy, games, size, seed, record=False):
    env, rng = SnakeEnv(size, size, seed=seed), np.random.default_rng(seed)
    scores, frames = [], []
    for g in range(games):
        s, done = env.reset(), False
        while not done:
            if record and g == 0:
                frames.append((list(env.snake), env.food))
            s, _, done, info = env.step(policy(env, s, rng))
        scores.append(info["score"])
    return np.array(scores), frames


# ------------------------------------------------------------------- outputs
def save_gif(frames, size, path, cell=24):
    from PIL import Image, ImageDraw

    imgs = []
    for snake, food in frames:
        img = Image.new("RGB", (size * cell, size * cell), (24, 24, 32))
        d = ImageDraw.Draw(img)
        fx, fy = food
        d.ellipse([fx * cell + 4, fy * cell + 4, (fx + 1) * cell - 4, (fy + 1) * cell - 4], fill=(235, 80, 80))
        for i, (x, y) in enumerate(snake):
            color = (120, 230, 120) if i == 0 else (60, 170, 90)
            d.rectangle([x * cell + 1, y * cell + 1, (x + 1) * cell - 2, (y + 1) * cell - 2], fill=color)
        imgs.append(img)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=60, loop=0)


def plot_curves(run_dirs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in run_dirs:
        f = run / "scores.json"
        if not f.exists():
            continue
        scores = np.array(json.loads(f.read_text())["scores"])
        k = 100
        avg = np.convolve(scores, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k, len(scores) + 1), avg, label=run.name, linewidth=2)
        ax.scatter(np.arange(1, len(scores) + 1), scores, s=2, alpha=0.15)
    ax.set_xlabel("episode")
    ax.set_ylabel("score (apples eaten)")
    ax.set_title("Snake DQN — training curve (moving average over 100 episodes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dqn")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123, help="different from training seed")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    net = MLP.load(str(ROOT / "runs" / args.run / "best_model.npz"))
    policies = {"random": random_policy, "greedy heuristic": greedy_heuristic, "DQN (ours)": net_policy(net)}

    (ROOT / "assets").mkdir(exist_ok=True)
    print(f"{args.games} games on a {args.size}x{args.size} board\n")
    print(f"{'policy':<18}{'mean':>7}{'median':>8}{'max':>6}")
    rows = []
    for name, pol in policies.items():
        sc, frames = play(pol, args.games, args.size, args.seed, record=args.gif and name == "DQN (ours)")
        rows.append((name, sc.mean(), np.median(sc), sc.max()))
        print(f"{name:<18}{sc.mean():>7.2f}{np.median(sc):>8.1f}{sc.max():>6d}")
        if frames:
            save_gif(frames, args.size, ROOT / "assets" / "demo.gif")

    plot_curves(sorted((ROOT / "runs").iterdir()), ROOT / "assets" / "training_curve.png")
    md = "| Policy | Mean score | Median | Max |\n|---|---|---|---|\n" + "\n".join(
        f"| {n} | {m:.2f} | {med:.1f} | {mx} |" for n, m, med, mx in rows)
    (ROOT / "assets" / "results.md").write_text(md + "\n")
    print("\nSaved assets/training_curve.png, assets/results.md" + (", assets/demo.gif" if args.gif else ""))


if __name__ == "__main__":
    main()
