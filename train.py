"""Train the DQN agent on Snake.

    python train.py                     # default: 1500 episodes, Double DQN
    python train.py --episodes 3000
    python train.py --no-double --tag vanilla   # ablation: plain DQN
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from snake.dqn import DQNAgent, DQNConfig
from snake.env import SnakeEnv

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--size", type=int, default=12, help="board is size x size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-double", action="store_true", help="disable Double DQN")
    ap.add_argument("--tag", default="dqn")
    args = ap.parse_args()

    cfg = DQNConfig(double_dqn=not args.no_double, seed=args.seed)
    env = SnakeEnv(args.size, args.size, seed=args.seed)
    agent = DQNAgent(env.state_size, env.n_actions, cfg)

    out_dir = ROOT / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, best_avg, t0 = [], -1.0, time.time()

    for ep in range(1, args.episodes + 1):
        s, done = env.reset(), False
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.remember(s, a, r, s2, float(done))
            agent.learn()
            s = s2
        scores.append(info["score"])

        avg = float(np.mean(scores[-100:]))
        if ep >= 100 and avg > best_avg:  # keep the best model by moving average
            best_avg = avg
            agent.q.save(str(out_dir / "best_model.npz"))
        if ep % 50 == 0:
            print(f"ep {ep:5d} | avg score (last 100) {avg:5.2f} | max {max(scores):3d} | "
                  f"eps {agent.epsilon:.3f} | steps {agent.total_steps:7d} | {time.time() - t0:5.0f}s")

    agent.q.save(str(out_dir / "final_model.npz"))
    (out_dir / "scores.json").write_text(json.dumps({"config": cfg.__dict__ | {"hidden": list(cfg.hidden)},
                                                      "board": args.size, "scores": scores}))
    print(f"\nDone in {time.time() - t0:.0f}s. Best avg-100 score: {best_avg:.2f}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
