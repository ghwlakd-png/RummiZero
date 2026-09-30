from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import LinearPolicyAgent, RandomDelayAgent, SolverAgent
from .arena import duel
from .training import League, train_linear


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rummizero")
    sub = p.add_subparsers(dest="command", required=True)

    arena = sub.add_parser("arena", help="SolverAgent vs weak random-delay baseline")
    arena.add_argument("--games", type=int, default=100)
    arena.add_argument("--seed", type=int, default=7)

    train = sub.add_parser("train", help="Bootstrap the v0.1 linear self-play policy")
    train.add_argument("--games", type=int, default=5000)
    train.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--out", type=Path, default=Path("models/linear_v1.json"))
    train.add_argument("--league-dir", type=Path, default=Path("models/league"))
    train.add_argument("--snapshot-every", type=int, default=500)

    ev = sub.add_parser("evaluate", help="Evaluate a saved linear policy vs SolverAgent")
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--games", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=99)
    return p


def main() -> None:
    args = _parser().parse_args()
    if args.command == "arena":
        result = duel(
            SolverAgent,
            lambda: RandomDelayAgent(0.45),
            games=args.games,
            seed=args.seed,
        )
    elif args.command == "train":
        league = League(args.league_dir)
        agent = train_linear(
            args.games,
            players=args.players,
            seed=args.seed,
            snapshot_every=args.snapshot_every,
            league=league,
        )
        agent.save(args.out)
        result = {"saved": str(args.out), "snapshots": len(league.snapshots), "weights": agent.weights}
    else:
        model_path = args.model
        result = duel(
            lambda: LinearPolicyAgent.load(model_path, training=False),
            SolverAgent,
            games=args.games,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
