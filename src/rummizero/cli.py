from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import CandidatePolicyAgent, LinearPolicyAgent, RandomDelayAgent, SolverAgent
from .arena import duel
from .backend import SolverBackend
from .candidates import FullTurnCandidateGenerator
from .training import League, train_candidate_policy, train_linear


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

    cand = sub.add_parser("candidates", help="Generate complete-turn candidates")
    cand.add_argument("--rack", type=str, required=True)
    cand.add_argument("--opening-done", action="store_true")
    cand.add_argument("--max-candidates", type=int, default=32)
    cand.add_argument("--max-solver-calls", type=int, default=256)

    neural = sub.add_parser(
        "train-action",
        help="Train the v0.3 action-conditioned neural self-play policy",
    )
    neural.add_argument("--games", type=int, default=100)
    neural.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    neural.add_argument("--seed", type=int, default=7)
    neural.add_argument("--hidden-size", type=int, default=32)
    neural.add_argument("--learning-rate", type=float, default=0.01)
    neural.add_argument("--out", type=Path, default=Path("models/action_v3.json"))
    neural.add_argument("--snapshot-dir", type=Path, default=Path("models/action_league"))
    neural.add_argument("--snapshot-every", type=int, default=0)
    neural.add_argument("--max-candidates", type=int, default=16)
    neural.add_argument("--max-solver-calls", type=int, default=64)

    neural_ev = sub.add_parser(
        "evaluate-action",
        help="Evaluate a v0.3 action policy vs SolverAgent",
    )
    neural_ev.add_argument("--model", type=Path, required=True)
    neural_ev.add_argument("--games", type=int, default=100)
    neural_ev.add_argument("--seed", type=int, default=99)
    neural_ev.add_argument("--max-candidates", type=int, default=16)
    neural_ev.add_argument("--max-solver-calls", type=int, default=64)
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
        result = {
            "saved": str(args.out),
            "snapshots": len(league.snapshots),
            "weights": agent.weights,
        }
    elif args.command == "evaluate":
        result = duel(
            lambda: LinearPolicyAgent.load(args.model, training=False),
            SolverAgent,
            games=args.games,
            seed=args.seed,
        )
    elif args.command == "candidates":
        rack = tuple(int(x.strip()) for x in args.rack.split(",") if x.strip())
        backend = SolverBackend()
        generated = FullTurnCandidateGenerator(
            backend,
            max_candidates=args.max_candidates,
            max_solver_calls=args.max_solver_calls,
        ).generate(rack, (), opening_done=args.opening_done)
        result = {
            "rack": rack,
            "candidate_count": len(generated.candidates),
            "solver_calls": generated.solver_calls,
            "truncated": generated.truncated,
            "candidates": [
                {
                    "rack_tiles": c.rack_tiles,
                    "table_sets": c.table_sets,
                    "free_jokers": c.free_jokers,
                }
                for c in generated.candidates
            ],
        }
    elif args.command == "train-action":
        agent = train_candidate_policy(
            args.games,
            players=args.players,
            seed=args.seed,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            snapshot_every=args.snapshot_every,
            snapshot_dir=args.snapshot_dir,
            candidate_max_candidates=args.max_candidates,
            candidate_max_solver_calls=args.max_solver_calls,
        )
        agent.save(args.out)
        result = {
            "saved": str(args.out),
            "games": args.games,
            "hidden_size": args.hidden_size,
        }
    else:
        result = duel(
            lambda: CandidatePolicyAgent.load(args.model, training=False),
            SolverAgent,
            games=args.games,
            seed=args.seed,
            simulator_kwargs={
                "candidate_max_candidates": args.max_candidates,
                "candidate_max_solver_calls": args.max_solver_calls,
            },
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
