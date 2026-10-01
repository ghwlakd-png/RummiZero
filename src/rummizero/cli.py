from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agents import CandidatePolicyAgent, LinearPolicyAgent, RandomDelayAgent, SolverAgent
from .arena import duel
from .backend import SolverBackend
from .benchmark import benchmark_selfplay
from .candidates import FullTurnCandidateGenerator
from .training import (
    ActionLeague,
    League,
    evaluate_promotion,
    train_candidate_league,
    train_candidate_policy,
    train_linear,
)


def _print_league_progress(info: dict) -> None:
    generation = info.get("generation")
    generation_text = f" | gen={generation}" if generation is not None else ""
    print(
        (
            f"[train-league] {info['game']}/{info['games']} games{generation_text} | "
            f"elapsed={info['elapsed_seconds']:.1f}s | "
            f"{info['seconds_per_game']:.2f}s/game | "
            f"wins={info['learner_wins']} | draws={info['draws']} | "
            f"snapshots={info['snapshots']}"
        ),
        file=sys.stderr,
        flush=True,
    )


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
    neural.add_argument("--out", type=Path, default=Path("models/action_v5.json"))
    neural.add_argument("--snapshot-dir", type=Path, default=Path("models/action_league_v5"))
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

    league = sub.add_parser(
        "train-league",
        help="Train against frozen history and optionally resume a checkpoint",
    )
    league.add_argument("--games", type=int, default=1000)
    league.add_argument("--players", type=int, default=2, choices=(2, 3, 4))
    league.add_argument("--seed", type=int, default=7)
    league.add_argument("--hidden-size", type=int, default=32)
    league.add_argument("--learning-rate", type=float, default=0.01)
    league.add_argument("--out", type=Path, default=Path("models/action_v5.json"))
    league.add_argument("--league-dir", type=Path, default=Path("models/action_league_v5"))
    league.add_argument("--resume", type=Path, default=None)
    league.add_argument("--snapshot-every", type=int, default=100)
    league.add_argument("--historical-fraction", type=float, default=0.55)
    league.add_argument("--solver-fraction", type=float, default=0.20)
    league.add_argument("--max-candidates", type=int, default=16)
    league.add_argument("--max-solver-calls", type=int, default=64)
    league.add_argument("--promotion-games", type=int, default=100)
    league.add_argument("--promotion-threshold", type=float, default=0.55)
    league.add_argument("--skip-promotion", action="store_true")
    league.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print one progress line every N games (0 disables progress output)",
    )

    promote = sub.add_parser(
        "promotion-gate",
        help="Head-to-head gate: challenger vs champion",
    )
    promote.add_argument("--challenger", type=Path, required=True)
    promote.add_argument("--champion", type=Path, required=True)
    promote.add_argument("--games", type=int, default=100)
    promote.add_argument("--seed", type=int, default=99)
    promote.add_argument("--threshold", type=float, default=0.55)
    promote.add_argument("--max-candidates", type=int, default=16)
    promote.add_argument("--max-solver-calls", type=int, default=64)

    bench = sub.add_parser(
        "benchmark",
        help="Measure end-to-end candidate-policy game throughput on this machine",
    )
    bench.add_argument("--games", type=int, default=3)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--starting-tiles", type=int, default=14)
    bench.add_argument("--max-turns", type=int, default=300)
    bench.add_argument("--max-candidates", type=int, default=16)
    bench.add_argument("--max-solver-calls", type=int, default=64)
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
    elif args.command == "evaluate-action":
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
    elif args.command == "train-league":
        agent, stats = train_candidate_league(
            args.games,
            players=args.players,
            seed=args.seed,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            league_dir=args.league_dir,
            snapshot_every=args.snapshot_every,
            historical_fraction=args.historical_fraction,
            solver_fraction=args.solver_fraction,
            candidate_max_candidates=args.max_candidates,
            candidate_max_solver_calls=args.max_solver_calls,
            progress_every=args.progress_every,
            progress_callback=_print_league_progress,
            resume_from=args.resume,
        )
        agent.save(args.out)
        promotion = None
        if not args.skip_promotion:
            promotion = ActionLeague(args.league_dir).consider_challenger(
                args.out,
                games=args.promotion_games,
                seed=args.seed + 1_000_003,
                threshold=args.promotion_threshold,
                candidate_max_candidates=args.max_candidates,
                candidate_max_solver_calls=args.max_solver_calls,
            )
        result = {"saved": str(args.out), **stats, "promotion": promotion}
    elif args.command == "benchmark":
        result = benchmark_selfplay(
            args.games,
            seed=args.seed,
            starting_tiles=args.starting_tiles,
            max_turns=args.max_turns,
            candidate_max_candidates=args.max_candidates,
            candidate_max_solver_calls=args.max_solver_calls,
        )
    else:
        result = evaluate_promotion(
            args.challenger,
            args.champion,
            games=args.games,
            seed=args.seed,
            threshold=args.threshold,
            candidate_max_candidates=args.max_candidates,
            candidate_max_solver_calls=args.max_solver_calls,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
