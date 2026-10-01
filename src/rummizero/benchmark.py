from __future__ import annotations

import random
import time
from typing import Any

from .agents import CandidatePolicyAgent, SolverAgent
from .simulator import GameSimulator


def benchmark_selfplay(
    games: int = 3,
    *,
    seed: int = 7,
    starting_tiles: int = 14,
    max_turns: int = 300,
    candidate_max_candidates: int = 16,
    candidate_max_solver_calls: int = 64,
) -> dict[str, Any]:
    """Measure end-to-end candidate-policy game throughput on this machine."""

    if games < 1:
        raise ValueError("games must be >= 1")
    rng = random.Random(seed)
    total_turns = 0
    completed = 0
    wins = {"candidate": 0, "solver": 0, "draw": 0}

    started = time.perf_counter()
    for game_i in range(games):
        candidate = CandidatePolicyAgent(
            hidden_size=32,
            training=False,
            seed=seed + game_i,
        )
        swapped = bool(game_i % 2)
        agents = [SolverAgent(), candidate] if swapped else [candidate, SolverAgent()]
        result = GameSimulator(
            agents,
            seed=rng.randrange(2**31),
            starting_tiles=starting_tiles,
            max_turns=max_turns,
            candidate_max_candidates=candidate_max_candidates,
            candidate_max_solver_calls=candidate_max_solver_calls,
        ).run()
        total_turns += result.turns
        completed += 1
        if result.winner is None:
            wins["draw"] += 1
        else:
            candidate_seat = 1 if swapped else 0
            if result.winner == candidate_seat:
                wins["candidate"] += 1
            else:
                wins["solver"] += 1

    elapsed = time.perf_counter() - started
    return {
        "games": completed,
        "elapsed_seconds": round(elapsed, 4),
        "seconds_per_game": round(elapsed / completed, 4),
        "turns": total_turns,
        "turns_per_game": round(total_turns / completed, 2),
        "games_per_hour": round(3600.0 * completed / elapsed, 2) if elapsed > 0 else None,
        "wins": wins,
        "settings": {
            "starting_tiles": starting_tiles,
            "max_turns": max_turns,
            "max_candidates": candidate_max_candidates,
            "max_solver_calls": candidate_max_solver_calls,
        },
    }
