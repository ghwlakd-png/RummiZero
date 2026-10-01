from __future__ import annotations

import random
from typing import Any, Callable

from .agents.base import Agent
from .simulator import GameSimulator
from .training.elo import update


def duel(
    make_a: Callable[[], Agent],
    make_b: Callable[[], Agent],
    *,
    games: int,
    seed: int,
    simulator_kwargs: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    wins_a = wins_b = draws = 0
    elo_a = elo_b = 1000.0
    sim_kwargs = simulator_kwargs or {}

    for i in range(games):
        swapped = bool(i % 2)
        agents = [make_b(), make_a()] if swapped else [make_a(), make_b()]
        result = GameSimulator(
            agents,
            seed=rng.randrange(2**31),
            **sim_kwargs,
        ).run()
        if result.winner is None:
            score_a = 0.5
            draws += 1
        else:
            a_seat = 1 if swapped else 0
            if result.winner == a_seat:
                wins_a += 1
                score_a = 1.0
            else:
                wins_b += 1
                score_a = 0.0
        elo_a, elo_b = update(elo_a, elo_b, score_a)
    return {
        "games": games,
        "a_wins": wins_a,
        "b_wins": wins_b,
        "draws": draws,
        "a_elo": round(elo_a, 1),
        "b_elo": round(elo_b, 1),
    }
