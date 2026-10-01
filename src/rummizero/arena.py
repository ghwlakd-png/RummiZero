from __future__ import annotations

import math
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
    paired_deals: bool = True,
) -> dict[str, float | int]:
    """Evaluate two agents, optionally reusing each deal with seats swapped."""

    if games < 1:
        raise ValueError("games must be >= 1")

    rng = random.Random(seed)
    wins_a = wins_b = draws = 0
    elo_a = elo_b = 1000.0
    score_sum = 0.0
    score_sq_sum = 0.0
    sim_kwargs = simulator_kwargs or {}
    pair_seed: int | None = None

    for i in range(games):
        swapped = bool(i % 2)
        if paired_deals:
            if not swapped or pair_seed is None:
                pair_seed = rng.randrange(2**31)
            game_seed = pair_seed
        else:
            game_seed = rng.randrange(2**31)

        agents = [make_b(), make_a()] if swapped else [make_a(), make_b()]
        result = GameSimulator(
            agents,
            seed=game_seed,
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

        score_sum += score_a
        score_sq_sum += score_a * score_a
        elo_a, elo_b = update(elo_a, elo_b, score_a)

    mean_score = score_sum / games
    if games > 1:
        variance = max(0.0, (score_sq_sum - games * mean_score * mean_score) / (games - 1))
        standard_error = math.sqrt(variance / games)
    else:
        standard_error = 0.0
    ci_margin = 1.96 * standard_error

    return {
        "games": games,
        "a_wins": wins_a,
        "b_wins": wins_b,
        "draws": draws,
        "a_score": round(mean_score, 4),
        "a_score_ci95_low": round(max(0.0, mean_score - ci_margin), 4),
        "a_score_ci95_high": round(min(1.0, mean_score + ci_margin), 4),
        # Kept for backwards compatibility. These are sequential online Elo
        # values and should not be used as the promotion decision.
        "a_elo": round(elo_a, 1),
        "b_elo": round(elo_b, 1),
        "paired_deals": paired_deals,
    }
