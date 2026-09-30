from __future__ import annotations

import random

from rummizero.agents import LinearPolicyAgent, SolverAgent
from rummizero.simulator import GameSimulator
from .league import League


def train_linear(
    games: int,
    *,
    players: int = 2,
    seed: int = 0,
    snapshot_every: int = 500,
    league: League | None = None,
    selfplay_fraction: float = 0.65,
) -> LinearPolicyAgent:
    """Bootstrap a policy with a mix of true self-play and solver sparring.

    Self-play games use the same current policy in every seat and update from all
    seat trajectories. The remaining games place the learner in a random seat
    against exact-solver baselines, which keeps early learning anchored.
    """
    learner = LinearPolicyAgent(training=True)
    root_rng = random.Random(seed)

    for game_i in range(1, games + 1):
        if root_rng.random() < selfplay_fraction:
            agents = [learner] * players
            train_seats = tuple(range(players))
        else:
            seat = root_rng.randrange(players)
            agents = [SolverAgent() for _ in range(players)]
            agents[seat] = learner
            train_seats = (seat,)

        sim = GameSimulator(agents, seed=root_rng.randrange(2**31))
        result = sim.run()
        for seat in train_seats:
            learner.update(
                result.trajectories[seat],
                None if result.winner is None else result.winner == seat,
            )

        if league and snapshot_every and game_i % snapshot_every == 0:
            league.snapshot(learner, game_i)

    return learner
