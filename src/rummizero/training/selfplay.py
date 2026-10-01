from __future__ import annotations

from pathlib import Path
import random

from rummizero.agents import CandidatePolicyAgent, LinearPolicyAgent, SolverAgent
from rummizero.simulator import GameSimulator
from rummizero.types import CandidateTransition
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


def train_candidate_policy(
    games: int,
    *,
    players: int = 2,
    seed: int = 0,
    hidden_size: int = 32,
    learning_rate: float = 0.01,
    selfplay_fraction: float = 0.65,
    snapshot_every: int = 0,
    snapshot_dir: Path | None = None,
    candidate_max_candidates: int = 16,
    candidate_max_solver_calls: int = 64,
) -> CandidatePolicyAgent:
    """Train the v0.3 action-conditioned neural policy by self-play.

    This first implementation is intentionally CPU/NumPy based. It validates
    candidate scoring, sampling, game execution, terminal learning and
    checkpointing before the later parallel/GPU learner.
    """
    learner = CandidatePolicyAgent(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        training=True,
        seed=seed,
    )
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

        sim = GameSimulator(
            agents,
            seed=root_rng.randrange(2**31),
            candidate_max_candidates=candidate_max_candidates,
            candidate_max_solver_calls=candidate_max_solver_calls,
        )
        result = sim.run()
        for seat in train_seats:
            trajectory = tuple(
                tr
                for tr in result.trajectories[seat]
                if isinstance(tr, CandidateTransition)
            )
            learner.update(
                trajectory,
                None if result.winner is None else result.winner == seat,
            )

        if (
            snapshot_dir is not None
            and snapshot_every
            and game_i % snapshot_every == 0
        ):
            learner.save(snapshot_dir / f"action_gen_{game_i:06d}.json")

    return learner
