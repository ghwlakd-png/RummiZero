from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Any, Callable

from rummizero.agents import CandidatePolicyAgent, LinearPolicyAgent, SolverAgent
from rummizero.simulator import GameSimulator
from rummizero.types import CandidateTransition
from .action_league import ActionLeague
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


def _candidate_trajectory(result, seat: int) -> tuple[CandidateTransition, ...]:
    return tuple(
        tr
        for tr in result.trajectories[seat]
        if isinstance(tr, CandidateTransition)
    )


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
            learner.update(
                _candidate_trajectory(result, seat),
                None if result.winner is None else result.winner == seat,
            )

        if (
            snapshot_dir is not None
            and snapshot_every
            and game_i % snapshot_every == 0
        ):
            learner.save(snapshot_dir / f"action_gen_{game_i:06d}.json")

    return learner


def train_candidate_league(
    games: int,
    *,
    players: int = 2,
    seed: int = 0,
    hidden_size: int = 32,
    learning_rate: float = 0.01,
    league_dir: Path = Path("models/action_league_v5"),
    snapshot_every: int = 100,
    historical_fraction: float = 0.55,
    solver_fraction: float = 0.20,
    candidate_max_candidates: int = 16,
    candidate_max_solver_calls: int = 64,
    progress_every: int = 0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[CandidatePolicyAgent, dict[str, Any]]:
    """v0.4 league self-play against frozen history, solver and current policy."""

    if not 0.0 <= historical_fraction <= 1.0:
        raise ValueError("historical_fraction must be within [0, 1]")
    if not 0.0 <= solver_fraction <= 1.0:
        raise ValueError("solver_fraction must be within [0, 1]")
    if historical_fraction + solver_fraction > 1.0:
        raise ValueError("historical_fraction + solver_fraction must be <= 1")
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    learner = CandidatePolicyAgent(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        training=True,
        seed=seed,
    )
    league = ActionLeague(league_dir)
    root_rng = random.Random(seed)
    opponent_counts = {"historical": 0, "solver": 0, "current": 0}
    learner_wins = draws = 0
    started = time.perf_counter()

    if snapshot_every:
        league.snapshot(learner, 0)

    for game_i in range(1, games + 1):
        learner_seat = root_rng.randrange(players)
        agents = []
        for seat in range(players):
            if seat == learner_seat:
                agents.append(learner)
                continue

            roll = root_rng.random()
            historical = league.sample(root_rng)
            if historical is not None and roll < historical_fraction:
                agents.append(CandidatePolicyAgent.load(historical, training=False))
                opponent_counts["historical"] += 1
            elif roll < historical_fraction + solver_fraction:
                agents.append(SolverAgent())
                opponent_counts["solver"] += 1
            else:
                agents.append(learner)
                opponent_counts["current"] += 1

        sim = GameSimulator(
            agents,
            seed=root_rng.randrange(2**31),
            candidate_max_candidates=candidate_max_candidates,
            candidate_max_solver_calls=candidate_max_solver_calls,
        )
        result = sim.run()
        learner.update(
            _candidate_trajectory(result, learner_seat),
            None if result.winner is None else result.winner == learner_seat,
        )
        if result.winner is None:
            draws += 1
        elif result.winner == learner_seat:
            learner_wins += 1

        if snapshot_every and game_i % snapshot_every == 0:
            league.snapshot(learner, game_i)

        if (
            progress_callback is not None
            and progress_every > 0
            and (game_i % progress_every == 0 or game_i == games)
        ):
            elapsed = time.perf_counter() - started
            progress_callback(
                {
                    "game": game_i,
                    "games": games,
                    "elapsed_seconds": elapsed,
                    "seconds_per_game": elapsed / game_i,
                    "learner_wins": learner_wins,
                    "draws": draws,
                    "snapshots": len(league.snapshots),
                }
            )

    stats = {
        "games": games,
        "learner_wins": learner_wins,
        "draws": draws,
        "opponents": opponent_counts,
        "snapshots": len(league.snapshots),
        "latest_snapshot": str(league.latest) if league.latest else None,
    }
    return learner, stats
