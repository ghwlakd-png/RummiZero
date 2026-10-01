from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Any, Callable

from rummizero.agents import CandidatePolicyAgent
from rummizero.agents.base import Agent
from rummizero.candidates import CandidateAction
from rummizero.simulator import GameSimulator
from rummizero.types import ActionKind, GameView, Transition


def _teacher_candidate(view: GameView) -> CandidateAction | None:
    move = view.solver_move
    if move is None or not move.rack_tiles:
        return None
    return CandidateAction(
        rack_tiles=tuple(sorted(move.rack_tiles)),
        table_sets=tuple(tuple(s) for s in move.table_sets),
        free_jokers=move.free_jokers,
    )


class SolverImitationAgent(Agent):
    """Plays exact solver moves while training a policy to imitate them."""

    def __init__(self, policy: CandidatePolicyAgent) -> None:
        self.policy = policy
        self.examples = 0
        self.correct = 0
        self.loss_sum = 0.0

    def choose(
        self,
        view: GameView,
        rng: random.Random,
    ) -> tuple[ActionKind, Transition | None]:
        return (ActionKind.PLAY_BEST if view.can_play else ActionKind.DRAW), None

    def choose_candidate(
        self,
        view: GameView,
        candidates: tuple[CandidateAction, ...],
        rng: random.Random,
    ) -> tuple[CandidateAction | None, None]:
        target = _teacher_candidate(view)
        if target is None:
            return None, None

        loss, correct = self.policy.supervised_update(view, candidates, target)
        self.examples += 1
        self.correct += int(correct)
        self.loss_sum += loss
        return target, None

    @property
    def accuracy(self) -> float:
        return self.correct / self.examples if self.examples else 0.0

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.examples if self.examples else 0.0


def train_solver_imitation(
    games: int,
    *,
    players: int = 2,
    seed: int = 0,
    hidden_size: int = 32,
    learning_rate: float = 0.01,
    resume_from: Path | None = None,
    candidate_max_candidates: int = 16,
    candidate_max_solver_calls: int = 64,
    progress_every: int = 0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[CandidatePolicyAgent, dict[str, Any]]:
    if games < 1:
        raise ValueError("games must be >= 1")
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    if resume_from is not None:
        policy = CandidatePolicyAgent.load(resume_from, training=True)
        policy.learning_rate = learning_rate
    else:
        policy = CandidatePolicyAgent(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            training=True,
            seed=seed,
        )

    teacher = SolverImitationAgent(policy)
    rng = random.Random(seed)
    started = time.perf_counter()
    total_turns = 0

    for game_i in range(1, games + 1):
        result = GameSimulator(
            [teacher] * players,
            seed=rng.randrange(2**31),
            candidate_max_candidates=candidate_max_candidates,
            candidate_max_solver_calls=candidate_max_solver_calls,
        ).run()
        total_turns += result.turns

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
                    "examples": teacher.examples,
                    "teacher_accuracy": teacher.accuracy,
                    "mean_loss": teacher.mean_loss,
                }
            )

    stats = {
        "games": games,
        "turns": total_turns,
        "examples": teacher.examples,
        "teacher_accuracy": round(teacher.accuracy, 4),
        "mean_loss": round(teacher.mean_loss, 6),
        "resumed_from": str(resume_from) if resume_from is not None else None,
    }
    return policy, stats
