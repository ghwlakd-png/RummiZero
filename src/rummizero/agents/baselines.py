from __future__ import annotations

import random

from .base import Agent
from rummizero.types import ActionKind, GameView, Transition


class SolverAgent(Agent):
    """Always takes the exact solver move when one exists."""

    def choose(self, view: GameView, rng: random.Random) -> tuple[ActionKind, Transition | None]:
        if view.can_play:
            return ActionKind.PLAY_BEST, None
        return ActionKind.DRAW, None


class RandomDelayAgent(Agent):
    """Weak baseline: sometimes declines a legal move and draws."""

    def __init__(self, play_probability: float = 0.5) -> None:
        self.play_probability = play_probability

    def choose(self, view: GameView, rng: random.Random) -> tuple[ActionKind, Transition | None]:
        if view.can_play and rng.random() < self.play_probability:
            return ActionKind.PLAY_BEST, None
        return ActionKind.DRAW, None
