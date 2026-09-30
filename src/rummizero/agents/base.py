from __future__ import annotations

from abc import ABC, abstractmethod
import random

from rummizero.types import ActionKind, GameView, Transition


class Agent(ABC):
    @abstractmethod
    def choose(self, view: GameView, rng: random.Random) -> tuple[ActionKind, Transition | None]:
        raise NotImplementedError

    def on_game_end(self, player_id: int, winner: int | None) -> None:
        return None
