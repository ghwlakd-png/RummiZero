from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ActionKind(str, Enum):
    PLAY_BEST = "play_best"
    DRAW = "draw"


@dataclass(frozen=True, slots=True)
class SolverMove:
    """A legal complete-table solution proposed by the exact solver."""

    rack_tiles: tuple[int, ...]
    table_sets: tuple[tuple[int, ...], ...]
    free_jokers: int = 0


@dataclass(frozen=True, slots=True)
class GameView:
    """Information legally observable by one player."""

    player_id: int
    rack: tuple[int, ...]
    table_sets: tuple[tuple[int, ...], ...]
    stock_count: int
    opponent_rack_counts: tuple[int, ...]
    opening_done: tuple[bool, ...]
    turn_index: int
    consecutive_passes: int
    solver_move: SolverMove | None
    joker_tile_id: int | None

    @property
    def can_play(self) -> bool:
        return self.solver_move is not None and bool(self.solver_move.rack_tiles)


@dataclass(frozen=True, slots=True)
class Transition:
    features: tuple[float, ...]
    played: int
    play_probability: float
    player_id: int


@dataclass(frozen=True, slots=True)
class GameResult:
    winner: int | None
    turns: int
    rack_sizes: tuple[int, ...]
    rack_penalties: tuple[int, ...]
    trajectories: tuple[tuple[Transition, ...], ...]
