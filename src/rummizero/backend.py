from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Iterable

from .types import SolverMove


class SolverBackend:
    """Thin adapter around the public `rummikub-solver` API.

    Tile IDs are stored as integers in RummiZero because upstream Tile objects
    are int subclasses. This keeps checkpoints/data independent of object reprs.
    """

    def __init__(self) -> None:
        try:
            from rummikub_solver import Joker, Number, RuleSet
        except ImportError as exc:  # pragma: no cover - integration environment
            raise RuntimeError(
                "rummikub-solver is required. Run: pip install rummikub-solver"
            ) from exc
        self._Joker = Joker
        self._Number = Number
        self.ruleset = RuleSet()
        self.tiles = tuple(self.ruleset.tiles)
        self.joker_id = int(self.tiles[-1]) if self.ruleset.jokers else None

    def fresh_deck(self) -> list[int]:
        deck: list[int] = []
        joker_id = self.joker_id
        for tile in self.tiles:
            tid = int(tile)
            if joker_id is not None and tid == joker_id:
                continue
            deck.extend([tid] * self.ruleset.repeats)
        if joker_id is not None:
            deck.extend([joker_id] * self.ruleset.jokers)
        return deck

    def solve(
        self,
        rack: Iterable[int],
        table_sets: Iterable[Iterable[int]],
        opening_done: bool,
    ) -> SolverMove | None:
        state = self.ruleset.new_game()
        table_flat = tuple(chain.from_iterable(table_sets))
        if table_flat:
            state.add_table(*table_flat)
        rack_t = tuple(rack)
        if rack_t:
            state.add_rack(*rack_t)
        state.initial = not opening_done
        solution = self.ruleset.solve(state)
        if not solution:
            return None
        return SolverMove(
            rack_tiles=tuple(int(t) for t in solution.tiles),
            table_sets=tuple(tuple(int(t) for t in s) for s in solution.sets),
            free_jokers=int(solution.free_jokers),
        )

    def rack_penalty(self, rack: Iterable[int]) -> int:
        total = 0
        for tid in rack:
            tile = self.tiles[tid - 1]
            if self.joker_id is not None and tid == self.joker_id:
                total += 30
            else:
                total += int(tile.value)
        return total

    def assert_inventory(
        self,
        racks: Iterable[Iterable[int]],
        table_sets: Iterable[Iterable[int]],
        stock: Iterable[int],
    ) -> None:
        expected = Counter(self.fresh_deck())
        actual: Counter[int] = Counter(stock)
        for rack in racks:
            actual.update(rack)
        for s in table_sets:
            actual.update(s)
        if actual != expected:
            raise AssertionError(f"tile inventory mismatch: {actual - expected} / {expected - actual}")
