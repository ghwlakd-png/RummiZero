from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

from .types import SolverMove


class CandidateBackend(Protocol):
    def solve(
        self,
        rack: Iterable[int],
        table_sets: Iterable[Iterable[int]],
        opening_done: bool,
    ) -> SolverMove | None: ...


@dataclass(frozen=True, slots=True)
class CandidateAction:
    """One complete legal turn candidate."""

    rack_tiles: tuple[int, ...]
    table_sets: tuple[tuple[int, ...], ...]
    free_jokers: int = 0

    @property
    def tile_count(self) -> int:
        return len(self.rack_tiles)

    @property
    def canonical_key(self) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]:
        sets = tuple(sorted(tuple(sorted(s)) for s in self.table_sets))
        return tuple(sorted(self.rack_tiles)), sets, self.free_jokers


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    candidates: tuple[CandidateAction, ...]
    solver_calls: int
    truncated: bool


class FullTurnCandidateGenerator:
    """Generate multiple complete legal turn candidates.

    v0.2a enumerates unique rack sub-multisets and asks the exact solver whether
    all tiles in that subset can legally be placed while preserving/rearranging
    the current table.

    Limitation: for a given rack subset the upstream MILP solver returns one
    arrangement. Enumerating multiple distinct table arrangements for the same
    subset is the next v0.2b step.
    """

    def __init__(
        self,
        backend: CandidateBackend,
        *,
        max_candidates: int = 128,
        max_solver_calls: int = 2048,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if max_solver_calls < 1:
            raise ValueError("max_solver_calls must be >= 1")
        self.backend = backend
        self.max_candidates = max_candidates
        self.max_solver_calls = max_solver_calls

    @staticmethod
    def _unique_subsets(rack: Iterable[int]) -> list[tuple[int, ...]]:
        counts = sorted(Counter(rack).items())
        out: list[tuple[int, ...]] = []

        def visit(index: int, current: list[int]) -> None:
            if index == len(counts):
                if current:
                    out.append(tuple(current))
                return
            tile, count = counts[index]
            for n in range(count + 1):
                if n:
                    current.extend([tile] * n)
                visit(index + 1, current)
                if n:
                    del current[-n:]

        visit(0, [])
        out.sort(key=lambda s: (-len(s), s))
        return out

    @staticmethod
    def _as_candidate(move: SolverMove) -> CandidateAction:
        return CandidateAction(
            rack_tiles=tuple(sorted(move.rack_tiles)),
            table_sets=tuple(tuple(s) for s in move.table_sets),
            free_jokers=move.free_jokers,
        )

    def generate(
        self,
        rack: Iterable[int],
        table_sets: Iterable[Iterable[int]],
        *,
        opening_done: bool,
    ) -> CandidateGenerationResult:
        rack_t = tuple(sorted(rack))
        table_t = tuple(tuple(s) for s in table_sets)
        subsets = self._unique_subsets(rack_t)

        seen: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]] = set()
        candidates: list[CandidateAction] = []
        solver_calls = 0
        truncated = False
        table_is_empty = not any(table_t)

        for subset in subsets:
            if len(candidates) >= self.max_candidates:
                truncated = True
                break
            if solver_calls >= self.max_solver_calls:
                truncated = True
                break
            if table_is_empty and len(subset) < 3:
                continue

            solver_calls += 1
            move = self.backend.solve(subset, table_t, opening_done)
            if move is None:
                continue

            if Counter(move.rack_tiles) != Counter(subset):
                continue

            candidate = self._as_candidate(move)
            if candidate.canonical_key in seen:
                continue
            seen.add(candidate.canonical_key)
            candidates.append(candidate)

        candidates.sort(key=lambda c: (-c.tile_count, c.canonical_key))
        return CandidateGenerationResult(tuple(candidates), solver_calls, truncated)
