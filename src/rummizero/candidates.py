from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

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

    Candidate rack subsets are streamed lazily, largest first. This is critical
    in long games where a rack may grow large: materializing every multiset
    subset is exponential and can exhaust RAM before max_solver_calls is applied.
    """

    def __init__(
        self,
        backend: CandidateBackend,
        *,
        max_candidates: int = 128,
        max_solver_calls: int = 2048,
        max_arrangements_per_subset: int = 8,
        arrangement_node_budget: int = 20_000,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if max_solver_calls < 1:
            raise ValueError("max_solver_calls must be >= 1")
        if max_arrangements_per_subset < 1:
            raise ValueError("max_arrangements_per_subset must be >= 1")
        if arrangement_node_budget < 1:
            raise ValueError("arrangement_node_budget must be >= 1")
        self.backend = backend
        self.max_candidates = max_candidates
        self.max_solver_calls = max_solver_calls
        self.max_arrangements_per_subset = max_arrangements_per_subset
        self.arrangement_node_budget = arrangement_node_budget

    @staticmethod
    def _iter_unique_subsets(rack: Iterable[int]) -> Iterator[tuple[int, ...]]:
        """Yield unique non-empty rack sub-multisets without materializing them.

        Ordering matches the useful property of the old implementation:
        larger subsets are considered before smaller subsets. Within a fixed
        size, lower tile IDs are preferred lexicographically.
        """

        counts = sorted(Counter(rack).items())
        if not counts:
            return

        remaining = [0] * (len(counts) + 1)
        for i in range(len(counts) - 1, -1, -1):
            remaining[i] = remaining[i + 1] + counts[i][1]

        current: list[int] = []

        def visit(index: int, need: int) -> Iterator[tuple[int, ...]]:
            if need == 0:
                yield tuple(current)
                return
            if index >= len(counts) or need > remaining[index]:
                return

            tile, count = counts[index]
            max_take = min(count, need)
            min_take = max(0, need - remaining[index + 1])

            # Prefer more copies of lower tile IDs first for lexical ordering.
            for n in range(max_take, min_take - 1, -1):
                if n:
                    current.extend([tile] * n)
                yield from visit(index + 1, need - n)
                if n:
                    del current[-n:]

        for target_size in range(remaining[0], 0, -1):
            yield from visit(0, target_size)

    @staticmethod
    def _as_candidate(move: SolverMove) -> CandidateAction:
        return CandidateAction(
            rack_tiles=tuple(sorted(move.rack_tiles)),
            table_sets=tuple(tuple(s) for s in move.table_sets),
            free_jokers=move.free_jokers,
        )

    def _append_candidate(
        self,
        candidate: CandidateAction,
        seen: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]],
        candidates: list[CandidateAction],
    ) -> bool:
        if candidate.canonical_key in seen:
            return False
        seen.add(candidate.canonical_key)
        candidates.append(candidate)
        return True

    def generate(
        self,
        rack: Iterable[int],
        table_sets: Iterable[Iterable[int]],
        *,
        opening_done: bool,
    ) -> CandidateGenerationResult:
        rack_t = tuple(sorted(rack))
        table_t = tuple(tuple(s) for s in table_sets)

        seen: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]] = set()
        candidates: list[CandidateAction] = []
        solver_calls = 0
        truncated = False
        table_is_empty = not any(table_t)
        enumerate_arrangements = getattr(self.backend, "enumerate_arrangements", None)

        for subset in self._iter_unique_subsets(rack_t):
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

            # The optimizer may choose only part of the supplied subset.
            # We only label an action with a subset when every supplied tile is used.
            if Counter(move.rack_tiles) != Counter(subset):
                continue

            added_layout = False
            if callable(enumerate_arrangements):
                room = self.max_candidates - len(candidates)
                per_subset_limit = min(self.max_arrangements_per_subset, room)
                arrangement_result = enumerate_arrangements(
                    subset,
                    table_t,
                    opening_done,
                    limit=per_subset_limit,
                    node_budget=self.arrangement_node_budget,
                )
                truncated = truncated or arrangement_result.truncated
                for layout in arrangement_result.arrangements:
                    candidate = CandidateAction(
                        rack_tiles=tuple(sorted(subset)),
                        table_sets=tuple(tuple(s) for s in layout),
                        free_jokers=0,
                    )
                    added_layout = self._append_candidate(
                        candidate, seen, candidates
                    ) or added_layout
                    if len(candidates) >= self.max_candidates:
                        truncated = True
                        break

            if not added_layout and len(candidates) < self.max_candidates:
                self._append_candidate(self._as_candidate(move), seen, candidates)

        candidates.sort(key=lambda c: (-c.tile_count, c.canonical_key))
        return CandidateGenerationResult(tuple(candidates), solver_calls, truncated)
