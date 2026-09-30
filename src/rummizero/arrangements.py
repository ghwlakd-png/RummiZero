from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ArrangementEnumeration:
    arrangements: tuple[tuple[tuple[int, ...], ...], ...]
    nodes: int
    truncated: bool


class ExactCoverArrangementEnumerator:
    """Enumerate legal complete table arrangements for a fixed multiset of tiles."""

    def __init__(
        self,
        valid_sets: Sequence[Sequence[int]],
        *,
        set_values: Sequence[int] | None = None,
        min_initial_value: int = 30,
    ) -> None:
        self.valid_sets = tuple(tuple(sorted(int(t) for t in s)) for s in valid_sets)
        self.set_counters = tuple(Counter(s) for s in self.valid_sets)
        self.set_values = (
            tuple(int(v) for v in set_values)
            if set_values is not None
            else tuple(0 for _ in self.valid_sets)
        )
        self.min_initial_value = int(min_initial_value)

        by_tile: dict[int, list[int]] = defaultdict(list)
        for i, counter in enumerate(self.set_counters):
            for tile in counter:
                by_tile[tile].append(i)
        self.by_tile = {tile: tuple(indices) for tile, indices in by_tile.items()}

    @staticmethod
    def _canonical(sets: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted(tuple(sorted(int(t) for t in s)) for s in sets))

    @staticmethod
    def _fits(need: Counter[int], remaining: Counter[int]) -> bool:
        return not (need - remaining)

    @staticmethod
    def _subtract(
        remaining: Counter[int], need: Counter[int]
    ) -> Counter[int]:
        out = remaining.copy()
        out.subtract(need)
        return +out

    def enumerate_pool(
        self,
        pool: Iterable[int],
        *,
        limit: int = 8,
        node_budget: int = 20_000,
        min_total_value: int | None = None,
    ) -> ArrangementEnumeration:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if node_budget < 1:
            raise ValueError("node_budget must be >= 1")

        remaining0 = Counter(int(t) for t in pool)
        if not remaining0:
            return ArrangementEnumeration((), 0, False)

        eligible = [
            i
            for i, need in enumerate(self.set_counters)
            if self._fits(need, remaining0)
        ]
        eligible_set = set(eligible)

        by_tile: dict[int, tuple[int, ...]] = {}
        for tile in remaining0:
            by_tile[tile] = tuple(
                i for i in self.by_tile.get(tile, ()) if i in eligible_set
            )

        found: list[tuple[tuple[int, ...], ...]] = []
        found_keys: set[tuple[tuple[int, ...], ...]] = set()
        nodes = 0
        truncated = False

        def recurse(
            remaining: Counter[int],
            chosen: list[int],
            value_sum: int,
        ) -> None:
            nonlocal nodes, truncated
            if len(found) >= limit:
                truncated = True
                return
            if nodes >= node_budget:
                truncated = True
                return
            nodes += 1

            if not remaining:
                if min_total_value is not None and value_sum < min_total_value:
                    return
                arrangement = self._canonical(self.valid_sets[i] for i in chosen)
                if arrangement not in found_keys:
                    found_keys.add(arrangement)
                    found.append(arrangement)
                return

            best_tile: int | None = None
            best_options: list[int] | None = None
            for tile in sorted(remaining):
                options = [
                    i
                    for i in by_tile.get(tile, ())
                    if self._fits(self.set_counters[i], remaining)
                ]
                if not options:
                    return
                if best_options is None or len(options) < len(best_options):
                    best_tile = tile
                    best_options = options
                    if len(options) == 1:
                        break

            assert best_tile is not None and best_options is not None
            for i in best_options:
                need = self.set_counters[i]
                recurse(
                    self._subtract(remaining, need),
                    chosen + [i],
                    value_sum + self.set_values[i],
                )
                if len(found) >= limit or nodes >= node_budget:
                    return

        recurse(remaining0, [], 0)
        found.sort()
        return ArrangementEnumeration(tuple(found), nodes, truncated)
