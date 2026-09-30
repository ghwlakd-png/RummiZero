from dataclasses import dataclass

from rummizero.candidates import FullTurnCandidateGenerator
from rummizero.types import SolverMove


@dataclass(frozen=True)
class ArrResult:
    arrangements: tuple
    nodes: int = 1
    truncated: bool = False


class MultiLayoutBackend:
    def solve(self, rack, table_sets, opening_done):
        rack = tuple(sorted(rack))
        if rack != (1, 2, 3, 4, 5, 6):
            return None
        return SolverMove(rack, ((1, 2, 3, 4, 5, 6),), 0)

    def enumerate_arrangements(
        self, rack, table_sets, opening_done, *, limit, node_budget
    ):
        return ArrResult(
            (
                ((1, 2, 3, 4, 5, 6),),
                ((1, 2, 3), (4, 5, 6)),
            )[:limit]
        )


def test_same_rack_subset_can_create_two_distinct_actions():
    gen = FullTurnCandidateGenerator(
        MultiLayoutBackend(),
        max_candidates=10,
        max_solver_calls=100,
        max_arrangements_per_subset=4,
    )
    result = gen.generate((1, 2, 3, 4, 5, 6), (), opening_done=True)
    same_subset = [
        c for c in result.candidates if c.rack_tiles == (1, 2, 3, 4, 5, 6)
    ]
    assert len(same_subset) == 2
    assert len({c.canonical_key for c in same_subset}) == 2
