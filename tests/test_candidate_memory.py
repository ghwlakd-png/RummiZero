from rummizero.candidates import FullTurnCandidateGenerator


class CountingNoMoveBackend:
    def __init__(self):
        self.calls = 0

    def solve(self, rack, table_sets, opening_done):
        self.calls += 1
        return None


def test_large_rack_subset_search_is_lazy_and_stops_at_solver_budget():
    backend = CountingNoMoveBackend()
    gen = FullTurnCandidateGenerator(
        backend,
        max_candidates=8,
        max_solver_calls=8,
    )

    # 30 distinct tiles would require 2**30 - 1 materialized subsets in the
    # old implementation and could exhaust RAM before the solver budget applied.
    result = gen.generate(tuple(range(1, 31)), (), opening_done=True)

    assert result.candidates == ()
    assert result.solver_calls == 8
    assert backend.calls == 8
    assert result.truncated is True


def test_lazy_subset_order_prefers_larger_then_lexicographic():
    subsets = FullTurnCandidateGenerator._iter_unique_subsets((1, 2, 3, 4))
    first_five = [next(subsets) for _ in range(5)]
    assert first_five == [
        (1, 2, 3, 4),
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
        (2, 3, 4),
    ]
