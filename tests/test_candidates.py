from rummizero.candidates import FullTurnCandidateGenerator
from rummizero.types import SolverMove


class FakeBackend:
    def solve(self, rack, table_sets, opening_done):
        rack = tuple(sorted(rack))
        if len(rack) < 3:
            return None
        return SolverMove(rack, (rack,), 0)


def test_unique_multiset_subsets_do_not_duplicate():
    gen = FullTurnCandidateGenerator(FakeBackend(), max_candidates=100)
    result = gen.generate((1, 1, 2, 3), (), opening_done=True)
    keys = [c.canonical_key for c in result.candidates]
    assert len(keys) == len(set(keys))
    assert any(c.rack_tiles == (1, 1, 2, 3) for c in result.candidates)


def test_candidate_budget_is_enforced():
    gen = FullTurnCandidateGenerator(
        FakeBackend(), max_candidates=2, max_solver_calls=100
    )
    result = gen.generate((1, 2, 3, 4, 5), (), opening_done=True)
    assert len(result.candidates) == 2
    assert result.truncated is True


class PartialBackend:
    def solve(self, rack, table_sets, opening_done):
        rack = tuple(sorted(rack))
        if len(rack) < 3:
            return None
        used = rack[:3]
        return SolverMove(used, (used,), 0)


def test_rejects_solver_result_that_does_not_consume_whole_subset():
    gen = FullTurnCandidateGenerator(PartialBackend(), max_candidates=100)
    result = gen.generate((1, 2, 3, 4), (), opening_done=True)
    assert all(len(c.rack_tiles) == 3 for c in result.candidates)
