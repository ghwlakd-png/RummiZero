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


class RecordingBackend(FakeBackend):
    def __init__(self):
        self.subsets = []

    def solve(self, rack, table_sets, opening_done):
        self.subsets.append(tuple(rack))
        return super().solve(rack, table_sets, opening_done)


def test_preferred_move_skips_impossible_larger_subsets():
    backend = RecordingBackend()
    gen = FullTurnCandidateGenerator(
        backend,
        max_candidates=4,
        max_solver_calls=8,
    )
    preferred = SolverMove((1, 2, 3), ((1, 2, 3),), 0)

    result = gen.generate(
        tuple(range(1, 15)),
        (),
        opening_done=True,
        preferred_move=preferred,
    )

    assert result.solver_calls <= 8
    assert backend.subsets
    assert all(len(subset) <= 3 for subset in backend.subsets)
    assert result.candidates


def test_preferred_move_interleaves_subset_sizes_within_budget():
    backend = RecordingBackend()
    gen = FullTurnCandidateGenerator(
        backend,
        max_candidates=20,
        max_solver_calls=4,
    )
    preferred = SolverMove((1, 2, 3, 4), ((1, 2, 3, 4),), 0)

    gen.generate(
        (1, 2, 3, 4, 5, 6),
        ((7, 8, 9),),
        opening_done=True,
        preferred_move=preferred,
    )

    assert [len(subset) for subset in backend.subsets] == [4, 3, 2, 1]
