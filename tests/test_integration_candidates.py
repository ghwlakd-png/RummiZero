import pytest

pytest.importorskip("rummikub_solver")

from rummizero.backend import SolverBackend
from rummizero.candidates import FullTurnCandidateGenerator


def test_real_solver_finds_multiple_run_candidates():
    backend = SolverBackend()
    gen = FullTurnCandidateGenerator(
        backend, max_candidates=20, max_solver_calls=50
    )
    result = gen.generate((1, 2, 3, 4), (), opening_done=True)
    played = {c.rack_tiles for c in result.candidates}
    assert (1, 2, 3, 4) in played
    assert (1, 2, 3) in played
    assert (2, 3, 4) in played
