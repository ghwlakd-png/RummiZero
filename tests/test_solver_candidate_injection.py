from rummizero.candidates import CandidateAction
from rummizero.simulator import _with_solver_candidate
from rummizero.types import SolverMove


def test_exact_solver_move_is_always_first_candidate_and_budget_is_kept():
    move = SolverMove(
        (1, 2, 3, 4),
        ((1, 2, 3, 4),),
        0,
    )
    alternatives = (
        CandidateAction((1, 2, 3), ((1, 2, 3),), 0),
        CandidateAction((2, 3, 4), ((2, 3, 4),), 0),
        CandidateAction((1, 3, 4), ((1, 3, 4),), 0),
        CandidateAction((1, 2, 4), ((1, 2, 4),), 0),
    )

    candidates = _with_solver_candidate(move, alternatives, 4)

    assert len(candidates) == 4
    assert candidates[0].rack_tiles == (1, 2, 3, 4)


def test_exact_solver_candidate_is_deduplicated():
    move = SolverMove((1, 2, 3), ((1, 2, 3),), 0)
    target = CandidateAction((1, 2, 3), ((1, 2, 3),), 0)

    candidates = _with_solver_candidate(move, (target,), 4)

    assert candidates == (target,)
