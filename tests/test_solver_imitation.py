import random

from rummizero.agents import CandidatePolicyAgent
from rummizero.candidates import CandidateAction
from rummizero.cli import _parser
from rummizero.training.imitation import SolverImitationAgent
from rummizero.types import GameView, SolverMove


def _view():
    move = SolverMove((1, 2, 3, 4), ((1, 2, 3, 4),), 0)
    return GameView(
        player_id=0,
        rack=(1, 2, 3, 4),
        table_sets=(),
        stock_count=70,
        opponent_rack_counts=(14,),
        opening_done=(True, True),
        turn_index=5,
        consecutive_passes=0,
        solver_move=move,
        joker_tile_id=53,
    )


def test_supervised_imitation_increases_teacher_probability():
    agent = CandidatePolicyAgent(hidden_size=8, seed=9, training=True, learning_rate=0.05)
    target = CandidateAction((1, 2, 3, 4), ((1, 2, 3, 4),), 0)
    alternatives = (
        target,
        CandidateAction((1, 2, 3), ((1, 2, 3),), 0),
    )

    _, _, before = agent.option_probabilities(_view(), alternatives)
    for _ in range(30):
        agent.supervised_update(_view(), alternatives, target)
    _, _, after = agent.option_probabilities(_view(), alternatives)

    assert float(after[0]) > float(before[0])


def test_imitation_agent_plays_teacher_action_and_tracks_examples():
    policy = CandidatePolicyAgent(hidden_size=8, seed=3, training=True)
    teacher = SolverImitationAgent(policy)
    target = CandidateAction((1, 2, 3, 4), ((1, 2, 3, 4),), 0)
    action, transition = teacher.choose_candidate(
        _view(),
        (target,),
        random.Random(1),
    )
    assert action == target
    assert transition is None
    assert teacher.examples == 1


def test_imitate_solver_cli_parses():
    args = _parser().parse_args(
        [
            "imitate-solver",
            "--games",
            "12",
            "--max-candidates",
            "4",
            "--max-solver-calls",
            "8",
        ]
    )
    assert args.command == "imitate-solver"
    assert args.games == 12
    assert args.max_candidates == 4
    assert args.max_solver_calls == 8
