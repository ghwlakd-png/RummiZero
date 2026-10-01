import random

import numpy as np

from rummizero.agents import CandidatePolicyAgent
from rummizero.candidates import CandidateAction
from rummizero.types import GameView, SolverMove


def _view():
    return GameView(
        player_id=0,
        rack=(1, 2, 3, 4),
        table_sets=(),
        stock_count=70,
        opponent_rack_counts=(14,),
        opening_done=(True, True),
        turn_index=5,
        consecutive_passes=0,
        solver_move=SolverMove((1, 2, 3, 4), ((1, 2, 3, 4),), 0),
        joker_tile_id=53,
    )


def test_policy_scores_candidates_plus_draw_and_learns(tmp_path):
    agent = CandidatePolicyAgent(hidden_size=8, seed=3, training=True)
    candidates = (
        CandidateAction((1, 2, 3), ((1, 2, 3),), 0),
        CandidateAction((1, 2, 3, 4), ((1, 2, 3, 4),), 0),
    )
    options, _, probs = agent.option_probabilities(_view(), candidates)
    assert len(options) == 3
    assert np.isclose(float(probs.sum()), 1.0)

    before = agent.w1.copy()
    _, transition = agent.choose_candidate(_view(), candidates, random.Random(7))
    assert transition is not None
    agent.update((transition,), True)
    assert not np.array_equal(before, agent.w1)

    path = tmp_path / "action.json"
    agent.save(path)
    loaded = CandidatePolicyAgent.load(path, training=False)
    _, _, loaded_probs = loaded.option_probabilities(_view(), candidates)
    _, _, new_probs = agent.option_probabilities(_view(), candidates)
    assert np.allclose(loaded_probs, new_probs)


def test_policy_with_no_play_candidates_can_draw():
    agent = CandidatePolicyAgent(hidden_size=4, seed=1, training=False)
    action, transition = agent.choose_candidate(_view(), (), random.Random(1))
    assert action is None
    assert transition is None
