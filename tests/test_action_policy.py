import random
from dataclasses import replace

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


def _transition(agent):
    candidates = (
        CandidateAction((1, 2, 3), ((1, 2, 3),), 0),
        CandidateAction((1, 2, 3, 4), ((1, 2, 3, 4),), 0),
    )
    _, transition = agent.choose_candidate(_view(), candidates, random.Random(7))
    assert transition is not None
    return candidates, transition


def test_policy_scores_candidates_plus_draw_and_learns(tmp_path):
    agent = CandidatePolicyAgent(hidden_size=8, seed=3, training=True)
    candidates, transition = _transition(agent)
    options, _, probs = agent.option_probabilities(_view(), candidates)
    assert len(options) == 3
    assert np.isclose(float(probs.sum()), 1.0)

    before = agent.w1.copy()
    agent.update((transition,), True)
    assert not np.array_equal(before, agent.w1)

    path = tmp_path / "action.json"
    agent.save(path)
    loaded = CandidatePolicyAgent.load(path, training=False)
    _, _, loaded_probs = loaded.option_probabilities(_view(), candidates)
    _, _, new_probs = agent.option_probabilities(_view(), candidates)
    assert np.allclose(loaded_probs, new_probs)


def test_game_update_is_normalized_by_trajectory_length():
    one = CandidatePolicyAgent(hidden_size=8, seed=11, training=True)
    many = CandidatePolicyAgent(hidden_size=8, seed=11, training=True)

    _, tr_one = _transition(one)
    _, tr_many = _transition(many)

    one.update((tr_one,), True)
    many.update((tr_many,) * 50, True)

    assert np.allclose(one.w1, many.w1)
    assert np.allclose(one.b1, many.b1)
    assert np.allclose(one.w2, many.w2)
    assert np.isclose(one.b2, many.b2)


def test_policy_with_no_play_candidates_can_draw():
    agent = CandidatePolicyAgent(hidden_size=4, seed=1, training=False)
    action, transition = agent.choose_candidate(_view(), (), random.Random(1))
    assert action is None
    assert transition is None


def test_training_temperature_increases_exploration_and_round_trips(tmp_path):
    cold = CandidatePolicyAgent(hidden_size=8, seed=3, training=True)
    hot = CandidatePolicyAgent(
        hidden_size=8,
        seed=3,
        training=True,
        training_temperature=3.0,
    )
    candidates, _ = _transition(cold)
    _, _, cold_probs = cold.option_probabilities(_view(), candidates)
    _, _, hot_probs = hot.option_probabilities(
        _view(), candidates, temperature=hot.training_temperature
    )
    assert hot_probs.max() < cold_probs.max()

    path = tmp_path / "hot.json"
    hot.save(path)
    loaded = CandidatePolicyAgent.load(path, training=True)
    assert loaded.training_temperature == 3.0


def test_teacher_focused_update_ignores_teacher_actions():
    focused = CandidatePolicyAgent(hidden_size=8, seed=13, training=True)
    expected = CandidatePolicyAgent(hidden_size=8, seed=13, training=True)
    _, transition = _transition(focused)
    teacher = replace(transition, chosen_index=0)
    deviation = replace(transition, chosen_index=1)

    focused.update((teacher, deviation), True, teacher_index=0)
    expected.update((deviation,), True)

    assert np.allclose(focused.w1, expected.w1)
    assert np.allclose(focused.w2, expected.w2)


def test_logit_scaling_preserves_ranking_and_softens_confidence():
    agent = CandidatePolicyAgent(hidden_size=8, seed=3, training=False)
    candidates, _ = _transition(CandidatePolicyAgent(hidden_size=8, seed=3))
    _, _, before = agent.option_probabilities(_view(), candidates)
    before_choice = int(np.argmax(before))

    agent.scale_logits(0.25)
    _, _, after = agent.option_probabilities(_view(), candidates)

    assert int(np.argmax(after)) == before_choice
    assert after.max() < before.max()
