from rummizero.action_features import STATE_ACTION_FEATURE_NAMES, encode_state_action
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
        solver_move=SolverMove((1, 2, 3), ((1, 2, 3),), 0),
        joker_tile_id=53,
    )


def test_state_action_encoding_has_fixed_shape():
    view = _view()
    candidate = CandidateAction((1, 2, 3), ((1, 2, 3),), 0)
    play = encode_state_action(view, candidate)
    draw = encode_state_action(view, None)
    assert len(play) == len(STATE_ACTION_FEATURE_NAMES)
    assert len(draw) == len(STATE_ACTION_FEATURE_NAMES)
    assert play != draw
