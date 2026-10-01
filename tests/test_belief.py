import pytest

from rummizero.belief import (
    STANDARD_TILE_TYPES,
    encode_belief,
    expected_opponent_tile_counts,
    unseen_tile_counts,
)
from rummizero.types import GameView


def _view():
    return GameView(
        player_id=0,
        rack=(1, 2),
        table_sets=((3, 3),),
        stock_count=88,
        opponent_rack_counts=(14,),
        opening_done=(True, True),
        turn_index=5,
        consecutive_passes=0,
        solver_move=None,
        joker_tile_id=53,
        opponent_last_play_tiles=(4,),
        opponent_draw_streaks=(2,),
    )


def test_unseen_counts_remove_private_and_public_tiles():
    unseen = unseen_tile_counts(_view())
    assert len(unseen) == STANDARD_TILE_TYPES
    assert unseen[0] == 1.0
    assert unseen[1] == 1.0
    assert unseen[2] == 0.0
    assert unseen[52] == 2.0


def test_uniform_belief_preserves_expected_opponent_rack_size():
    expected = expected_opponent_tile_counts(_view())
    assert len(expected) == STANDARD_TILE_TYPES
    assert sum(expected) == pytest.approx(14.0)
    assert expected[2] == 0.0


def test_belief_encoder_is_bounded():
    encoded = encode_belief(_view())
    assert len(encoded) == STANDARD_TILE_TYPES
    assert all(0.0 <= value <= 1.0 for value in encoded)
