from rummizero.features import FEATURE_NAMES, encode
from rummizero.types import GameView, SolverMove


def test_feature_shape_and_separation():
    view = GameView(
        player_id=0,
        rack=(1, 2, 53),
        table_sets=((3, 4, 5),),
        stock_count=60,
        opponent_rack_counts=(12,),
        opening_done=(False, True),
        turn_index=4,
        consecutive_passes=0,
        solver_move=SolverMove((1, 2), ((1, 2, 3),)),
        joker_tile_id=53,
    )
    f = encode(view)
    assert len(f) == len(FEATURE_NAMES)
    assert f[3] > 0
    assert f[6] > 0
