import random

from rummizero.agents.linear import LinearPolicyAgent
from rummizero.types import GameView, SolverMove


def _view():
    return GameView(
        player_id=0,
        rack=tuple(range(1, 11)),
        table_sets=(),
        stock_count=50,
        opponent_rack_counts=(8,),
        opening_done=(True, True),
        turn_index=20,
        consecutive_passes=0,
        solver_move=SolverMove((1, 2, 3), ((1, 2, 3),)),
        joker_tile_id=53,
    )


def test_linear_checkpoint(tmp_path):
    a = LinearPolicyAgent(training=True)
    _, tr = a.choose(_view(), random.Random(1))
    assert tr is not None
    a.update((tr,), won=True)
    p = tmp_path / "a.json"
    a.save(p)
    b = LinearPolicyAgent.load(p)
    assert b.weights == a.weights
