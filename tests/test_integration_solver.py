import pytest

pytest.importorskip("rummikub_solver")

from rummizero.agents import RandomDelayAgent, SolverAgent
from rummizero.simulator import GameSimulator


def test_headless_game_finishes_and_conserves_tiles():
    result = GameSimulator(
        [SolverAgent(), RandomDelayAgent(0.6)],
        seed=123,
        validate_inventory=True,
        max_turns=1000,
    ).run()
    assert result.turns > 0
    assert len(result.rack_sizes) == 2
