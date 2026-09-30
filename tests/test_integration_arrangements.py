import pytest

pytest.importorskip("rummikub_solver")

from rummizero.backend import SolverBackend


def test_real_ruleset_can_rearrange_same_pool_multiple_ways():
    backend = SolverBackend()
    result = backend.enumerate_arrangements(
        (1, 2, 3, 4, 5, 6, 7, 8),
        (),
        True,
        limit=10,
        node_budget=5000,
    )
    layouts = set(result.arrangements)
    assert ((1, 2, 3), (4, 5, 6, 7, 8)) in layouts
    assert ((1, 2, 3, 4), (5, 6, 7, 8)) in layouts
    assert ((1, 2, 3, 4, 5), (6, 7, 8)) in layouts
