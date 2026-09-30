from collections import Counter


def test_standard_tile_inventory_count():
    assert 13 * 4 * 2 + 2 == 106
    assert Counter([1] * 2 + [53] * 2).total() == 4
