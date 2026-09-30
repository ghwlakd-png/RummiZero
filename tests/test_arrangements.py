from rummizero.arrangements import ExactCoverArrangementEnumerator


def test_exact_cover_enumerates_multiple_layouts_for_same_tiles():
    enumerator = ExactCoverArrangementEnumerator(
        [
            (1, 2, 3, 4, 5, 6),
            (1, 2, 3),
            (4, 5, 6),
        ],
        set_values=(21, 6, 15),
    )
    result = enumerator.enumerate_pool((1, 2, 3, 4, 5, 6), limit=10)
    assert result.truncated is False
    assert ((1, 2, 3, 4, 5, 6),) in result.arrangements
    assert ((1, 2, 3), (4, 5, 6)) in result.arrangements


def test_exact_cover_respects_initial_value_filter():
    enumerator = ExactCoverArrangementEnumerator(
        [(1, 2, 3), (10, 11, 12)],
        set_values=(6, 33),
        min_initial_value=30,
    )
    low = enumerator.enumerate_pool((1, 2, 3), min_total_value=30)
    high = enumerator.enumerate_pool((10, 11, 12), min_total_value=30)
    assert low.arrangements == ()
    assert high.arrangements == (((10, 11, 12),),)
