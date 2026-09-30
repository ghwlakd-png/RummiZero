from rummizero.training.elo import expected, update


def test_elo_is_symmetric():
    assert expected(1000, 1000) == 0.5
    a, b = update(1000, 1000, 1.0)
    assert a > 1000
    assert b < 1000
    assert round(a + b, 8) == 2000
