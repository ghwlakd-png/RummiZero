from types import SimpleNamespace

import rummizero.arena as arena


class FakeSimulator:
    seeds = []

    def __init__(self, agents, *, seed, **kwargs):
        self.agents = agents
        self.seed = seed
        FakeSimulator.seeds.append(seed)

    def run(self):
        return SimpleNamespace(winner=0)


def test_duel_reuses_each_deal_with_seats_swapped(monkeypatch):
    FakeSimulator.seeds = []
    monkeypatch.setattr(arena, "GameSimulator", FakeSimulator)

    result = arena.duel(
        lambda: object(),
        lambda: object(),
        games=4,
        seed=123,
        paired_deals=True,
    )

    assert FakeSimulator.seeds[0] == FakeSimulator.seeds[1]
    assert FakeSimulator.seeds[2] == FakeSimulator.seeds[3]
    assert FakeSimulator.seeds[0] != FakeSimulator.seeds[2]
    assert result["a_wins"] == 2
    assert result["b_wins"] == 2
    assert result["a_score"] == 0.5
    assert result["paired_deals"] is True
