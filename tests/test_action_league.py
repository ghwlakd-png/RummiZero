import json
import random

from rummizero.agents import CandidatePolicyAgent
from rummizero.training.action_league import ActionLeague, promotion_score


def test_action_league_snapshots_refresh_and_sample(tmp_path):
    agent = CandidatePolicyAgent(hidden_size=4, seed=1)
    league = ActionLeague(tmp_path)
    assert league.latest is None

    p0 = league.snapshot(agent, 0)
    p10 = league.snapshot(agent, 10)
    assert p0.exists()
    assert p10.exists()
    assert league.latest == p10
    assert league.sample(random.Random(1)) in {p0, p10}

    reloaded = ActionLeague(tmp_path)
    assert reloaded.snapshots == [p0, p10]


def test_promotion_score_counts_draw_as_half():
    result = {"games": 10, "a_wins": 5, "b_wins": 3, "draws": 2}
    assert promotion_score(result) == 0.6


def test_first_challenger_bootstraps_champion(tmp_path):
    challenger = tmp_path / "challenger.json"
    CandidatePolicyAgent(hidden_size=4, seed=7, training=False).save(challenger)

    league = ActionLeague(tmp_path / "league")
    result = league.consider_challenger(challenger)

    assert result["bootstrapped"] is True
    assert result["promoted"] is True
    assert league.champion_path.exists()

    challenger_data = json.loads(challenger.read_text(encoding="utf-8"))
    champion_data = json.loads(league.champion_path.read_text(encoding="utf-8"))
    assert challenger_data["w1"] == champion_data["w1"]
    assert challenger_data["w2"] == champion_data["w2"]
