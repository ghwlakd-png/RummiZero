import json

from rummizero.agents import CandidatePolicyAgent
from rummizero.training import experiment


def test_auto_experiment_runs_rounds_and_resumes(monkeypatch, tmp_path):
    baseline = tmp_path / "baseline.json"
    CandidatePolicyAgent(hidden_size=4, seed=1).save(baseline)
    calls = []

    def fake_train(games, **kwargs):
        calls.append((games, kwargs["resume_from"], kwargs["learning_rate"]))
        return CandidatePolicyAgent.load(kwargs["resume_from"], training=True), {
            "games": games,
        }

    scores = iter((0.40, 0.50, 0.60))

    def fake_consider(self, challenger, **kwargs):
        score = next(scores)
        if score >= kwargs["threshold"]:
            self.set_champion(CandidatePolicyAgent.load(challenger, training=False))
        return {
            "score": score,
            "promoted": score >= kwargs["threshold"],
            "games": kwargs["games"],
        }

    monkeypatch.setattr(experiment, "train_candidate_league", fake_train)
    monkeypatch.setattr(experiment.ActionLeague, "consider_challenger", fake_consider)
    work = tmp_path / "auto"
    first = experiment.run_league_experiments(
        baseline=baseline,
        work_dir=work,
        rounds=2,
        games_per_round=4,
        promotion_games=2,
        seed=11,
        learning_rate=0.004,
    )
    assert first["completed_rounds"] == 2
    assert [call[2] for call in calls] == [0.004, 0.002]
    assert first["next_learning_rate"] == 0.003

    resumed = experiment.run_league_experiments(
        baseline=baseline,
        work_dir=work,
        rounds=3,
        games_per_round=4,
        promotion_games=2,
        seed=11,
        learning_rate=0.004,
    )
    assert resumed["completed_rounds"] == 3
    assert len(calls) == 3
    assert calls[-1][2] == 0.003
    assert calls[-1][1].name == "challenger_round_002.json"
    saved = json.loads((work / "experiment_state.json").read_text(encoding="utf-8"))
    assert len(saved["rounds"]) == 3
    assert saved["promotions"] == 1


def test_auto_experiment_cli_defaults():
    from rummizero.cli import _parser

    args = _parser().parse_args(
        ["auto-experiment", "--baseline", "base.json", "--work-dir", "work"]
    )
    assert args.rounds == 3
    assert args.games_per_round == 20
    assert args.promotion_games == 40
    assert args.continuation_floor == 0.45
    assert args.initial_logit_scale == 1.0
