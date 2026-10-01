from pathlib import Path

from rummizero.cli import _parser
from rummizero.training.selfplay import _snapshot_generation


def test_train_league_resume_flag_parses():
    args = _parser().parse_args(
        [
            "train-league",
            "--games",
            "90",
            "--resume",
            r"models\action_smoke10.json",
            "--league-dir",
            r"models\league_smoke10",
        ]
    )
    assert args.resume == Path(r"models\action_smoke10.json")
    assert args.games == 90


def test_snapshot_generation_parses_checkpoint_suffix():
    assert _snapshot_generation(Path("action_gen_000010.json")) == 10
    assert _snapshot_generation(Path("not_a_snapshot.json")) is None
