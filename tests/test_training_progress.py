from rummizero.cli import _parser, _print_league_progress


def test_train_league_progress_flag_parses():
    args = _parser().parse_args(
        ["train-league", "--games", "100", "--progress-every", "5"]
    )
    assert args.command == "train-league"
    assert args.progress_every == 5


def test_progress_printer_writes_human_readable_line(capsys):
    _print_league_progress(
        {
            "game": 10,
            "games": 100,
            "elapsed_seconds": 123.4,
            "seconds_per_game": 12.34,
            "learner_wins": 4,
            "draws": 1,
            "snapshots": 2,
        }
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[train-league] 10/100 games" in captured.err
    assert "12.34s/game" in captured.err
