def test_cli_import_and_benchmark_parser_do_not_cycle():
    from rummizero.cli import _parser

    args = _parser().parse_args(
        [
            "benchmark",
            "--games",
            "1",
            "--seed",
            "7",
            "--max-turns",
            "4",
            "--max-candidates",
            "2",
            "--max-solver-calls",
            "4",
        ]
    )
    assert args.command == "benchmark"
    assert args.games == 1
