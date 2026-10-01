from rummizero.benchmark import benchmark_selfplay


def test_benchmark_smoke_runs_end_to_end():
    result = benchmark_selfplay(
        games=1,
        seed=3,
        starting_tiles=3,
        max_turns=4,
        candidate_max_candidates=2,
        candidate_max_solver_calls=4,
    )
    assert result["games"] == 1
    assert result["turns"] <= 4
    assert result["seconds_per_game"] >= 0.0
    assert sum(result["wins"].values()) == 1
