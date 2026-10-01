from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from rummizero.agents import CandidatePolicyAgent

from .action_league import ActionLeague
from .selfplay import train_candidate_league


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _next_learning_rate(
    current: float,
    score: float,
    *,
    threshold: float,
    continuation_floor: float,
    minimum: float,
    maximum: float,
) -> float:
    """Conservatively adapt the next experiment after a completed gate."""

    if score < continuation_floor:
        return max(minimum, current * 0.5)
    if score < threshold:
        return min(maximum, current * 1.5)
    return current


def run_league_experiments(
    *,
    baseline: Path,
    work_dir: Path,
    rounds: int,
    games_per_round: int,
    promotion_games: int,
    seed: int,
    learning_rate: float,
    minimum_learning_rate: float = 0.0001,
    maximum_learning_rate: float = 0.02,
    promotion_threshold: float = 0.55,
    continuation_floor: float = 0.45,
    players: int = 2,
    snapshot_every: int = 10,
    historical_fraction: float = 0.4,
    solver_fraction: float = 0.4,
    candidate_max_candidates: int = 4,
    candidate_max_solver_calls: int = 8,
    max_turns: int = 300,
    training_temperature: float = 1.0,
    initial_logit_scale: float = 1.0,
    progress_every: int = 0,
    training_progress_callback: Callable[[dict[str, Any]], None] | None = None,
    evaluation_progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run bounded train/evaluate/promote rounds without human intervention.

    State is written after every promotion gate. Re-running the same command
    resumes at the next unfinished round, and never replaces the accepted
    champion without clearing the promotion gate. Non-catastrophic challengers
    may continue as a separate research lineage.
    """

    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if games_per_round < 1:
        raise ValueError("games_per_round must be >= 1")
    if promotion_games < 2 or promotion_games % 2:
        raise ValueError("promotion_games must be an even number >= 2")
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    if not 0 < minimum_learning_rate <= learning_rate <= maximum_learning_rate:
        raise ValueError("learning-rate bounds are inconsistent")
    if initial_logit_scale <= 0:
        raise ValueError("initial_logit_scale must be > 0")
    if not 0.0 <= continuation_floor <= promotion_threshold:
        raise ValueError("continuation_floor must be within [0, promotion_threshold]")

    work_dir.mkdir(parents=True, exist_ok=True)
    league = ActionLeague(work_dir / "league")
    state_path = work_dir / "experiment_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = len(state.get("rounds", []))
        current_learning_rate = float(state["next_learning_rate"])
    else:
        completed = 0
        current_learning_rate = learning_rate
        state = {
            "version": 1,
            "baseline": str(baseline),
            "champion": str(league.champion_path),
            "requested_rounds": rounds,
            "games_per_round": games_per_round,
            "promotion_games": promotion_games,
            "initial_logit_scale": initial_logit_scale,
            "continuation_floor": continuation_floor,
            "rounds": [],
            "next_learning_rate": current_learning_rate,
        }

    state["requested_rounds"] = rounds
    if not league.has_champion:
        champion = CandidatePolicyAgent.load(baseline, training=False)
        champion.scale_logits(initial_logit_scale)
        league.set_champion(champion)

    if "training_parent" in state:
        training_parent = Path(state["training_parent"])
    elif state["rounds"] and float(state["rounds"][-1]["promotion"]["score"]) >= continuation_floor:
        training_parent = Path(state["rounds"][-1]["challenger"])
    else:
        training_parent = league.champion_path

    for round_index in range(completed + 1, rounds + 1):
        round_seed = seed + (round_index - 1) * 1_000_003
        challenger = work_dir / f"challenger_round_{round_index:03d}.json"
        agent, training = train_candidate_league(
            games_per_round,
            players=players,
            seed=round_seed,
            learning_rate=current_learning_rate,
            league_dir=league.root,
            snapshot_every=snapshot_every,
            historical_fraction=historical_fraction,
            solver_fraction=solver_fraction,
            candidate_max_candidates=candidate_max_candidates,
            candidate_max_solver_calls=candidate_max_solver_calls,
            max_turns=max_turns,
            training_temperature=training_temperature,
            progress_every=progress_every,
            progress_callback=training_progress_callback,
            resume_from=training_parent,
        )
        agent.save(challenger)
        promotion = league.consider_challenger(
            challenger,
            games=promotion_games,
            seed=round_seed + 500_009,
            threshold=promotion_threshold,
            candidate_max_candidates=candidate_max_candidates,
            candidate_max_solver_calls=candidate_max_solver_calls,
            max_turns=max_turns,
            progress_every=progress_every,
            progress_callback=evaluation_progress_callback,
        )
        score = float(promotion["score"])
        next_learning_rate = _next_learning_rate(
            current_learning_rate,
            score,
            threshold=promotion_threshold,
            continuation_floor=continuation_floor,
            minimum=minimum_learning_rate,
            maximum=maximum_learning_rate,
        )
        if promotion["promoted"]:
            next_training_parent = league.champion_path
        elif score >= continuation_floor:
            next_training_parent = challenger
        else:
            next_training_parent = league.champion_path
        state["rounds"].append(
            {
                "round": round_index,
                "seed": round_seed,
                "learning_rate": current_learning_rate,
                "challenger": str(challenger),
                "resumed_from": str(training_parent),
                "training": training,
                "promotion": promotion,
                "next_learning_rate": next_learning_rate,
            }
        )
        state["next_learning_rate"] = next_learning_rate
        state["training_parent"] = str(next_training_parent)
        state["completed_rounds"] = round_index
        state["promotions"] = sum(
            bool(item["promotion"]["promoted"]) for item in state["rounds"]
        )
        _save_state(state_path, state)
        current_learning_rate = next_learning_rate
        training_parent = next_training_parent

    return {
        **state,
        "state_file": str(state_path),
        "champion": str(league.champion_path),
    }
