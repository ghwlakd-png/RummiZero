from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

from rummizero.agents import CandidatePolicyAgent
from rummizero.arena import duel


@dataclass
class ActionLeague:
    """Persistent pool of frozen action-policy checkpoints and one champion."""

    root: Path
    snapshots: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> tuple[Path, ...]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = sorted(self.root.glob("action_gen_*.json"))
        return tuple(self.snapshots)

    def snapshot(self, agent: CandidatePolicyAgent, generation: int) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"action_gen_{generation:06d}.json"
        agent.save(path)
        if path not in self.snapshots:
            self.snapshots.append(path)
            self.snapshots.sort()
        return path

    def sample(self, rng: random.Random) -> Path | None:
        if not self.snapshots:
            return None
        return rng.choice(self.snapshots)

    @property
    def latest(self) -> Path | None:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def champion_path(self) -> Path:
        return self.root / "champion.json"

    @property
    def has_champion(self) -> bool:
        return self.champion_path.exists()

    def set_champion(self, agent: CandidatePolicyAgent) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        agent.save(self.champion_path)
        return self.champion_path

    def consider_challenger(
        self,
        challenger: str | Path,
        *,
        games: int = 100,
        seed: int = 99,
        threshold: float = 0.55,
        candidate_max_candidates: int = 16,
        candidate_max_solver_calls: int = 64,
    ) -> dict[str, Any]:
        """Promote challenger when it clears the head-to-head gate.

        The first challenger bootstraps the league champion without a match.
        Later challengers must meet the configured score threshold.
        """

        challenger = Path(challenger)
        if not challenger.exists():
            raise FileNotFoundError(challenger)

        if not self.has_champion:
            agent = CandidatePolicyAgent.load(challenger, training=False)
            self.set_champion(agent)
            return {
                "bootstrapped": True,
                "promoted": True,
                "challenger": str(challenger),
                "champion": str(self.champion_path),
                "score": None,
                "threshold": threshold,
                "games": 0,
            }

        result = evaluate_promotion(
            challenger,
            self.champion_path,
            games=games,
            seed=seed,
            threshold=threshold,
            candidate_max_candidates=candidate_max_candidates,
            candidate_max_solver_calls=candidate_max_solver_calls,
        )
        if result["promoted"]:
            agent = CandidatePolicyAgent.load(challenger, training=False)
            self.set_champion(agent)
        return {
            **result,
            "bootstrapped": False,
            "challenger": str(challenger),
            "champion": str(self.champion_path),
        }


def promotion_score(result: dict[str, Any]) -> float:
    games = int(result["games"])
    if games <= 0:
        raise ValueError("games must be > 0")
    return (int(result["a_wins"]) + 0.5 * int(result["draws"])) / games


def evaluate_promotion(
    challenger: str | Path,
    champion: str | Path,
    *,
    games: int = 100,
    seed: int = 99,
    threshold: float = 0.55,
    candidate_max_candidates: int = 16,
    candidate_max_solver_calls: int = 64,
) -> dict[str, Any]:
    """Head-to-head gate used before replacing a league champion."""

    result = duel(
        lambda: CandidatePolicyAgent.load(challenger, training=False),
        lambda: CandidatePolicyAgent.load(champion, training=False),
        games=games,
        seed=seed,
        simulator_kwargs={
            "candidate_max_candidates": candidate_max_candidates,
            "candidate_max_solver_calls": candidate_max_solver_calls,
        },
    )
    score = promotion_score(result)
    return {
        **result,
        "score": round(score, 4),
        "threshold": threshold,
        "promoted": score >= threshold,
    }
