from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

from rummizero.agents import CandidatePolicyAgent, SolverAgent
from rummizero.arena import duel


@dataclass
class ActionLeague:
    """Persistent pool of frozen v0.3+ action-policy checkpoints."""

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
