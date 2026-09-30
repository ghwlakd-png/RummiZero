from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random

from .base import Agent
from rummizero.features import FEATURE_NAMES, encode
from rummizero.types import ActionKind, GameView, Transition


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class LinearPolicyAgent(Agent):
    """Tiny policy-gradient bootstrap agent.

    It learns only PLAY_BEST vs DRAW. This validates self-play plumbing before a
    neural action-conditioned policy is introduced.
    """

    weights: list[float] = field(default_factory=lambda: [0.0] * len(FEATURE_NAMES))
    learning_rate: float = 0.02
    training: bool = True

    def choose(self, view: GameView, rng: random.Random) -> tuple[ActionKind, Transition | None]:
        if not view.can_play:
            return ActionKind.DRAW, None
        feats = encode(view)
        logit = sum(w * x for w, x in zip(self.weights, feats, strict=True))
        p = _sigmoid(logit)
        played = 1 if (p >= 0.5 if not self.training else rng.random() < p) else 0
        action = ActionKind.PLAY_BEST if played else ActionKind.DRAW
        tr = Transition(feats, played, p, view.player_id) if self.training else None
        return action, tr

    def update(self, trajectory: tuple[Transition, ...], won: bool | None) -> None:
        if not self.training or won is None or not trajectory:
            return
        advantage = 1.0 if won else -1.0
        for tr in trajectory:
            grad = tr.played - tr.play_probability
            for i, x in enumerate(tr.features):
                self.weights[i] += self.learning_rate * advantage * grad * x
        self.weights[:] = [max(-12.0, min(12.0, w)) for w in self.weights]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "version": 1,
                    "feature_names": FEATURE_NAMES,
                    "weights": self.weights,
                    "learning_rate": self.learning_rate,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, training: bool = False) -> "LinearPolicyAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["feature_names"]) != FEATURE_NAMES:
            raise ValueError("checkpoint feature schema does not match this RummiZero version")
        return cls(
            weights=[float(x) for x in data["weights"]],
            learning_rate=float(data.get("learning_rate", 0.02)),
            training=training,
        )
