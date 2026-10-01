from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

from .base import Agent
from rummizero.action_features import STATE_ACTION_FEATURE_NAMES, encode_state_action
from rummizero.candidates import CandidateAction
from rummizero.types import ActionKind, CandidateTransition, GameView, Transition


class CandidatePolicyAgent(Agent):
    """Small action-conditioned neural policy."""

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        learning_rate: float = 0.01,
        training: bool = True,
        seed: int = 0,
        max_grad_norm: float = 1.0,
    ) -> None:
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.training = training
        self.max_grad_norm = max_grad_norm
        rng = np.random.default_rng(seed)
        n_in = len(STATE_ACTION_FEATURE_NAMES)
        scale = 1.0 / max(1, n_in) ** 0.5
        self.w1 = rng.normal(0.0, scale, size=(n_in, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=float)
        self.w2 = rng.normal(0.0, 1.0 / hidden_size**0.5, size=hidden_size)
        self.b2 = 0.0

    def choose(
        self,
        view: GameView,
        rng: random.Random,
    ) -> tuple[ActionKind, Transition | None]:
        return (ActionKind.PLAY_BEST if view.can_play else ActionKind.DRAW), None

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exp = np.exp(np.clip(shifted, -60.0, 60.0))
        return exp / exp.sum()

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.w1 + self.b1)
        scores = hidden @ self.w2 + self.b2
        return hidden, scores

    def option_probabilities(
        self,
        view: GameView,
        candidates: tuple[CandidateAction, ...],
    ) -> tuple[tuple[CandidateAction | None, ...], np.ndarray, np.ndarray]:
        options: tuple[CandidateAction | None, ...] = tuple(candidates) + (None,)
        x = np.asarray(
            [encode_state_action(view, candidate) for candidate in options],
            dtype=float,
        )
        _, scores = self._forward(x)
        probabilities = self._softmax(scores)
        return options, x, probabilities

    def choose_candidate(
        self,
        view: GameView,
        candidates: tuple[CandidateAction, ...],
        rng: random.Random,
    ) -> tuple[CandidateAction | None, CandidateTransition | None]:
        options, x, probabilities = self.option_probabilities(view, candidates)
        if self.training:
            draw = rng.random()
            cumulative = 0.0
            chosen_index = len(options) - 1
            for i, probability in enumerate(probabilities):
                cumulative += float(probability)
                if draw <= cumulative:
                    chosen_index = i
                    break
        else:
            chosen_index = int(np.argmax(probabilities))

        transition = None
        if self.training:
            transition = CandidateTransition(
                option_features=tuple(tuple(float(v) for v in row) for row in x),
                chosen_index=chosen_index,
                probabilities=tuple(float(v) for v in probabilities),
                player_id=view.player_id,
            )
        return options[chosen_index], transition

    def update(
        self,
        trajectory: tuple[CandidateTransition, ...],
        won: bool | None,
    ) -> None:
        """Apply one normalized REINFORCE update per completed game.

        The previous implementation updated once per turn, so a 100-turn game
        could move the policy roughly 100x more than a short game. It also
        changed the weights between transitions from the same trajectory.
        Accumulating gradients against one frozen set of weights, averaging by
        trajectory length, and clipping the final gradient makes each game a
        bounded learning step.
        """

        if not self.training or won is None or not trajectory:
            return

        advantage = 1.0 if won else -1.0
        grad_w1 = np.zeros_like(self.w1)
        grad_b1 = np.zeros_like(self.b1)
        grad_w2 = np.zeros_like(self.w2)
        grad_b2 = 0.0

        for tr in trajectory:
            x = np.asarray(tr.option_features, dtype=float)
            hidden, scores = self._forward(x)
            probabilities = self._softmax(scores)

            grad_scores = probabilities.copy()
            grad_scores[tr.chosen_index] -= 1.0
            grad_scores *= advantage

            grad_w2 += hidden.T @ grad_scores
            grad_b2 += float(grad_scores.sum())
            grad_hidden = np.outer(grad_scores, self.w2)
            grad_pre = grad_hidden * (1.0 - hidden * hidden)
            grad_w1 += x.T @ grad_pre
            grad_b1 += grad_pre.sum(axis=0)

        scale = 1.0 / len(trajectory)
        grad_w1 *= scale
        grad_b1 *= scale
        grad_w2 *= scale
        grad_b2 *= scale

        grad_norm = float(
            np.sqrt(
                np.sum(grad_w1 * grad_w1)
                + np.sum(grad_b1 * grad_b1)
                + np.sum(grad_w2 * grad_w2)
                + grad_b2 * grad_b2
            )
        )
        if grad_norm > self.max_grad_norm:
            clip = self.max_grad_norm / grad_norm
            grad_w1 *= clip
            grad_b1 *= clip
            grad_w2 *= clip
            grad_b2 *= clip

        lr = self.learning_rate
        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        self.w1 -= lr * grad_w1
        self.b1 -= lr * grad_b1

        np.clip(self.w1, -8.0, 8.0, out=self.w1)
        np.clip(self.w2, -8.0, 8.0, out=self.w2)
        np.clip(self.b1, -8.0, 8.0, out=self.b1)
        self.b2 = float(max(-8.0, min(8.0, self.b2)))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "version": 3,
                    "feature_names": STATE_ACTION_FEATURE_NAMES,
                    "hidden_size": self.hidden_size,
                    "learning_rate": self.learning_rate,
                    "max_grad_norm": self.max_grad_norm,
                    "w1": self.w1.tolist(),
                    "b1": self.b1.tolist(),
                    "w2": self.w2.tolist(),
                    "b2": self.b2,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        training: bool = False,
    ) -> "CandidatePolicyAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["feature_names"]) != STATE_ACTION_FEATURE_NAMES:
            raise ValueError("checkpoint feature schema does not match this RummiZero version")
        agent = cls(
            hidden_size=int(data["hidden_size"]),
            learning_rate=float(data.get("learning_rate", 0.01)),
            training=training,
            seed=0,
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
        )
        agent.w1 = np.asarray(data["w1"], dtype=float)
        agent.b1 = np.asarray(data["b1"], dtype=float)
        agent.w2 = np.asarray(data["w2"], dtype=float)
        agent.b2 = float(data["b2"])
        return agent
