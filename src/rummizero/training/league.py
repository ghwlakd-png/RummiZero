from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rummizero.agents.linear import LinearPolicyAgent


@dataclass
class League:
    root: Path
    snapshots: list[Path] = field(default_factory=list)

    def snapshot(self, agent: LinearPolicyAgent, generation: int) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"linear_gen_{generation:06d}.json"
        agent.save(path)
        self.snapshots.append(path)
        return path
