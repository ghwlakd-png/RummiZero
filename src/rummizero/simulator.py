from __future__ import annotations

from collections import Counter
import random
from typing import Sequence

from .agents.base import Agent
from .backend import SolverBackend
from .types import ActionKind, GameResult, GameView, Transition


class GameSimulator:
    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        seed: int = 0,
        starting_tiles: int = 14,
        max_turns: int = 2000,
        validate_inventory: bool = False,
    ) -> None:
        if not 2 <= len(agents) <= 4:
            raise ValueError("RummiZero currently supports 2 to 4 players")
        self.agents = tuple(agents)
        self.rng = random.Random(seed)
        self.backend = SolverBackend()
        self.starting_tiles = starting_tiles
        self.max_turns = max_turns
        self.validate_inventory = validate_inventory

    def run(self) -> GameResult:
        deck = self.backend.fresh_deck()
        self.rng.shuffle(deck)
        racks: list[list[int]] = [[] for _ in self.agents]
        for _ in range(self.starting_tiles):
            for rack in racks:
                rack.append(deck.pop())
        table_sets: tuple[tuple[int, ...], ...] = ()
        opening_done = [False] * len(self.agents)
        trajectories: list[list[Transition]] = [[] for _ in self.agents]
        consecutive_no_play = 0

        winner: int | None = None
        turn = 0
        while turn < self.max_turns:
            pid = turn % len(self.agents)
            rack = racks[pid]
            move = self.backend.solve(rack, table_sets, opening_done[pid])
            opponents = tuple(len(racks[i]) for i in range(len(racks)) if i != pid)
            view = GameView(
                player_id=pid,
                rack=tuple(sorted(rack)),
                table_sets=table_sets,
                stock_count=len(deck),
                opponent_rack_counts=opponents,
                opening_done=tuple(opening_done),
                turn_index=turn,
                consecutive_passes=consecutive_no_play,
                solver_move=move,
                joker_tile_id=self.backend.joker_id,
            )
            action, transition = self.agents[pid].choose(view, self.rng)
            if transition is not None:
                trajectories[pid].append(transition)

            played = False
            if action is ActionKind.PLAY_BEST and move is not None and move.rack_tiles:
                rack_counts = Counter(rack)
                needed = Counter(move.rack_tiles)
                if needed - rack_counts:
                    raise AssertionError("solver proposed tiles not present in rack")
                rack_counts.subtract(needed)
                racks[pid] = list(rack_counts.elements())
                table_sets = move.table_sets
                opening_done[pid] = True
                played = True
                if not racks[pid]:
                    winner = pid
            elif deck:
                racks[pid].append(deck.pop())

            consecutive_no_play = 0 if played else consecutive_no_play + 1
            turn += 1

            if self.validate_inventory:
                self.backend.assert_inventory(racks, table_sets, deck)
            if winner is not None:
                break
            if not deck and consecutive_no_play >= len(self.agents):
                penalties = [self.backend.rack_penalty(r) for r in racks]
                best = min(penalties)
                leaders = [i for i, p in enumerate(penalties) if p == best]
                winner = leaders[0] if len(leaders) == 1 else None
                break

        penalties = tuple(self.backend.rack_penalty(r) for r in racks)
        return GameResult(
            winner=winner,
            turns=turn,
            rack_sizes=tuple(len(r) for r in racks),
            rack_penalties=penalties,
            trajectories=tuple(tuple(t) for t in trajectories),
        )
