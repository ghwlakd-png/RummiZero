from __future__ import annotations

from collections import Counter
import random
from typing import Sequence

from .agents.base import Agent
from .backend import SolverBackend
from .candidates import CandidateAction, FullTurnCandidateGenerator
from .types import ActionKind, GameResult, GameView, Transition


def _solver_candidate(move) -> CandidateAction:
    return CandidateAction(
        rack_tiles=tuple(sorted(move.rack_tiles)),
        table_sets=tuple(tuple(s) for s in move.table_sets),
        free_jokers=move.free_jokers,
    )


def _with_solver_candidate(
    move,
    candidates: tuple[CandidateAction, ...],
    limit: int,
) -> tuple[CandidateAction, ...]:
    """Guarantee the exact solver move is available to candidate policies."""

    target = _solver_candidate(move)
    merged = [target]
    for candidate in candidates:
        if candidate.canonical_key == target.canonical_key:
            continue
        merged.append(candidate)
        if len(merged) >= limit:
            break
    return tuple(merged)


class GameSimulator:
    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        seed: int = 0,
        starting_tiles: int = 14,
        max_turns: int = 2000,
        validate_inventory: bool = False,
        candidate_max_candidates: int = 16,
        candidate_max_solver_calls: int = 64,
        candidate_arrangements_per_subset: int = 4,
        candidate_arrangement_node_budget: int = 5000,
    ) -> None:
        if not 2 <= len(agents) <= 4:
            raise ValueError("RummiZero currently supports 2 to 4 players")
        self.agents = tuple(agents)
        self.rng = random.Random(seed)
        self.backend = SolverBackend()
        self.candidate_generator = FullTurnCandidateGenerator(
            self.backend,
            max_candidates=candidate_max_candidates,
            max_solver_calls=candidate_max_solver_calls,
            max_arrangements_per_subset=candidate_arrangements_per_subset,
            arrangement_node_budget=candidate_arrangement_node_budget,
        )
        self.starting_tiles = starting_tiles
        self.max_turns = max_turns
        self.validate_inventory = validate_inventory

    @staticmethod
    def _consume_rack_tiles(rack: list[int], tiles: tuple[int, ...]) -> list[int]:
        rack_counts = Counter(rack)
        needed = Counter(tiles)
        if needed - rack_counts:
            raise AssertionError("candidate proposed tiles not present in rack")
        rack_counts.subtract(needed)
        return list(rack_counts.elements())

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
        last_play_tiles = [0] * len(self.agents)
        draw_streaks = [0] * len(self.agents)

        winner: int | None = None
        turn = 0
        while turn < self.max_turns:
            pid = turn % len(self.agents)
            rack = racks[pid]
            move = self.backend.solve(rack, table_sets, opening_done[pid])
            opponents = tuple(len(racks[i]) for i in range(len(racks)) if i != pid)
            opponent_last_play_tiles = tuple(
                last_play_tiles[i] for i in range(len(racks)) if i != pid
            )
            opponent_draw_streaks = tuple(
                draw_streaks[i] for i in range(len(racks)) if i != pid
            )
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
                opponent_last_play_tiles=opponent_last_play_tiles,
                opponent_draw_streaks=opponent_draw_streaks,
            )

            agent = self.agents[pid]
            candidate_choose = getattr(agent, "choose_candidate", None)
            played = False
            tiles_played = 0

            if callable(candidate_choose):
                if move is None:
                    candidates = ()
                else:
                    generated = self.candidate_generator.generate(
                        rack,
                        table_sets,
                        opening_done=opening_done[pid],
                        preferred_move=move,
                    )
                    candidates = _with_solver_candidate(
                        move,
                        generated.candidates,
                        self.candidate_generator.max_candidates,
                    )
                candidate, transition = candidate_choose(view, candidates, self.rng)
                if transition is not None:
                    trajectories[pid].append(transition)

                if candidate is not None and candidate.rack_tiles:
                    racks[pid] = self._consume_rack_tiles(rack, candidate.rack_tiles)
                    table_sets = candidate.table_sets
                    opening_done[pid] = True
                    played = True
                    tiles_played = len(candidate.rack_tiles)
                    if not racks[pid]:
                        winner = pid
                elif deck:
                    racks[pid].append(deck.pop())
            else:
                action, transition = agent.choose(view, self.rng)
                if transition is not None:
                    trajectories[pid].append(transition)

                if action is ActionKind.PLAY_BEST and move is not None and move.rack_tiles:
                    racks[pid] = self._consume_rack_tiles(rack, move.rack_tiles)
                    table_sets = move.table_sets
                    opening_done[pid] = True
                    played = True
                    tiles_played = len(move.rack_tiles)
                    if not racks[pid]:
                        winner = pid
                elif deck:
                    racks[pid].append(deck.pop())

            if played:
                last_play_tiles[pid] = tiles_played
                draw_streaks[pid] = 0
            else:
                last_play_tiles[pid] = 0
                draw_streaks[pid] += 1

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
