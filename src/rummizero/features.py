from __future__ import annotations

from .types import GameView

FEATURE_NAMES = (
    "bias",
    "rack_size",
    "stock_fraction",
    "best_move_tiles",
    "opponent_min_rack",
    "opponent_mean_rack",
    "joker_fraction",
    "opening_done",
    "late_game",
)


def encode(view: GameView, deck_size: int = 106) -> tuple[float, ...]:
    opp = view.opponent_rack_counts or (0,)
    move_tiles = len(view.solver_move.rack_tiles) if view.solver_move else 0
    jokers = (
        sum(1 for t in view.rack if t == view.joker_tile_id)
        if view.joker_tile_id is not None
        else 0
    )
    return (
        1.0,
        len(view.rack) / 20.0,
        view.stock_count / float(deck_size),
        move_tiles / 14.0,
        min(opp) / 20.0,
        (sum(opp) / len(opp)) / 20.0,
        jokers / 2.0,
        1.0 if view.opening_done[view.player_id] else 0.0,
        1.0 if view.stock_count <= 20 else 0.0,
    )
