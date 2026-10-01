from __future__ import annotations

from .belief import STANDARD_TILE_TYPES, encode_belief
from .candidates import CandidateAction
from .features import FEATURE_NAMES, encode
from .types import GameView

BELIEF_FEATURE_NAMES = tuple(
    f"opponent_belief_tile_{tile_id:02d}"
    for tile_id in range(1, STANDARD_TILE_TYPES + 1)
)

HISTORY_FEATURE_NAMES = (
    "opponent_last_play_mean",
    "opponent_last_play_max",
    "opponent_draw_streak_mean",
    "opponent_draw_streak_max",
)

ACTION_FEATURE_NAMES = (
    "draw_action",
    "rack_tiles_used",
    "rack_tiles_remaining",
    "final_set_count",
    "final_table_tiles",
    "set_count_delta",
    "joker_play_fraction",
    "free_jokers",
    "max_set_length",
)

STATE_ACTION_FEATURE_NAMES = (
    FEATURE_NAMES
    + BELIEF_FEATURE_NAMES
    + HISTORY_FEATURE_NAMES
    + ACTION_FEATURE_NAMES
)


def _history_features(view: GameView) -> tuple[float, ...]:
    plays = view.opponent_last_play_tiles or (0,)
    streaks = view.opponent_draw_streaks or (0,)
    return (
        (sum(plays) / len(plays)) / 14.0,
        max(plays) / 14.0,
        (sum(streaks) / len(streaks)) / 10.0,
        max(streaks) / 10.0,
    )


def encode_state_action(
    view: GameView,
    candidate: CandidateAction | None,
) -> tuple[float, ...]:
    state = encode(view) + encode_belief(view) + _history_features(view)
    if candidate is None:
        action = (
            1.0,
            0.0,
            len(view.rack) / 20.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        return state + action

    used = len(candidate.rack_tiles)
    final_sets = candidate.table_sets
    final_table_tiles = sum(len(s) for s in final_sets)
    old_set_count = len(view.table_sets)
    joker_plays = (
        sum(1 for t in candidate.rack_tiles if t == view.joker_tile_id)
        if view.joker_tile_id is not None
        else 0
    )
    max_set_len = max((len(s) for s in final_sets), default=0)
    action = (
        0.0,
        used / 14.0,
        max(0, len(view.rack) - used) / 20.0,
        len(final_sets) / 20.0,
        final_table_tiles / 106.0,
        (len(final_sets) - old_set_count) / 10.0,
        joker_plays / 2.0,
        candidate.free_jokers / 2.0,
        max_set_len / 13.0,
    )
    return state + action
