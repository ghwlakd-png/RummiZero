from __future__ import annotations

from collections import Counter

from .types import GameView

STANDARD_TILE_TYPES = 53
STANDARD_COPIES_PER_TYPE = 2


def unseen_tile_counts(
    view: GameView,
    *,
    tile_types: int = STANDARD_TILE_TYPES,
    copies_per_type: int = STANDARD_COPIES_PER_TYPE,
) -> tuple[float, ...]:
    """Return how many copies of each standard tile are still hidden.

    Hidden means: not in the observing player's rack and not visible on table.
    In standard Rummikub every tile identity, including the joker identity, has
    two physical copies.
    """

    visible = Counter(view.rack)
    for meld in view.table_sets:
        visible.update(meld)

    counts = []
    for tile_id in range(1, tile_types + 1):
        counts.append(float(max(0, copies_per_type - visible[tile_id])))
    return tuple(counts)


def expected_opponent_tile_counts(
    view: GameView,
    *,
    tile_types: int = STANDARD_TILE_TYPES,
    copies_per_type: int = STANDARD_COPIES_PER_TYPE,
) -> tuple[float, ...]:
    """Uniform hidden-information prior for tiles held by all opponents.

    Given only legal public information, each hidden physical tile is initially
    treated as equally likely to occupy an opponent-rack slot or a stock slot.
    The result is the expected number of copies of every tile identity across
    all opponent racks.
    """

    unseen = unseen_tile_counts(
        view,
        tile_types=tile_types,
        copies_per_type=copies_per_type,
    )
    hidden_total = sum(unseen)
    opponent_slots = float(sum(view.opponent_rack_counts))
    if hidden_total <= 0.0 or opponent_slots <= 0.0:
        return (0.0,) * tile_types

    opponent_share = min(1.0, opponent_slots / hidden_total)
    return tuple(count * opponent_share for count in unseen)


def encode_belief(
    view: GameView,
    *,
    tile_types: int = STANDARD_TILE_TYPES,
    copies_per_type: int = STANDARD_COPIES_PER_TYPE,
) -> tuple[float, ...]:
    expected = expected_opponent_tile_counts(
        view,
        tile_types=tile_types,
        copies_per_type=copies_per_type,
    )
    return tuple(value / copies_per_type for value in expected)
