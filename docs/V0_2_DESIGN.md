# v0.2 candidate-action design

## Goal

Replace the v0.1 binary choice (play the solver's single best move vs draw) with
many complete legal turn candidates.

## v0.2a: exact rack-subset candidates

For each unique non-empty sub-multiset of the rack:

1. Give that subset plus the current table to the exact MILP solver.
2. Accept it only when the solver can consume every tile in the subset.
3. Store the solver's complete final table arrangement as one candidate action.
4. Canonicalize and deduplicate equivalent actions.
5. Stop at configurable candidate and solver-call budgets.

This can produce actions such as play 3 tiles, play 4 tiles, play 7 tiles,
instead of forcing the single maximum-tile solution.

## Why it matters

The future network can compare long-term strategic alternatives:

- empty the rack aggressively,
- keep flexible connectors,
- hold or release a joker,
- avoid exposing useful table structure,
- react to an opponent who has very few tiles left.

## v0.2a limitation

The upstream MILP API yields one arrangement for a given rack subset. Two
different legal table rearrangements that consume the same subset are therefore
collapsed to one candidate.

## v0.2b

Add a dedicated exact-cover or constrained arrangement enumerator capable of
returning several distinct canonical table layouts for the same rack subset.

## Performance policy

Candidate generation is budgeted by max_solver_calls and max_candidates. A
14-tile rack has 16,383 non-empty subsets before duplicate-tile reduction, so
large-scale self-play will require structural prefilters, caching, and eventually
native candidate generation.
