# RummiZero roadmap

## v0.1 — bootstrap pipeline — DONE
- headless multiplayer simulator
- exact solver baseline
- information-separated observation
- trajectory collection
- tiny learning policy
- Elo arena
- league checkpoints

## v0.2 — Full-Turn Candidate Generator — DONE
- rack-subset candidate generation
- multiple complete table arrangements for the same rack subset
- exact-cover enumeration
- candidate budgets and deduplication
- integration tests

## v0.3 — Action-conditioned neural policy — DONE (MVP)
- state encoder + candidate action encoder
- small neural scorer
- candidate softmax selection including draw
- terminal policy-gradient update
- checkpoint save/load

## v0.4 — League self-play — DONE (MVP)
- frozen historical checkpoint pool
- current policy vs historical snapshots / solver / current policy
- periodic action-policy snapshots
- persistent champion
- automatic promotion gate
- deterministic evaluation seeds

## v0.5 — Imperfect-information strength — IN PROGRESS
- uniform remaining-tile belief vector from private rack + public table
- public opponent action-history features
- version-isolated v0.5 league/checkpoints
- next: action-conditioned belief updates and longer history encoder
- later: NFSP/DeepCFR experiments

## v0.6 — search
- determinization / information-set MCTS
- neural value/action priors
- latency budget suitable for live recommendation

## integration
- adapter from LDPlayer/OpenCV recognition to GameView
- return top-3 candidates with value/confidence
- preserve the existing Rummikub Guide UI
