# RummiZero roadmap

## v0.1 — bootstrap pipeline — DONE
- headless multiplayer simulator
- exact solver baseline
- information-separated observation
- trajectory collection
- tiny learning policy
- Elo arena
- league checkpoints

## v0.2 — Full-Turn Candidate Generator — IN PROGRESS

### v0.2a — rack-subset candidate generation
- generate many legal complete-turn alternatives
- no fixed max-3-rack-tile action restriction
- exact-subset verification through MILP
- canonical action keys and deduplication
- configurable candidate and solver-call budgets
- unit and real-solver integration tests

### v0.2b — multiple arrangements per rack subset
- native exact-cover / arrangement enumeration
- multiple canonical table layouts for the same used rack tiles
- structural prefilters
- memoization / transposition cache
- top-K pruning for self-play speed
- property tests for tile conservation and legal sets

## v0.3 — Action-conditioned neural policy
- state encoder + candidate action encoder
- Deep Monte-Carlo baseline inspired by DouZero
- parallel CPU actors and GPU learner
- replay buffer and frozen evaluation opponents

## v0.4 — League self-play
- current policy vs historical snapshots and exploiters
- Elo/TrueSkill-style ratings
- automatic promotion gates
- deterministic benchmark seeds

## v0.5 — imperfect-information strength
- remaining-tile belief model
- opponent rack-size/action-history features
- recurrent or Transformer history encoder
- NFSP/DeepCFR experiments

## v0.6 — search
- determinization / information-set MCTS
- neural value/action priors
- latency budget suitable for live recommendation

## integration
- adapter from LDPlayer/OpenCV recognition to GameView
- return top-3 candidates with value/confidence
- preserve the existing Rummikub Guide UI
