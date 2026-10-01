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
- train-action and evaluate-action CLI

## v0.4 — League self-play — IN PROGRESS
- frozen historical checkpoint pool
- current policy vs historical snapshots / solver / current policy
- periodic action-policy snapshots
- head-to-head promotion gate
- deterministic evaluation seeds
- next: persistent champion, automatic promotion and larger benchmark suites

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
