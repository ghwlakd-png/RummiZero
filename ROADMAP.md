# RummiZero roadmap

## v0.1 — bootstrap pipeline (this package)
- headless multiplayer simulator
- exact solver baseline
- legal information-separated observation
- trajectory collection
- tiny learning policy
- Elo arena
- league checkpoints

## v0.2 — Full-Turn Candidate Generator
- emit multiple complete final table arrangements, not only one best MILP result
- remove any fixed "max 3 rack tiles" restriction
- canonical action encoding independent of set ordering
- deduplicate equivalent arrangements
- candidate pruning/top-K without losing forced tactical moves
- property tests: inventory conservation + every set legal + all old table tiles preserved

## v0.3 — Action-conditioned neural policy
- input: state encoder + candidate action encoder
- Deep Monte-Carlo baseline inspired by DouZero
- parallel CPU actors, GPU learner
- replay buffer and frozen evaluation opponents

## v0.4 — League self-play
- current policy vs historical snapshots and exploiters
- Elo/TrueSkill-style ratings
- automatic promotion gates
- deterministic benchmark seeds

## v0.5 — imperfect-information strength
- remaining-tile belief model
- opponent rack-size/action-history features
- optional recurrent/Transformer history encoder
- NFSP/DeepCFR experiments

## v0.6 — search
- determinization or information-set MCTS over sampled hidden racks
- neural value/action priors
- latency budget suitable for live recommendation

## integration
- adapter from LDPlayer/OpenCV recognition output to `GameView`
- return top-3 candidate actions with confidence/value
- preserve current Rummikub Guide UI as the presentation layer
