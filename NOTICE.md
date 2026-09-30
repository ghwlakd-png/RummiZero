# Research references / attribution

RummiZero v0.1 is original glue code and does not vendor upstream source code.
It is intentionally designed around the public API of:

- `mjpieters/rummikub-solver` (MIT): exact MILP-based Rummikub solving.
  https://github.com/mjpieters/rummikub-solver
- `w3rr0/Rummikub-AI` (MIT): useful reference for Gymnasium + MaskablePPO and
  C++-accelerated move generation.
  https://github.com/w3rr0/Rummikub-AI
- `kwai/DouZero`: architectural inspiration for action-conditioned learning and
  large-scale self-play in an imperfect-information shedding game.
  https://github.com/kwai/DouZero
- `datamllab/rlcard`: reference for NFSP/CFR/DeepCFR style imperfect-information
  training infrastructure.
  https://github.com/datamllab/rlcard

No upstream implementation is copied into this repository in v0.1.
