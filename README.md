# RummiZero v0.1

A **headless Rummikub self-play research scaffold**. The goal is to evolve the
existing screen-recognition/recommendation project into a league-trained agent
that can eventually use full-turn candidate actions, opponent belief, and
information-set search.

## What v0.1 already does

- Runs complete 2–4 player games without a GUI.
- Uses `rummikub-solver` as a strong exact/MILP baseline for legal best moves.
- Keeps **private rack**, public table, stock size, opponent rack sizes, and each
  player's opening-meld state separate in the observation.
- Provides baseline agents (`SolverAgent`, `RandomDelayAgent`).
- Provides a tiny trainable `LinearPolicyAgent` so the **self-play → update →
  checkpoint → arena** pipeline is executable before a neural network is added.
- Includes Elo arena and league snapshot infrastructure.
- Uses deterministic seeds for reproducible experiments.

## Deliberate v0.1 limitation

The solver exposes the best move, not the full set of alternative legal final
arrangements. Therefore the trainable action space in v0.1 is only:

1. `PLAY_BEST` — apply the solver's proposed legal move.
2. `DRAW` — decline the move and draw (when stock remains).

This is **not the final AlphaZero/DouZero action representation**. The next major
engine milestone is a Full-Turn Candidate Generator that emits many legal final
arrangements; the network will then score `(state, candidate_action)` pairs.
The interfaces in this repo are arranged so that replacement does not require
rewriting the simulator or league system.

## Install (Windows / Python 3.11+)

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

`rummikub-solver` uses CVXPY and by default can use SciPy/HiGHS. For a newer
HiGHS backend:

```powershell
pip install "rummikub-solver[highs]"
```

## Smoke test

```powershell
rummizero arena --games 100 --seed 7
```

## Self-play bootstrap

```powershell
rummizero train --games 5000 --players 2 --seed 7 --out models/linear_v1.json
rummizero evaluate --model models/linear_v1.json --games 1000 --seed 99
```

The v0.1 learner is intentionally tiny; its purpose is validating the data and
league pipeline. It learns when to take the current exact solver move versus
holding/drawing. A later DMC/PPO action-conditioned neural model will replace it.

## Architecture

```text
GameSimulator
  ├─ public table / stock / turn history
  ├─ private rack per player
  ├─ SolverBackend (exact baseline)
  └─ Agent interface
       ├─ SolverAgent
       ├─ RandomDelayAgent
       └─ LinearPolicyAgent  <-- v0.1 self-play learner

Arena -> Elo
League -> saved policy snapshots

NEXT:
FullTurnCandidateGenerator
   state + candidate action
          ↓
   action-conditioned neural net
          ↓
   DMC / PPO / NFSP league self-play
          ↓
   opponent belief + IS-MCTS
```

## Why the observation is different from early Rummikub RL repos

A tile in your rack is strategically different from the same tile on the table.
The table's meld structure, opponent rack counts, stock count, opening status,
and action history are also relevant. RummiZero keeps these channels separate
instead of collapsing rack + table into one count vector.

## Project status

`v0.1 = executable research scaffold`, not a finished superhuman agent.
The important result is that we can now run headless games, collect trajectories,
train a policy, retain generations, and evaluate generations reproducibly.
