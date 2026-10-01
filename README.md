# RummiZero v0.2-dev

A **headless Rummikub self-play research project**.

The current development branch adds a first multi-action candidate generator so
the agent is no longer limited to only "play the solver's single best move" vs
"draw".

## Stable v0.1

- complete 2-4 player headless games
- exact MILP solver baseline through rummikub-solver
- separated private/public observation channels
- baseline agents
- bootstrap self-play learner
- Elo arena
- league snapshots

## v0.2a candidate generator

The development branch adds FullTurnCandidateGenerator.

For a rack, it enumerates unique tile sub-multisets and asks the exact solver
whether every tile in that subset can be legally played while preserving or
rearranging the existing table. Every accepted result becomes a complete turn
candidate.

That changes the future learning problem from:

1. play the one solver move
2. draw

to something closer to:

1. play 3 tiles with arrangement A
2. play 4 tiles with arrangement B
3. play 7 tiles with arrangement C
4. draw

The next v0.2b milestone will generate multiple distinct table arrangements even
when they consume the same rack subset.

## Install

On Windows, using Python 3.13 without PowerShell activation:

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -U pip
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

## Tests

    .\.venv\Scripts\python.exe -m pytest -q

## Baseline arena

    .\.venv\Scripts\rummizero.exe arena --games 10 --seed 7

## Candidate demo

After checking out the v0.2 development branch:

    .\.venv\Scripts\rummizero.exe candidates --rack 1,2,3,4 --opening-done

For a standard ruleset, tile IDs 1,2,3,4 are the same-colour run values 1-4.
The output should include several legal alternatives, such as playing all four
or valid three-tile runs.

See docs/V0_2_DESIGN.md for the design and limitations.

## Resumable automatic experiments

`auto-experiment` runs bounded train, paired evaluation, and promotion rounds
without waiting for user input between stages. It writes
`experiment_state.json` after every completed gate, resumes unfinished round
sequences, protects the accepted champion, and can continue a non-catastrophic
challenger as a research lineage without promoting it.

    .\.venv\Scripts\rummizero.exe auto-experiment `
      --baseline models\action_imitation100.json `
      --work-dir models\auto_imitation100 `
      --rounds 3 --games-per-round 20 --promotion-games 40 `
      --max-candidates 4 --max-solver-calls 8 --max-turns 300

Evaluation and promotion commands also accept `--progress-every` and
`--max-turns`, so a slow or looping deal remains observable and bounded.
