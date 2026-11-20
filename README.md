# CMP3005 MediDrone CVRP

Capacitated Vehicle Routing for medical drone delivery: Clarke-Wright baseline, Genetic Algorithm with Split decoder, and Tabu Search, all verified by an independent feasibility checker.

See `CLAUDE.md` for project rules and layout.

## Requirements

Python 3.10+ (developed on 3.14), matplotlib (only for plots). No VRP solver libraries.

## Status

Implemented: `solution.py`, `feasibility.py`, `parser_generator.py`. Other modules are skeletons.

## Data

- `data/benchmarks/`: put CVRPLIB files here (`A-n32-k5`, `A-n45-k6`, `A-n60-k9`, `A-n80-k10`, `.vrp` + `.sol`).
- `data/medidrone/`, `data/tiny/`: generated. Regenerate (MediDrone seed 2026, tiny n=7,8,9 with exact optima in `.sol`):

```
python -m src.parser_generator
```

## Tests

Run from the project root:

```
python -m tests.test_feasibility
python -m tests.test_parser_generator
```
