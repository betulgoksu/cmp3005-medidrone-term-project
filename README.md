# CMP3005 MediDrone CVRP

Capacitated Vehicle Routing for medical drone delivery: Clarke-Wright baseline, Genetic Algorithm with Split decoder, and Tabu Search, all verified by an independent feasibility checker.

See `CLAUDE.md` for project rules and layout.

## Requirements

Python 3.10+, numpy, matplotlib (scipy optional). No VRP solver libraries.

## Status

Implemented: `solution.py`, `feasibility.py`. Other modules are skeletons.

## Tests

```
python -m tests.test_feasibility
```
