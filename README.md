# CMP3005 MediDrone CVRP

Capacitated Vehicle Routing for medical drone delivery: Clarke-Wright baseline, Genetic Algorithm with Split decoder, and Tabu Search, all verified by an independent feasibility checker.

See `CLAUDE.md` for project rules and layout.

## Requirements

Python 3.10+ (developed on 3.14), matplotlib (only for plots). No VRP solver libraries.

## Status

Implemented: `solution.py`, `feasibility.py`, `parser_generator.py`, `clarke_wright.py`, `ga_split.py`. Other modules are skeletons.

Run on every instance in `data/` (feasibility-checked):

```
python -m src.clarke_wright
python -m src.ga_split
```

Note: Clarke-Wright exceeds the fleet limit on A-n45-k6 (7 routes, K=6); this is reported, not hidden.

GA from Python (memetic by default: 2-opt + granular relocate on decoded routes):

```python
from src.ga_split import run_ga
res = run_ga(inst, seed=1, pop_size=50, generations=200, pm=0.2, elite=2,
             ls_prob=0.2,        # chance a child gets local search; 0 = plain GA
             neighbor_count=10,  # nearest neighbours tried by relocate
             time_limit=30)      # seconds, checked once per generation
res.best, res.fitness, res.history, res.generations
```

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
python -m tests.test_clarke_wright
python -m tests.test_ga_split
```
