# CMP3005 MediDrone CVRP

Capacitated Vehicle Routing for medical drone delivery: Clarke-Wright baseline, Genetic Algorithm with Split decoder, and Tabu Search, all verified by an independent feasibility checker.

Layout: `src/parser_generator.py` (CVRPLIB parser, MediDrone and tiny generators, exact DP), `src/solution.py` (representation, cost), `src/feasibility.py` (independent checker), `src/clarke_wright.py`, `src/ga_split.py`, `src/tabu_search.py`, `src/complexity_experiments.py` (experiments, scaling, CSV, plots), `src/analysis.py` (Wilcoxon test, tenure study, Bosphorus crossings, convergence and route plots).

All code is our own. Published ideas it follows: Clarke & Wright (1964) savings, Prins (2004) giant tour + Split and the duplicate-fitness rule, Held-Karp DP for the tiny optima.

## Requirements

Python 3.10+ (developed on 3.14), matplotlib (only for plots). No VRP solver libraries.

## Status

Implemented: `solution.py`, `feasibility.py`, `parser_generator.py`, `clarke_wright.py`, `ga_split.py`, `tabu_search.py`, `complexity_experiments.py`.

Run on every instance in `data/` (feasibility-checked):

```
python -m src.clarke_wright
python -m src.ga_split
python -m src.tabu_search
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

Tabu Search from Python (relocate, swap, 2-opt; starts from Clarke-Wright by default; `kick_start(inst, seed)` gives the seeded CW + random-relocate start used in the experiments):

```python
from src.tabu_search import run_tabu
res = run_tabu(inst, seed=1, start=None,   # or a Solution to start from
               max_iterations=1000,
               tenure=(7, 15),            # int = fixed, (low, high) = dynamic
               time_limit=30)             # seconds, checked before each iteration
res.best, res.fitness, res.history, res.iterations
```

## Experiments

One command reproduces every table and figure in the report (about 2 hours, sequential):

```
python -m src.complexity_experiments --seeds 10
```

It runs, in order:

1. Equal-budget runs: Clarke-Wright once per instance; memetic GA and Tabu Search once per seed 1..10
   with the same wall-clock budget, 30 s for n <= 50 and 60 s for larger n (`--time-limit X` overrides).
2. Tabu tenure study on A-n60-k9 (tenures 3, 7, 15, 30, 60 and [7, 15]; 5 seeds, 20 s each).
3. Scaling study on MediDrone instances n = 25, 50, 100, 200, 400 (seed 11372).

| Output in `results/` | Report item |
|---|---|
| `runs.csv` | raw results: one row per run, with final routes and best-cost-vs-time trace |
| `summary.csv` | benchmark, tiny and MediDrone tables (best/mean/std, gap, runtime, optimum hits) |
| `gap_by_instance.png`, `medidrone_costs.png` | gap and MediDrone cost figures |
| `wilcoxon.csv` | GA vs Tabu signed-rank test per instance |
| `crossings.csv`, `routes_medidrone.png` | Bosphorus crossings and best MediDrone route map |
| `convergence_<instance>.png` | convergence curves (A-n45-k6, A-n60-k9, MediDrone) |
| `tenure_study.csv`, `tenure_study.png` | parameter sensitivity |
| `scaling.csv`, `scaling.png` | time per step / iteration / generation vs n, fitted log-log slopes |
| `hardware.txt` | machine and Python version used |

Parts can be run alone: `--skip-runs`, `--skip-study`, `--skip-scaling`, `--instances A-n32-k5 ...`,
`--sizes 25 50 100`, `--study-instance`, `--study-time`, `--study-seeds`, `--out DIR`.
`--analysis-only` rebuilds summary, Wilcoxon, crossings and plots from an existing `runs.csv`
without new runs (every stored solution is re-checked with `feasibility.py`).

## Data

- `data/benchmarks/`: put CVRPLIB files here (`A-n32-k5`, `A-n45-k6`, `A-n60-k9`, `A-n80-k10`, `.vrp` + `.sol`).
- `data/medidrone/`, `data/tiny/`: generated. Regenerate (MediDrone seed 11372, tiny n=7,8,9 with exact optima in `.sol`):

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
python -m tests.test_tabu_search
python -m tests.test_complexity_experiments
python -m tests.test_analysis
```


