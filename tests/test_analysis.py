"""Tests for analysis.py.

Run from the project root:  python -m tests.test_analysis   (pytest also works)
"""

import tempfile
from pathlib import Path

from src.analysis import (
    best_valid_row, crossings, parse_routes, plot_tenure, tenure_study, wilcoxon_exact, wilcoxon_table,
)
from src.complexity_experiments import routes_to_str, tabu_start
from src.parser_generator import ROOT, parse_cvrplib
from src.solution import ProblemInstance


def test_wilcoxon_exact_known_values():
    # all 10 differences positive: W = 0, two-sided p = 2 / 2^10
    w, p, m = wilcoxon_exact(list(range(2, 12)), [1] * 10)
    assert (w, m) == (0, 10) and abs(p - 2 / 1024) < 1e-12
    assert wilcoxon_exact([1, 2, 3], [1, 2, 3]) == (0.0, 1.0, 0)  # no non-zero differences
    # d = [2, -1, 2, -1]: average ranks 3.5, 1.5, 3.5, 1.5 -> W+ = 7, W- = 3; 8 of 16 patterns as extreme
    assert wilcoxon_exact([5, 3, 8, 1], [3, 4, 6, 2]) == (3.0, 0.5, 4)


def test_wilcoxon_table_pairs_valid_seeds_only():
    base = dict(instance="X", is_valid=True)
    rows = [dict(base, algorithm="memetic_ga", seed=s, cost=100) for s in (1, 2, 3)]
    rows += [dict(base, algorithm="tabu_search", seed=1, cost=110), dict(base, algorithm="tabu_search", seed=2, cost=90),
             dict(base, algorithm="tabu_search", seed=3, cost=50, is_valid=False)]
    (t,) = wilcoxon_table(rows)
    assert (t["pairs"], t["ga_better"], t["ts_better"]) == (2, 1, 1)


def test_crossings_by_hand():
    # depot (AS) at x=500; customer 1 in EU, 2 in AS
    inst = ProblemInstance.from_coords(name="t", coords=[(500, 500), (100, 500), (900, 500)],
                                       demand=[0, 1, 1], capacity=10, bosphorus_penalty=150,
                                       sides=["AS", "EU", "AS"])
    assert crossings([[1, 2]], inst) == (1, 2)    # 0->1 and 1->2 cross
    assert crossings([[1], [2]], inst) == (1, 2)  # only the EU route crosses, twice
    assert crossings([[2]], inst) == (0, 0)


def test_routes_text_round_trip():
    routes = [[3, 1, 2], [5], [4, 6]]
    assert parse_routes(routes_to_str(routes)) == routes


def test_best_valid_row():
    rows = [dict(instance="X", algorithm="a", seed=2, cost=90, is_valid=False),
            dict(instance="X", algorithm="a", seed=3, cost=100, is_valid=True),
            dict(instance="X", algorithm="b", seed=1, cost=100, is_valid=True)]
    assert best_valid_row(rows, "X")["seed"] == 1 and best_valid_row(rows, "Y") is None


def test_tenure_study_tiny():
    vrp = ROOT / "data/tiny/tiny-n7-s1.vrp"
    inst = parse_cvrplib(vrp)
    rows = tenure_study(inst, 324, [3, (7, 15)], [1, 2], 0.05, tabu_start)
    assert len(rows) == 4 and all(r["is_valid"] and r["gap_pct"] >= 0 for r in rows)
    path = Path(tempfile.mkdtemp()) / "t.png"
    plot_tenure(rows, path)
    assert path.stat().st_size > 0


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
