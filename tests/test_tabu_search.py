"""Tests for tabu_search.py.

Run from the project root:  python -m tests.test_tabu_search   (pytest also works)
"""

import random
import time

from src.clarke_wright import clarke_wright
from src.feasibility import check_solution
from src.parser_generator import ROOT, generate_medidrone, parse_cvrplib, parse_solution
from src.solution import Solution, solution_cost
from src.tabu_search import (
    TabuResult, _apply, draw_tenure, is_admissible, relocate_delta, run_tabu, swap_delta,
    two_opt_delta,
)


def test_deltas_match_full_recompute():
    """Every O(1) delta must equal the real cost change (Bosphorus penalty included)."""
    rng = random.Random(0)
    inst, _ = generate_medidrone(seed=2026)
    routes = [r[:] for r in clarke_wright(inst).routes]
    load = [sum(inst.demand[c] for c in r) for r in routes]
    for _ in range(500):
        paths = [[0, *r, 0] for r in routes]
        a, b = rng.sample(range(len(routes)), 2)
        la, lb = len(routes[a]), len(routes[b])
        kind = rng.choice(["relocate", "swap", "2opt"])
        if kind == "relocate" and la >= 2:
            move = ("relocate", a, rng.randint(1, la), b, rng.randint(0, lb))
            d = relocate_delta(paths[a], move[2], paths[b], move[4], inst.cost)
        elif kind == "swap" and la and lb:
            move = ("swap", a, rng.randint(1, la), b, rng.randint(1, lb))
            d = swap_delta(paths[a], move[2], paths[b], move[4], inst.cost)
        elif kind == "2opt" and la >= 2:
            i = rng.randint(1, la - 1)
            move = ("2opt", a, i, rng.randint(i + 1, la))
            d = two_opt_delta(paths[a], move[2], move[3], inst.cost)
        else:
            continue
        before = solution_cost(routes, inst)
        _apply(move, routes, paths, load, inst.demand)
        assert solution_cost(routes, inst) - before == d, move
    assert sorted(c for r in routes for c in r) == list(range(1, inst.n + 1))


def test_tenure_fixed_and_dynamic():
    rng = random.Random(1)
    assert all(draw_tenure(7, rng) == 7 for _ in range(10))
    draws = {draw_tenure((5, 9), rng) for _ in range(200)}
    assert draws == {5, 6, 7, 8, 9}


def test_tabu_and_aspiration():
    key = ("node", 3, 1)
    tabu = {key: 10}  # tabu through iteration 10
    best = (0, 100)
    assert not is_admissible([key], tabu, 5, (0, 120), best)   # tabu, not better than best
    assert not is_admissible([key], tabu, 10, (0, 100), best)  # equal to best is not enough
    assert is_admissible([key], tabu, 5, (0, 99), best)        # aspiration: new best
    assert is_admissible([key], tabu, 11, (0, 120), best)      # tenure expired
    assert is_admissible([("node", 4, 1)], tabu, 5, (0, 120), best)  # other attribute
    assert is_admissible([key], tabu, 5, (0, 500), (1, 50))    # fleet-feasible beats fleet-excess


def test_tiny_reaches_optimum():
    for vrp in sorted((ROOT / "data/tiny").glob("*.vrp")):
        inst = parse_cvrplib(vrp)
        # worst possible start: one route per customer, limited by K only through fitness
        start = Solution([[c] for c in range(1, inst.n + 1)])
        res = run_tabu(inst, seed=1, start=start, max_iterations=200)
        assert res.best.cost == parse_solution(vrp.with_suffix(".sol")).cost, inst.name
        assert check_solution(res.best, inst).is_valid


def test_deterministic_and_monotone():
    inst, _ = generate_medidrone(seed=2026)
    a = run_tabu(inst, seed=3, max_iterations=60)
    b = run_tabu(inst, seed=3, max_iterations=60)
    assert a.best.routes == b.best.routes and a.history == b.history
    assert all(x >= y for x, y in zip(a.history, a.history[1:]))
    assert a.best.cost <= clarke_wright(inst).cost == a.history[0]
    assert check_solution(a.best, inst).is_valid


def test_time_limit_and_result_contract():
    inst, _ = generate_medidrone(seed=2026)
    t = time.perf_counter()
    res = run_tabu(inst, seed=1, max_iterations=10**6, time_limit=1.0)
    assert time.perf_counter() - t < 1.5  # checked before each iteration (~7 ms here)
    assert isinstance(res, TabuResult) and 0 < res.iterations < 10**6
    assert len(res.history) == res.iterations + 1
    assert res.fitness == (0, res.best.cost)


def test_benchmarks():
    names = ["A-n32-k5", "A-n45-k6", "A-n60-k9", "A-n80-k10"]
    if not all((ROOT / f"data/benchmarks/{n}.vrp").exists() for n in names):
        print("  skip: benchmark files not in data/benchmarks/")
        return
    for name in names:
        inst = parse_cvrplib(ROOT / f"data/benchmarks/{name}.vrp")
        res = run_tabu(inst, seed=1, max_iterations=150)
        report = check_solution(res.best, inst)
        # Only a fleet excess may remain, and only if the fitness says so (A-n45-k6 starts at 7 > 6 routes).
        assert [e for e in report.errors if "fleet" not in e] == [], (name, report.errors)
        assert report.is_valid == (res.fitness[0] == 0), name
        assert res.best.cost == solution_cost(res.best.routes, inst)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
