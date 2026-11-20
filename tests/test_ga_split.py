"""Tests for ga_split.py.

Run from the project root:  python -m tests.test_ga_split   (pytest also works)
"""

import itertools
import math
import random
import time

from src.clarke_wright import clarke_wright
from src.feasibility import check_solution
from src.ga_split import (
    GAResult, decode, local_search, min_routes, mutate, nearest_neighbors, order_crossover, run_ga,
    split, split_bounded, two_opt,
)
from src.parser_generator import ROOT, generate_medidrone, generate_tiny, parse_cvrplib, parse_solution
from src.solution import ProblemInstance, Solution, solution_cost


def hand_instance(**kw) -> ProblemInstance:
    """depot (0,0); 1 (3,4); 2 (6,8); 3 (0,5). d01=5 d02=10 d03=5 d12=5 d13=3 d23=7."""
    args = dict(name="hand", coords=[(0, 0), (3, 4), (6, 8), (0, 5)], demand=[0, 4, 3, 5],
                capacity=10, num_vehicles=2)
    args.update(kw)
    return ProblemInstance.from_coords(**args)


def all_cuts(tour):
    """Every way to cut a tour into consecutive routes."""
    for cuts in itertools.product([False, True], repeat=len(tour) - 1):
        routes, cur = [], [tour[0]]
        for c, cut in zip(tour[1:], cuts):
            if cut:
                routes.append(cur)
                cur = []
            cur.append(c)
        yield routes + [cur]


def brute_split(tour, inst, max_routes=None):
    """(cost, routes count) of the cheapest capacity-feasible cut, optionally with <= max_routes routes."""
    best = (math.inf, None)
    for routes in all_cuts(tour):
        if all(sum(inst.demand[c] for c in r) <= inst.capacity for r in routes) and (
            max_routes is None or len(routes) <= max_routes
        ):
            best = min(best, (solution_cost(routes, inst), len(routes)))
    return best


def test_split_by_hand():
    # [1,2]+[3] = 20+10 = 30;  [1]+[2,3] = 10+22 = 32;  [1]+[2]+[3] = 40;  [1,2,3] load 12 > 10
    assert split([1, 2, 3], hand_instance()) == [[1, 2], [3]]


def test_split_matches_brute_force():
    rng = random.Random(0)
    for seed in range(10):
        inst = generate_tiny(8, seed)
        tour = rng.sample(range(1, 9), 8)
        assert solution_cost(split(tour, inst), inst) == brute_split(tour, inst)[0]


def test_split_bounded_matches_brute_force():
    rng = random.Random(1)
    for seed in range(10):
        inst = generate_tiny(8, seed)
        tour = rng.sample(range(1, 9), 8)
        for k in range(1, 6):
            routes = split_bounded(tour, inst, k)
            cost, _ = brute_split(tour, inst, k)
            if cost == math.inf:
                assert routes is None
            else:
                assert len(routes) <= k and solution_cost(routes, inst) == cost
                assert sorted(c for r in routes for c in r) == list(range(1, 9))


def test_min_routes_matches_brute_force():
    rng = random.Random(2)
    for seed in range(10):
        inst = generate_tiny(8, seed)
        tour = rng.sample(range(1, 9), 8)
        feasible = [len(r) for r in all_cuts(tour)
                    if all(sum(inst.demand[c] for c in x) <= inst.capacity for x in r)]
        assert min_routes(tour, inst) == min(feasible)


def test_decode_respects_fleet_when_possible():
    # Unlimited-fleet optimum for [1,2,3] is 2 routes; K=1 is impossible (load 12 > 10).
    fit, sol = decode([1, 2, 3], hand_instance(num_vehicles=1))
    assert fit == (1, 30) and len(sol.routes) == 2
    fit, sol = decode([1, 2, 3], hand_instance())
    assert fit == (0, 30)


def test_operators_keep_permutation():
    rng = random.Random(3)
    for _ in range(200):
        p1, p2 = rng.sample(range(1, 21), 20), rng.sample(range(1, 21), 20)
        child = order_crossover(p1, p2, rng)
        assert sorted(child) == list(range(1, 21))
        mutate(child, rng)
        assert sorted(child) == list(range(1, 21))


def test_ox_keeps_parent1_segment():
    class FixedCuts(random.Random):
        def sample(self, population, k):
            return [2, 5]
    child = order_crossover([1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1], FixedCuts())
    # keep p1[2:5] = [3,4,5]; fill from p2 starting at position 5: 2,1,7,6 -> positions 5,6,0,1
    assert child == [7, 6, 3, 4, 5, 2, 1]


def test_two_opt_removes_crossing():
    # depot (0,0), 1 (0,10), 2 (10,0), 3 (10,10). 0-1-2-3-0 crosses itself: 10+14+10+14 = 48.
    # Uncrossed 0-1-3-2-0 = 40.
    inst = ProblemInstance.from_coords(name="sq", coords=[(0, 0), (0, 10), (10, 0), (10, 10)],
                                       demand=[0, 1, 1, 1], capacity=10)
    route = [1, 2, 3]
    assert two_opt(route, inst.cost)
    assert solution_cost([route], inst) == 40
    assert not two_opt(route, inst.cost)  # already 2-optimal


def test_relocate_merges_and_respects_capacity():
    # depot (0,0), 1 (10,0), 2 (11,0): [[1],[2]] = 20+22 = 42, [[1,2]] = 10+1+11 = 22.
    kw = dict(name="line", coords=[(0, 0), (10, 0), (11, 0)], demand=[0, 3, 3])
    inst = ProblemInstance.from_coords(**kw, capacity=10)
    nb = nearest_neighbors(inst, 5)
    routes = local_search([[1], [2]], inst, nb)
    assert len(routes) == 1 and solution_cost(routes, inst) == 22
    tight = ProblemInstance.from_coords(**kw, capacity=5)  # 3+3 > 5: no merge allowed
    assert local_search([[1], [2]], tight, nb) == [[1], [2]]


def test_local_search_never_worse_and_stays_valid():
    rng = random.Random(4)
    for seed in range(10):
        inst, _ = generate_medidrone(seed)
        nb = nearest_neighbors(inst, 10)
        routes = split(rng.sample(range(1, 101), 100), inst)
        improved = local_search(routes, inst, nb)
        assert solution_cost(improved, inst) <= solution_cost(routes, inst)
        assert len(improved) <= len(routes)
        sol = Solution(improved)
        sol.evaluate(inst)
        report = check_solution(sol, inst)
        assert [e for e in report.errors if "fleet" not in e] == [], report.errors


def test_memetic_beats_plain_ga():
    inst, _ = generate_medidrone(seed=2026)
    plain = run_ga(inst, seed=1, pop_size=20, generations=10, ls_prob=0)
    memetic = run_ga(inst, seed=1, pop_size=20, generations=10)
    assert check_solution(memetic.best, inst).is_valid
    assert memetic.best.cost < plain.best.cost


def test_time_limit_and_result_contract():
    inst, _ = generate_medidrone(seed=2026)
    start = time.perf_counter()
    res = run_ga(inst, seed=1, generations=10_000, time_limit=1.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, elapsed  # limit is checked once per generation (~0.15 s here)
    assert isinstance(res, GAResult) and 0 < res.generations < 10_000
    assert len(res.history) == res.generations + 1
    assert res.fitness == (0, res.best.cost) and check_solution(res.best, inst).is_valid


def test_ga_tiny_reaches_optimum():
    for vrp in sorted((ROOT / "data/tiny").glob("*.vrp")):
        inst = parse_cvrplib(vrp)
        res = run_ga(inst, seed=1, pop_size=20, generations=50)
        assert check_solution(res.best, inst).is_valid
        assert res.best.cost == parse_solution(vrp.with_suffix(".sol")).cost, inst.name


def test_ga_deterministic_and_monotone():
    inst, _ = generate_medidrone(seed=2026)
    a = run_ga(inst, seed=5, pop_size=20, generations=15)
    b = run_ga(inst, seed=5, pop_size=20, generations=15)
    assert a.best.routes == b.best.routes and a.history == b.history
    assert all(x >= y for x, y in zip(a.history, a.history[1:]))  # elitism: never worse
    assert check_solution(a.best, inst).is_valid
    assert a.best.cost <= clarke_wright(inst).cost  # CW order is in the initial population


def test_ga_benchmarks_valid():
    names = ["A-n32-k5", "A-n45-k6", "A-n60-k9", "A-n80-k10"]
    if not all((ROOT / f"data/benchmarks/{n}.vrp").exists() for n in names):
        print("  skip: benchmark files not in data/benchmarks/")
        return
    for name in names:
        inst = parse_cvrplib(ROOT / f"data/benchmarks/{name}.vrp")
        res = run_ga(inst, seed=1, pop_size=30, generations=30)
        report = check_solution(res.best, inst)
        assert report.is_valid, (name, res.fitness, report.errors)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
