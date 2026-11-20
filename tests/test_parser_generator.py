"""Tests for parser_generator.py.

Run from the project root:  python -m tests.test_parser_generator   (pytest also works)
The A-n32-k5 test is skipped until data/benchmarks/A-n32-k5.vrp and .sol are present.
"""

import itertools
import math
import tempfile
from pathlib import Path

from src.feasibility import check_solution
from src.parser_generator import (
    ROOT, generate_medidrone, generate_tiny, parse_cvrplib, parse_solution,
    solve_exact, write_instance, write_solution,
)
from src.solution import Solution, solution_cost

# Depot is file node 3 on purpose, to test re-indexing (depot -> 0, others keep file order).
SAMPLE_VRP = """NAME : sample-n3-k2
COMMENT : hand-made
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1 3 4
2 6 8
3 0 0
4 0 5
DEMAND_SECTION
1 4
2 3
3 0
4 5
DEPOT_SECTION
3
-1
EOF
"""


def _tmp(name: str) -> Path:
    return Path(tempfile.mkdtemp()) / name


def test_parse_sample():
    path = _tmp("sample-n3-k2.vrp")
    path.write_text(SAMPLE_VRP)
    inst = parse_cvrplib(path)
    assert inst.n == 3 and inst.capacity == 10 and inst.num_vehicles == 2  # K from "-k2"
    assert inst.coords[0] == (0, 0) and inst.demand == [0, 4, 3, 5]
    assert inst.dist[0][1] == 5 and inst.dist[1][3] == 3 and inst.dist[2][3] == 7  # EUC_2D rounding
    assert inst.sides is None and inst.bosphorus_penalty == 0


def test_parse_rejects_non_euc2d():
    path = _tmp("bad.vrp")
    path.write_text(SAMPLE_VRP.replace("EUC_2D", "GEO"))
    try:
        parse_cvrplib(path)
    except ValueError:
        return
    raise AssertionError("expected ValueError for GEO")


def test_benchmark_a_n32_k5():
    vrp, sol_path = ROOT / "data/benchmarks/A-n32-k5.vrp", ROOT / "data/benchmarks/A-n32-k5.sol"
    if not (vrp.exists() and sol_path.exists()):
        print("  skip: A-n32-k5 .vrp/.sol not in data/benchmarks/")
        return
    inst = parse_cvrplib(vrp)
    assert (inst.n, inst.capacity, inst.num_vehicles) == (31, 100, 5)
    sol = parse_solution(sol_path)
    report = check_solution(sol, inst)
    assert report.is_valid, report.errors


def test_medidrone_properties():
    inst, labels = generate_medidrone(seed=7)
    assert inst.n == 100 and inst.coords[0] == (500, 500)
    assert all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in inst.coords)
    assert all(1 <= q <= 30 for q in inst.demand[1:])
    assert inst.capacity == 100 and inst.num_vehicles == math.ceil(sum(inst.demand) / 100) + 1
    assert inst.bosphorus_penalty == 150
    assert inst.sides == ["EU" if x < 500 else "AS" for x, _ in inst.coords]
    assert labels[1:].count(-1) == 20                     # 20% uniform
    assert 4 <= len(set(labels) - {-1}) <= 6               # 4-6 clusters
    assert {"EU", "AS"} <= set(inst.sides)
    eu, as_ = inst.sides.index("EU"), inst.sides.index("AS")
    assert inst.cost[eu][as_] == inst.dist[eu][as_] + 150  # crossing penalty applied


def test_medidrone_seeded():
    a, _ = generate_medidrone(seed=1)
    b, _ = generate_medidrone(seed=1)
    c, _ = generate_medidrone(seed=2)
    assert a.coords == b.coords and a.demand == b.demand
    assert a.coords != c.coords


def test_write_parse_round_trip():
    inst, _ = generate_medidrone(seed=3)
    path = _tmp("m.vrp")
    write_instance(inst, path)
    back = parse_cvrplib(path)
    assert back.dist == inst.dist and back.cost == inst.cost
    assert (back.demand, back.capacity, back.num_vehicles) == (inst.demand, inst.capacity, inst.num_vehicles)
    assert (back.sides, back.bosphorus_penalty) == (inst.sides, inst.bosphorus_penalty)

    sol = Solution([[1, 2], [3]], 42)
    write_solution(sol, path.with_suffix(".sol"))
    assert parse_solution(path.with_suffix(".sol")) == sol


def brute_force_optimum(inst) -> int:
    """Try every customer order and every way to cut it into routes."""
    best = math.inf
    for perm in itertools.permutations(range(1, inst.n + 1)):
        for cuts in itertools.product([False, True], repeat=inst.n - 1):
            routes, cur = [], [perm[0]]
            for c, cut in zip(perm[1:], cuts):
                if cut:
                    routes.append(cur)
                    cur = []
                cur.append(c)
            routes.append(cur)
            if len(routes) <= inst.num_vehicles and all(
                sum(inst.demand[c] for c in r) <= inst.capacity for r in routes
            ):
                best = min(best, solution_cost(routes, inst))
    return best


def test_exact_matches_brute_force():
    for seed in range(5):
        inst = generate_tiny(6, seed)
        sol = solve_exact(inst)
        assert check_solution(sol, inst).is_valid
        assert sol.cost == brute_force_optimum(inst), inst.name


def test_exact_respects_fleet_limit():
    inst = generate_tiny(6, 0)
    inst.num_vehicles = math.ceil(sum(inst.demand) / inst.capacity)  # tighten K
    sol = solve_exact(inst)
    assert len(sol.routes) <= inst.num_vehicles
    assert sol.cost == brute_force_optimum(inst)


def test_saved_tiny_optima_valid():
    files = sorted((ROOT / "data/tiny").glob("*.vrp"))
    assert len(files) >= 3, "run: python -m src.parser_generator"
    for vrp in files:
        inst = parse_cvrplib(vrp)
        sol = parse_solution(vrp.with_suffix(".sol"))
        assert inst.n in (7, 8, 9)
        assert check_solution(sol, inst).is_valid
        assert sol.cost == solve_exact(inst).cost


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
