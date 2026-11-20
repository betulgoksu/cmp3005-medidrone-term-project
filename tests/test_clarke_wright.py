"""Tests for clarke_wright.py.

Run from the project root:  python -m tests.test_clarke_wright   (pytest also works)
"""

from src.clarke_wright import clarke_wright, join, repair_fleet, savings_list
from src.feasibility import (
    check_capacity, check_cost, check_coverage, check_indices, check_solution, recompute_cost,
)
from src.parser_generator import ROOT, generate_medidrone, generate_tiny, parse_cvrplib, parse_solution
from src.solution import ProblemInstance, Solution


def hand_instance(**kw) -> ProblemInstance:
    """depot (0,0); 1 (3,4); 2 (6,8); 3 (0,5). d01=5 d02=10 d03=5 d12=5 d13=3 d23=7."""
    args = dict(name="hand", coords=[(0, 0), (3, 4), (6, 8), (0, 5)], demand=[0, 4, 3, 5],
                capacity=10, num_vehicles=2)
    args.update(kw)
    return ProblemInstance.from_coords(**args)


def test_savings_by_hand():
    # s12 = 5+10-5 = 10, s23 = 10+5-7 = 8, s13 = 5+5-3 = 7
    assert savings_list(hand_instance()) == [(10, 1, 2), (8, 2, 3), (7, 1, 3)]


def test_hand_solution():
    # merge (1,2): load 7. (2,3) and (1,3) would give load 12 > 10, so they are rejected.
    inst = hand_instance()
    sol = clarke_wright(inst)
    assert sol.routes == [[1, 2], [3]] and sol.cost == 30
    assert check_solution(sol, inst).is_valid


def test_join_orientations():
    assert join([1, 2], [3, 4], 2, 3) == [1, 2, 3, 4]
    assert join([1, 2], [3, 4], 1, 3) == [2, 1, 3, 4]
    assert join([1, 2], [3, 4], 2, 4) == [1, 2, 4, 3]
    assert join([1, 2], [3, 4], 1, 4) == [2, 1, 4, 3]


def test_fleet_limit_forces_non_positive_merge():
    # Customers on opposite sides of the depot: saving 10 + 10 - 20 = 0.
    kw = dict(name="line", coords=[(0, 0), (10, 0), (-10, 0)], demand=[0, 1, 1], capacity=10)
    free = ProblemInstance.from_coords(**kw, num_vehicles=None)
    assert len(clarke_wright(free).routes) == 2  # zero saving: no merge
    tight = ProblemInstance.from_coords(**kw, num_vehicles=1)
    sol = clarke_wright(tight)
    assert len(sol.routes) == 1 and check_solution(sol, tight).is_valid


def test_repair_removes_route_when_capacity_allows():
    # K=2, three single routes. Donor = route [2] (all have 1 customer; q2=3 is smallest).
    # Insert 2 into [1]: before 1 -> 10+5-5 = 10, after 1 -> 5+10-5 = 10 (tie: first position wins).
    # Insert 2 into [3]: 10+7-5 = 12. So the result is [[2, 1], [3]].
    inst = hand_instance()
    routes = repair_fleet([[1], [2], [3]], inst)
    assert routes == [[2, 1], [3]]
    sol = Solution(routes)
    sol.evaluate(inst)
    assert check_solution(sol, inst).is_valid


def test_repair_gives_up_cleanly_when_nothing_fits():
    # Q=5: customer 2 (q=3) fits neither with 1 (q=4) nor with 3 (q=5).
    inst = hand_instance(capacity=5)
    routes = repair_fleet([[1], [2], [3]], inst)
    assert routes == [[1], [2], [3]]  # unchanged, no capacity violation forced
    sol = Solution(routes)
    sol.evaluate(inst)
    assert check_solution(sol, inst).errors == ["3 routes used, fleet size K=2"]


def test_repair_partial_move_keeps_capacity():
    # Donor [1, 2] (2 customers) is smaller than [3, 4, 5]. Customer 1 (q=1) fits (load 6+1),
    # customer 2 (q=9) does not (16 > 10), so it stays alone on the donor route.
    inst = ProblemInstance.from_coords(
        name="partial", coords=[(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (0, 3)],
        demand=[0, 1, 9, 2, 2, 2], capacity=10, num_vehicles=1)
    routes = repair_fleet([[1, 2], [3, 4, 5]], inst)
    assert routes[0] == [2] and sorted(routes[1]) == [1, 3, 4, 5]
    sol = Solution(routes)
    sol.evaluate(inst)
    assert check_solution(sol, inst).errors == ["2 routes used, fleet size K=1"]


def test_clarke_wright_uses_repair():
    # Same as the gives-up case through the full algorithm: no merge fits Q=5, K=2.
    inst = hand_instance(capacity=5)
    sol = clarke_wright(inst)
    assert len(sol.routes) == 3 and sol.cost == sum(2 * inst.cost[0][i] for i in (1, 2, 3))
    assert check_solution(sol, inst).errors == ["3 routes used, fleet size K=2"]


def test_tiny_valid_and_not_below_optimum():
    for vrp in sorted((ROOT / "data/tiny").glob("*.vrp")):
        inst = parse_cvrplib(vrp)
        sol = clarke_wright(inst)
        assert check_solution(sol, inst).is_valid, inst.name
        assert sol.cost >= parse_solution(vrp.with_suffix(".sol")).cost, inst.name


def test_random_tiny_valid():
    for n in (5, 7, 9):
        for seed in range(10):
            inst = generate_tiny(n, seed)
            report = check_solution(clarke_wright(inst), inst)
            assert report.is_valid, (inst.name, report.errors)


def test_medidrone_valid_and_deterministic():
    for seed in (11372, 1, 2):
        inst, _ = generate_medidrone(seed)
        sol = clarke_wright(inst)
        report = check_solution(sol, inst)
        assert report.is_valid, (inst.name, report.errors)
        assert clarke_wright(inst).routes == sol.routes  # deterministic


BENCH = ROOT / "data/benchmarks"


def test_benchmarks_valid():
    names = ["A-n32-k5", "A-n60-k9", "A-n80-k10"]
    if not all((BENCH / f"{name}.vrp").exists() for name in names):
        print("  skip: benchmark files not in data/benchmarks/")
        return
    for name in names:
        inst = parse_cvrplib(BENCH / f"{name}.vrp")
        report = check_solution(clarke_wright(inst), inst)
        assert report.is_valid, (name, report.errors)


def test_benchmark_a_n45_k6_fleet_bottleneck():
    """Greedy CW + insertion repair cannot reach K=6 on A-n45-k6 (sum q = 593 of 600).

    Everything except the fleet limit must still hold: ids, coverage, capacity, cost.
    """
    path = BENCH / "A-n45-k6.vrp"
    if not path.exists():
        print("  skip: A-n45-k6.vrp not in data/benchmarks/")
        return
    inst = parse_cvrplib(path)
    sol = clarke_wright(inst)
    assert check_indices(sol.routes, inst.n) == []
    assert check_coverage(sol.routes, inst.n) == []
    assert check_capacity(sol.routes, inst) == []
    assert check_cost(sol.cost, recompute_cost(sol.routes, inst)) == []
    assert inst.num_vehicles == 6 and len(sol.routes) == 7
    assert check_solution(sol, inst).errors == ["7 routes used, fleet size K=6"]

    # Why the repair gives up: the stranded customer needs 19, the best spare capacity is 11.
    loads = sorted((len(r), sum(inst.demand[c] for c in r)) for r in sol.routes)
    stranded_demand = loads[0][1]
    max_spare = max(inst.capacity - load for _, load in loads[1:])
    assert (stranded_demand, max_spare) == (19, 11)


def test_published_solutions_valid():
    for sol_path in sorted(BENCH.glob("*.sol")):
        inst = parse_cvrplib(sol_path.with_suffix(".vrp"))
        report = check_solution(parse_solution(sol_path), inst)
        assert report.is_valid, (sol_path.name, report.errors)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")

