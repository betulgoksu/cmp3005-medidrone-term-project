"""Tests for clarke_wright.py.

Run from the project root:  python -m tests.test_clarke_wright   (pytest also works)
"""

from src.clarke_wright import clarke_wright, join, savings_list
from src.feasibility import check_solution
from src.parser_generator import ROOT, generate_medidrone, generate_tiny, parse_cvrplib, parse_solution
from src.solution import ProblemInstance


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
    for seed in (2026, 1, 2):
        inst, _ = generate_medidrone(seed)
        sol = clarke_wright(inst)
        report = check_solution(sol, inst)
        assert report.is_valid, (inst.name, report.errors)
        assert clarke_wright(inst).routes == sol.routes  # deterministic


def test_benchmarks_valid():
    files = sorted((ROOT / "data/benchmarks").glob("*.vrp"))
    if not files:
        print("  skip: no files in data/benchmarks/")
        return
    for vrp in files:
        inst = parse_cvrplib(vrp)
        assert check_solution(clarke_wright(inst), inst).is_valid, inst.name


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
