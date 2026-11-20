"""Tests for solution.py and feasibility.py.

Run from the project root:  python -m tests.test_feasibility   (pytest also works)

Hand-checked instance (EUC_2D):
    depot 0 (0,0); 1 (3,4); 2 (6,8); 3 (0,5)
    d01=5  d02=10  d03=5  d12=5  d13=3 (3.16)  d23=7 (6.71)
    demand [0, 4, 3, 5], Q=10, K=2
    routes [[1, 2], [3]] -> (5+5+10) + (5+5) = 30
"""

from src.feasibility import check_solution
from src.solution import ProblemInstance, Solution, euc_2d, from_giant_tour, to_giant_tour

COORDS = [(0, 0), (3, 4), (6, 8), (0, 5)]
DEMAND = [0, 4, 3, 5]


def make_inst(**kw) -> ProblemInstance:
    args = dict(name="tiny", coords=COORDS, demand=DEMAND, capacity=10, num_vehicles=2)
    args.update(kw)
    return ProblemInstance.from_coords(**args)


def errors_for(routes, cost=None, **kw) -> list[str]:
    inst = make_inst(**kw)
    sol = Solution(routes, cost)
    if cost is None:
        sol.evaluate(inst)
    report = check_solution(sol, inst)
    assert report.is_valid == (not report.errors)
    return report.errors


def assert_one_error(errors: list[str], fragment: str) -> None:
    assert any(fragment in e for e in errors), f"expected '{fragment}' in {errors}"


# --- solution.py ---

def test_euc_2d_rounding():
    assert euc_2d((0, 0), (1, 1)) == 1      # 1.414
    assert euc_2d((0, 0), (1.5, 2)) == 3    # 2.5 rounds up
    assert euc_2d((0, 0), (1, 3)) == 3      # 3.162


def test_costs_and_loads():
    inst = make_inst()
    sol = Solution([[1, 2], [3]])
    assert sol.evaluate(inst) == 30
    assert sol.loads(inst) == [7, 5]


def test_bosphorus_penalty():
    inst = make_inst(sides=["EU", "EU", "EU", "AS"], bosphorus_penalty=100)
    assert Solution([[1, 2], [3]]).evaluate(inst) == 230  # route [3] crosses twice
    assert Solution([[1, 2, 3]]).evaluate(inst) == 5 + 5 + 7 + 5 + 200


def test_giant_tour_round_trip():
    routes = [[1, 2], [3]]
    assert to_giant_tour(routes) == [1, 2, 0, 3]
    assert from_giant_tour(to_giant_tour(routes)) == routes


# --- feasibility.py ---

def test_valid_solution():
    assert errors_for([[1, 2], [3]]) == []
    assert errors_for([[1, 2], [3]], sides=["EU", "EU", "EU", "AS"], bosphorus_penalty=100) == []
    assert errors_for([[1, 2], [], [3]]) == []  # empty route ignored by fleet rule


def test_fail_indices():
    assert_one_error(errors_for([[1, 0, 2], [3]], cost=30), "invalid customer id 0")
    assert_one_error(errors_for([[1, 2], [3, 4]], cost=30), "invalid customer id 4")
    assert_one_error(errors_for([[1, 2], [3, -1]], cost=30), "invalid customer id -1")


def test_fail_coverage():
    assert_one_error(errors_for([[1, 2]]), "customer 3 not visited")
    assert_one_error(errors_for([[1, 2], [3, 1]]), "customer 1 visited 2 times")


def test_fail_capacity():
    assert_one_error(errors_for([[1, 2, 3]]), "load 12 exceeds capacity 10")


def test_fail_fleet():
    assert_one_error(errors_for([[1], [2], [3]]), "3 routes used, fleet size K=2")
    assert errors_for([[1], [2], [3]], num_vehicles=None) == []  # K unlimited


def test_fail_cost():
    assert_one_error(errors_for([[1, 2], [3]], cost=29), "reported cost 29 != recomputed cost 30")
    inst = make_inst()
    assert_one_error(check_solution(Solution([[1, 2], [3]]), inst).errors, "no reported cost")


def test_bad_instance_rejected():
    for kw in (dict(demand=[1, 4, 3, 5]), dict(sides=["EU"]), dict(bosphorus_penalty=5)):
        try:
            make_inst(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")

