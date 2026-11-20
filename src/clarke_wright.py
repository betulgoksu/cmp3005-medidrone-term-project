"""Deterministic Clarke-Wright savings heuristic (parallel version), the baseline.

Idea: start with one route per customer (0 -> i -> 0). Joining the route ending at i
with the route starting at j removes arcs (i, 0) and (0, j) and adds (i, j), saving
    s(i, j) = c(0, i) + c(0, j) - c(i, j).
Process pairs by decreasing saving and merge whenever it is allowed.

``c`` is ``inst.cost`` (distance + Bosphorus penalty), so savings measure the real
objective; for standard CVRPLIB instances c = d.

Time complexity (n customers):
    computing savings   O(n^2)
    sorting savings     O(n^2 log n)   <- dominant
    merging             O(n^2) pair checks at O(1) each, plus at most n-1 merges
                        costing O(n) each (relabelling route ids) -> O(n^2)
    total               O(n^2 log n) time, O(n^2) memory for the savings list.
"""

from __future__ import annotations

from src.solution import ProblemInstance, Solution


def savings_list(inst: ProblemInstance) -> list[tuple[int, int, int]]:
    """All pairs (s, i, j) with i < j, sorted by saving descending; ties broken by (i, j)."""
    c = inst.cost
    pairs = [
        (c[0][i] + c[0][j] - c[i][j], i, j)
        for i in range(1, inst.n + 1)
        for j in range(i + 1, inst.n + 1)
    ]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return pairs


def join(ri: list[int], rj: list[int], i: int, j: int) -> list[int]:
    """Concatenate two routes so that i and j become adjacent.

    Both i and j must be endpoints. Costs are symmetric, so reversing a route is free.
    """
    if ri[-1] != i:
        ri = ri[::-1]  # now ri ends with i
    if rj[0] != j:
        rj = rj[::-1]  # now rj starts with j
    return ri + rj


def clarke_wright(inst: ProblemInstance) -> Solution:
    """Build a solution with the parallel Clarke-Wright savings algorithm.

    A pair (i, j) is merged when i and j are in different routes, both are route
    endpoints (not interior), and the combined load is <= Q. Only positive savings
    are used, except while there are more routes than K: then non-positive savings
    are also used, to reduce the fleet. If capacity makes K unreachable, the result
    has too many routes and feasibility.py will report it.

    Returns:
        Solution with cost already evaluated.
    """
    routes: dict[int, list[int]] = {i: [i] for i in range(1, inst.n + 1)}  # route id -> customers
    route_of = list(range(inst.n + 1))  # customer -> route id
    load = {i: inst.demand[i] for i in routes}
    max_routes = inst.num_vehicles if inst.num_vehicles is not None else inst.n

    for s, i, j in savings_list(inst):
        if s <= 0 and len(routes) <= max_routes:
            break  # remaining savings are not positive and the fleet is already small enough
        a, b = route_of[i], route_of[j]
        if a == b or load[a] + load[b] > inst.capacity:
            continue
        ra, rb = routes[a], routes[b]
        if i not in (ra[0], ra[-1]) or j not in (rb[0], rb[-1]):
            continue
        routes[a] = join(ra, rb, i, j)
        load[a] += load.pop(b)
        for c in routes.pop(b):
            route_of[c] = a

    sol = Solution([routes[k] for k in sorted(routes)])
    sol.evaluate(inst)
    return sol


def main() -> None:
    """Run Clarke-Wright on every instance in data/ and print feasibility-checked costs."""
    from src.feasibility import check_solution
    from src.parser_generator import ROOT, parse_cvrplib

    for path in sorted((ROOT / "data").glob("*/*.vrp")):
        inst = parse_cvrplib(path)
        sol = clarke_wright(inst)
        report = check_solution(sol, inst)
        status = "valid" if report.is_valid else f"INVALID {report.errors}"
        print(f"{inst.name:24} cost {sol.cost:7}  routes {len(sol.routes):3}  K={inst.num_vehicles}  {status}")


if __name__ == "__main__":
    main()
