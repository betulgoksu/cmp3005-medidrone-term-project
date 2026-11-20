"""Independent feasibility & cost checker. Every reported solution must pass this.

The five checks are separate functions so each can be explained (and tested) alone:
  1. check_indices   - every route entry is an int customer id in 1..n (no depot 0 inside).
  2. check_coverage  - every customer 1..n is visited exactly once.
  3. check_capacity  - each route's demand <= Q.
  4. check_fleet     - number of non-empty routes <= K (empty routes are ignored; K=None = unlimited).
  5. check_cost      - total cost recomputed from scratch equals the reported cost.

Depot start/end is implicit in the representation, so check 1 is what enforces it.
The cost is recomputed from ``dist`` and ``sides`` directly, NOT from
``ProblemInstance.cost`` or solution.py helpers, so a bug there cannot hide here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.solution import ProblemInstance, Solution


@dataclass
class FeasibilityReport:
    """Result of ``check_solution``.

    Attributes:
        is_valid: True iff all five checks passed.
        errors: One human-readable message per violation.
        recomputed_cost: Cost computed independently; ``None`` if ids were invalid.
    """

    is_valid: bool
    errors: list[str]
    recomputed_cost: int | None


def check_indices(routes: list[list[int]], n: int) -> list[str]:
    """Check 1: every entry is an int in 1..n."""
    errors = []
    for k, route in enumerate(routes):
        for c in route:
            if type(c) is not int or not 1 <= c <= n:
                errors.append(f"route {k}: invalid customer id {c!r} (must be int in 1..{n})")
    return errors


def check_coverage(routes: list[list[int]], n: int) -> list[str]:
    """Check 2: each customer 1..n appears exactly once over all routes."""
    counts = Counter(c for route in routes for c in route)
    errors = []
    for c in range(1, n + 1):
        if counts[c] == 0:
            errors.append(f"customer {c} not visited")
        elif counts[c] > 1:
            errors.append(f"customer {c} visited {counts[c]} times")
    return errors


def check_capacity(routes: list[list[int]], inst: ProblemInstance) -> list[str]:
    """Check 3: sum of demands on each route <= Q. Invalid ids are skipped (check 1 reports them)."""
    errors = []
    for k, route in enumerate(routes):
        load = sum(inst.demand[c] for c in route if type(c) is int and 1 <= c <= inst.n)
        if load > inst.capacity:
            errors.append(f"route {k}: load {load} exceeds capacity {inst.capacity}")
    return errors


def check_fleet(routes: list[list[int]], num_vehicles: int | None) -> list[str]:
    """Check 4: non-empty routes <= K. Empty routes use no vehicle, so they are not counted."""
    used = sum(1 for r in routes if r)
    if num_vehicles is not None and used > num_vehicles:
        return [f"{used} routes used, fleet size K={num_vehicles}"]
    return []


def recompute_cost(routes: list[list[int]], inst: ProblemInstance) -> int:
    """Total cost from scratch: sum over arcs 0->r[0]->...->r[-1]->0 of dist + crossing penalty."""
    total = 0
    for route in routes:
        if not route:
            continue
        path = [0, *route, 0]
        for a, b in zip(path, path[1:]):
            total += inst.dist[a][b]
            if inst.sides is not None and inst.sides[a] != inst.sides[b]:
                total += inst.bosphorus_penalty
    return total


def check_cost(reported: int | None, recomputed: int) -> list[str]:
    """Check 5: reported cost equals the independently recomputed cost."""
    if reported is None:
        return ["no reported cost (call Solution.evaluate or set cost)"]
    if reported != recomputed:
        return [f"reported cost {reported} != recomputed cost {recomputed}"]
    return []


def check_solution(sol: Solution, inst: ProblemInstance) -> FeasibilityReport:
    """Run all five checks and return a report. Never raises on bad routes."""
    errors = check_indices(sol.routes, inst.n)
    ids_ok = not errors
    errors += check_coverage(sol.routes, inst.n)
    errors += check_capacity(sol.routes, inst)
    errors += check_fleet(sol.routes, inst.num_vehicles)
    recomputed = recompute_cost(sol.routes, inst) if ids_ok else None
    if recomputed is None:
        errors.append("cost not checked: invalid customer ids")
    else:
        errors += check_cost(sol.cost, recomputed)
    return FeasibilityReport(not errors, errors, recomputed)
