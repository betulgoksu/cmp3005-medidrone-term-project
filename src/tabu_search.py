"""Route-based Tabu Search for the CVRP.

Start: Clarke-Wright solution (or a given Solution). Each iteration evaluates the
whole neighborhood, takes the best admissible move (even if it worsens the cost),
and makes the reverse move tabu for ``tenure`` iterations.

Moves (all costs from ``inst.cost``, so the Bosphorus penalty is included; every
delta is O(1) because only the arcs next to the changed positions change):

  relocate  move customer u from route A to any position in another route B
            (load_B + q_u <= Q). May empty route A (one route fewer).
  swap      exchange customer u in route A with v in another route B
            (both new loads <= Q).
  2-opt     reverse a segment inside one route.

Moves inside one route other than 2-opt (intra relocate/swap) are not used: 2-opt
already reorders a route, and leaving them out keeps every delta formula simple.

Fitness = (routes_over_K, cost), compared lexicographically, same as the GA: any
solution with <= K non-empty routes beats any solution with more, no penalty weight.

Tabu attributes (stored as tabu_until[key] = last iteration it is still tabu):
  ("node", u, r)   customer u may not re-enter route r   (set by relocate and swap)
  ("arc", a, b)    arc {a, b} may not be re-added         (set by 2-opt)
Aspiration: a tabu move is allowed if its fitness beats the best fitness found so far.
Tenure: fixed int, or (low, high) for a random tenure drawn per move (dynamic).

Neighborhood size and time per iteration (n customers, R routes, L = max route length):
  relocate  n * (n + R) candidate positions   O(n^2)
  swap      <= n^2 / 2 customer pairs          O(n^2)
  2-opt     sum over routes of L^2 / 2         O(n * L)
  => O(n^2) candidates, each evaluated in O(1), so O(n^2) per iteration
     (plus O(n) to apply the chosen move).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.clarke_wright import clarke_wright
from src.feasibility import check_solution
from src.solution import ProblemInstance, Solution

Fitness = tuple[int, int]  # (routes over K, total cost)
Cost = list[list[int]]


# --------------------------------------------------------------------------- deltas
# P is a route as a path with the depot at both ends: [0, c1, ..., cL, 0].
# Customer indices in P run from 1 to L.

def relocate_delta(pa: list[int], i: int, pb: list[int], j: int, c: Cost) -> int:
    """Cost change of moving pa[i] to between pb[j] and pb[j+1] (different routes).

    removal:   - c(p, u) - c(u, x) + c(p, x)     p, x = neighbours of u in A
    insertion: + c(a, u) + c(u, b) - c(a, b)     a, b = pb[j], pb[j+1]
    """
    p, u, x = pa[i - 1], pa[i], pa[i + 1]
    a, b = pb[j], pb[j + 1]
    return c[p][x] - c[p][u] - c[u][x] + c[a][u] + c[u][b] - c[a][b]


def swap_delta(pa: list[int], i: int, pb: list[int], j: int, c: Cost) -> int:
    """Cost change of exchanging u = pa[i] and v = pb[j] (different routes): each takes the other's neighbours."""
    u, v = pa[i], pb[j]
    pu, nu, pv, nv = pa[i - 1], pa[i + 1], pb[j - 1], pb[j + 1]
    return (c[pu][v] + c[v][nu] - c[pu][u] - c[u][nu]
            + c[pv][u] + c[u][nv] - c[pv][v] - c[v][nv])


def two_opt_delta(p: list[int], i: int, j: int, c: Cost) -> int:
    """Cost change of reversing p[i..j] (1 <= i < j <= L): arcs (p[i-1], p[i]), (p[j], p[j+1])
    become (p[i-1], p[j]), (p[i], p[j+1]). Symmetric costs, so the middle is unchanged."""
    return c[p[i - 1]][p[j]] + c[p[i]][p[j + 1]] - c[p[i - 1]][p[i]] - c[p[j]][p[j + 1]]


# --------------------------------------------------------------------------- tabu helpers

def draw_tenure(tenure: int | tuple[int, int], rng: random.Random) -> int:
    """Fixed tenure, or a random tenure in [low, high] (dynamic)."""
    return tenure if isinstance(tenure, int) else rng.randint(*tenure)


def is_admissible(keys: list[tuple], tabu_until: dict[tuple, int], it: int,
                  fit: Fitness, best_fit: Fitness) -> bool:
    """A move is allowed if none of its attributes is tabu at iteration ``it``,
    or (aspiration) if it would give a new best-so-far fitness."""
    return fit < best_fit or all(tabu_until.get(k, -1) < it for k in keys)


def _arc(a: int, b: int) -> tuple[str, int, int]:
    return ("arc", min(a, b), max(a, b))


# --------------------------------------------------------------------------- search

@dataclass
class TabuResult:
    """Output of ``run_tabu``.

    Attributes:
        best: Best solution found (cost evaluated).
        fitness: (routes over K, cost) of ``best``.
        history: Best cost after each iteration (index 0 = start solution).
        iterations: Iterations actually run.
    """

    best: Solution
    fitness: Fitness
    history: list[int]
    iterations: int


def run_tabu(
    inst: ProblemInstance,
    seed: int,
    start: Solution | None = None,
    max_iterations: int = 1000,
    tenure: int | tuple[int, int] = (7, 15),
    time_limit: float | None = None,
) -> TabuResult:
    """Run Tabu Search.

    Args:
        inst: Problem instance.
        seed: Seed for ``random.Random``: used for dynamic tenure and for breaking
            ties between equally good moves. Same seed -> same run.
        start: Starting solution; default is the Clarke-Wright solution.
        max_iterations: Maximum number of iterations.
        tenure: Fixed tenure (int) or (low, high) for dynamic tenure.
        time_limit: Optional wall-clock limit in seconds, checked before each iteration.

    If every move is tabu and none aspirates, the iteration makes no move and the
    tabu entries simply expire. Each new best is checked with feasibility.py, which
    also confirms that the incrementally tracked cost equals the recomputed cost.
    """
    rng = random.Random(seed)
    t0 = time.perf_counter()
    c, q, cap = inst.cost, inst.demand, inst.capacity
    k_max = inst.num_vehicles

    routes = [r[:] for r in (start or clarke_wright(inst)).routes if r]
    load = [sum(q[x] for x in r) for r in routes]
    cost = sum(_route_cost(r, c) for r in routes)
    used = len(routes)

    def fitness(cost_: int, used_: int) -> Fitness:
        return (max(0, used_ - k_max) if k_max is not None else 0, cost_)

    best_fit = fitness(cost, used)
    best = _snapshot(routes, cost)
    _verify(best, best_fit, inst)
    history = [cost]
    tabu_until: dict[tuple, int] = {}

    it = 0
    while it < max_iterations and (time_limit is None or time.perf_counter() - t0 < time_limit):
        it += 1
        paths = [[0, *r, 0] for r in routes]
        chosen = None  # (fit, delta, move); ties broken by reservoir sampling
        ties = 0

        def consider(fit: Fitness, delta: int, move: tuple, keys: list[tuple]) -> None:
            nonlocal chosen, ties
            if not is_admissible(keys, tabu_until, it, fit, best_fit):
                return
            if chosen is None or fit < chosen[0]:
                chosen, ties = (fit, delta, move), 1
            elif fit == chosen[0]:
                ties += 1
                if rng.random() * ties < 1:
                    chosen = (fit, delta, move)

        for a, pa in enumerate(paths):
            la = len(pa) - 2
            for i in range(1, la + 1):
                u = pa[i]
                for b, pb in enumerate(paths):
                    if b == a:
                        continue
                    lb = len(pb) - 2
                    # relocate u -> route b
                    if load[b] + q[u] <= cap:
                        used_after = used - (la == 1) + (lb == 0)
                        for j in range(lb + 1):
                            d = relocate_delta(pa, i, pb, j, c)
                            consider(fitness(cost + d, used_after), d, ("relocate", a, i, b, j),
                                     [("node", u, b)])
                    # swap u <-> v (each unordered pair once)
                    if b > a:
                        for j in range(1, lb + 1):
                            v = pb[j]
                            if load[a] - q[u] + q[v] <= cap and load[b] - q[v] + q[u] <= cap:
                                d = swap_delta(pa, i, pb, j, c)
                                consider(fitness(cost + d, used), d, ("swap", a, i, b, j),
                                         [("node", u, b), ("node", v, a)])
            # 2-opt inside route a
            for i in range(1, la):
                for j in range(i + 1, la + 1):
                    d = two_opt_delta(pa, i, j, c)
                    consider(fitness(cost + d, used), d, ("2opt", a, i, j),
                             [_arc(pa[i - 1], pa[j]), _arc(pa[i], pa[j + 1])])

        if chosen is None:
            history.append(best_fit[1])
            continue
        fit, delta, move = chosen
        ten = draw_tenure(tenure, rng)
        for key in _apply(move, routes, paths, load, q):
            tabu_until[key] = it + ten
        cost += delta
        used = sum(1 for r in routes if r)
        if fit < best_fit:
            best_fit, best = fit, _snapshot(routes, cost)
            _verify(best, best_fit, inst)
        history.append(best_fit[1])

    return TabuResult(best, best_fit, history, it)


def _apply(move: tuple, routes: list[list[int]], paths: list[list[int]],
           load: list[int], q: list[int]) -> list[tuple]:
    """Apply a move in place; return the tabu attributes that forbid undoing it."""
    kind = move[0]
    if kind == "relocate":
        _, a, i, b, j = move
        u = routes[a].pop(i - 1)
        routes[b].insert(j, u)
        load[a] -= q[u]
        load[b] += q[u]
        return [("node", u, a)]
    if kind == "swap":
        _, a, i, b, j = move
        u, v = routes[a][i - 1], routes[b][j - 1]
        routes[a][i - 1], routes[b][j - 1] = v, u
        load[a] += q[v] - q[u]
        load[b] += q[u] - q[v]
        return [("node", u, a), ("node", v, b)]
    _, a, i, j = move
    p = paths[a]
    routes[a][i - 1:j] = routes[a][i - 1:j][::-1]
    return [_arc(p[i - 1], p[i]), _arc(p[j], p[j + 1])]


def _route_cost(route: list[int], c: Cost) -> int:
    path = [0, *route, 0]
    return sum(c[x][y] for x, y in zip(path, path[1:]))


def _snapshot(routes: list[list[int]], cost: int) -> Solution:
    """Copy of the current routes (empty ones dropped) with the tracked cost as reported cost."""
    return Solution([r[:] for r in routes if r], cost)


def _verify(sol: Solution, fit: Fitness, inst: ProblemInstance) -> None:
    """feasibility.py check; only a fleet violation may remain, and only if fitness says so."""
    report = check_solution(sol, inst)
    allowed = [e for e in report.errors if "fleet size" in e] if fit[0] > 0 else []
    assert report.errors == allowed, report.errors


def main() -> None:
    """Run Tabu Search (seed 1) on every instance in data/ and print feasibility-checked results."""
    from src.parser_generator import ROOT, parse_cvrplib

    for path in sorted((ROOT / "data").glob("*/*.vrp")):
        inst = parse_cvrplib(path)
        res = run_tabu(inst, seed=1, time_limit=30)
        report = check_solution(res.best, inst)
        status = "valid" if report.is_valid else f"INVALID {report.errors}"
        print(f"{inst.name:24} cost {res.best.cost:7}  routes {len(res.best.routes):3}  "
              f"K={inst.num_vehicles}  iters {res.iterations:5}  {status}")


if __name__ == "__main__":
    main()
