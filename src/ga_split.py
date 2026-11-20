"""Giant-tour Genetic Algorithm with the Split DP decoder (Prins 2004 style).

Chromosome: a permutation of customers 1..n with no depots (a "giant tour").
Decoder: Split cuts the giant tour into consecutive capacity-feasible routes at
minimum total cost. So the GA searches over customer orders, and Split solves
"where should each route start and end" exactly for each order.

Fitness (lower is better) is the tuple (excess_routes, cost):
- excess_routes = how many routes over K even the best cut of this order needs (0 = feasible)
- cost = total cost of the decoded routes (``inst.cost``, Bosphorus penalty included)
Comparing tuples lexicographically means any fleet-feasible solution beats any
infeasible one, with no penalty weight to tune.

GA loop (one generation):
    elitism: the best ``elite`` individuals survive unchanged
    repeat until population is full:
        parents   <- binary tournament selection (x2)
        child     <- Order Crossover OX(p1, p2)
        child     <- mutation with probability pm (swap or inversion)
        decode child with Split, compute fitness
        with probability ls_prob: local search (2-opt + relocate) on the decoded
        routes, then re-encode them as the child's giant tour (memetic GA)
Every new best solution is checked with feasibility.py.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from src.clarke_wright import clarke_wright
from src.feasibility import check_solution
from src.solution import ProblemInstance, Solution, to_giant_tour

Fitness = tuple[int, int]  # (excess routes over K, total cost)


# --------------------------------------------------------------------------- Split

def split(tour: list[int], inst: ProblemInstance) -> list[list[int]]:
    """Optimal split of a giant tour into capacity-feasible routes (unlimited fleet).

    DAG: nodes 0..m (m = len(tour)); node i means "the first i customers are served".
    Arc (i, j), i < j, is the route tour[i], ..., tour[j-1] and exists only if its
    load <= Q. Its weight is that route's cost:
        w(i, j) = c(0, t_i) + sum_{k=i}^{j-2} c(t_k, t_{k+1}) + c(t_{j-1}, 0)
    The best split is the shortest path 0 -> m (Bellman on a DAG, nodes in order):
        V[0] = 0,   V[j] = min_{i < j, arc (i,j) exists} V[i] + w(i, j)

    Complexity: O(m^2) arcs in the worst case, each weight built in O(1) by extending
    the previous route by one customer. The inner loop stops once load > Q, so with
    at most L customers per route it is O(m * L) in practice (L << m when Q is tight).
    """
    m, cost, q = len(tour), inst.cost, inst.demand
    value = [math.inf] * (m + 1)
    pred = [-1] * (m + 1)
    value[0] = 0
    for i in range(m):
        load = route = 0
        for j in range(i + 1, m + 1):
            c = tour[j - 1]
            load += q[c]
            if load > inst.capacity:
                break
            route = cost[0][c] if j == i + 1 else route + cost[tour[j - 2]][c]
            cand = value[i] + route + cost[c][0]
            if cand < value[j]:
                value[j], pred[j] = cand, i
    if value[m] == math.inf:
        raise ValueError("a customer's demand exceeds Q")
    return _rebuild(tour, pred, m)


def split_bounded(tour: list[int], inst: ProblemInstance, max_routes: int) -> list[list[int]] | None:
    """Optimal split using at most ``max_routes`` routes, or None if impossible.

    Same DAG as ``split`` but the label also counts routes:
        V[r][j] = cheapest way to serve the first j customers with exactly r routes
        V[r][j] = min_i V[r-1][i] + w(i, j)
    Complexity O(K * m * L). Used only when ``split`` returns more than K routes.
    """
    m, cost, q = len(tour), inst.cost, inst.demand
    value = [[math.inf] * (m + 1) for _ in range(max_routes + 1)]
    pred = [[-1] * (m + 1) for _ in range(max_routes + 1)]
    value[0][0] = 0
    for r in range(1, max_routes + 1):
        for i in range(m):
            if value[r - 1][i] == math.inf:
                continue
            load = route = 0
            for j in range(i + 1, m + 1):
                c = tour[j - 1]
                load += q[c]
                if load > inst.capacity:
                    break
                route = cost[0][c] if j == i + 1 else route + cost[tour[j - 2]][c]
                cand = value[r - 1][i] + route + cost[c][0]
                if cand < value[r][j]:
                    value[r][j], pred[r][j] = cand, i
    best_r = min(range(max_routes + 1), key=lambda r: value[r][m])
    if value[best_r][m] == math.inf:
        return None
    routes, j = [], m
    for r in range(best_r, 0, -1):
        i = pred[r][j]
        routes.append(tour[i:j])
        j = i
    return routes[::-1]


def _rebuild(tour: list[int], pred: list[int], m: int) -> list[list[int]]:
    """Follow predecessor labels back from m to 0 to recover the routes."""
    routes, j = [], m
    while j > 0:
        routes.append(tour[pred[j]:j])
        j = pred[j]
    return routes[::-1]


def min_routes(tour: list[int], inst: ProblemInstance) -> int:
    """Fewest routes any split of this order can use: fill each route greedily (optimal for counting)."""
    count, load = 1, 0
    for c in tour:
        if load + inst.demand[c] > inst.capacity:
            count, load = count + 1, 0
        load += inst.demand[c]
    return count


def decode(tour: list[int], inst: ProblemInstance) -> tuple[Fitness, Solution]:
    """Giant tour -> (fitness, evaluated Solution). Respects K when this order allows it."""
    routes = split(tour, inst)
    excess = 0
    k = inst.num_vehicles
    if k is not None and len(routes) > k:
        bounded = split_bounded(tour, inst, k)
        if bounded is not None:
            routes = bounded
        else:
            excess = min_routes(tour, inst) - k
    sol = Solution(routes)
    sol.evaluate(inst)
    return (excess, sol.cost), sol


# --------------------------------------------------------------------------- local search

def nearest_neighbors(inst: ProblemInstance, count: int) -> list[list[int]]:
    """For each customer u, its ``count`` cheapest other customers (by inst.cost). Index 0 unused."""
    customers = range(1, inst.n + 1)
    return [[]] + [
        sorted((v for v in customers if v != u), key=lambda v: (inst.cost[u][v], v))[:count]
        for u in customers
    ]


def two_opt(route: list[int], cost: list[list[int]]) -> bool:
    """Intra-route 2-opt, first improvement, until no improving move. Changes ``route`` in place.

    Reversing path[i+1..j] replaces arcs (a, b) and (c, d) by (a, c) and (b, d):
        delta = cost(a, c) + cost(b, d) - cost(a, b) - cost(c, d)
    (costs are symmetric, so the reversed middle part costs the same).
    Returns True if the route changed.
    """
    path = [0, *route, 0]
    changed, again = False, True
    while again:
        again = False
        for i in range(len(path) - 3):
            for j in range(i + 2, len(path) - 1):
                a, b, c, d = path[i], path[i + 1], path[j], path[j + 1]
                if cost[a][c] + cost[b][d] - cost[a][b] - cost[c][d] < 0:
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    again = changed = True
    route[:] = path[1:-1]
    return changed


def relocate_pass(routes: list[list[int]], inst: ProblemInstance, neighbors: list[list[int]]) -> bool:
    """One pass of inter-route relocate over all customers, first improvement per customer.

    Customer u is moved next to (before or after) one of its nearest neighbors v in
    another route with room for it (load + q_u <= Q). Gain from removing u between p and x:
        cost(p, u) + cost(u, x) - cost(p, x)
    Cost of inserting u between a and b:  cost(a, u) + cost(u, b) - cost(a, b)
    The move is made if insertion cost < removal gain. Only neighbors are tried
    ("granular" search), so a pass costs O(n * G * L) instead of O(n^2).
    Routes may become empty; the caller drops them. Returns True if anything moved.
    """
    cost, q = inst.cost, inst.demand
    load = [sum(q[c] for c in r) for r in routes]
    route_of = {c: k for k, r in enumerate(routes) for c in r}
    moved = False
    for u in range(1, inst.n + 1):
        ka = route_of[u]
        ra = routes[ka]
        i = ra.index(u)
        p, x = (ra[i - 1] if i else 0), (ra[i + 1] if i + 1 < len(ra) else 0)
        gain = cost[p][u] + cost[u][x] - cost[p][x]
        for v in neighbors[u]:
            kb = route_of[v]
            if kb == ka or load[kb] + q[u] > inst.capacity:
                continue
            rb = routes[kb]
            j = rb.index(v)
            options = ((rb[j - 1] if j else 0, v, j), (v, rb[j + 1] if j + 1 < len(rb) else 0, j + 1))
            move = next(((a, b, pos) for a, b, pos in options
                         if cost[a][u] + cost[u][b] - cost[a][b] < gain), None)
            if move is not None:
                ra.pop(i)
                rb.insert(move[2], u)
                load[ka] -= q[u]
                load[kb] += q[u]
                route_of[u] = kb
                moved = True
                break
    return moved


def local_search(routes: list[list[int]], inst: ProblemInstance, neighbors: list[list[int]]
                 ) -> list[list[int]]:
    """Memetic improvement: alternate 2-opt (inside routes) and relocate (between routes)
    until neither improves. Never violates capacity, never adds a route, never increases cost.

    Returns new route lists (input is not modified), with empty routes removed.
    """
    routes = [r[:] for r in routes]
    improved = True
    while improved:
        improved = False
        for r in routes:
            improved |= two_opt(r, inst.cost)
        improved |= relocate_pass(routes, inst, neighbors)
    return [r for r in routes if r]


# --------------------------------------------------------------------------- operators

def order_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """OX: copy p1[a:b], then fill the other positions with p2's remaining customers in p2 order, starting after b."""
    m = len(p1)
    a, b = sorted(rng.sample(range(m + 1), 2))
    child = [0] * m
    child[a:b] = p1[a:b]
    kept = set(p1[a:b])
    fill = [c for c in p2[b:] + p2[:b] if c not in kept]
    for pos, c in zip([*range(b, m), *range(a)], fill):
        child[pos] = c
    return child


def mutate(tour: list[int], rng: random.Random) -> None:
    """In place: swap two customers, or reverse a segment (inversion), with equal probability."""
    i, j = sorted(rng.sample(range(len(tour)), 2))
    if rng.random() < 0.5:
        tour[i], tour[j] = tour[j], tour[i]
    else:
        tour[i:j + 1] = tour[i:j + 1][::-1]


def tournament(pop: list[tuple[Fitness, list[int], Solution]], rng: random.Random) -> list[int]:
    """Binary tournament: pick two random individuals, return the fitter one's tour."""
    a, b = rng.sample(pop, 2)
    return min(a, b, key=lambda ind: ind[0])[1]


# --------------------------------------------------------------------------- GA

@dataclass
class GAResult:
    """Output of ``run_ga``.

    Attributes:
        best: Best solution found (fleet-feasible if any was found).
        fitness: (excess routes, cost) of ``best``.
        history: Best cost after each generation (for convergence plots).
        generations: Generations actually run.
    """

    best: Solution
    fitness: Fitness
    history: list[int]
    generations: int


def run_ga(
    inst: ProblemInstance,
    seed: int,
    pop_size: int = 50,
    generations: int = 200,
    pm: float = 0.2,
    elite: int = 2,
    ls_prob: float = 0.2,
    neighbor_count: int = 10,
    time_limit: float | None = None,
) -> GAResult:
    """Run the giant-tour GA.

    Args:
        inst: Problem instance.
        seed: Seed for the GA's own ``random.Random`` (required, no global state).
        pop_size: Population size (>= 2).
        generations: Maximum number of generations.
        pm: Mutation probability per child.
        elite: Individuals copied unchanged to the next generation.
        ls_prob: Probability that a child gets local search (memetic GA); 0 = plain GA.
        neighbor_count: G, nearest neighbors tried by relocate.
        time_limit: Optional wall-clock limit in seconds (checked once per generation).

    Initial population: the Clarke-Wright giant tour plus random permutations, all
    improved by local search when ls_prob > 0.
    The best solution is checked with feasibility.py whenever it improves; a capacity,
    coverage or cost error there is a bug and raises AssertionError. A fleet violation
    is allowed only if no fleet-feasible order was found (fitness[0] > 0).
    """
    rng = random.Random(seed)
    start = time.perf_counter()
    customers = list(range(1, inst.n + 1))
    neighbors = nearest_neighbors(inst, neighbor_count)

    tours = [[c for c in to_giant_tour(clarke_wright(inst).routes) if c != 0]]
    while len(tours) < pop_size:
        tours.append(rng.sample(customers, len(customers)))
    pop = [make_individual(t, inst, neighbors, ls_prob > 0) for t in tours]
    pop.sort(key=lambda ind: ind[0])
    best = pop[0]
    _verify(best, inst)
    history = [best[0][1]]

    gen = 0
    while gen < generations and (time_limit is None or time.perf_counter() - start < time_limit):
        gen += 1
        children = pop[:elite]
        while len(children) < pop_size:
            child = order_crossover(tournament(pop, rng), tournament(pop, rng), rng)
            if rng.random() < pm:
                mutate(child, rng)
            children.append(make_individual(child, inst, neighbors, rng.random() < ls_prob))
        pop = sorted(children, key=lambda ind: ind[0])
        if pop[0][0] < best[0]:
            best = pop[0]
            _verify(best, inst)
        history.append(best[0][1])

    return GAResult(best[2], best[0], history, gen)


def make_individual(tour: list[int], inst: ProblemInstance, neighbors: list[list[int]],
                    improve: bool) -> tuple[Fitness, list[int], Solution]:
    """Decode a tour; if ``improve``, run local search and re-encode (memetic step).

    After local search the routes are concatenated into a new giant tour and decoded
    again; Split can only match or beat the local-search cut of that order. The
    improved individual is kept only if its fitness is not worse.
    """
    fit, sol = decode(tour, inst)
    if improve:
        new_tour = [c for r in local_search(sol.routes, inst, neighbors) for c in r]
        new_fit, new_sol = decode(new_tour, inst)
        if new_fit <= fit:
            return new_fit, new_tour, new_sol
    return fit, tour, sol


def _verify(ind: tuple[Fitness, list[int], Solution], inst: ProblemInstance) -> None:
    """Check an incumbent with feasibility.py; only a fleet violation may remain, and only if fitness says so."""
    report = check_solution(ind[2], inst)
    allowed = [e for e in report.errors if "fleet size" in e] if ind[0][0] > 0 else []
    assert report.errors == allowed, report.errors


def main() -> None:
    """Run the GA (seed 1) on every instance in data/ and print feasibility-checked results."""
    from src.parser_generator import ROOT, parse_cvrplib

    for path in sorted((ROOT / "data").glob("*/*.vrp")):
        inst = parse_cvrplib(path)
        res = run_ga(inst, seed=1, time_limit=30)
        report = check_solution(res.best, inst)
        status = "valid" if report.is_valid else f"INVALID {report.errors}"
        print(f"{inst.name:24} cost {res.best.cost:7}  routes {len(res.best.routes):3}  "
              f"K={inst.num_vehicles}  gens {res.generations:4}  {status}")


if __name__ == "__main__":
    main()
