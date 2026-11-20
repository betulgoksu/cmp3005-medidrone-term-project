"""Problem instance and solution representation for the CVRP.

Conventions (see CLAUDE.md):
- Node 0 is the depot, customers are 1..n.
- A solution is ``routes: list[list[int]]``; the depot is omitted from each route
  (every route implicitly starts and ends at 0).
- Distances use TSPLIB EUC_2D nearest-integer rounding.
- MediDrone only: crossing the Bosphorus (an arc between nodes on different sides)
  adds ``bosphorus_penalty`` to the arc cost. With penalty 0 / no sides, this is plain CVRP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def euc_2d(a: tuple[float, float], b: tuple[float, float]) -> int:
    """TSPLIB EUC_2D distance: Euclidean distance rounded to the nearest integer."""
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


@dataclass
class ProblemInstance:
    """A CVRP instance (standard CVRPLIB or MediDrone).

    Attributes:
        name: Instance name.
        capacity: Vehicle capacity Q.
        demand: Demand per node, length n+1; ``demand[0]`` (depot) must be 0.
        dist: Integer distance matrix, (n+1) x (n+1).
        num_vehicles: Fleet size K; ``None`` means unlimited.
        coords: Optional node coordinates (needed only for plotting).
        bosphorus_penalty: Extra cost added to every arc that crosses sides.
        sides: Optional side label per node (e.g. "EU"/"AS"), length n+1.
        cost: Arc cost matrix = dist + crossing penalty. Computed, not passed in.
    """

    name: str
    capacity: int
    demand: list[int]
    dist: list[list[int]]
    num_vehicles: int | None = None
    coords: list[tuple[float, float]] | None = None
    bosphorus_penalty: int = 0
    sides: list[str] | None = None
    cost: list[list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        size = len(self.demand)
        if self.demand[0] != 0:
            raise ValueError("depot demand (demand[0]) must be 0")
        if len(self.dist) != size or any(len(row) != size for row in self.dist):
            raise ValueError(f"dist must be {size}x{size}")
        if self.sides is not None and len(self.sides) != size:
            raise ValueError(f"sides must have length {size}")
        if self.bosphorus_penalty and self.sides is None:
            raise ValueError("bosphorus_penalty needs sides")
        # Precomputed so every algorithm reads one matrix; penalty is 0 for standard CVRPLIB.
        self.cost = [
            [self.dist[i][j] + self.crossing_penalty(i, j) for j in range(size)]
            for i in range(size)
        ]

    @classmethod
    def from_coords(
        cls,
        name: str,
        coords: list[tuple[float, float]],
        demand: list[int],
        capacity: int,
        num_vehicles: int | None = None,
        bosphorus_penalty: int = 0,
        sides: list[str] | None = None,
    ) -> ProblemInstance:
        """Build an instance whose distance matrix is computed with EUC_2D rounding."""
        dist = [[euc_2d(a, b) for b in coords] for a in coords]
        return cls(name, capacity, demand, dist, num_vehicles, coords, bosphorus_penalty, sides)

    @property
    def n(self) -> int:
        """Number of customers (depot excluded)."""
        return len(self.demand) - 1

    def crossing_penalty(self, i: int, j: int) -> int:
        """Bosphorus penalty for arc (i, j): the penalty if i and j are on different sides, else 0."""
        if self.sides is None or self.sides[i] == self.sides[j]:
            return 0
        return self.bosphorus_penalty


def route_cost(route: list[int], inst: ProblemInstance) -> int:
    """Cost of one route 0 -> route[0] -> ... -> route[-1] -> 0. An empty route costs 0."""
    if not route:
        return 0
    total = inst.cost[0][route[0]] + inst.cost[route[-1]][0]
    for a, b in zip(route, route[1:]):
        total += inst.cost[a][b]
    return total


def route_load(route: list[int], inst: ProblemInstance) -> int:
    """Total customer demand served by one route."""
    return sum(inst.demand[c] for c in route)


def solution_cost(routes: list[list[int]], inst: ProblemInstance) -> int:
    """Total cost of all routes."""
    return sum(route_cost(r, inst) for r in routes)


def to_giant_tour(routes: list[list[int]]) -> list[int]:
    """Join routes into one depot-delimited sequence, e.g. [[1, 2], [3]] -> [1, 2, 0, 3].

    The GA's customer permutation (no depots) is ``[c for c in tour if c != 0]``.
    """
    tour: list[int] = []
    for k, route in enumerate(routes):
        if k:
            tour.append(0)
        tour.extend(route)
    return tour


def from_giant_tour(tour: list[int]) -> list[list[int]]:
    """Inverse of ``to_giant_tour``: split at every 0, dropping empty routes."""
    routes: list[list[int]] = [[]]
    for node in tour:
        if node == 0:
            routes.append([])
        else:
            routes[-1].append(node)
    return [r for r in routes if r]


@dataclass
class Solution:
    """A CVRP solution.

    Attributes:
        routes: Customer sequences, depot omitted.
        cost: Reported total cost (set by the algorithm or by ``evaluate``);
            ``None`` until evaluated. feasibility.py recomputes and compares it.
    """

    routes: list[list[int]]
    cost: int | None = None

    def evaluate(self, inst: ProblemInstance) -> int:
        """Compute, store and return the total cost."""
        self.cost = solution_cost(self.routes, inst)
        return self.cost

    def loads(self, inst: ProblemInstance) -> list[int]:
        """Demand served by each route."""
        return [route_load(r, inst) for r in self.routes]
