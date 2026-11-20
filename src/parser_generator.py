"""Instance I/O, instance generators and the exact solver for tiny instances.

- parse_cvrplib / write_instance: TSPLIB/CVRPLIB EUC_2D files <-> ProblemInstance.
  Depot becomes node 0, other nodes keep file order as 1..n.
  Extra optional keys for MediDrone: VEHICLES, BOSPHORUS_PENALTY, SIDE_SECTION.
- parse_solution / write_solution: CVRPLIB .sol files ("Route #k: ..." / "Cost X").
- generate_medidrone: n=100 clustered Istanbul-style instance with Bosphorus penalty.
- generate_tiny + solve_exact: small instances with a provably optimal solution (DP).

Run ``python -m src.parser_generator`` to (re)generate data/medidrone and data/tiny.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path

from src.solution import ProblemInstance, Solution

ROOT = Path(__file__).resolve().parent.parent
BOSPHORUS_X = 500  # x < 500 is Europe, x >= 500 is Asia


# --------------------------------------------------------------------------- parsing

def parse_cvrplib(path: str | Path) -> ProblemInstance:
    """Parse a TSPLIB/CVRPLIB EUC_2D file into a ProblemInstance.

    Fleet size K comes from a ``VEHICLES`` key if present, else from ``-k<K>`` in the
    name, else is ``None`` (unlimited).
    """
    path = Path(path)
    header: dict[str, str] = {}
    sections: dict[str, list[list[str]]] = {}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if line.endswith("_SECTION"):
            current = line
            sections[current] = []
        elif ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().upper()] = value.strip()
        elif current is not None:
            sections[current].append(line.split())

    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError(f"{path}: only EDGE_WEIGHT_TYPE EUC_2D is supported")
    for needed in ("NODE_COORD_SECTION", "DEMAND_SECTION"):
        if needed not in sections:
            raise ValueError(f"{path}: missing {needed}")

    coords = {int(r[0]): (float(r[1]), float(r[2])) for r in sections["NODE_COORD_SECTION"]}
    demand = {int(r[0]): int(r[1]) for r in sections["DEMAND_SECTION"]}
    depot = int(sections["DEPOT_SECTION"][0][0]) if "DEPOT_SECTION" in sections else 1
    order = [depot] + [i for i in coords if i != depot]  # file id of new index 0, 1, ..., n

    sides = None
    if "SIDE_SECTION" in sections:
        side_of = {int(r[0]): r[1] for r in sections["SIDE_SECTION"]}
        sides = [side_of[i] for i in order]

    name = header.get("NAME", path.stem)
    if "VEHICLES" in header:
        k: int | None = int(header["VEHICLES"])
    else:
        match = re.search(r"-k(\d+)", name)
        k = int(match.group(1)) if match else None

    return ProblemInstance.from_coords(
        name=name,
        coords=[coords[i] for i in order],
        demand=[demand[i] for i in order],
        capacity=int(header["CAPACITY"]),
        num_vehicles=k,
        bosphorus_penalty=int(header.get("BOSPHORUS_PENALTY", 0)),
        sides=sides,
    )


def write_instance(inst: ProblemInstance, path: str | Path) -> None:
    """Write an instance in CVRPLIB format (depot = file node 1). Inverse of parse_cvrplib."""
    if inst.coords is None:
        raise ValueError("write_instance needs coords")
    lines = [
        f"NAME : {inst.name}",
        "TYPE : CVRP",
        f"DIMENSION : {inst.n + 1}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {inst.capacity}",
    ]
    if inst.num_vehicles is not None:
        lines.append(f"VEHICLES : {inst.num_vehicles}")
    if inst.bosphorus_penalty:
        lines.append(f"BOSPHORUS_PENALTY : {inst.bosphorus_penalty}")
    lines.append("NODE_COORD_SECTION")
    lines += [f"{i + 1} {x:g} {y:g}" for i, (x, y) in enumerate(inst.coords)]
    lines.append("DEMAND_SECTION")
    lines += [f"{i + 1} {d}" for i, d in enumerate(inst.demand)]
    if inst.sides is not None:
        lines.append("SIDE_SECTION")
        lines += [f"{i + 1} {s}" for i, s in enumerate(inst.sides)]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
    Path(path).write_text("\n".join(lines) + "\n")


def parse_solution(path: str | Path) -> Solution:
    """Parse a CVRPLIB .sol file. Customer ids there already use our 1..n numbering."""
    routes: list[list[int]] = []
    cost = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("Route"):
            routes.append([int(c) for c in line.split(":", 1)[1].split()])
        elif line.startswith("Cost"):
            cost = int(float(line.split()[1]))
    return Solution(routes, cost)


def write_solution(sol: Solution, path: str | Path) -> None:
    """Write a solution in CVRPLIB .sol format."""
    lines = [f"Route #{k + 1}: {' '.join(map(str, r))}" for k, r in enumerate(sol.routes)]
    lines.append(f"Cost {sol.cost}")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- MediDrone

def generate_medidrone(seed: int, n: int = 100) -> tuple[ProblemInstance, list[int]]:
    """Generate a MediDrone instance on [0, 1000]^2 with the depot at (500, 500).

    - 80% of customers in 4-6 Gaussian clusters, centres alternating Europe/Asia
      so both sides always get clusters; 20% uniform over the square.
    - Demand uniform in [1, 30], Q = 100, K = ceil(sum(q) / Q) + 1.
    - Side: "EU" if x < 500 else "AS" (so the depot, x = 500, is on the Asian side).
    - Arcs between sides cost an extra 150 (Bosphorus crossing).

    Returns:
        (instance, cluster label per node); label -1 = depot or uniform customer.
    """
    rng = random.Random(seed)
    num_clusters = rng.randint(4, 6)
    centres = [
        (rng.uniform(100, 450) if c % 2 == 0 else rng.uniform(550, 900), rng.uniform(100, 900))
        for c in range(num_clusters)
    ]
    n_uniform = round(0.2 * n)

    coords: list[tuple[float, float]] = [(500, 500)]
    labels = [-1]
    for i in range(n - n_uniform):
        c = i % num_clusters  # round-robin keeps cluster sizes balanced
        cx, cy = centres[c]
        coords.append((_clip(rng.gauss(cx, 50)), _clip(rng.gauss(cy, 50))))
        labels.append(c)
    for _ in range(n_uniform):
        coords.append((rng.randint(0, 1000), rng.randint(0, 1000)))
        labels.append(-1)

    demand = [0] + [rng.randint(1, 30) for _ in range(n)]
    capacity = 100
    inst = ProblemInstance.from_coords(
        name=f"medidrone-n{n}-s{seed}",
        coords=coords,
        demand=demand,
        capacity=capacity,
        num_vehicles=math.ceil(sum(demand) / capacity) + 1,
        bosphorus_penalty=150,
        sides=["EU" if x < BOSPHORUS_X else "AS" for x, _ in coords],
    )
    return inst, labels


def _clip(v: float) -> int:
    """Round to an integer coordinate inside [0, 1000]."""
    return min(1000, max(0, round(v)))


def plot_medidrone(inst: ProblemInstance, labels: list[int], path: str | Path) -> None:
    """Save a plot: both sides shaded, Bosphorus line at x=500, clusters, uniform customers, depot."""
    import matplotlib  # imported here so the rest of the module works without matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axvspan(0, BOSPHORUS_X, color="tab:blue", alpha=0.06, label="Europe (x < 500)")
    ax.axvspan(BOSPHORUS_X, 1000, color="tab:orange", alpha=0.06, label="Asia (x >= 500)")
    ax.axvline(BOSPHORUS_X, color="tab:blue", lw=2, ls="--", label="Bosphorus (x = 500)")

    for c in sorted(set(labels) - {-1}):
        pts = [inst.coords[i] for i in range(1, inst.n + 1) if labels[i] == c]
        ax.scatter(*zip(*pts), s=25, label=f"cluster {c + 1}")
    uniform = [inst.coords[i] for i in range(1, inst.n + 1) if labels[i] == -1]
    if uniform:
        ax.scatter(*zip(*uniform), s=25, marker="x", color="gray", label="uniform")
    ax.scatter(*inst.coords[0], s=250, marker="*", color="red", edgecolor="black", zorder=5, label="depot")

    ax.set(xlim=(0, 1000), ylim=(0, 1000), aspect="equal", xlabel="x", ylabel="y",
           title=f"{inst.name}  (Q={inst.capacity}, K={inst.num_vehicles}, P={inst.bosphorus_penalty})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- tiny + exact

def generate_tiny(n: int, seed: int) -> ProblemInstance:
    """Small random instance: coords in [0, 100]^2, depot (50, 50), demand 1..10, Q = 20."""
    rng = random.Random(seed)
    coords = [(50, 50)] + [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    demand = [0] + [rng.randint(1, 10) for _ in range(n)]
    capacity = 20
    return ProblemInstance.from_coords(
        name=f"tiny-n{n}-s{seed}",
        coords=coords,
        demand=demand,
        capacity=capacity,
        num_vehicles=math.ceil(sum(demand) / capacity) + 1,
    )


def solve_exact(inst: ProblemInstance) -> Solution:
    """Provably optimal CVRP solution by dynamic programming (for n <= 12 only).

    Step 1 (Held-Karp): for every customer subset S with load <= Q, the cheapest
            single route visiting exactly S.
    Step 2 (set partitioning): f[r][M] = cheapest way to serve set M with r routes,
            f[r][M] = min over S in M containing M's lowest customer of route[S] + f[r-1][M \\ S].
    Answer: min over r <= K of f[r][all]. Customer i is bit i-1.
    """
    n = inst.n
    if n > 12:
        raise ValueError("solve_exact is exponential; use it only for n <= 12")
    c, inf, full = inst.cost, math.inf, 1 << n
    load = [0] * full
    for mask in range(1, full):
        low = mask & -mask
        load[mask] = load[mask ^ low] + inst.demand[low.bit_length()]

    # Step 1: path[mask][j] = cheapest 0 -> ... -> j visiting exactly mask.
    path = [[inf] * n for _ in range(full)]
    prev = [[-1] * n for _ in range(full)]
    for j in range(n):
        if inst.demand[j + 1] <= inst.capacity:
            path[1 << j][j] = c[0][j + 1]
    for mask in range(1, full):
        for j in range(n):
            if path[mask][j] == inf:
                continue
            for k in range(n):
                nxt = mask | (1 << k)
                if nxt == mask or load[nxt] > inst.capacity:
                    continue
                cand = path[mask][j] + c[j + 1][k + 1]
                if cand < path[nxt][k]:
                    path[nxt][k], prev[nxt][k] = cand, j
    route_cost = [inf] * full
    route_last = [-1] * full
    for mask in range(1, full):
        for j in range(n):
            cand = path[mask][j] + c[j + 1][0]
            if cand < route_cost[mask]:
                route_cost[mask], route_last[mask] = cand, j

    # Step 2: partition customers into at most K routes.
    max_routes = inst.num_vehicles if inst.num_vehicles is not None else n
    f = [[inf] * full for _ in range(max_routes + 1)]
    choice = [[0] * full for _ in range(max_routes + 1)]
    f[0][0] = 0
    for r in range(1, max_routes + 1):
        for mask in range(1, full):
            low = mask & -mask
            rest = mask ^ low
            sub = rest
            while True:
                s = sub | low
                cand = route_cost[s] + f[r - 1][mask ^ s]
                if cand < f[r][mask]:
                    f[r][mask], choice[r][mask] = cand, s
                if sub == 0:
                    break
                sub = (sub - 1) & rest
    best_r = min(range(max_routes + 1), key=lambda r: f[r][full - 1])
    if f[best_r][full - 1] == inf:
        raise ValueError(f"{inst.name}: no feasible solution")

    routes, mask = [], full - 1
    for r in range(best_r, 0, -1):
        s = choice[r][mask]
        routes.append(_rebuild_route(s, route_last[s], prev))
        mask ^= s
    return Solution(routes, int(f[best_r][full - 1]))


def _rebuild_route(mask: int, last: int, prev: list[list[int]]) -> list[int]:
    """Follow Held-Karp predecessors back from ``last`` to recover the route order."""
    route = []
    while mask:
        route.append(last + 1)
        mask, last = mask ^ (1 << last), prev[mask][last]
    return route[::-1]


# --------------------------------------------------------------------------- main

MEDIDRONE_SEED = 11372
TINY_SPECS = [(7, 1), (8, 2), (9, 3)]  # (n, seed)


def main() -> None:
    """Generate and save the MediDrone instance (+ plot) and the tiny instances (+ optima)."""
    from src.feasibility import check_solution

    med_dir, tiny_dir = ROOT / "data" / "medidrone", ROOT / "data" / "tiny"
    inst, labels = generate_medidrone(seed=MEDIDRONE_SEED)
    write_instance(inst, med_dir / f"{inst.name}.vrp")
    print(f"{inst.name}: n={inst.n}, sum(q)={sum(inst.demand)}, Q={inst.capacity}, K={inst.num_vehicles}")
    try:
        plot_medidrone(inst, labels, med_dir / f"{inst.name}.png")
    except ImportError:
        print("  matplotlib not installed: plot skipped (pip install matplotlib)")

    for n, seed in TINY_SPECS:
        tiny = generate_tiny(n, seed)
        sol = solve_exact(tiny)
        report = check_solution(sol, tiny)
        assert report.is_valid, report.errors
        write_instance(tiny, tiny_dir / f"{tiny.name}.vrp")
        write_solution(sol, tiny_dir / f"{tiny.name}.sol")
        print(f"{tiny.name}: optimal cost {sol.cost}, {len(sol.routes)} routes, K={tiny.num_vehicles}")


if __name__ == "__main__":
    main()

