"""Analysis on top of the experiment rows: statistics, parameter study, Bosphorus, plots.

- ``wilcoxon_exact`` / ``wilcoxon_table``: exact two-sided Wilcoxon signed-rank test,
  GA vs Tabu paired by seed (standard library only, 2^m sign patterns, m <= 10 here).
- ``tenure_study``: Tabu gap for several tabu tenures (parameter sensitivity).
- ``crossings`` / ``crossing_table``: how many routes cross the Bosphorus and how much
  of the cost is penalty.
- ``plot_convergence``, ``plot_routes``, ``plot_tenure``: figures for the report.
- ``read_runs``: reload results/runs.csv, so the analysis can be redone without new runs.

Every solution taken from the rows is re-checked with feasibility.py (rule 2).
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import platform
import statistics
import sys
from pathlib import Path

from src.feasibility import check_solution
from src.solution import ProblemInstance, Solution
from src.tabu_search import run_tabu

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALGO_COLORS = {"memetic_ga": "#eb6834", "tabu_search": "#1baf7a"}
ALGO_LABELS = {"clarke_wright": "Clarke-Wright", "memetic_ga": "Memetic GA", "tabu_search": "Tabu Search"}


# --------------------------------------------------------------------------- I/O

def hardware_info() -> str:
    """One text block describing the machine and Python used for the timings."""
    return (f"platform: {platform.platform()}\nprocessor: {platform.processor()}\n"
            f"machine: {platform.machine()}\nlogical CPUs: {os.cpu_count()}\n"
            f"python: {sys.version.split()[0]} ({platform.python_implementation()})\n"
            "runs are sequential and single-threaded\n")


def read_runs(path: Path) -> list[dict]:
    """Load runs.csv back into typed rows (ints, floats, bools, None for empty cells)."""
    def num(v: str, kind=int):
        return None if v == "" else kind(v)

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r.update(n=int(r["n"]), K=num(r["K"]), seed=num(r["seed"]), cost=int(r["cost"]),
                     routes=int(r["routes"]), is_valid=r["is_valid"] == "True", bks=num(r["bks"]),
                     gap_pct=num(r["gap_pct"], float), runtime_s=float(r["runtime_s"]),
                     iterations=num(r["iterations"]), time_limit_s=num(r["time_limit_s"], float))
            rows.append(r)
    return rows


def read_tenure_study(path: Path) -> list[dict]:
    """Load tenure_study.csv back into typed rows (for re-plotting without new runs)."""
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r, seed=int(r["seed"]), cost=int(r["cost"]), is_valid=r["is_valid"] == "True",
                     gap_pct=float(r["gap_pct"]), iterations=int(r["iterations"]),
                     time_limit_s=float(r["time_limit_s"])) for r in csv.DictReader(f)]


def parse_routes(text: str) -> list[list[int]]:
    """Inverse of ``routes_to_str``: '1 2|3' -> [[1, 2], [3]]."""
    return [[int(c) for c in part.split()] for part in text.split("|") if part.strip()]


def checked_solution(row: dict, inst: ProblemInstance) -> Solution:
    """Solution stored in a row, re-validated with feasibility.py (must be valid)."""
    sol = Solution(parse_routes(row["solution"]), row["cost"])
    report = check_solution(sol, inst)
    assert report.is_valid, (row["instance"], row["algorithm"], row["seed"], report.errors)
    return sol


def best_valid_row(rows: list[dict], instance: str, algorithm: str | None = None) -> dict | None:
    """Cheapest valid row for an instance (optionally one algorithm); ties -> lowest seed."""
    cands = [r for r in rows if r["instance"] == instance and r["is_valid"]
             and (algorithm is None or r["algorithm"] == algorithm)]
    return min(cands, key=lambda r: (r["cost"], r["seed"] or 0)) if cands else None


# --------------------------------------------------------------------------- statistics

def wilcoxon_exact(x: list[float], y: list[float]) -> tuple[float, float, int]:
    """Exact two-sided Wilcoxon signed-rank test on pairs (x_i, y_i).

    Zero differences are dropped; tied |d| get average ranks. W = min(W+, W-).
    p = share of the 2^m equally likely sign patterns with min(W+, W-) <= observed W.
    Returns (W, p, m). With m = 0 (all pairs equal) p = 1.
    """
    d = [a - b for a, b in zip(x, y) if a != b]
    m = len(d)
    if m == 0:
        return 0.0, 1.0, 0
    order = sorted(range(m), key=lambda i: abs(d[i]))
    ranks = [0.0] * m
    i = 0
    while i < m:  # average ranks over runs of equal |d|
        j = i
        while j + 1 < m and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    total = sum(ranks)
    w_plus = sum(r for r, di in zip(ranks, d) if di > 0)
    w = min(w_plus, total - w_plus)
    hits = 0
    for signs in itertools.product((0, 1), repeat=m):
        wp = sum(r for r, s in zip(ranks, signs) if s)
        hits += min(wp, total - wp) <= w + 1e-9
    return w, hits / 2**m, m


def wilcoxon_table(rows: list[dict]) -> list[dict]:
    """Per instance: GA vs Tabu costs paired by seed (only seeds where both are valid)."""
    out = []
    for instance in sorted({r["instance"] for r in rows}):
        ga = {r["seed"]: r["cost"] for r in rows
              if r["instance"] == instance and r["algorithm"] == "memetic_ga" and r["is_valid"]}
        ts = {r["seed"]: r["cost"] for r in rows
              if r["instance"] == instance and r["algorithm"] == "tabu_search" and r["is_valid"]}
        seeds = sorted(ga.keys() & ts.keys())
        if not seeds:
            continue
        x, y = [ga[s] for s in seeds], [ts[s] for s in seeds]
        w, p, m = wilcoxon_exact(x, y)
        out.append({"instance": instance, "pairs": len(seeds), "nonzero_pairs": m,
                    "ga_better": sum(a < b for a, b in zip(x, y)), "ts_better": sum(a > b for a, b in zip(x, y)),
                    "median_ga": statistics.median(x), "median_ts": statistics.median(y),
                    "W": w, "p_value": round(p, 4), "significant_5pct": p < 0.05})
    return out


# --------------------------------------------------------------------------- Bosphorus

def crossings(routes: list[list[int]], inst: ProblemInstance) -> tuple[int, int]:
    """(routes with at least one crossing arc, total crossing arcs); depot arcs included."""
    if inst.sides is None:
        return 0, 0
    crossing_routes = arcs = 0
    for r in routes:
        path = [0, *r, 0]
        k = sum(inst.sides[a] != inst.sides[b] for a, b in zip(path, path[1:]))
        arcs += k
        crossing_routes += k > 0
    return crossing_routes, arcs


def crossing_table(rows: list[dict], instances: dict[str, ProblemInstance]) -> list[dict]:
    """For each instance with sides and each algorithm: crossings of its best valid solution."""
    out = []
    for name, inst in instances.items():
        if inst.sides is None:
            continue
        for algorithm in ALGO_LABELS:
            row = best_valid_row(rows, name, algorithm)
            if row is None:
                continue
            sol = checked_solution(row, inst)
            cr, arcs = crossings(sol.routes, inst)
            eu_routes = sum(any(inst.sides[c] == "EU" for c in r) for r in sol.routes)
            out.append({"instance": name, "algorithm": algorithm, "seed": row["seed"], "cost": row["cost"],
                        "routes": len(sol.routes), "crossing_routes": cr, "crossing_arcs": arcs,
                        "routes_with_eu_customer": eu_routes,
                        "penalty_cost": arcs * inst.bosphorus_penalty,
                        "penalty_share_pct": round(100 * arcs * inst.bosphorus_penalty / row["cost"], 2)})
    return out


# --------------------------------------------------------------------------- parameter study

def tenure_study(inst: ProblemInstance, bks: int, tenures: list, seeds: list[int],
                 time_limit: float, start_fn) -> list[dict]:
    """Tabu gap per tenure setting (fixed int or (low, high)); same start and budget for all.

    ``start_fn(inst, seed)`` gives the start (the experiments use the CW + kick start).
    """
    out = []
    for tenure in tenures:
        for seed in seeds:
            res = run_tabu(inst, seed=seed, start=start_fn(inst, seed), max_iterations=10**9,
                           tenure=tuple(tenure) if isinstance(tenure, list) else tenure,
                           time_limit=time_limit)
            report = check_solution(res.best, inst)
            out.append({"instance": inst.name, "tenure": str(tenure), "seed": seed, "cost": res.best.cost,
                        "is_valid": report.is_valid, "gap_pct": round(100 * (res.best.cost - bks) / bks, 3),
                        "iterations": res.iterations, "time_limit_s": time_limit})
            print(f"tenure {str(tenure):9} seed {seed:2} cost {res.best.cost} valid {report.is_valid}", flush=True)
    return out


# --------------------------------------------------------------------------- plots

def _axes(figsize: tuple[float, float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return plt, fig, ax


def plot_convergence(rows: list[dict], instance: str, path: Path) -> None:
    """Best cost vs time for every GA and Tabu seed on one instance (step lines)."""
    plt, fig, ax = _axes((7, 4.3))
    sel = [r for r in rows if r["instance"] == instance and r["algorithm"] in ALGO_COLORS]
    if not sel:
        plt.close(fig)
        return
    for r in sel:
        trace = json.loads(r["trace"])
        if not trace:
            continue
        ts = [t for t, _ in trace] + [r["time_limit_s"] or trace[-1][0]]
        cs = [c for _, c in trace] + [trace[-1][1]]
        ax.plot(ts, cs, drawstyle="steps-post", color=ALGO_COLORS[r["algorithm"]], linewidth=1.2,
                alpha=0.75 if r["is_valid"] else 0.3, linestyle="-" if r["is_valid"] else ":")
    bks = sel[0]["bks"]
    if bks is not None:
        ax.axhline(bks, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(f"BKS {bks}", (ax.get_xlim()[1], bks), ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(left=0)
    ax.set_xlabel("seconds (symlog)", color=INK_2)
    ax.set_ylabel("best fleet-feasible cost so far", color=INK_2)
    ax.set_title(f"{instance}: convergence, one line per seed (dotted = run ended fleet-infeasible)",
                 color=INK, fontsize=10, loc="left")
    handles = [plt.Line2D([], [], color=c, linewidth=2) for c in ALGO_COLORS.values()]
    ax.legend(handles, [ALGO_LABELS[a] for a in ALGO_COLORS], frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_routes(sol: Solution, inst: ProblemInstance, title: str, path: Path) -> None:
    """Route map: colour + line style per route (8 hues x 3 styles), Bosphorus line at x = 500."""
    plt, fig, ax = _axes((7.2, 7.2))
    ax.grid(False)
    xy = inst.coords
    ax.axvline(500, color=INK_2, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(250, 1015, "Europe (x < 500)", ha="center", fontsize=9, color=INK_2)
    ax.text(750, 1015, "Asia (x >= 500)", ha="center", fontsize=9, color=INK_2)
    for k, r in enumerate(sol.routes):
        path_ = [0, *r, 0]
        style = ["-", "--", ":"][k // len(HUES) % 3]
        ax.plot([xy[i][0] for i in path_], [xy[i][1] for i in path_], color=HUES[k % len(HUES)],
                linewidth=1.6, linestyle=style, marker="o", markersize=4, markeredgecolor=SURFACE, zorder=2)
    ax.scatter([xy[0][0]], [xy[0][1]], marker="s", s=90, color=INK, zorder=3)
    ax.annotate("depot", xy[0], xytext=(8, -12), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(-20, 1020)
    ax.set_ylim(-20, 1040)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tenure(rows: list[dict], path: Path) -> None:
    """Mean gap (valid runs) per tenure setting, error bars = std over seeds, dots = seeds."""
    plt, fig, ax = _axes((7, 4))
    ax.grid(axis="x", visible=False)
    labels = list(dict.fromkeys(r["tenure"] for r in rows))
    for x, lab in enumerate(labels):
        gaps = [r["gap_pct"] for r in rows if r["tenure"] == lab and r["is_valid"]]
        if not gaps:
            ax.text(x, 0.2, "infeasible", rotation=90, ha="center", fontsize=7, color=INK_2)
            continue
        mean = statistics.mean(gaps)
        std = statistics.stdev(gaps) if len(gaps) > 1 else 0
        ax.bar(x, mean, 0.6, color="#1baf7a", yerr=[[min(std, mean)], [std]],
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
        ax.scatter([x] * len(gaps), gaps, s=14, color=INK, zorder=3)
        ax.annotate(f"{mean:.2f}", (x + 0.32, mean), va="center", ha="left", fontsize=8, color=INK)
    inst = rows[0]["instance"]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("tabu tenure (fixed, or [low, high] drawn per move)", color=INK_2)
    ax.set_ylabel("gap to BKS (%)", color=INK_2)
    ax.set_title(f"Tabu tenure sensitivity on {inst} ({rows[0]['time_limit_s']:.0f} s per run; dots = seeds)",
                 color=INK, fontsize=10, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
