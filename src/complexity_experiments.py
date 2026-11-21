"""Experiments: equal-budget multi-seed comparison, runtime scaling, CSV and plots.

1. Equal-budget runs (``run_experiments``)
   Clarke-Wright runs once per instance (deterministic). The memetic GA and Tabu
   Search run once per seed with the SAME wall-clock budget and no iteration cap,
   so they compete on equal time. Budget (``budget_for``): 30 s for n <= 50, 60 s
   for larger n, unless ``--time-limit`` overrides it. Tabu starts from the
   Clarke-Wright solution plus a seeded kick of random relocates (``kick_start``),
   otherwise every seed would repeat the same trajectory.
   Each row stores the final routes and the best-cost-vs-time trace, so route plots,
   convergence plots and crossing counts are rebuilt from runs.csv.
   Every solution is checked with
   feasibility.py; cost statistics in the summary use valid runs only (rule 2),
   and the feasible rate shows how many runs were valid.
   gap % = 100 * (cost - BKS) / BKS, with BKS from the instance's .sol file
   (CVRPLIB best known, or the exact optimum for tiny instances). MediDrone has no BKS.

2. Runtime scaling (``measure_scaling``)
   On MediDrone instances of growing n, time one step of each algorithm and fit
   the empirical exponent b in  t ~ a * n^b  (least squares on log t vs log n):
       Clarke-Wright full run      theory O(n^2 log n)  -> b slightly above 2
       Split, Q = 100 (per call)   theory O(n * L)      -> b about 1 (L stays ~ Q / mean q)
       Split, Q = inf (per call)   theory O(n^2)        -> b about 2 (L = n)
       Tabu, one iteration         theory O(n^2)        -> b about 2
       GA, one generation          theory O(P * (n + nL + ls_prob * LS))  (P = population)
       one relocate delta          theory O(1)
       full cost re-evaluation     theory O(n)

3. Analysis (``src/analysis.py``): Wilcoxon test GA vs Tabu, tenure study,
   convergence and route plots, Bosphorus crossing counts, hardware info.

Outputs (default ``results/``): runs.csv, summary.csv, scaling.csv, wilcoxon.csv,
tenure_study.csv, crossings.csv, hardware.txt, gap_by_instance.png,
medidrone_costs.png, scaling.png, convergence_*.png, routes_medidrone.png,
tenure_study.png.

Usage:  python -m src.complexity_experiments --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

from src import analysis
from src.clarke_wright import clarke_wright
from src.feasibility import check_solution
from src.ga_split import run_ga, split
from src.parser_generator import ROOT, generate_medidrone, parse_cvrplib, parse_solution
from src.solution import ProblemInstance, Solution, solution_cost
from src.tabu_search import kick_start, relocate_delta, run_tabu

ALGORITHMS = ["clarke_wright", "memetic_ga", "tabu_search"]
COLORS = {"clarke_wright": "#2a78d6", "memetic_ga": "#eb6834", "tabu_search": "#1baf7a"}
LABELS = {"clarke_wright": "Clarke-Wright", "memetic_ga": "Memetic GA", "tabu_search": "Tabu Search"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DEFAULT_SEEDS = list(range(1, 11))
UNLIMITED = 10**9  # iteration cap high enough that only time_limit stops GA / Tabu
TENURES = [3, 7, 15, 30, 60, (7, 15)]  # tenure study: fixed values and the default dynamic range


# --------------------------------------------------------------------------- instances

def load_instances(names: list[str] | None = None) -> list[tuple[ProblemInstance, int | None]]:
    """All .vrp files in data/benchmarks, data/medidrone, data/tiny as (instance, BKS).

    BKS is the cost in the matching .sol file, or None if there is none.
    ``names`` keeps only instances with those names.
    """
    out = []
    for folder in ("benchmarks", "medidrone", "tiny"):
        for vrp in sorted((ROOT / "data" / folder).glob("*.vrp")):
            inst = parse_cvrplib(vrp)
            if names and inst.name not in names:
                continue
            sol = vrp.with_suffix(".sol")
            out.append((inst, parse_solution(sol).cost if sol.exists() else None))
    return out


def gap_pct(cost: int, bks: int | None) -> float | None:
    """Percentage gap to the best known solution; None without a BKS."""
    return None if bks is None else 100 * (cost - bks) / bks


# --------------------------------------------------------------------------- runs

def budget_for(n: int) -> float:
    """Wall-clock seconds per GA / Tabu run: 30 s for n <= 50, 60 s for larger n."""
    return 30.0 if n <= 50 else 60.0


def tabu_start(inst: ProblemInstance, seed: int) -> Solution:
    """Tabu starting solution for one run: Clarke-Wright plus a seeded kick (``kick_start``).

    The start is checked with feasibility.py: capacity, coverage and cost must hold;
    only a fleet excess (inherited from Clarke-Wright) is tolerated.
    """
    start = kick_start(inst, seed)
    errors = [e for e in check_solution(start, inst).errors if "fleet size" not in e]
    assert errors == [], errors
    return start


def routes_to_str(routes: list[list[int]]) -> str:
    """Routes as text for the CSV: customers separated by spaces, routes by '|'."""
    return "|".join(" ".join(map(str, r)) for r in routes)


def run_one(algorithm: str, inst: ProblemInstance, bks: int | None, seed: int | None,
            time_limit: float) -> dict:
    """Run one algorithm once and return a flat result row (validated with feasibility.py)."""
    t0 = time.perf_counter()
    trace: list[tuple[float, int]] = []
    if algorithm == "clarke_wright":
        sol, iterations, start_label = clarke_wright(inst), None, ""
    elif algorithm == "memetic_ga":
        res = run_ga(inst, seed=seed, generations=UNLIMITED, time_limit=time_limit)
        sol, iterations, start_label, trace = res.best, res.generations, "cw_plus_random_population", res.trace
    elif algorithm == "tabu_search":
        start, start_label = tabu_start(inst, seed), "cw_kick"
        res = run_tabu(inst, seed=seed, start=start, max_iterations=UNLIMITED, time_limit=time_limit)
        sol, iterations, trace = res.best, res.iterations, res.trace
    else:
        raise ValueError(f"unknown algorithm {algorithm}")
    runtime = time.perf_counter() - t0
    report = check_solution(sol, inst)
    gap = gap_pct(sol.cost, bks)
    return {
        "algorithm": algorithm,
        "instance": inst.name,
        "n": inst.n,
        "K": inst.num_vehicles,
        "seed": seed,
        "start": start_label,
        "time_limit_s": None if algorithm == "clarke_wright" else time_limit,
        "cost": sol.cost,
        "routes": len(sol.routes),
        "is_valid": report.is_valid,
        "errors": "; ".join(report.errors),
        "bks": bks,
        "gap_pct": None if gap is None else round(gap, 3),
        "runtime_s": round(runtime, 4),
        "iterations": iterations,
        "solution": routes_to_str(sol.routes),
        "trace": json.dumps([[round(t, 3), c] for t, c in trace]),
    }


def run_experiments(instances: list[tuple[ProblemInstance, int | None]], seeds: list[int],
                    time_limit: float | None = None, verbose: bool = True) -> list[dict]:
    """Clarke-Wright once per instance, GA and Tabu once per (instance, seed) with the same
    budget: ``time_limit`` if given, else ``budget_for(n)``."""
    rows = []
    for inst, bks in instances:
        limit = time_limit if time_limit is not None else budget_for(inst.n)
        jobs = [("clarke_wright", None)] + [(a, s) for a in ALGORITHMS[1:] for s in seeds]
        for algorithm, seed in jobs:
            row = run_one(algorithm, inst, bks, seed, limit)
            rows.append(row)
            if verbose:
                print(f"{row['instance']:22} {algorithm:14} seed {str(seed):4} cost {row['cost']:7} "
                      f"gap {row['gap_pct']!s:7} valid {row['is_valid']}", flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per (instance, algorithm): runs, feasible rate, and mean/std/best/worst of cost and gap
    over VALID runs only; ``bks_hits`` = valid runs whose cost equals the BKS (the exact
    optimum on tiny instances); mean runtime and iterations over all runs."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["instance"], r["algorithm"]), []).append(r)
    out = []
    for (instance, algorithm), rs in groups.items():
        valid = [r for r in rs if r["is_valid"]]
        costs = [r["cost"] for r in valid]
        gaps = [r["gap_pct"] for r in valid if r["gap_pct"] is not None]
        iters = [r["iterations"] for r in rs if r["iterations"] is not None]
        out.append({
            "instance": instance,
            "algorithm": algorithm,
            "n": rs[0]["n"],
            "K": rs[0]["K"],
            "bks": rs[0]["bks"],
            "runs": len(rs),
            "feasible_rate": len(valid) / len(rs),
            "bks_hits": sum(c == rs[0]["bks"] for c in costs) if rs[0]["bks"] is not None else None,
            "mean_cost": _r(statistics.mean(costs)) if costs else None,
            "std_cost": _r(statistics.stdev(costs)) if len(costs) > 1 else (0.0 if costs else None),
            "best_cost": min(costs) if costs else None,
            "worst_cost": max(costs) if costs else None,
            "mean_gap_pct": _r(statistics.mean(gaps)) if gaps else None,
            "std_gap_pct": _r(statistics.stdev(gaps)) if len(gaps) > 1 else (0.0 if gaps else None),
            "best_gap_pct": _r(min(gaps)) if gaps else None,
            "mean_runtime_s": _r(statistics.mean(r["runtime_s"] for r in rs)),
            "mean_iterations": _r(statistics.mean(iters)) if iters else None,
        })
    return out


def _r(x: float) -> float:
    return round(x, 3)


def write_csv(rows: list[dict], path: Path) -> None:
    """Write rows (same keys) to CSV; None is written as an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)


# --------------------------------------------------------------------------- scaling

def _best_time(fn, reps: int) -> float:
    """Minimum wall-clock time of ``reps`` calls (the minimum filters out OS noise)."""
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure_scaling(sizes: list[int], seed: int, reps: int = 3) -> list[dict]:
    """Time each algorithm step on MediDrone instances of size n (instance seed ``seed``)."""
    rows = []
    for n in sizes:
        inst, _ = generate_medidrone(seed=seed, n=n)
        tour = random.Random(seed).sample(range(1, n + 1), n)
        routes = split(tour, inst)
        mean_len = n / len(routes)
        unbounded = ProblemInstance(inst.name, sum(inst.demand), inst.demand, inst.dist, None,
                                    inst.coords, inst.bosphorus_penalty, inst.sides)
        cw = clarke_wright(inst)
        tabu_iters, ga_gens, evals = 5, 3, 1000
        paths = [[0, *r, 0] for r in cw.routes]
        steps = [
            ("clarke_wright_run", "O(n^2 log n)", lambda: clarke_wright(inst)),
            ("split_Q100_call", "O(n*L)", lambda: split(tour, inst)),
            ("split_Qinf_call", "O(n^2)", lambda: split(tour, unbounded)),
            ("tabu_iteration", "O(n^2)",
             lambda: run_tabu(inst, seed=seed, start=cw, max_iterations=tabu_iters)),
            ("ga_generation", "O(P*(n + nL + LS))", lambda: run_ga(inst, seed=seed, generations=ga_gens)),
            ("relocate_delta_call", "O(1)",
             lambda: [relocate_delta(paths[0], 1, paths[1], 1, inst.cost) for _ in range(evals)]),
            ("full_cost_eval", "O(n)", lambda: [solution_cost(cw.routes, inst) for _ in range(evals)]),
        ]
        for step, theory, fn in steps:
            t = _best_time(fn, reps)
            if step == "tabu_iteration":
                t /= tabu_iters
            elif step == "ga_generation":
                # generations=0 builds only the initial population; the difference is ga_gens generations
                t = max(t - _best_time(lambda: run_ga(inst, seed=seed, generations=0), reps), 1e-9) / ga_gens
            elif step in ("relocate_delta_call", "full_cost_eval"):
                t /= evals
            rows.append({"step": step, "theory": theory, "n": n, "seconds": t,
                         "mean_route_len": round(mean_len, 2) if step == "split_Q100_call" else None})
    return rows


def fit_exponent(ns: list[float], ts: list[float]) -> float:
    """Slope b of the least-squares line log t = log a + b log n."""
    slope, _ = statistics.linear_regression([math.log(n) for n in ns], [math.log(t) for t in ts])
    return slope


def add_exponents(scaling: list[dict]) -> list[dict]:
    """Add the fitted exponent of each step to every row of that step."""
    for step in {r["step"] for r in scaling}:
        rs = [r for r in scaling if r["step"] == step]
        b = round(fit_exponent([r["n"] for r in rs], [r["seconds"] for r in rs]), 2)
        for r in rs:
            r["fitted_exponent"] = b
    return scaling


# --------------------------------------------------------------------------- plots

def _style(ax) -> None:
    """Recessive grid and axes; data stays in front."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_gaps(summary: list[dict], path: Path) -> None:
    """Grouped bars: mean gap % (valid runs) per instance with a BKS, error bar = std over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    instances = sorted({s["instance"] for s in summary if s["bks"] is not None},
                       key=lambda name: next(s["n"] for s in summary if s["instance"] == name))
    if not instances:
        return
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(instances)), 4.2), facecolor=SURFACE)
    for k, algorithm in enumerate(ALGORITHMS):
        for x, name in enumerate(instances):
            s = next((s for s in summary if s["instance"] == name and s["algorithm"] == algorithm), None)
            pos = x + (k - 1) * width
            if s is None:
                continue
            if s["mean_gap_pct"] is None:  # no valid run: say so instead of drawing a bar
                ax.text(pos, 0.3, "infeasible", rotation=90, ha="center", va="bottom", fontsize=7, color=INK_2)
                continue
            std = s["std_gap_pct"] or 0
            ax.bar(pos, s["mean_gap_pct"], width - 0.03, color=COLORS[algorithm],
                   yerr=[[min(std, s["mean_gap_pct"])], [std]] if std else None,  # gap >= 0: clip at 0
                   error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2},
                   label=LABELS[algorithm] if x == 0 else None)
            if s["mean_gap_pct"] == 0:  # a zero-height bar is invisible: mark it
                ax.text(pos, 0.05, "0", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(instances)), instances, rotation=20, ha="right")
    ax.set_ylabel("gap to BKS (%)", color=INK_2)
    ax.set_title("Mean gap to best known solution (valid runs; bars = std over seeds)", color=INK, fontsize=11, loc="left")
    _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[a]) for a in ALGORITHMS]
    ax.legend(handles, [LABELS[a] for a in ALGORITHMS], frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_medidrone(rows: list[dict], path: Path) -> None:
    """Cost of every valid run on MediDrone instances, one column per algorithm (dots = seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({r["instance"] for r in rows if r["instance"].startswith("medidrone")})
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.2), facecolor=SURFACE, squeeze=False)
    for ax, name in zip(axes[0], names):
        for x, algorithm in enumerate(ALGORITHMS):
            costs = [r["cost"] for r in rows if r["instance"] == name and r["algorithm"] == algorithm and r["is_valid"]]
            if not costs:
                continue
            jitter = random.Random(x)
            ax.scatter([x + jitter.uniform(-0.12, 0.12) for _ in costs], costs, s=36,
                       color=COLORS[algorithm], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            mean = statistics.mean(costs)
            ax.hlines(mean, x - 0.25, x + 0.25, color=INK, linewidth=2, zorder=2)
            ax.annotate(f"{mean:.0f}", (x + 0.27, mean), va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(ALGORITHMS)), [LABELS[a] for a in ALGORITHMS])
        ax.set_xlim(-0.6, len(ALGORITHMS) - 0.3)
        ax.set_ylabel("total cost (distance + Bosphorus penalty)", color=INK_2)
        ax.set_title(f"{name}: cost per seed (line = mean)", color=INK, fontsize=11, loc="left")
        _style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


SCALING_PANELS = [
    ("Algorithm steps", [("clarke_wright_run", "#2a78d6", "o"), ("split_Q100_call", "#eb6834", "s"),
                         ("split_Qinf_call", "#1baf7a", "^"), ("tabu_iteration", "#6250d6", "D"),
                         ("ga_generation", "#eda100", "v")]),
    ("Delta vs full evaluation", [("relocate_delta_call", "#e87ba4", "o"), ("full_cost_eval", "#008300", "s")]),
]


def plot_scaling(scaling: list[dict], path: Path) -> None:
    """Log-log runtime vs n per step (two panels), with fitted exponent and theory in the legend."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    sizes = sorted({r["n"] for r in scaling})
    for ax, (title, steps) in zip(axes, SCALING_PANELS):
        for step, color, marker in steps:
            rs = sorted((r for r in scaling if r["step"] == step), key=lambda r: r["n"])
            if not rs:
                continue
            ax.plot([r["n"] for r in rs], [r["seconds"] for r in rs], color=color, linewidth=2,
                    marker=marker, markersize=6, markeredgecolor=SURFACE,
                    label=f"{step}: slope {rs[0]['fitted_exponent']} (theory {rs[0]['theory']})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("n (customers)", color=INK_2)
        ax.set_ylabel("seconds per call (best of reps)", color=INK_2)
        ax.set_title(title + " (log-log)", color=INK, fontsize=11, loc="left")
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.15))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main

CONVERGENCE_INSTANCES = ["A-n45-k6", "A-n60-k9", "medidrone-n100-s11372"]


def analyze(rows: list[dict], out: Path) -> None:
    """Summary, plots, Wilcoxon test, Bosphorus crossings and route map from run rows."""
    instances = {inst.name: inst for inst, _ in load_instances(sorted({r["instance"] for r in rows}))}
    summary = summarize(rows)
    write_csv(summary, out / "summary.csv")
    plot_gaps(summary, out / "gap_by_instance.png")
    plot_medidrone(rows, out / "medidrone_costs.png")
    wil = analysis.wilcoxon_table(rows)
    if wil:
        write_csv(wil, out / "wilcoxon.csv")
    cross = analysis.crossing_table(rows, instances)
    if cross:
        write_csv(cross, out / "crossings.csv")
    for name in CONVERGENCE_INSTANCES:
        if name in instances:
            analysis.plot_convergence(rows, name, out / f"convergence_{name}.png")
    for name, inst in instances.items():
        best = analysis.best_valid_row(rows, name)
        if name.startswith("medidrone") and best is not None:
            sol = analysis.checked_solution(best, inst)
            cr, arcs = analysis.crossings(sol.routes, inst)
            title = (f"{name}: best solution, {analysis.ALGO_LABELS[best['algorithm']]} seed {best['seed']}, "
                     f"cost {best['cost']}, {len(sol.routes)} routes, {cr} cross the Bosphorus")
            analysis.plot_routes(sol, inst, title, out / "routes_medidrone.png")
    print(f"wrote summary.csv, wilcoxon.csv, crossings.csv and plots to {out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; see module docstring."""
    p = argparse.ArgumentParser(description="CVRP experiments: equal-budget runs, scaling, CSV, plots.")
    p.add_argument("--time-limit", type=float, default=None,
                   help="seconds per GA / Tabu run (default: 30 s for n <= 50, 60 s otherwise)")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds, used as 1..N (default 10)")
    p.add_argument("--instances", nargs="*", help="instance names to run (default: all in data/)")
    p.add_argument("--sizes", type=int, nargs="*", default=[25, 50, 100, 200, 400],
                   help="n values for the scaling study")
    p.add_argument("--scaling-seed", type=int, default=11372, help="MediDrone seed for scaling instances")
    p.add_argument("--study-instance", default="A-n60-k9", help="instance for the tenure study")
    p.add_argument("--study-time", type=float, default=20.0, help="seconds per tenure-study run")
    p.add_argument("--study-seeds", type=int, default=5, help="seeds 1..N per tenure setting")
    p.add_argument("--skip-runs", action="store_true", help="do not run the equal-budget experiments")
    p.add_argument("--analysis-only", action="store_true",
                   help="rebuild summary, tests and plots from an existing runs.csv (no new runs)")
    p.add_argument("--skip-study", action="store_true", help="do not run the tenure study")
    p.add_argument("--skip-scaling", action="store_true", help="do not run the scaling study")
    p.add_argument("--out", type=Path, default=ROOT / "results", help="output folder (default results/)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hardware.txt").write_text(analysis.hardware_info(), encoding="utf-8")

    if args.analysis_only:
        analyze(analysis.read_runs(args.out / "runs.csv"), args.out)
        if (args.out / "tenure_study.csv").exists():
            analysis.plot_tenure(analysis.read_tenure_study(args.out / "tenure_study.csv"),
                                 args.out / "tenure_study.png")
        return
    if not args.skip_runs:
        rows = run_experiments(load_instances(args.instances), list(range(1, args.seeds + 1)), args.time_limit)
        write_csv(rows, args.out / "runs.csv")
        analyze(rows, args.out)
    if not args.skip_study:
        loaded = load_instances([args.study_instance])
        if loaded:
            (inst, bks), = loaded
            study = analysis.tenure_study(inst, bks, TENURES, list(range(1, args.study_seeds + 1)),
                                          args.study_time, tabu_start)
            write_csv(study, args.out / "tenure_study.csv")
            analysis.plot_tenure(study, args.out / "tenure_study.png")
            print(f"wrote tenure_study.csv, tenure_study.png to {args.out}", flush=True)
    if not args.skip_scaling:
        scaling = add_exponents(measure_scaling(args.sizes, args.scaling_seed))
        write_csv(scaling, args.out / "scaling.csv")
        plot_scaling(scaling, args.out / "scaling.png")
        for step in sorted({r["step"] for r in scaling}):
            r = next(r for r in scaling if r["step"] == step)
            print(f"{step:20} fitted exponent {r['fitted_exponent']:5}  theory {r['theory']}")
        print(f"wrote scaling.csv, scaling.png to {args.out}", flush=True)


if __name__ == "__main__":
    main()

