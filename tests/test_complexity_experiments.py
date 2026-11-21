"""Tests for complexity_experiments.py (short budgets, tiny instances only).

Run from the project root:  python -m tests.test_complexity_experiments   (pytest also works)
"""

import csv
import tempfile
from pathlib import Path

from src.analysis import parse_routes, read_runs
from src.complexity_experiments import (
    add_exponents, budget_for, fit_exponent, gap_pct, load_instances, main, measure_scaling,
    plot_gaps, plot_medidrone, plot_scaling, run_experiments, summarize, tabu_start, write_csv,
)
from src.solution import Solution
from src.feasibility import check_solution

TINY = ["tiny-n7-s1", "tiny-n8-s2", "tiny-n9-s3"]


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def test_gap_pct():
    assert gap_pct(110, 100) == 10.0
    assert gap_pct(100, 100) == 0.0
    assert gap_pct(100, None) is None


def test_load_instances_bks():
    loaded = {inst.name: bks for inst, bks in load_instances()}
    assert loaded["tiny-n7-s1"] == 324 and loaded["medidrone-n100-s11372"] is None
    if "A-n32-k5" in loaded:
        assert loaded["A-n32-k5"] == 784


def test_run_experiments_tiny():
    rows = run_experiments(load_instances(TINY), seeds=[1, 2], time_limit=0.2, verbose=False)
    assert len(rows) == 3 * (1 + 2 * 2)  # CW once + GA and Tabu per seed, per instance
    for r in rows:
        assert r["is_valid"] and r["errors"] == "", r
        assert r["gap_pct"] >= 0  # BKS is the exact optimum: nothing can beat it
        if r["algorithm"] != "clarke_wright":
            assert r["runtime_s"] < 0.2 + 0.5 and r["iterations"] > 0
        # the stored routes are the reported solution: re-check them independently
        (inst, _), = load_instances([r["instance"]])
        assert check_solution(Solution(parse_routes(r["solution"]), r["cost"]), inst).is_valid


def test_budget_rule():
    assert budget_for(7) == budget_for(50) == 30 and budget_for(51) == budget_for(100) == 60


def test_tabu_start_per_seed():
    rows = run_experiments(load_instances(["tiny-n9-s3"]), [1, 2, 3], 0.1, verbose=False)
    starts = {r["seed"]: r["start"] for r in rows if r["algorithm"] == "tabu_search"}
    assert starts == {1: "cw_kick", 2: "cw_kick", 3: "cw_kick"}
    (inst, _), = load_instances(["medidrone-n100-s11372"])
    s2, s3 = tabu_start(inst, 2), tabu_start(inst, 3)
    assert s2.routes != s3.routes and check_solution(s2, inst).is_valid


def test_equal_seed_runs_repeat():
    # Same seed and enough budget to converge -> same cost (exact optimum on tiny).
    a = run_experiments(load_instances(["tiny-n7-s1"]), [3], 0.2, verbose=False)
    b = run_experiments(load_instances(["tiny-n7-s1"]), [3], 0.2, verbose=False)
    assert [r["cost"] for r in a] == [r["cost"] for r in b]


def test_summarize_by_hand():
    base = dict(instance="X", n=5, K=2, bks=100, runtime_s=1.0)
    rows = [
        dict(base, algorithm="ga", cost=110, gap_pct=10.0, is_valid=True, iterations=4),
        dict(base, algorithm="ga", cost=120, gap_pct=20.0, is_valid=True, iterations=6),
        dict(base, algorithm="ga", cost=90, gap_pct=-10.0, is_valid=False, iterations=5),  # ignored in cost stats
    ]
    (s,) = summarize(rows)
    assert s["runs"] == 3 and abs(s["feasible_rate"] - 2 / 3) < 1e-9 and s["bks_hits"] == 0
    assert (s["mean_cost"], s["best_cost"], s["worst_cost"]) == (115, 110, 120)
    assert s["std_cost"] == 7.071 and s["mean_gap_pct"] == 15 and s["best_gap_pct"] == 10
    assert s["mean_iterations"] == 5 and s["mean_runtime_s"] == 1


def test_summarize_no_valid_runs():
    rows = [dict(instance="X", algorithm="cw", n=5, K=2, bks=100, runtime_s=0.1,
                 cost=90, gap_pct=-10.0, is_valid=False, iterations=None)]
    (s,) = summarize(rows)
    assert s["feasible_rate"] == 0 and s["mean_cost"] is None and s["mean_gap_pct"] is None


def test_runs_csv_round_trip():
    rows = run_experiments(load_instances(["tiny-n7-s1"]), [1], 0.1, verbose=False)
    path = _tmp() / "runs.csv"
    write_csv(rows, path)
    back = read_runs(path)
    keys = ("algorithm", "seed", "cost", "is_valid", "bks", "gap_pct", "iterations", "solution", "trace")
    assert [{k: r[k] for k in keys} for r in back] == [{k: r[k] for k in keys} for r in rows]


def test_write_csv_round_trip():
    path = _tmp() / "out.csv"
    write_csv([{"a": 1, "b": None}, {"a": 2, "b": "x"}], path)
    with open(path, newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == [{"a": "1", "b": ""}, {"a": "2", "b": "x"}]


def test_fit_exponent():
    ns = [10, 20, 40, 80]
    assert abs(fit_exponent(ns, [3e-6 * n**2 for n in ns]) - 2) < 1e-9
    assert abs(fit_exponent(ns, [5e-4 * n for n in ns]) - 1) < 1e-9


def test_scaling_rows_and_plots():
    scaling = add_exponents(measure_scaling([20, 40], seed=1, reps=1))
    assert {r["step"] for r in scaling} == {"clarke_wright_run", "split_Q100_call", "split_Qinf_call",
                                           "tabu_iteration", "ga_generation", "relocate_delta_call",
                                           "full_cost_eval"}
    assert all(r["seconds"] > 0 and "fitted_exponent" in r for r in scaling)

    out = _tmp()
    rows = run_experiments(load_instances(["tiny-n7-s1", "medidrone-n100-s11372"]), [1], 0.3, verbose=False)
    plot_gaps(summarize(rows), out / "gap.png")
    plot_medidrone(rows, out / "med.png")
    plot_scaling(scaling, out / "scaling.png")
    assert all((out / f).stat().st_size > 0 for f in ("gap.png", "med.png", "scaling.png"))


def test_main_end_to_end():
    out = _tmp()
    main(["--time-limit", "0.1", "--seeds", "2", "--instances", "tiny-n7-s1", "medidrone-n100-s11372",
          "--sizes", "10", "20", "--study-instance", "tiny-n7-s1", "--study-time", "0.05",
          "--study-seeds", "1", "--out", str(out)])
    for name in ("runs.csv", "summary.csv", "scaling.csv", "wilcoxon.csv", "crossings.csv", "hardware.txt",
                 "tenure_study.csv", "gap_by_instance.png", "scaling.png", "tenure_study.png",
                 "routes_medidrone.png", "convergence_medidrone-n100-s11372.png"):
        assert (out / name).exists(), name
    before = (out / "summary.csv").read_text(encoding="utf-8")
    main(["--analysis-only", "--out", str(out)])  # rebuilt from runs.csv alone
    assert (out / "summary.csv").read_text(encoding="utf-8") == before


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")

