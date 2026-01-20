from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Lab3Params, Lab4Params
from .lab3_sim import run_lab3
from .lab4_analytic import analytic_table_for_X, mean_batch_service_time_minutes
from .lab4_mc import run_lab4_mc
from .search_x import find_min_X


def _ensure_out_dir() -> Path:
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_lab3_and_save(out: Path) -> None:
    p3 = Lab3Params()
    parts, summary = run_lab3(seed=20250119, params=p3)

    summary.to_csv(out / "lab3_summary.csv", index=False)
    parts.to_csv(out / "lab3_parts.csv", index=False)

    # Plot: lead time histogram
    plt.figure()
    if not parts.empty:
        plt.hist(parts["lead_time"], bins=25)
    plt.title("Lab3: Lead time distribution (minutes)")
    plt.xlabel("Lead time, min")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "lab3_plot_lead_time_hist.png", dpi=180)
    plt.close()

    # Plot: cumulative finished parts over time
    plt.figure()
    if not parts.empty:
        s = parts.sort_values("finished_at")
        y = np.arange(1, len(s) + 1)
        plt.plot(s["finished_at"], y)
    plt.title("Lab3: Cumulative finished parts vs time")
    plt.xlabel("Time, min")
    plt.ylabel("Finished parts, pcs")
    plt.tight_layout()
    plt.savefig(out / "lab3_plot_cumulative_finished.png", dpi=180)
    plt.close()


def run_lab4_and_save(out: Path) -> None:
    p4 = Lab4Params()

    # --- Search X (Monte-Carlo) ---
    best, hist = find_min_X(p4, X_low=4.0, X_high=40.0, tol=0.5, max_iter=10)
    hist.to_csv(out / "lab4_search_x_history.csv", index=False)

    # Run MC at best X and at a few nearby values to build a "family" curve
    X_grid = sorted({round(best.X + d, 1) for d in [-4, -2, 0, 2, 4] if best.X + d > 0})
    mc_rows = []
    for X in X_grid:
        mc = run_lab4_mc(p4, X=X, runs=p4.mc_runs, seed0=30000 + int(X * 10))
        mc["X"] = X
        mc_rows.append(mc)
    mc_df = pd.concat(mc_rows, ignore_index=True)

    # Summary per X
    grp = mc_df.groupby("X").agg(
        pass_fraction=("max_queue", lambda s: float((s <= p4.queue_limit).mean())),
        qmax_mean=("max_queue", "mean"),
        qmax_p95=("max_queue", lambda s: float(np.percentile(s, 95))),
        mean_wait=("mean_wait_min", "mean"),
        mean_service=("mean_service_min", "mean"),
    ).reset_index()

    grp.to_csv(out / "lab4_search_x_results.csv", index=False)

    # --- Analytic model for same X values ---
    analytic = analytic_table_for_X(p4, X_values=list(grp["X"].values))
    comp = pd.DataFrame({"X": grp["X"]})
    comp = comp.merge(grp, on="X").merge(analytic.rename(columns={"X_mean_interarrival_min": "X"}), on="X")
    comp.to_csv(out / "lab4_analytic_vs_mc.csv", index=False)

    # Plot 1 (family): Qmax distributions per X (boxplot)
    plt.figure()
    data = [mc_df.loc[mc_df["X"] == X, "max_queue"].values for X in grp["X"]]
    plt.boxplot(data, labels=[str(X) for X in grp["X"]])
    plt.title("Lab4: Max queue per shift for different X (100 runs each)")
    plt.xlabel("X (mean interarrival), min")
    plt.ylabel("Max queue length, batches")
    plt.tight_layout()
    plt.savefig(out / "lab4_plot_max_queue_boxplot.png", dpi=180)
    plt.close()

    # Plot 2 (family): pass fraction vs X and p95 of max queue vs X
    plt.figure()
    plt.plot(grp["X"], grp["pass_fraction"], marker="o")
    plt.axhline(p4.pass_fraction)
    plt.title("Lab4: P(Qmax <= 20) vs X")
    plt.xlabel("X (mean interarrival), min")
    plt.ylabel("Pass fraction")
    plt.tight_layout()
    plt.savefig(out / "lab4_plot_pass_fraction.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(grp["X"], grp["qmax_p95"], marker="o")
    plt.axhline(p4.queue_limit)
    plt.title("Lab4: 95th percentile of Qmax vs X")
    plt.xlabel("X (mean interarrival), min")
    plt.ylabel("Qmax 95th percentile, batches")
    plt.tight_layout()
    plt.savefig(out / "lab4_plot_qmax_p95.png", dpi=180)
    plt.close()

    # Plot 3: analytic Lq vs MC mean max queue (different metrics, but trend check)
    plt.figure()
    plt.plot(comp["X"], comp["Lq"], marker="o")
    plt.title("Lab4 analytic (M/M/4): mean queue Lq vs X")
    plt.xlabel("X (mean interarrival), min")
    plt.ylabel("Lq (analytic), batches")
    plt.tight_layout()
    plt.savefig(out / "lab4_plot_analytic_Lq.png", dpi=180)
    plt.close()

    # Small text note
    meanS = mean_batch_service_time_minutes(p4)
    with open(out / "lab4_note.txt", "w", encoding="utf-8") as f:
        f.write("Lab4 найденное X (минимальное, по критерию pass_fraction>=0.95):\n")
        f.write(f"X* = {best.X:.2f} мин\n")
        f.write(f"Pass fraction = {best.pass_fraction:.3f}\n")
        f.write(f"Mean(Qmax) = {best.qmax_mean:.2f}\n")
        f.write(f"P95(Qmax) = {best.qmax_p95:.2f}\n\n")
        f.write(f"Analytic mean batch service time S̄ ≈ {meanS:.2f} мин (used for M/M/4 approximation).\n")


def main() -> None:
    out = _ensure_out_dir()
    run_lab3_and_save(out)
    run_lab4_and_save(out)
    print(f"Done. See outputs in: {out}")


if __name__ == "__main__":
    main()
