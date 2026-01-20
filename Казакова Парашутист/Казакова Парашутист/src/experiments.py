"""Computer experiments for Lab #2.

Produces:
- >=2 plots with families of curves (3-5 curves each)
- an alternative diagram (bar chart)
- a summary table (CSV)
- verification report comparing numerical and theoretical terminal velocities

Run via:
  python -m src.run_all
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .model import Params, simulate, time_to_constant_speed, terminal_velocity


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_family_time_series(
    outpath: str,
    title: str,
    xlabel: str,
    ylabel: str,
    curves: List[Tuple[np.ndarray, np.ndarray, str]],
) -> None:
    """Generic helper to draw a family of curves on one chart."""
    plt.figure(figsize=(10, 6))
    for x, y, label in curves:
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_bar_chart(outpath: str, title: str, xlabel: str, ylabel: str, labels: List[str], values: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def verification_terminal_velocity(p: Params) -> pd.DataFrame:
    """Verify numerical steady-state against theoretical terminal velocity.

    We take the last sample from each phase and compare with v_term formula.
    """
    res = simulate(p)
    t = res["t"]
    v = res["v"]
    phase = res["phase"]

    # theoretical
    vterm1 = float(res["vterm1"][0])
    vterm2 = float(res["vterm2"][0])

    # numerical: take last value in each phase (if exists)
    idx1 = np.where(phase == 1)[0]
    idx2 = np.where(phase == 2)[0]

    v1_num = float(v[idx1[-1]]) if len(idx1) else float("nan")
    v2_num = float(v[idx2[-1]]) if len(idx2) else float("nan")

    df = pd.DataFrame(
        [
            {
                "phase": "без парашута",
                "v_term_theory_mps": vterm1,
                "v_last_numeric_mps": v1_num,
                "abs_error_mps": abs(v1_num - vterm1),
                "rel_error_%": (abs(v1_num - vterm1) / vterm1 * 100.0) if vterm1 > 0 else np.nan,
            },
            {
                "phase": "з парашутом",
                "v_term_theory_mps": vterm2,
                "v_last_numeric_mps": v2_num,
                "abs_error_mps": abs(v2_num - vterm2),
                "rel_error_%": (abs(v2_num - vterm2) / vterm2 * 100.0) if vterm2 > 0 else np.nan,
            },
        ]
    )
    return df


def run_all_experiments(out_dir: str = "outputs") -> None:
    ensure_dir(out_dir)

    # Base scenario (inside allowed ranges)
    base = Params(
        m=75.0,
        H=1.75,
        S=0.45,
        R=2.8,
        h0=3000.0,
        t_open=20.0,
        dt=0.02,
        eps_rel=0.01,
    )

    # Experiment 1: family of v(t) for different masses (5 curves)
    masses = [50, 60, 75, 90, 100]
    curves_v_mass = []
    summary_rows = []

    for m in masses:
        p = Params(**{**asdict(base), "m": float(m)})
        res = simulate(p)

        t = res["t"]
        v = res["v"]
        h = res["h"]
        phase = res["phase"]

        # time when v becomes practically constant in each phase
        vterm1 = float(res["vterm1"][0])
        vterm2 = float(res["vterm2"][0])
        idx_open = int(np.searchsorted(t, p.t_open))

        t_const1 = time_to_constant_speed(t, v, vterm1, p.eps_rel, start_index=0)
        t_const2 = time_to_constant_speed(t, v, vterm2, p.eps_rel, start_index=idx_open)

        curves_v_mass.append((t, v, f"m={m} кг"))

        # landing time & final speed
        t_land = float(t[-1])
        v_land = float(v[-1])

        summary_rows.append(
            {
                "scenario": "vary_mass",
                "m_kg": m,
                "H_m": p.H,
                "S_m": p.S,
                "R_m": p.R,
                "t_open_s": p.t_open,
                "v_term_free_mps": vterm1,
                "v_term_parachute_mps": vterm2,
                "t_const_free_s": t_const1,
                "t_const_parachute_s": t_const2,
                "t_land_s": t_land,
                "v_land_mps": v_land,
            }
        )

    plot_family_time_series(
        outpath=os.path.join(out_dir, "plot1_speed_vs_time_mass.png"),
        title="Швидкість падіння v(t) при різній масі (без/з парашутом)",
        xlabel="Час t, с",
        ylabel="Швидкість v, м/с",
        curves=curves_v_mass,
    )

    # Experiment 2: family of h(t) for different parachute radius (4 curves)
    radii = [2.2, 2.6, 3.0, 3.4]
    curves_h_r = []

    for R in radii:
        p = Params(**{**asdict(base), "R": float(R)})
        res = simulate(p)
        t = res["t"]
        h = res["h"]
        curves_h_r.append((t, h, f"R={R:.1f} м"))

        vterm1 = float(res["vterm1"][0])
        vterm2 = float(res["vterm2"][0])
        idx_open = int(np.searchsorted(t, p.t_open))
        t_const1 = time_to_constant_speed(t, res["v"], vterm1, p.eps_rel, start_index=0)
        t_const2 = time_to_constant_speed(t, res["v"], vterm2, p.eps_rel, start_index=idx_open)

        summary_rows.append(
            {
                "scenario": "vary_radius",
                "m_kg": p.m,
                "H_m": p.H,
                "S_m": p.S,
                "R_m": R,
                "t_open_s": p.t_open,
                "v_term_free_mps": vterm1,
                "v_term_parachute_mps": vterm2,
                "t_const_free_s": t_const1,
                "t_const_parachute_s": t_const2,
                "t_land_s": float(t[-1]),
                "v_land_mps": float(res["v"][-1]),
            }
        )

    plot_family_time_series(
        outpath=os.path.join(out_dir, "plot2_height_vs_time_radius.png"),
        title="Висота польоту h(t) при різному радіусі парашута",
        xlabel="Час t, с",
        ylabel="Висота h, м",
        curves=curves_h_r,
    )

    # Alternative diagram (bar chart): time to constant speed after opening vs R
    # Use the same radii set
    tconst2_vals = []
    labels = []
    for R in radii:
        p = Params(**{**asdict(base), "R": float(R)})
        res = simulate(p)
        t = res["t"]
        v = res["v"]
        vterm2 = float(res["vterm2"][0])
        idx_open = int(np.searchsorted(t, p.t_open))
        t_const2 = time_to_constant_speed(t, v, vterm2, p.eps_rel, start_index=idx_open)
        tconst2_vals.append(t_const2)
        labels.append(f"{R:.1f}")

    plot_bar_chart(
        outpath=os.path.join(out_dir, "diagram_time_to_const_vs_radius.png"),
        title="Час виходу на сталу швидкість після розкриття vs радіус парашута",
        xlabel="Радіус парашута R, м",
        ylabel="t_const (після розкриття), с",
        labels=labels,
        values=tconst2_vals,
    )

    # Summary table
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    # Verification
    df_ver = verification_terminal_velocity(base)
    df_ver.to_csv(os.path.join(out_dir, "verification_terminal_velocity.csv"), index=False)

    # Small text report
    report_path = os.path.join(out_dir, "report_short.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Лабораторна робота №2 — короткий звіт\n")
        f.write("================================\n\n")
        f.write("1) Комп'ютерні експерименти: побудовано 2 графіки з сімействами кривих\n")
        f.write("   - v(t) для 5 значень маси\n")
        f.write("   - h(t) для 4 значень радіуса парашута\n")
        f.write("2) Альтернативна діаграма: стовпчикова діаграма t_const після розкриття vs R\n")
        f.write("3) Підсумкова таблиця: summary_table.csv\n\n")
        f.write("Верифікація (порівняння з теорією):\n")
        f.write(df_ver.to_string(index=False))
        f.write("\n\n")
        f.write("Примітка: t_const визначено за критерієм |v - v_term|/v_term <= 1% (eps_rel=0.01).\n")

